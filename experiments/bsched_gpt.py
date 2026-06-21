"""
Real-transformer test of batch-size scheduling on FineWeb (GPT-2 tokens).

A small, standard GPT trained with AdamW at a FIXED constant learning rate.
The only thing that varies between runs is the batch-size schedule b(t)
(tokens per optimizer step), at a MATCHED total token budget D. This breaks the
circularity of the noisy-quadratic surrogate: the network is not constructed to
satisfy the FSL model.

Reads modded-nanogpt's FineWeb .bin shards (256 int32 header, then uint16 tokens).

Usage (single GPU):
  python bsched_gpt.py --schedule const --batch_tokens 524288 --out logs/const_hi.json --device cuda:0
  python bsched_gpt.py --schedule const --batch_tokens 65536  --out logs/const_lo.json --device cuda:1
  python bsched_gpt.py --schedule ramp                          --out logs/ramp.json    --device cuda:3
"""
import os, glob, json, time, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# data: FineWeb .bin shards
# --------------------------------------------------------------------------- #
def load_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520 and header[1] == 1, f"bad header {path}"
    ntok = int(header[2])
    toks = np.fromfile(path, dtype=np.uint16, count=ntok, offset=256 * 4)
    assert len(toks) == ntok
    return toks


class Loader:
    """Sequential contiguous (n_seq, T) batches over concatenated shards."""
    def __init__(self, pattern, T):
        self.files = sorted(glob.glob(pattern))
        assert self.files, f"no files: {pattern}"
        self.T = T
        self.fi = 0
        self.tok = load_shard(self.files[0])
        self.pos = 0

    def next(self, n_seq):
        need = n_seq * self.T + 1
        if self.pos + need > len(self.tok):
            self.fi = (self.fi + 1) % len(self.files)
            self.tok = load_shard(self.files[self.fi])
            self.pos = 0
        buf = torch.from_numpy(self.tok[self.pos:self.pos + need].astype(np.int64))
        self.pos += n_seq * self.T
        x = buf[:-1].view(n_seq, self.T)
        y = buf[1:].view(n_seq, self.T)
        return x, y


# --------------------------------------------------------------------------- #
# model: standard GPT
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.ln2 = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 4 * dim, bias=False)
        self.fcproj = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(a)
        x = x + self.fcproj(F.gelu(self.fc(self.ln2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, dim, n_layer, n_head, block_size):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab, dim)
        self.wpe = nn.Embedding(block_size, dim)
        self.blocks = nn.ModuleList([Block(dim, n_head) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.wte.weight = self.head.weight  # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None]
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.lnf(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# --------------------------------------------------------------------------- #
# batch-size schedule: returns tokens-per-optimizer-step until budget D is spent
# --------------------------------------------------------------------------- #
def make_schedule(kind, D, batch_tokens, b_min, b_max, ramp_fracs):
    """Yield b_k (tokens this step). All schedules spend ~D tokens total."""
    sched = []
    if kind == "const":
        spent = 0
        while spent < D:
            sched.append(batch_tokens)
            spent += batch_tokens
    elif kind == "ramp":
        # late-switch doubling: small batch carries most tokens, then ramps up.
        stages, b = [], b_min
        while b < b_max:
            stages.append(b); b *= 2
        stages.append(b_max)
        fracs = ramp_fracs[:len(stages)]
        fracs = [f / sum(fracs) for f in fracs]
        for b, fr in zip(stages, fracs):
            spent_stage, target = 0, fr * D
            while spent_stage < target:
                sched.append(b); spent_stage += b
    else:
        raise ValueError(kind)
    return sched


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", choices=["const", "ramp"], default="const")
    ap.add_argument("--batch_tokens", type=int, default=524288)   # const batch
    ap.add_argument("--b_min", type=int, default=65536)           # ramp endpoints
    ap.add_argument("--b_max", type=int, default=524288)
    ap.add_argument("--ramp_fracs", type=float, nargs="+",
                    default=[0.50, 0.25, 0.15, 0.10])
    ap.add_argument("--total_tokens", type=float, default=3.0e8)
    ap.add_argument("--lr", type=float, default=3e-4)             # constant
    ap.add_argument("--warmup_tokens", type=float, default=8e6)
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--micro_seq", type=int, default=32)          # seqs per fwd/bwd
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=50304)
    ap.add_argument("--eval_tokens", type=float, default=1.0e7)   # eval cadence
    ap.add_argument("--val_tokens", type=int, default=2_000_000)  # val set size
    ap.add_argument("--data", default="data/fineweb10B")
    ap.add_argument("--out", default="logs/run.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision("high")
    dev = torch.device(args.device)
    T = args.block_size
    micro_tokens = args.micro_seq * T

    D = int(args.total_tokens)
    sched = make_schedule(args.schedule, D, args.batch_tokens,
                          args.b_min, args.b_max, args.ramp_fracs)
    for b in sched:
        assert b % micro_tokens == 0, f"batch {b} not multiple of micro {micro_tokens}"

    model = GPT(args.vocab, args.dim, args.n_layer, args.n_head, T).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    n_params_nonemb = n_params - model.wte.weight.numel() - model.wpe.weight.numel()

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": 0.1},
        {"params": nodecay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)

    train = Loader(os.path.join(args.data, "fineweb_train_*.bin"), T)
    # fixed val set: first val_tokens of the val shard
    val_tok = load_shard(sorted(glob.glob(os.path.join(args.data, "fineweb_val_*.bin")))[0])
    val_tok = torch.from_numpy(val_tok[:args.val_tokens].astype(np.int64))

    @torch.no_grad()
    def eval_val():
        model.eval()
        losses, n = 0.0, 0
        for i in range(0, len(val_tok) - 1 - T, args.micro_seq * T):
            chunk = val_tok[i:i + args.micro_seq * T + 1]
            if len(chunk) < args.micro_seq * T + 1:
                break
            x = chunk[:-1].view(args.micro_seq, T).to(dev)
            y = chunk[1:].view(args.micro_seq, T).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, l = model(x, y)
            losses += l.item(); n += 1
        model.train()
        return losses / max(n, 1)

    log = {"args": vars(args), "n_params": n_params, "n_params_nonemb": n_params_nonemb,
           "records": []}
    consumed = 0
    next_eval = 0
    ema = None
    t0 = time.time()
    print(f"[{args.out}] {args.schedule} params={n_params/1e6:.1f}M "
          f"(non-emb {n_params_nonemb/1e6:.1f}M) steps={len(sched)} D={D/1e6:.0f}M", flush=True)

    for step, b in enumerate(sched):
        # constant lr with token-space warmup (identical across runs)
        lr = args.lr * min(1.0, consumed / args.warmup_tokens) if args.warmup_tokens > 0 else args.lr
        for g in opt.param_groups:
            g["lr"] = lr

        accum = b // micro_tokens
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(accum):
            x, y = train.next(args.micro_seq)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / accum).backward()
            step_loss += loss.item() / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        consumed += b
        ema = step_loss if ema is None else 0.9 * ema + 0.1 * step_loss

        if consumed >= next_eval or step == len(sched) - 1:
            vl = eval_val()
            rec = {"tokens": consumed, "step": step + 1, "batch": b,
                   "train_ema": ema, "val": vl, "lr": lr, "sec": time.time() - t0}
            log["records"].append(rec)
            print(f"  t={consumed/1e6:7.1f}M step={step+1:5d} b={b:7d} "
                  f"train={ema:.4f} val={vl:.4f} ({time.time()-t0:.0f}s)", flush=True)
            next_eval += int(args.eval_tokens)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(log, f)
    print(f"[{args.out}] done. final val={log['records'][-1]['val']:.4f} "
          f"steps={len(sched)} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
