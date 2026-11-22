#!/usr/bin/env python3
"""Subset local icon fonts after PurgeCSS, then fingerprint the final CSS.

Run from the repository root after `bundle exec jekyll build` and PurgeCSS.
Original fonts stay in assets/ so every build can include newly used icons.
"""

import hashlib
import io
from pathlib import Path
import re
import sys

from fontTools import subset
from fontTools.ttLib import TTFont


ROOT = Path(__file__).resolve().parents[1]
FONTS = {
    "fa-solid-900": "assets/webfonts/fa-solid-900.woff2",
    "fa-brands-400": "assets/webfonts/fa-brands-400.woff2",
    "fa-regular-400": "assets/webfonts/fa-regular-400.woff2",
    "academicons": "assets/fonts/academicons.ttf",
    "scholar-icons": "assets/fonts/scholar-icons.ttf",
}
FONT_FACE = re.compile(r"@font-face\s*\{[^{}]*\}", re.I)


def fingerprint(data):
    return hashlib.sha256(data).hexdigest()[:16]


def optimize(site):
    styles = sorted((site / "assets/css").glob("*.css"))
    if not styles:
        raise SystemExit(f"No built CSS in {site}; run the site build first.")
    css = {path: path.read_text() for path in styles}
    # PurgeCSS keeps rules used by any generated HTML or local JavaScript,
    # including icons revealed by interactions. Retain all their code points.
    combined = "\n".join(css.values())
    codepoints = {int(value, 16) for value in re.findall(r"\\([0-9a-fA-F]{1,6})", combined)}
    codepoints.update(ord(char) for char in combined if ord(char) > 127)
    destinations = {}
    reports = []
    output_dir = site / "assets/fonts/subsets"
    output_dir.mkdir(parents=True, exist_ok=True)

    def replace_face(match):
        face = match.group()
        name = next((name for name in FONTS if re.search(
            rf"/{re.escape(name)}(?:\.[0-9a-f]{{16}})?\.(?:woff2?|ttf|eot|svg)", face
        )), None)
        if name is None:
            return face
        if name not in destinations:
            source = ROOT / FONTS[name]
            font = TTFont(source, recalcTimestamp=False)
            required = codepoints & set(font.getBestCmap())
            if not required:
                # No retained CSS rule uses this font.
                font.close()
                destinations[name] = None
                return ""
            options = subset.Options()
            options.name_IDs = ["*"]  # Preserve copyright and license metadata.
            subsetter = subset.Subsetter(options=options)
            subsetter.populate(unicodes=required)
            subsetter.subset(font)
            font.flavor = "woff2"
            buffer = io.BytesIO()
            font.save(buffer)
            font.close()
            data = buffer.getvalue()
            # Fail the build if subsetting loses a requested glyph.
            with TTFont(io.BytesIO(data)) as result:
                missing = required - set(result.getBestCmap())
                if missing:
                    raise ValueError(f"Missing code points in {name}: {missing}")
            filename = f"{name}.{fingerprint(data)}.woff2"
            (output_dir / filename).write_bytes(data)
            destinations[name] = filename
            reports.append((name, len(required), source.stat().st_size, len(data)))
        filename = destinations[name]
        if filename is None:
            return ""
        # Replace all legacy fallback formats, including their format hints.
        face = re.sub(r"src\s*:[^;}]+;?", "", face)
        return face[:-1].rstrip().rstrip(";") + f';src:url("../fonts/subsets/{filename}") format("woff2");}}'

    versions = {}
    for path, content in css.items():
        optimized = FONT_FACE.sub(replace_face, content)
        path.write_text(optimized)
        versions[path.name] = fingerprint(optimized.encode())

    # Jekyll fingerprints source CSS before PurgeCSS and font rewriting.
    # Fingerprint the final files so newly used icons invalidate cached CSS.
    css_link = re.compile(r'(assets/css/)([^/\s"\'<>?]+\.css)(?:\?v=[^\s"\'<>]+)?')
    for path in site.rglob("*.html"):
        original = path.read_text()
        updated = css_link.sub(lambda m: (
            f"{m[1]}{m[2]}?v={versions[m[2]]}" if m[2] in versions else m[0]
        ), original)
        if updated != original:
            path.write_text(updated)
    for name, glyphs, before, after in reports:
        print(f"{name}: {glyphs} code points, {before:,} -> {after:,} bytes")


if __name__ == "__main__":
    optimize(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "_site")
