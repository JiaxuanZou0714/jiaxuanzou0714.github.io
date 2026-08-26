# Jiaxuan Zou

Source for [jiaxuanzou0714.github.io](https://jiaxuanzou0714.github.io), an academic personal site built with Jekyll and the [al-folio](https://github.com/alshedivat/al-folio) theme.

## Local development

Docker is the supported local workflow:

```bash
docker compose pull
docker compose up
```

The site is available at <http://localhost:8080>.

## Post translations

Posts are written in Chinese. [`scripts/translate.mjs`](scripts/translate.mjs) renders English
versions into `_en_posts/`, served under `/en/blog/`. Math, code, Liquid tags, and link targets are
withheld from the model and restored afterwards; every markdown block is cached by content hash in
`_data/i18n/zh-en.json`, so editing one paragraph re-translates one paragraph.

Both the generated markdown and the cache are committed. Deployment therefore needs no API key and
makes no API calls, and a translation the model gets wrong can be corrected by editing the file in
`_en_posts/` directly.

```bash
cp .env.example .env      # then add DEEPSEEK_API_KEY
npm run translate:check   # structure check, no API calls
npm run translate
```

The same run is available as a manual GitHub Action,
[`.github/workflows/translate.yml`](.github/workflows/translate.yml), which reads the key from the
`DEEPSEEK_API_KEY` repository secret and commits the result.

## Deployment

Pushes to `main` or `master` trigger [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml), which builds the production site, runs PurgeCSS, and publishes `_site` to the `gh-pages` branch.

## License

Theme code remains available under the terms in [LICENSE](LICENSE).
