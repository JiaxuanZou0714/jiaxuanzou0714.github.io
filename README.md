# Jiaxuan Zou

Source for [jiaxuanzou0714.github.io](https://jiaxuanzou0714.github.io), an academic personal site built with Jekyll and the [al-folio](https://github.com/alshedivat/al-folio) theme.

## Local development

Docker is the supported local workflow:

```bash
docker compose pull
docker compose up
```

The site is available at <http://localhost:8080>.

## Deployment

Pushes to `main` or `master` trigger [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml), which builds the production site, runs PurgeCSS, and publishes `_site` to the `gh-pages` branch.

## License

Theme code remains available under the terms in [LICENSE](LICENSE).
