# AccentCL Demo

Project page for **AccentCL: Robust Accent Classification with Incremental Expansion**.

Built with Vue 3 + Vite + Bootstrap, deployed to GitHub Pages.

## Filling in real content

- **Figures**: replace `public/figures/architecture-placeholder.svg` and
  `public/figures/results-placeholder.svg` (or point `src/PaperView.vue` at new filenames).
- **Audio demo**: drop `.wav` files into `public/demo/<stage>/<accent-slug>/`, where
  `<stage>` is one of `base`, `spanish`, `chinese`, matching the sample IDs and accent
  class names defined in `src/PaperView.vue` (`samplesByStageAndClass`, `baseClasses`, `stages`).
- **Predictions**: update the placeholder `predicted` / `confidence` fields in
  `src/PaperView.vue` with real model outputs.
- **Links**: fill in the paper/code links in the header, and the BibTeX citation block.

## Project Setup

```sh
npm install
```

### Compile and Hot-Reload for Development

```sh
npm run dev
```

### Compile and Minify for Production

```sh
npm run build
```

### Deploy to GitHub Pages

Push to `main` — the workflow in `.github/workflows/deploy.yml` builds and publishes
`dist/` automatically. Make sure GitHub Pages is set to "GitHub Actions" as the source
in the repo settings.
