<template>
  <div class="page">
    <!-- Hero -->
    <header class="hero">
      <div class="wrap">
        <h1 class="title">
          <span class="title-name">AccentCL</span>: Robust Accent Classification with Incremental
          Expansion
        </h1>

        <div class="authors">
          <span v-for="author in authors" :key="author.name" class="author">
            <a :href="author.url" target="_blank" rel="noopener noreferrer">{{ author.name }}</a
            ><sup>1</sup>
          </span>
        </div>
        <div class="affiliations"><sup>1</sup>Texas A&amp;M University</div>

        <div class="venue">
          <span class="venue-name">IEEE Spoken Language Technology (SLT) 2026</span>
        </div>

        <nav class="links">
          <template v-for="link in links" :key="link.label">
            <span v-if="!link.href" class="link-btn link-btn-disabled" aria-disabled="true">
              <svg viewBox="0 0 16 16" aria-hidden="true"><path :d="link.icon" /></svg>
              <span>{{ link.label }}</span>
              <span class="soon">Soon</span>
            </span>
            <a
              v-else
              :href="link.href"
              :target="link.external ? '_blank' : undefined"
              :rel="link.external ? 'noopener noreferrer' : undefined"
              class="link-btn"
            >
              <svg viewBox="0 0 16 16" aria-hidden="true"><path :d="link.icon" /></svg>
              <span>{{ link.label }}</span>
            </a>
          </template>
        </nav>
      </div>
    </header>

    <main>
      <!-- Abstract -->
      <section class="section">
        <div class="wrap">
          <h2 class="section-title">Abstract</h2>
          <p class="abstract">
            Accent classifiers are typically trained with a fixed label inventory and cannot
            accommodate new accent categories as new data becomes available. Moreover, accented
            speech corpora often exhibit substantial class imbalance and/or domain shift due to
            differences in recording conditions across corpora. We present AccentCL, a
            class-incremental learning framework for English accent classification that is robust to
            class imbalance and cross-corpus domain shift. AccentCL extracts multi-layer
            representations from a frozen Whisper-Large-v3 encoder, optimized with an
            imbalance-aware cross-entropy loss to reduce bias toward the majority accent classes and
            a domain mean alignment loss that minimizes distributional mean shift across training
            corpora. The label space is then expanded via replay-based continual learning, using the
            frozen base model for knowledge retention and an old-to-new margin loss to reduce
            overprediction on newly added classes. On a five-class accent classification task,
            AccentCL achieves 77.1% balanced accuracy and a 76.9% macro-averaged F1 score. We
            further evaluate the model's ability to incrementally incorporate two new accent
            categories: Spanish-accented and Chinese-accented English. When adding Spanish-accented
            English to the pretrained model, AccentCL attains an F1 of 83.3% on the new class while
            retaining 77.3% balanced accuracy on the base classes. When subsequently adding
            Chinese-accented English, it achieves 61.8% F1 on the new class while preserving 77.6%
            balanced accuracy on the previously learned classes. These results show that AccentCL
            enables robust regional accent classification while allowing new accent categories to be
            added without full retraining.
          </p>
        </div>
      </section>

      <!-- Method -->
      <section class="section section-alt" id="method">
        <div class="wrap wrap-wide">
          <h2 class="section-title">Method</h2>
          <figure class="figure">
            <img
              :src="getAssetUrl('/figures/thumbnail.svg')"
              alt="AccentCL model architecture"
              loading="lazy"
            />
            <figcaption>
              <b>Figure 1.</b> Model architecture of AccentCL. A frozen Whisper-Large-v3 encoder
              provides multi-layer speech representations, which are projected, concatenated, and
              pooled with attentive statistics pooling to produce an accent embedding.
              <b>Phase 1</b> trains a base regional accent classifier with domain mean alignment.
              <b>Phase 2</b> expands the classifier to a new accent class using replay-based
              continual training with retention and an old-to-new margin loss.
            </figcaption>
          </figure>
        </div>
      </section>

      <!-- Results -->
      <section class="section" id="results">
        <div class="wrap wrap-wide">
          <h2 class="section-title">Results</h2>
          <p class="section-lead">
            A five-class regional accent classifier, expanded to seven classes through two
            class-incremental steps.
          </p>

          <!-- Table I -->
          <h3 class="subsection-title">Robust regional accent classification</h3>
          <div class="table-wrap">
            <table class="results-table">
              <thead>
                <tr>
                  <th rowspan="2">Method</th>
                  <th colspan="3" class="group-th">All</th>
                  <th colspan="3" class="group-th">OOD</th>
                </tr>
                <tr>
                  <th class="num">Acc ↑</th>
                  <th class="num">Bal Acc ↑</th>
                  <th class="num">Macro-F1 ↑</th>
                  <th class="num">Acc ↑</th>
                  <th class="num">Bal Acc ↑</th>
                  <th class="num">Macro-F1 ↑</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="row in regionalRows"
                  :key="row.method"
                  :class="{ 'row-ours': row.ours, 'row-ablation': row.ablation }"
                >
                  <td>{{ row.method }}</td>
                  <td
                    v-for="(v, i) in row.values"
                    :key="i"
                    class="num"
                    :class="{ best: v === regionalBest[i] }"
                  >
                    {{ v.toFixed(1) }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <p class="table-caption">
            <b>Table 1.</b> Results on the shared 5-class regional label space (%). <b>All</b> uses
            every in-label test utterance from our benchmark; <b>OOD</b> uses source-held-out Speech
            Accent Archive and IDEA, which no model trains on. CommonAccent and Voxlect predictions
            are mapped to the same regional labels, and predictions outside the label set count as
            errors.
          </p>
          <p class="results-text">
            Compared with Voxlect, AccentCL base improves accuracy by <b>+11.2</b>, balanced
            accuracy by <b>+11.3</b>, and macro-F1 by <b>+8.8</b> points on the full test set, and
            OOD accuracy by <b>+5.9</b> points on unseen recording sources. The ablations show that
            logit adjustment and domain mean alignment (DMA) both improve OOD robustness. DMA has
            the larger effect, and combining them generalizes best.
          </p>

          <!-- Table II -->
          <h3 class="subsection-title">Class-incremental accent expansion</h3>
          <div class="table-wrap">
            <table class="results-table">
              <thead>
                <tr>
                  <th>Method</th>
                  <th class="num">New F1 ↑</th>
                  <th class="num">Old BAcc ↑</th>
                  <th class="num">Δ<sub>avg</sub> ↑</th>
                  <th class="num">Δ<sub>worst</sub> ↑</th>
                </tr>
              </thead>
              <tbody v-for="step in incrementalSteps" :key="step.label">
                <tr class="step-row">
                  <td colspan="5">{{ step.label }}</td>
                </tr>
                <tr v-for="row in step.rows" :key="row.method" :class="{ 'row-ours': row.ours }">
                  <td>{{ row.method }}</td>
                  <td
                    v-for="(v, i) in row.values"
                    :key="i"
                    class="num"
                    :class="{ best: v !== null && v === step.best[i], muted: v === null }"
                  >
                    <template v-if="v === null">&ndash;</template>
                    <template v-else>
                      {{ v.toFixed(1) }}
                      <span v-if="i === 3" class="worst-cls">({{ row.worstClass }})</span>
                    </template>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <p class="table-caption">
            <b>Table 2.</b> Class-incremental results (5 → 6 → 7 classes). Old BAcc is computed over
            the classes learned before each step. Δ<sub>avg</sub> and Δ<sub>worst</sub> are the
            average and worst-class accuracy change on old classes relative to the previous model;
            more negative means more forgetting. Voxlect is a fixed-label reference, not a continual
            learner.
          </p>
          <p class="results-text">
            AccentCL gets the best new-class F1 and old-class balanced accuracy at both steps, using
            a replay memory of only ~10% of the original training data. Retention and the old-to-new
            margin loss help in different ways, and combining them gives the most consistent balance
            between learning the new accent and keeping the old ones.
          </p>

          <div class="bounds">
            <div class="bounds-title">Reference bounds (New F1 / Old BAcc)</div>
            <div class="bounds-grid">
              <div></div>
              <div class="bounds-head">5 → 6</div>
              <div class="bounds-head">6 → 7</div>
              <div class="bounds-label">Naive new-class-only fine-tuning</div>
              <div class="num">15.0 / 0.3</div>
              <div class="num">5.7 / 5.7</div>
              <div class="bounds-label">Joint retraining (upper bound)</div>
              <div class="num">89.3 / 77.7</div>
              <div class="num">70.3 / 80.0</div>
              <div class="bounds-label bounds-ours">AccentCL</div>
              <div class="num bounds-ours">83.3 / 77.3</div>
              <div class="num bounds-ours">61.8 / 77.6</div>
            </div>
          </div>

          <figure class="figure figure-pair">
            <div class="figure-grid">
              <img
                :src="getAssetUrl('/figures/chinese_cm.svg')"
                alt="Confusion matrix after adding Chinese-accented English"
                loading="lazy"
              />
              <img
                :src="getAssetUrl('/figures/chinese_tsne_feat.svg')"
                alt="t-SNE of accent embeddings after adding Chinese-accented English"
                loading="lazy"
              />
            </div>
            <figcaption>
              <b>Figure 2.</b> <b>Top:</b> Confusion matrix of the final 7-way classifier (5 regions
              + Spanish- and Chinese-accented English), showing old-class predictions remain
              concentrated on the diagonal after two rounds of class expansion. <b>Bottom:</b> t-SNE
              projection of accent embeddings at the same stage, showing the newly added
              Chinese-accented English cluster is well separated from the five base regional
              clusters and from Spanish-accented English.
            </figcaption>
          </figure>
        </div>
      </section>

      <!-- BibTeX -->
      <section class="section section-alt" id="bibtex">
        <div class="wrap">
          <div class="bibtex-header">
            <h2 class="section-title">BibTeX</h2>
            <button class="copy-btn" @click="copyBibtex">
              {{ copied ? 'Copied' : 'Copy' }}
            </button>
          </div>
          <pre class="bibtex"><code v-text="bibtex"></code></pre>
        </div>
      </section>
    </main>

    <footer class="footer">
      <div class="wrap">
        © 2026 Texas A&amp;M University ·
        <a href="https://github.com/PSI-TAMU/AccentCL" target="_blank" rel="noopener noreferrer"
          >GitHub</a
        >
      </div>
    </footer>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const getAssetUrl = (path) => {
  const base = import.meta.env.BASE_URL || '/'
  const cleanPath = path.startsWith('/') ? path.slice(1) : path
  const cleanBase = base.endsWith('/') ? base : base + '/'
  return cleanBase + cleanPath
}

const authors = [
  { name: 'Mu-Ruei Tseng', url: 'https://github.com/Morris88826' },
  { name: 'Waris Quamer', url: 'https://github.com/warisqr007' },
  { name: 'Ghady Nasrallah', url: 'https://github.com/Ghadynasrallah' },
  {
    name: 'Ricardo Gutierrez-Osuna',
    url: 'https://scholar.google.com/citations?user=UnuQfEwAAAAJ&hl=en',
  },
]

const regionalRows = [
  { method: 'CommonAccent', values: [56.0, 48.3, 48.6, 79.0, 53.6, 58.2] },
  { method: 'Voxlect', values: [64.8, 65.8, 68.1, 83.8, 77.5, 82.6] },
  { method: 'AccentCL base', values: [76.0, 77.1, 76.9, 89.7, 79.6, 83.0], ours: true },
  { method: 'w/o logit adjustment', values: [75.8, 76.0, 76.9, 87.6, 77.6, 80.1], ablation: true },
  { method: 'w/o DMA', values: [75.7, 77.7, 75.8, 84.8, 72.8, 75.9], ablation: true },
  {
    method: 'w/o logit adjustment & DMA',
    values: [75.6, 76.0, 76.5, 81.9, 65.1, 69.2],
    ablation: true,
  },
]

// Highest value per column, used to bold the best result.
const columnMax = (rows) =>
  rows[0].values.map((_, i) => Math.max(...rows.map((r) => r.values[i]).filter((v) => v !== null)))

const regionalBest = columnMax(regionalRows)

const incrementalSteps = [
  {
    label: 'Base model: 5 regional classes',
    rows: [{ method: 'Frozen base', values: [null, 77.1, null, null] }],
  },
  {
    label: 'Step 1: 5 → 6, adding Spanish-accented English',
    rows: [
      { method: 'Voxlect', values: [52.7, 66.0, null, null] },
      { method: 'Replay only', values: [81.4, 76.2, -1.6, -4.4], worstClass: 'BI' },
      { method: 'Replay + retention', values: [82.8, 77.1, -0.7, -3.6], worstClass: 'BI' },
      { method: 'Replay + margin', values: [82.2, 76.1, -1.8, -4.6], worstClass: 'BI' },
      { method: 'AccentCL', values: [83.3, 77.3, -0.8, -3.9], worstClass: 'BI', ours: true },
    ],
  },
  {
    label: 'Step 2: 6 → 7, adding Chinese-accented English',
    rows: [
      { method: 'Voxlect', values: [null, 63.7, null, null] },
      { method: 'Replay only', values: [41.9, 75.1, -3.7, -7.5], worstClass: 'SPA' },
      { method: 'Replay + retention', values: [48.5, 75.9, -2.9, -5.7], worstClass: 'SPA' },
      { method: 'Replay + margin', values: [59.9, 77.1, -1.4, -3.8], worstClass: 'NA' },
      { method: 'AccentCL', values: [61.8, 77.6, -1.7, -4.2], worstClass: 'NA', ours: true },
    ],
  },
].map((step) => ({
  ...step,
  // Bold only within incremental steps that compare several methods.
  best: step.rows.length > 1 ? columnMax(step.rows) : [],
}))

const links = [
  {
    label: 'Paper',
    href: 'https://arxiv.org/abs/2610.07426',
    external: true,
    icon: 'M4 0h5.5L14 4.5V14a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V2a2 2 0 0 1 2-2zm5 1.5V5h3.5L9 1.5zM5 8.5a.5.5 0 0 0 0 1h6a.5.5 0 0 0 0-1H5zm0 2a.5.5 0 0 0 0 1h6a.5.5 0 0 0 0-1H5zm0 2a.5.5 0 0 0 0 1h4a.5.5 0 0 0 0-1H5z',
  },
  {
    label: 'Code',
    href: 'https://github.com/PSI-TAMU/AccentCL',
    external: true,
    icon: 'M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z',
  },
  {
    label: 'Results',
    href: '#results',
    icon: 'M1 14.5a.5.5 0 0 0 .5.5h13a.5.5 0 0 0 0-1H2V1.5a.5.5 0 0 0-1 0v13zM4 9.5a.5.5 0 0 1 .5-.5h1a.5.5 0 0 1 .5.5V13H4V9.5zm3.5-3a.5.5 0 0 1 .5-.5h1a.5.5 0 0 1 .5.5V13h-2V6.5zm3.5-3a.5.5 0 0 1 .5-.5h1a.5.5 0 0 1 .5.5V13h-2V3.5z',
  },
  {
    label: 'BibTeX',
    href: '#bibtex',
    icon: 'M3.5 1A1.5 1.5 0 0 0 2 2.5v11a.5.5 0 0 0 .8.4L8 10.1l5.2 3.8a.5.5 0 0 0 .8-.4v-11A1.5 1.5 0 0 0 12.5 1h-9z',
  },
]

const bibtex = `@misc{tseng2026accentclrobustaccentclassification,
  title         = {AccentCL: Robust Accent Classification with Incremental Expansion},
  author        = {Mu-Ruei Tseng and Waris Quamer and Ghady Nasrallah and Ricardo Gutierrez-Osuna},
  year          = {2026},
  eprint        = {2610.07426},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.07426},
}`

const copied = ref(false)

const copyBibtex = async () => {
  try {
    await navigator.clipboard.writeText(bibtex)
    copied.value = true
    setTimeout(() => (copied.value = false), 2000)
  } catch {
    // Clipboard unavailable (e.g. non-HTTPS); text remains selectable.
  }
}
</script>

<style scoped>
.page {
  --ink: #1a1a1a;
  --muted: #5f6368;
  --faint: #9aa0a6;
  --line: #e6e6e6;
  --alt: #fafafa;
  --link: #1a5fb4;

  min-height: 100vh;
  background: #ffffff;
  color: var(--ink);
  font-family:
    'Noto Sans',
    -apple-system,
    BlinkMacSystemFont,
    'Segoe UI',
    sans-serif;
  font-size: 16px;
  line-height: 1.6;
}

.wrap {
  max-width: 860px;
  margin: 0 auto;
  padding: 0 20px;
}

.wrap-wide {
  max-width: 1040px;
}

a {
  color: var(--link);
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

/* Hero */
.hero {
  padding: 72px 0 48px;
  text-align: center;
}

.title {
  font-family: 'Google Sans', 'Noto Sans', sans-serif;
  font-size: 40px;
  font-weight: 600;
  line-height: 1.25;
  letter-spacing: -0.5px;
  margin: 0 auto 24px;
  max-width: 820px;
}

.title-name {
  font-weight: 700;
}

.authors {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 4px 22px;
  font-size: 18px;
}

.author {
  white-space: nowrap;
}

.authors sup,
.affiliations sup {
  font-size: 0.65em;
  margin-left: 1px;
  color: var(--muted);
}

.affiliations {
  margin-top: 6px;
  font-size: 16px;
  color: var(--muted);
}

.venue {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
  gap: 10px;
  margin-top: 18px;
}

.venue-name {
  font-size: 17px;
  font-weight: 600;
}

.links {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 10px;
  margin-top: 28px;
}

.link-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 20px;
  border-radius: 999px;
  background: #363636;
  color: #ffffff;
  font-size: 15px;
  font-weight: 500;
  transition: background 0.15s ease;
}

.link-btn:hover {
  background: #111111;
  color: #ffffff;
  text-decoration: none;
}

.link-btn svg {
  width: 16px;
  height: 16px;
  fill: currentColor;
}

.link-btn-disabled,
.link-btn-disabled:hover {
  background: #8a8a8a;
  cursor: default;
}

.soon {
  font-size: 11px;
  font-weight: 600;
  padding: 0 7px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.22);
}

.link-btn:focus-visible {
  outline: 2px solid var(--link);
  outline-offset: 2px;
}

/* Sections */
.section {
  padding: 56px 0;
}

.section-alt {
  background: var(--alt);
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
}

.section-title {
  font-family: 'Google Sans', 'Noto Sans', sans-serif;
  font-size: 28px;
  font-weight: 600;
  text-align: center;
  margin: 0 0 24px;
}

.section-lead {
  text-align: center;
  color: var(--muted);
  margin: -12px auto 28px;
  max-width: 640px;
}

.abstract {
  text-align: justify;
  hyphens: auto;
  margin: 0;
}

/* Figures */
.figure {
  /* Override Bootstrap's .figure { display: inline-block }. */
  display: block;
  margin: 0;
}

.figure img {
  display: block;
  width: 100%;
  height: auto;
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 6px;
}

.figure figcaption {
  max-width: 860px;
  margin: 18px auto 0;
  font-size: 14.5px;
  line-height: 1.65;
  color: #3c4043;
  text-align: justify;
}

.figure-pair {
  margin-top: 36px;
}

.figure-grid {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
}

.figure-grid img {
  max-width: 860px;
}

/* Results table */
.table-wrap {
  max-width: 860px;
  margin: 0 auto;
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
}

.results-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 15px;
}

.results-table th,
.results-table td {
  padding: 11px 16px;
  text-align: left;
  border-bottom: 1px solid var(--line);
}

.results-table tbody tr:last-child td {
  border-bottom: none;
}

.results-table th {
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.6px;
  text-transform: uppercase;
  color: var(--muted);
  background: var(--alt);
  white-space: nowrap;
}

.results-table .num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.results-table .muted {
  color: var(--muted);
}

.subsection-title {
  font-family: 'Google Sans', 'Noto Sans', sans-serif;
  font-size: 20px;
  font-weight: 600;
  max-width: 860px;
  margin: 40px auto 14px;
}

.section-lead + .subsection-title {
  margin-top: 0;
}

.results-table .group-th {
  text-align: center;
  border-left: 1px solid var(--line);
}

.results-table thead tr:last-child th:nth-child(1),
.results-table thead tr:last-child th:nth-child(4) {
  border-left: 1px solid var(--line);
}

.results-table .best {
  font-weight: 700;
}

.results-table .row-ours td {
  background: #f3f7fc;
}

.results-table .row-ours td:first-child {
  font-weight: 600;
}

.results-table .row-ablation td:first-child {
  padding-left: 32px;
  color: var(--muted);
}

.results-table .step-row td {
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.4px;
  text-transform: uppercase;
  color: var(--muted);
  background: var(--alt);
}

.results-table tbody + tbody tr:first-child td {
  border-top: 1px solid var(--line);
}

.worst-cls {
  font-size: 12.5px;
  font-weight: 400;
  color: var(--muted);
}

.table-caption,
.results-text {
  max-width: 860px;
  margin: 12px auto 0;
}

.table-caption {
  font-size: 14px;
  line-height: 1.6;
  color: #3c4043;
}

.results-text {
  margin-top: 14px;
}

.bounds {
  max-width: 860px;
  margin: 24px auto 0;
  padding: 16px 20px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--alt);
}

.bounds-title {
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.6px;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 10px;
}

.bounds-grid {
  display: grid;
  grid-template-columns: 1fr auto auto;
  gap: 6px 32px;
  font-size: 14.5px;
}

.bounds-grid .num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.bounds-head {
  text-align: right;
  font-size: 13px;
  font-weight: 600;
  color: var(--muted);
}

.bounds-ours {
  font-weight: 600;
}

/* BibTeX */
.bibtex-header {
  position: relative;
}

.bibtex-header .section-title {
  margin-bottom: 20px;
}

.copy-btn {
  position: absolute;
  right: 0;
  top: 50%;
  transform: translateY(-50%);
  padding: 4px 14px;
  border: 1px solid #d0d0d0;
  border-radius: 6px;
  background: #ffffff;
  color: var(--ink);
  font-size: 13px;
  cursor: pointer;
}

.copy-btn:hover {
  border-color: var(--ink);
}

.copy-btn:focus-visible {
  outline: 2px solid var(--link);
  outline-offset: 2px;
}

.bibtex {
  margin: 0;
  padding: 18px 22px;
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 8px;
  font-size: 13.5px;
  line-height: 1.6;
  color: var(--ink);
  overflow-x: auto;
}

/* Footer */
.footer {
  padding: 32px 0 40px;
  text-align: center;
  font-size: 13px;
  color: var(--faint);
}

.footer a {
  color: var(--muted);
}

/* Responsive */
@media (max-width: 640px) {
  .hero {
    padding: 48px 0 32px;
  }

  .title {
    font-size: 28px;
  }

  .authors {
    font-size: 16px;
  }

  .section {
    padding: 40px 0;
  }

  .section-title {
    font-size: 24px;
  }

  .abstract,
  .figure figcaption {
    text-align: left;
  }

  .copy-btn {
    top: 0;
    transform: none;
  }
}
</style>
