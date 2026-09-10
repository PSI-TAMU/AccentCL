<template>
  <div class="min-vh-100 bg-white">
    <div class="bg-white" id="paper">
      <div class="container py-5">
        <!-- Header -->
        <div class="text-center mb-5">
          <h2 class="demo-title">
            AccentCL: Robust Accent Classification with Incremental Expansion
          </h2>
          <p class="demo-subtitle">Anonymous Submission</p>
          <div class="header-links">
            <a href="#" class="header-link">Paper (Coming Soon)</a>
            <span class="header-link-sep">·</span>
            <a href="#" class="header-link">Code (Coming Soon)</a>
          </div>
        </div>

        <!-- Abstract -->
        <div class="row justify-content-center mb-5">
          <div class="col-lg-10">
            <div class="abstract-block">
              <div class="abstract-title">Abstract</div>
              <p class="abstract-text">
                Accent classifiers are typically trained with a fixed label inventory and cannot
                accommodate new accent categories as new data becomes available. Moreover, accented
                speech corpora often exhibit substantial class imbalance and/or domain shift due to
                differences in recording conditions across corpora. We present AccentCL, a
                class-incremental learning framework for English accent classification that is
                robust to class imbalance and cross-corpus domain shift. AccentCL extracts
                multi-layer representations from a frozen Whisper-Large-v3 encoder, optimized with
                an imbalance-aware cross-entropy loss to reduce bias toward the majority accent
                classes and a domain mean alignment loss that minimizes distributional mean shift
                across training corpora. The label space is then expanded via replay-based continual
                learning, using the frozen base model for knowledge retention and an old-to-new
                margin loss to reduce overprediction on newly added classes. On a five-class accent
                classification task, AccentCL achieves 77.1% balanced accuracy and a 76.9%
                macro-averaged F1 score. We further evaluate the model's ability to incrementally
                incorporate two new accent categories: Spanish-accented and Chinese-accented
                English. When adding Spanish-accented English to the pretrained model, AccentCL
                attains an F1 of 83.3% on the new class while retaining 77.3% balanced accuracy on
                the base classes. When subsequently adding Chinese-accented English, it achieves
                61.8% F1 on the new class while preserving 77.6% balanced accuracy on the
                previously learned classes. These results show that AccentCL enables robust regional
                accent classification while allowing new accent categories to be added without full
                retraining.
              </p>
            </div>
          </div>
        </div>

        <!-- Architecture -->
        <div class="row justify-content-center mb-5">
          <div class="col-lg-10">
            <div class="figure-block">
              <div class="figure-title">Method Overview</div>
              <div class="figure-container">
                <img
                  :src="getAssetUrl('/figures/thumbnail.svg')"
                  alt="AccentCL Architecture"
                  class="figure-img"
                />
              </div>
              <p class="figure-caption">
                <b>Model architecture of AccentCL.</b> A frozen Whisper-Large-v3 encoder provides
                multi-layer speech representations, which are projected, concatenated, and pooled
                with attentive statistics pooling to produce an accent embedding. Phase 1 trains a
                base regional accent classifier with domain mean alignment. Phase 2 expands the
                classifier to a new accent class using replay-based continual training with
                retention and an old-to-new margin loss
              </p>
            </div>
          </div>
        </div>

        <!-- Results -->
        <div class="row justify-content-center mb-5">
          <div class="col-lg-10">
            <div class="figure-block">
              <div class="figure-title">Results</div>
              <div class="table-responsive">
                <table class="results-table">
                  <thead>
                    <tr>
                      <th>Stage</th>
                      <th>Old-Class Balanced Accuracy</th>
                      <th>New-Class F1</th>
                      <th>Notes</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Base regional classifier (5-way)</td>
                      <td>77.1%</td>
                      <td>&ndash;</td>
                      <td>Macro-F1: 76.9%</td>
                    </tr>
                    <tr>
                      <td>+ Spanish-accented English</td>
                      <td>77.3% <span class="delta">(&minus;0.8 pts avg. drop)</span></td>
                      <td>83.3%</td>
                      <td></td>
                    </tr>
                    <tr>
                      <td>+ Chinese-accented English</td>
                      <td>77.6%</td>
                      <td>61.8%</td>
                      <td></td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div class="row mt-4 g-3">
                <div class="col-md-6">
                  <div class="figure-container">
                    <img
                      :src="getAssetUrl('/figures/chinese_cm.svg')"
                      alt="Confusion matrix after adding Chinese-accented English"
                      class="figure-img"
                    />
                  </div>
                </div>
                <div class="col-md-6">
                  <div class="figure-container">
                    <img
                      :src="getAssetUrl('/figures/chinese_tsne_feat.svg')"
                      alt="t-SNE of accent embeddings after adding Chinese-accented English"
                      class="figure-img"
                    />
                  </div>
                </div>
              </div>
              <p class="figure-caption">
                <b>Left:</b> Confusion matrix of the final 7-way classifier (5 regions + Spanish-
                and Chinese-accented English), showing old-class predictions remain concentrated on
                the diagonal after two rounds of class expansion. <b>Right:</b> t-SNE projection of
                accent embeddings at the same stage, showing the newly added Chinese-accented
                English cluster is well separated from the five base regional clusters and from
                Spanish-accented English.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const getAssetUrl = (path) => {
  const base = import.meta.env.BASE_URL || '/'
  const cleanPath = path.startsWith('/') ? path.slice(1) : path
  const cleanBase = base.endsWith('/') ? base : base + '/'
  return cleanBase + cleanPath
}

const slugify = (label) =>
  label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '')

const baseClasses = ['Region A', 'Region B', 'Region C', 'Region D', 'Region E']

const stages = [
  {
    key: 'base',
    label: 'Stage 1: Base (5-way regional)',
    classes: baseClasses,
  },
  {
    key: 'spanish',
    label: 'Stage 2: + Spanish-accented English',
    classes: [...baseClasses, 'Spanish-accented English'],
  },
  {
    key: 'chinese',
    label: 'Stage 3: + Chinese-accented English',
    classes: [...baseClasses, 'Spanish-accented English', 'Chinese-accented English'],
  },
]

// Placeholder samples/predictions per stage+class. Replace with real sample IDs and model outputs.
const samplesByStageAndClass = {}
for (const stage of stages) {
  samplesByStageAndClass[stage.key] = {}
  for (const cls of stage.classes) {
    samplesByStageAndClass[stage.key][cls] = [
      { id: 'sample_01', predicted: cls, confidence: '0.00 (placeholder)' },
      { id: 'sample_02', predicted: cls, confidence: '0.00 (placeholder)' },
    ]
  }
}

const selectedStageKey = ref(stages[0].key)
const selectedClass = ref(stages[0].classes[0])
const selectedSampleId = ref(samplesByStageAndClass[stages[0].key][stages[0].classes[0]][0].id)

const currentStage = computed(
  () => stages.find((s) => s.key === selectedStageKey.value) ?? stages[0],
)

const currentSamples = computed(() => {
  const classes = samplesByStageAndClass[selectedStageKey.value]
  return classes?.[selectedClass.value] ?? []
})

const currentSample = computed(
  () =>
    currentSamples.value.find((s) => s.id === selectedSampleId.value) ?? currentSamples.value[0],
)
</script>

<style scoped>
/* Abstract */
.abstract-block {
  background: #f6f8fa;
  border: 1px solid #e1e4e8;
  border-left: 4px solid #0969da;
  border-radius: 12px;
  padding: 28px 32px;
}

.abstract-title {
  font-size: 13px;
  font-weight: 700;
  color: #0969da;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  margin-bottom: 14px;
}

.abstract-text {
  font-size: 15px;
  color: #24292f;
  line-height: 1.75;
  margin: 0;
  letter-spacing: -0.1px;
}

/* Figure blocks */
.figure-block {
  background: #ffffff;
  border: 1px solid #e1e4e8;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.figure-title {
  font-size: 18px;
  font-weight: 600;
  color: #24292f;
  margin-bottom: 16px;
  letter-spacing: -0.3px;
  padding-bottom: 12px;
  border-bottom: 2px solid #f6f8fa;
}

.figure-container {
  width: 100%;
  border-radius: 8px;
  overflow: hidden;
  background: #f6f8fa;
}

.figure-img {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 8px;
}

.figure-caption {
  font-size: 13.5px;
  color: #57606a;
  line-height: 1.7;
  margin: 16px 0 0 0;
  padding-top: 14px;
  border-top: 1px solid #e1e4e8;
}

/* Header */
.demo-title {
  font-size: 32px;
  font-weight: 700;
  color: #24292f;
  letter-spacing: -1px;
  margin-bottom: 8px;
  background: linear-gradient(135deg, #24292f 0%, #0969da 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.demo-subtitle {
  font-size: 16px;
  font-weight: 400;
  color: #57606a;
  letter-spacing: -0.2px;
  line-height: 1.4;
  margin: 0 0 12px 0;
}

.header-links {
  font-size: 14px;
}

.header-link {
  color: #0969da;
  text-decoration: none;
  font-weight: 500;
}

.header-link:hover {
  text-decoration: underline;
}

.header-link-sep {
  color: #d0d7de;
  margin: 0 10px;
}

/* Results table */
.results-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14.5px;
}

.results-table th,
.results-table td {
  padding: 12px 14px;
  text-align: left;
  border-bottom: 1px solid #e1e4e8;
}

.results-table th {
  color: #57606a;
  font-weight: 600;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  background: #f6f8fa;
}

.results-table td {
  color: #24292f;
}

.delta {
  color: #57606a;
  font-size: 12.5px;
}

/* Demo section */
.demo-note {
  font-size: 13.5px;
  color: #57606a;
  margin-bottom: 20px;
}

.demo-note code {
  background: #f6f8fa;
  border: 1px solid #e1e4e8;
  border-radius: 4px;
  padding: 1px 5px;
  font-size: 12.5px;
}

.models-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.models-title {
  font-size: 20px;
  font-weight: 600;
  color: #24292f;
  margin: 0;
  letter-spacing: -0.3px;
}

/* Selector panel */
.selector-panel {
  background: #ffffff;
  border: 1px solid #e1e4e8;
  border-radius: 12px;
  padding: 20px 24px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.selector-group {
  display: flex;
  align-items: center;
  gap: 16px;
}

.selector-label {
  font-size: 14px;
  font-weight: 600;
  color: #24292f;
  letter-spacing: -0.2px;
  white-space: nowrap;
  min-width: 120px;
}

.pair-select {
  flex: 1;
  max-width: 400px;
  padding: 12px 14px;
  font-size: 15px;
  border: 1.5px solid #d0d7de;
  border-radius: 8px;
  background: #ffffff;
  color: #24292f;
  font-weight: 500;
  letter-spacing: -0.2px;
  transition: all 0.2s ease;
}

.pair-select:focus {
  outline: none;
  border-color: #0969da;
  box-shadow: 0 0 0 3px rgba(9, 105, 218, 0.1);
}

/* Audio section */
.audio-section {
  background: #ffffff;
  border: 1px solid #e1e4e8;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.audio-section-title {
  font-size: 19px;
  font-weight: 600;
  color: #24292f;
  margin-bottom: 20px;
  letter-spacing: -0.4px;
  padding-bottom: 12px;
  border-bottom: 2px solid #f6f8fa;
}

.src-tgt-row {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}

.audio-column {
  display: flex;
  flex-direction: column;
  gap: 10px;
  background: #f6f8fa;
  padding: 14px;
  border-radius: 10px;
  border: 1px solid #e1e4e8;
}

.src-column {
  border-left: 3px solid #0969da;
}

.tgt-column {
  border-left: 3px solid #1a7f37;
}

.audio-label {
  font-size: 11px;
  font-weight: 600;
  color: #57606a;
  letter-spacing: 0.3px;
  text-transform: uppercase;
}

.audio-column audio {
  width: 100%;
  height: 40px;
  outline: none;
}

.prediction-value {
  font-size: 15px;
  font-weight: 600;
  color: #24292f;
}

.results-separator {
  height: 1px;
  background: linear-gradient(to right, transparent, #d0d7de 20%, #d0d7de 80%, transparent);
  margin: 24px 0;
  position: relative;
}

.results-separator::before {
  content: '';
  position: absolute;
  top: -4px;
  left: 50%;
  transform: translateX(-50%);
  width: 8px;
  height: 8px;
  background: #0969da;
  border-radius: 50%;
  box-shadow:
    0 0 0 3px #ffffff,
    0 0 0 4px #d0d7de;
}

.results-label {
  font-size: 12px;
  font-weight: 600;
  color: #57606a;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 10px;
}

.prediction-row {
  display: flex;
  align-items: baseline;
  gap: 12px;
}

.prediction-main {
  font-size: 18px;
  font-weight: 700;
  color: #0969da;
}

.prediction-confidence {
  font-size: 13px;
  color: #57606a;
}

/* Citation */
.citation-block {
  background: #f6f8fa;
  border: 1px solid #e1e4e8;
  border-radius: 8px;
  padding: 16px 20px;
  font-size: 13px;
  color: #24292f;
  overflow-x: auto;
  margin: 0;
}

/* Responsive */
@media (max-width: 768px) {
  .demo-title {
    font-size: 24px;
    letter-spacing: -0.5px;
  }

  .selector-group {
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
  }

  .pair-select {
    max-width: 100%;
    width: 100%;
  }

  .src-tgt-row {
    grid-template-columns: 1fr;
  }
}
</style>
