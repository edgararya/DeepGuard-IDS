// frontend/src/renderer/components/ResultPanel.jsx
// Presentational card for a DetectionResult returned by the backend.

import { Fragment } from 'react';

// Human-readable names for the per-model scores in DetectionResult.
const MODEL_LABELS = {
  mesonet_1: 'Meso4_DF',
  mesonet_2: 'Meso4_F2F',
  mesonet_3: 'MesoInception_DF',
  xception: 'XceptionNet',
};

const SCORE_KEYS = ['mesonet_1', 'mesonet_2', 'mesonet_3', 'xception'];

// confidence / per-model scores / ela_score / blur_score are all 0..1 floats
// (or null when a model did not load). Display them as percentages.
function percent(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'N/A';
  }
  return `${(value * 100).toFixed(1)}%`;
}

function ResultPanel({ result }) {
  if (!result) {
    return null;
  }

  const isFake = result.label === 'fake';
  const variant = isFake ? 'fake' : 'real';

  return (
    <section className={`result-panel result-panel--${variant}`} aria-live="polite">
      <header className="result-panel__header">
        <span className={`result-panel__verdict result-panel__verdict--${variant}`}>
          {isFake ? 'Fake' : 'Real'}
        </span>
        <span className="result-panel__confidence">
          Confidence: {percent(result.confidence)}
        </span>
      </header>

      <dl className="result-panel__scores">
        <dt>ELA</dt>
        <dd>{percent(result.ela_score)}</dd>

        <dt>Blur</dt>
        <dd>{percent(result.blur_score)}</dd>

        {SCORE_KEYS.map((key) => (
          <Fragment key={key}>
            <dt>{MODEL_LABELS[key]}</dt>
            <dd>{percent(result.per_model_scores?.[key])}</dd>
          </Fragment>
        ))}
      </dl>
    </section>
  );
}

export default ResultPanel;