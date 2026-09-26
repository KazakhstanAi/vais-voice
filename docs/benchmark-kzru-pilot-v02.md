# VAIS Voice KZ/RU Benchmark Pilot v0.2 protocol

Status: **research**. Protocol version: **0.2**. Benchmark ID:
`vais-kzru-voice-benchmark`. This protocol records reproducible evidence; it is not a claim of
production antifraud readiness or a calibrated probability of authorship.

## Scope and eligibility

The supported tasks are paired TTS runtime diagnostics and synthetic-speech detector diagnostics
for Kazakh (`kk`) and Russian (`ru`). A model is eligible only when its model revision, runtime,
source, intended role, and license status are recorded. Unknown values remain null. Runtime changes
create distinct records even when model weights are identical.

Prompts are assigned immutable IDs and categories: general speech, numbers/entities,
code-switching, and stress/robustness. Inclusion requires a real audio artifact, SHA-256, source
artifact, language, runtime, and quality-gate status. Failed generation and negative findings are
retained.

## Evidence and dimensions

Every record covers `naturalness`, `pronunciation`, `prosody`, `numbers_entities`,
`language_adherence`, `stability`, `code_switching`, `robustness`, and
`synthetic_detectability`. Values are `pass`, `warning`, `fail`, `not_evaluated`, `planned`, or
`derived`. Evidence types are `human_review`, `automatic_metric`, `derived_finding`, and
`not_evaluated`.

`not_evaluated` means no valid observation exists; it is not a failure or a numeric zero. `planned`
identifies future work. A label inferred from free-form review text is always `derived_finding`,
never a direct measurement. Numeric human scores may be published only with scale, instructions,
reviewer count, aggregation, disagreement/variance, and sample count.

Human reviewers must be real people with recorded language competence. Pseudonymous reviewer IDs
are allowed. Individual records and disagreement are retained; edits after aggregation require an
audit record. Automated metrics and human review are stored separately.

## Detector evaluation

Positive class is synthetic and higher score means more synthetic evidence. Report ROC-AUC, EER,
FPR@TPR95, class counts, score range, unique scores, the validation-selected threshold, and the
confusion matrix at that fixed threshold. Single-class slices receive null ROC-AUC/EER/FPR values.
The current score is a **Synthetic evidence score**, not a probability (`calibrated: false`).

Checkpoint selection uses validation ROC-AUC and validation loss. Threshold selection uses
validation EER. Test and protected data are never used for either choice.

## Comparability

Direct ranking is allowed only when language, prompt set, benchmark version, evaluation method,
and dataset scope all match. Cross-language and different-runtime records may appear beside each
other but are not ranked. The published Piper pair is a one-prompt runtime diagnostic of the same
model, not a model-family comparison or population estimate.

## Protected evaluation

The protected test requires an independent held-out generator, at least 50 quality-gated synthetic
samples, at least 50 compatible real samples, complete provenance and legal status, and zero use in
training, validation, checkpoint selection, or threshold selection. Its current state is
**planned / undefined**. No protected metric may be published until the gate passes.

## Publication and retention

Publication states are limited to `research` for v0.2. Each public WAV and snapshot has a SHA-256.
The published JSON is immutable; corrections use `pilot-v0.2.1` or a later version. Source
manifests, configs, reviews, failed runs, checkpoint metadata, and audit reports are retained.

## Limitations

- The detector diagnostic is Kazakh-only and uses one seen Piper generator.
- FLEURS speaker identity is unavailable, so speaker-disjoint splitting cannot be proven.
- Real and synthetic duration distributions differ and may be a shortcut.
- Human review identity/count metadata for the published pair is absent; dimension labels derived
  from its text are marked accordingly.
- Russian detector evaluation and protected unseen-generator evaluation are not available.
