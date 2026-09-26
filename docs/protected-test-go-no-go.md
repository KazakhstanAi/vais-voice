# Protected unseen-generator evaluation: NO-GO

Decision date: 2026-09-26. Protected status: **planned / undefined**.

VoxCPM 1.5 Kazakh LoRA is independent from Piper and has pinned model/runtime provenance. Only five
Kazakh pilot files exist: two passed review and three carry warnings for English pronunciation of
Arabic numerals. This is below the required 50 quality-gated synthetic samples. Its upstream
KazakhTTS dataset card also lacks an explicit SPDX license, so commercial clearance remains under
review. It is not an eligible protected corpus.

No quality-gated Silero KK protected corpus with sufficient count and complete provenance was
found. Qwen material is Russian, has runaway-generation failures, and is outside the current
Kazakh detector scope. Neither is eligible.

Because no candidate meets count, quality, leakage, compatibility, and legal gates, no retraining
or unseen ROC-AUC is performed. The safe alternative is to publish the seen-generator diagnostic,
the confound audit, and a planned protected-test state. A future GO requires a frozen independent
generator corpus with at least 50 synthetic and 50 compatible real samples, human quality records,
license review, and a pre-registered one-shot evaluation.
