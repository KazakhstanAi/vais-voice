# Digital Bridge evidence audit

Audit date: 2026-09-26. Result: the two frontend WAV files are exact copies of research artifacts.
No hash blocker was found.

| Sample | Research source | Runtime | Bytes | Audio | SHA-256 |
|---|---|---:|---:|---|---|
| `diag_kk_piper_1_2_0` | `runs/piper-real-v01/runtime-compat-diagnostic/kk_piper_1.2.0.wav` | Piper standalone 1.2.0 | 125716 | mono, 22050 Hz, 2.849705 s | `5c0469c08ec66969a7e758367af7d123b1c856eb0fb0c20fc8c2925b81e53048` |
| `diag_kk_piper_1_8_0` | `runs/piper-real-v01/runtime-compat-diagnostic/kk_piper_1.8.0.wav` | piper-tts 1.8.0 | 113708 | mono, 22050 Hz, 2.577415 s | `22e237710069cd6786ba49044d1fd76dd71378ca6d395c2805875da1c150420f` |

Both use prompt `runtime-compat-kk-hello-v01`, text `Сәлеметсіз бе. Бұл қазақ тіліндегі
тест.`, Kazakh, `kk_KZ-issai-high`, speaker 0, and model revision
`320d5f7f7751a17ef6512d5c23863056c6a11c0f`. Model SHA-256 is
`4dee767c893e8535da821447d12cb030e3569e11254c14030a1da5d8b2222c16`.

The raw findings are in
`runs/piper-real-v01/runtime-compat-diagnostic/pronunciation_review.jsonl`. They contain a review
status and free-form finding but no reviewer ID, reviewer count, timestamp, or structured
dimension ratings. Therefore quality gates are direct record fields, while only `naturalness`
is projected as `derived_finding`. `synthetic_detectability` is `not_evaluated` because perceived
synthetic character is not a detector measurement. Pronunciation, prosody,
numbers/entities, language adherence, stability, code-switching, and robustness are
`not_evaluated`. No numeric human rating or consensus is claimed.

Frontend audit covered `src/lib/evidence.ts`, `public/audio/benchmark/manifest.json`, and both WAVs
at commit `fcf550a`. The original frontend hashes and sizes matched. The v0.2 publication contract
adds evidence type and nullable reviewer count without treating unknown values as failure.
