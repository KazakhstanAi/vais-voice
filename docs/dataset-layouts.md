# Verified dataset provenance and layouts

Verification date: 2026-09-19. Raw data and generated outputs are gitignored.

| Source | Pinned release | Adapter layout | Local verification |
| --- | --- | --- | --- |
| KSC2 | Official ISSAI KSC2, HF revision `cececbec1049f93f34a7421552500da01971ead8` | `issai_ksc2_sidecar_v1` | Two real FLAC/TXT pairs decoded and ingested |
| FLEURS Kazakh | `google/fleurs`, config `kk_kz`, revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd` | `google_fleurs_tsv_v1` | Two real validation WAVs decoded and ingested |
| FLEURS Russian | `google/fleurs`, config `ru_ru`, same revision | `google_fleurs_tsv_v1` | Two real validation WAVs decoded and ingested |
| ASVspoof 2021 | Speech Deepfake (DF) v1.0, Zenodo `4835108` | `asvspoof2021_df_keys_v1` | One real bona fide and one real spoof FLAC decoded and ingested with official keys |

## KSC2

The official split archive begins with this observed structure:

```text
ISSAI_KSC2/
  Test/
    crowdsourced/
      <utterance-id>.flac
      <utterance-id>.txt
```

The adapter recursively discovers supported audio. The same-stem UTF-8 transcript is provenance
context but is not required by the canonical audio manifest. The verified subset exposed no
utterance-level speaker identifier, so each row records `speaker_id` as missing and receives a
deterministic sample-local placeholder. It defaults conservatively to `kk`; it does not infer
`kk_ru` from transcript text.

Official provenance: [ISSAI KSC2 page](https://issai.nu.edu.kz/kz-speech-corpus/), data DOI
[`10.48342/m90y-aj02`](https://doi.org/10.48342/m90y-aj02), and paper DOI
[`10.21437/Interspeech.2022-421`](https://doi.org/10.21437/Interspeech.2022-421). ISSAI states
CC BY 4.0 and specifies attribution language for commercial products. The current Hugging Face
card displays `mit`; the source config follows the dataset-specific ISSAI project page and records
this discrepancy rather than treating the two labels as interchangeable.

## FLEURS

The official source loader defines a headerless, tab-separated seven-field record:

```text
id  filename  raw_transcription  transcription  characters  num_samples  gender
```

Audio is under the corresponding split directory. FLEURS does not expose a stable speaker ID in
this representation, so missing speaker provenance is explicit. Dataset configs `kk_kz` and
`ru_ru` normalize only to internal `kk` and `ru` respectively.

Official provenance: [`google/fleurs`](https://huggingface.co/datasets/google/fleurs), CC BY 4.0,
and paper DOI [`10.1109/SLT54892.2023.10023141`](https://doi.org/10.1109/SLT54892.2023.10023141).

## ASVspoof 2021 DF

The pinned source is the Speech Deepfake evaluation track, not LA or PA. The official
`keys/DF/CM/trial_metadata.txt` rows begin with:

```text
speaker trial codec source attack label trim subset ...
```

The parser reads `attack` from field 5 and `bonafide`/`spoof` from field 6, while retaining
compatibility with compact five-column exports. Audio is expected under
`ASVspoof2021_DF_eval/flac`. ASVspoof remains `external_sanity` and cannot silently enter the
primary KZ/RU build.

Official provenance: [ASVspoof 2021](https://www.asvspoof.org/index2021.html), DF v1.0 DOI
[`10.5281/zenodo.4835108`](https://doi.org/10.5281/zenodo.4835108), and paper DOI
[`10.21437/ASVSPOOF.2021-8`](https://doi.org/10.21437/ASVSPOOF.2021-8).
The challenge webpage describes an Open Data Commons Attribution licence, but the downloaded DF
archive's `LICENSE.DF.txt` incorporates ODbL 1.0 and applies DbCL 1.0 to the contents; Zenodo's API
also identifies `odc-odbl`. The source config records the packaged ODbL/DbCL terms rather than
silently resolving that discrepancy in the less restrictive direction.

## First real pilot build

The first local pilot contains six real recordings: two KSC2 Kazakh, two FLEURS `kk_kz`, and two
FLEURS `ru_ru`. It totals 56.112 seconds. There are no verified `kk_ru` samples, no known speaker
IDs, no exact duplicates, and no invalid files. The generated artifacts are:

```text
data/manifests/local/kzru_real_v01.parquet
data/processed/kzru_real_v01/dataset_report.json
data/processed/kzru_real_v01/preparation.json
```

The pilot manifest SHA-256 is
`9a12f82ac7f05ae273d7969dc225a79ee174f2355835d3881b02413af284aa76`.
These local artifacts are intentionally not committed.
