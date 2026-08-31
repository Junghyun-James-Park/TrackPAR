# TrackPAR

Person-attribute auto-labelling for overhead retail CCTV. SAM 3 tracks every
person, then a vision-language model labels four attributes per track.

| attribute | shape | how | measured |
|---|---|---|---|
| gender | one per track | Qwen3.5-9B + identity LoRA, K=4 frames per call | 0.9456 accuracy |
| age | one per track | same adapter, separate integer prompt, K=4 | MAE 3.63 years |
| exposed | one per **frame** | base 9B, `eyes` prompt, K=8 | F1 0.728 |
| watched | one per **frame** | base 9B, `svfd` prompt, **K=1** | F1 0.740 |

gender and age are held-out scores on 349 tracks. The age number is worth reading
against the best single constant guess, which scores MAE 10.46. exposed and
watched are measured over all 5,168 annotated instances, with bootstrap
confidence intervals — see [docs/RESULTS.md](docs/RESULTS.md), which also explains
why the two use different prompts and different K.

### Why tracking, and not just classifying crops

Every identity number above is measured **per track**, with K frames of the same
person in one call. That is the reason the pipeline tracks at all, and it is worth
a number rather than an assertion.

Same model, same 62 held-out tracks, only the way frames are used changes:

| model | frames used | gender | age MAE | within 5 yr | within 10 yr |
|---|---|---|---|---|---|
| base-4B | one frame at a time (K=1) | 0.8710 | 11.17 | 0.298 | 0.577 |
| | per frame, then majority vote | 0.9194 | 9.74 | 0.339 | 0.645 |
| | **K frames in one call** | **0.9839** | **8.69** | 0.435 | 0.694 |
| base-9B, no fine-tune | one frame at a time | 0.8548 | 11.51 | 0.363 | 0.573 |
| | per frame, then majority vote | 0.8871 | 9.45 | 0.468 | 0.645 |
| | **K frames in one call** | **0.9516** | 9.71 | 0.403 | 0.661 |
| 9B, fine-tuned single-image | one frame at a time | 0.8548 | 8.46 | 0.500 | 0.738 |
| | per frame, then majority vote | 0.9032 | **6.65** | 0.613 | 0.806 |
| | K frames in one call | 0.9000 | 7.70 | 0.400 | 0.800 |
| 9B, fine-tuned multi-image | one frame at a time | 0.9032 | 7.12 | 0.577 | 0.802 |
| | per frame, then majority vote | 0.9677 | 5.98 | 0.661 | 0.887 |
| | **K frames in one call** | **0.9677** | **5.37** | 0.710 | 0.855 |

Reading down each model, the gain from tracking is consistent:

| model | gender | age MAE |
|---|---|---|
| base-4B | 0.8710 → 0.9839 (**+0.113**) | 11.17 → 8.69 (**−2.48**) |
| base-9B, no fine-tune | 0.8548 → 0.9516 (**+0.097**) | 11.51 → 9.71 (**−1.80**) |
| 9B, FT single-image | 0.8548 → 0.9000 (+0.045) | 8.46 → 7.70 (−0.76) |
| 9B, FT multi-image | 0.9032 → 0.9677 (**+0.065**) | 7.12 → 5.37 (**−1.75**) |

Three things this shows.

**Grouping frames beats voting over them.** Majority vote already helps — it
averages away per-frame noise — but handing K frames to the model in one call
helps more in every row but one. The model is not just voting; it is using views
of the same person together.

**The gain is largest where the model is weakest.** base-4B gains +0.113 gender,
the fine-tuned multi-image model +0.065. Tracking substitutes for model capacity,
which is what makes it worth the tracking cost on a small backbone.

**Fine-tuning and tracking are not interchangeable.** The one row where grouping
does not help on age is the model fine-tuned on *single* images (6.65 → 7.70): it
was trained to read one crop, so being handed four is off-distribution. The
adapter this pipeline ships was fine-tuned on multi-image inputs for that reason,
and it is the only arm that is best at both (0.9677 / 5.37).

Note these are 62 tracks, a development set from an earlier stage, so read the
direction and the size of the effect rather than the third decimal. The shipped
numbers at the top of this README are on the 349-track held-out set.

---

## What has to be true before anything runs

Three things are not in this repository and have to be obtained separately.

**1. The corpus.** Overhead CCTV frames plus per-frame person boxes. Not
redistributable here; obtain from the authors. The pipeline needs:

```
LOTTE_IMAGES   frames, one directory per recording session
LOTTE_ANNOT    per-frame person instances (JSON), used to build tracks
LOTTE_CSV      ground truth, only needed to SCORE — a labelling run never reads it
```

**2. SAM 3**, for stage 1 only. It is public but the weights are gated:

```bash
git clone https://github.com/facebookresearch/sam3
pip install -e sam3
huggingface-cli login          # after accepting terms at
                               # https://huggingface.co/facebook/sam3
```

If you already have tracks, skip all of this and start at stage 2 with
`bash run_all.sh --from fragments`.

**3. The identity LoRA adapter**, 1.2 GB, for gender and age.

```bash
pip install gdown
bash setup/fetch_weights.sh --gdrive <SHARE_URL>
```

The link is in [docs/WEIGHTS.md](docs/WEIGHTS.md). The script accepts a bare file
id or any share-URL shape, unpacks the archive, and prints the sha256 of both
weight files so you can check the download against the published hashes — a
truncated transfer is otherwise indistinguishable from a complete one.

The archive holds only what inference reads: the LoRA weights, the trained vision
tower, and the tokenizer/processor config. Optimiser state is not included, so
this cannot be used to resume training.

If you would rather install from a local copy:

```bash
bash setup/fetch_weights.sh /path/to/adapter_dir
```

Without the adapter, unset `IDENTITY_ADAPTER` to run identity on the base model —
the pipeline still works, the numbers drop.

The base model (`Qwen/Qwen3.5-9B`) downloads from HuggingFace on first use.

---

## Setup

Two conda environments, because SAM 3 and the VLM stack disagree on
`transformers`: SAM 3 needs 5.8.x, the VLM side is pinned to 5.3.0.

```bash
git clone <this repo> TrackPAR && cd TrackPAR

# stages 2-6 (the VLM pipeline)
conda create -n trackpar python=3.11 -y
conda activate trackpar
pip install -r requirements.txt

# stage 1 only (SAM 3 tracking)
conda create -n sam3 python=3.12 -y
conda activate sam3
pip install -r requirements-sam3.txt
pip install -e /path/to/sam3
```

Then point the config at your machine and check it:

```bash
$EDITOR config/paths.sh        # LOTTE_*, SAM3_SRC, IDENTITY_ADAPTER, TRACKPAR_GPUS
conda activate trackpar
source config/paths.sh
python setup/check_env.py
```

`check_env.py` is not decorative. Every check in it corresponds to a mistake that
cost real GPU time during development:

- **`HF_HOME` is writable.** A shell profile pointing it at an unwritable mount
  makes every model load fail with a `PermissionError` that reads like a
  HuggingFace outage. `config/paths.sh` overrides an inherited value that fails
  this test rather than passing it through.
- **`non_lora_state_dict.bin` is present.** `PeftModel.from_pretrained` loads
  `adapter_model.safetensors` and silently ignores that file, which holds the
  trained vision tower and merger. Nothing raises. You end up evaluating a base
  vision tower under a fine-tuned adapter — a model that never existed — and it
  scores plausibly enough to publish.
- **All 15 pipeline modules import.** Missing files otherwise surface only when
  the run reaches them, which in practice meant a crash after model load. Three
  files were missing on the first packaging attempt, every one pulled in
  transitively rather than named anywhere obvious.

---

## Running it

```bash
source config/paths.sh
bash run_all.sh                    # all six stages
bash run_all.sh --from fragments   # reuse existing tracks
bash run_all.sh --score            # and grade afterwards
```

Every stage skips itself when its output exists, so an interrupted run resumes.

| stage | script | output | cost |
|---|---|---|---|
| 1 | `track_sam3_chunked.py` | `out/track_sam3/<session>/track.json` | hours, one GPU, **sam3 env** |
| 2 | `phase1_build_all_fragments.py` | `out/phase1_fragments.json` | minutes, CPU |
| 3 | `multiimg_eval.py` | `out/identity.json` | ~2.5 h |
| 4 | `age_eval.py` | `out/age.json` | ~40 min |
| 5a | `exp20_unified_infer.py` | `out/momentary_exposed_*.json` | ~3 h, two GPUs |
| 5b | `momentary_k1_control.py` | `out/momentary_watched_sh*.json` | **~11 h**, two GPUs |
| 6 | `merge_labels.py` | `out/labels.json` | seconds |

Stage 5b is the expensive one and the cost is structural, not an inefficiency:
`watched` only works at K=1, which means one model call per frame — about 18,000
calls at ~4.4 s each. `exposed` at K=8 covers eight frames per call, hence ~3 h
for the same corpus.

Set `TRACKPAR_GPUS` to the two cards you want. Stages 5a and 5b shard across
both; 3 and 4 use whatever `device_map="auto"` picks.

---

## The prompts

All twelve prompts that were scored are in `prompts/`, so any of them can be
swapped in through `config/paths.sh`.

```
prompts/crop/     one cropped person per call
  combined.txt    eyes / nose / mouth / gaze reported separately, rule applied in code
  eyes.txt        eye visibility and gaze direction only
  features.txt    eyes / nose / mouth, no gaze
  metav4.txt      OPRO-optimised; what an earlier version of this pipeline shipped
  plain.txt       ask for exposed and watched directly
  subattr.txt     the 40-attribute schema, momentary derived from it
  svfd.txt        graded face-visibility tiers plus an explicit rarity prior
  trueonly.txt    sparse output format: list only the frames that are true

prompts/padq/     full scene, target named by bounding box, ONE ATTRIBUTE PER CALL
  exposed_v3.txt  watched_v2.txt  gender_v3.txt  age_v1.txt
```

`prompts/padq/*` use a different representation from `prompts/crop/*`, so a
comparison across the two groups mixes prompt with representation. They take
`{n}` and `{bboxes}` placeholders; the crop prompts take `{K}`.

To change what ships, edit `EXPOSED_PROMPT` / `EXPOSED_K` / `WATCHED_PROMPT` /
`WATCHED_K` in `config/paths.sh`. **K matters as much as the prompt** — on
identical frames the same prompt moves by up to 0.667 watched F1 between K=8 and
K=1.

---

## Scoring

```bash
python eval/full_grid.py               # every prompt, bootstrap CIs
python eval/momentary_deploy_grid.py   # K=8 deployment path
python eval/k1_grid.py                 # K=1 against K=8, same frames
```

Two constraints are built into these and should not be worked around.

**Scoring is restricted to the sessions that are actually annotated.** In the
corpus this was built on, 4 of 11 sessions carry usable `exposed`/`watched`
labels. The other 7 store an explicit `False` on every row, and that value is
wrong: one of them is the same camera as an annotated session that reads 13.7%
exposed, and faces are plainly visible in its crops. Scoring against it would be
scoring against noise. The four annotated sessions hold 99.8% of exposed
positives and 100% of watched positives, so the restriction costs almost nothing.

**Comparisons carry confidence intervals.** `full_grid.py` resamples the
instances 2,000 times, scoring both arms on the same resample each time, and
reports the 95% interval. Where intervals overlap it says so rather than printing
a ranking. This is load-bearing: on exposed the top four arms are within 0.033 of
each other and cannot be separated by this data, which is not visible from the
point estimates.

---

## Layout

```
config/paths.sh          every absolute path, in one file
setup/check_env.py       pre-flight; refuses to run on a broken environment
setup/fetch_weights.sh   install the identity adapter
setup/patch_paths.py     rewrite baked-in literals to read the environment
pipeline/                the scripts a run calls
eval/                    scoring and comparison tables
prompts/                 all twelve scored prompts
src/                     model loading shared with the training tree
run_all.sh               six stages, resumable
docs/RESULTS.md          measured numbers and how to read them
docs/WEIGHTS.md          adapter download link and checksums
docs/FINAL_REPORT.md     the delivered system: pipeline, model, data, evaluation
```

[docs/FINAL_REPORT.md](docs/FINAL_REPORT.md) is the document to hand to someone
who wants the system rather than its history. It covers the pipeline and the one
structural decision behind it, the model and how it was fine-tuned, the corpus
and its annotation limits, the delivered output, the evaluation methodology, and
a comparison against a public PAR benchmark the model has never seen.

`setup/patch_paths.py --check` verifies no hardcoded path has crept back in.

---

## Data terms

The CCTV corpus is not redistributable through this repository. If you use RAP v2
via `pipeline/rap2_eval.py`, its terms apply: do not redistribute the dataset and
do not use it for commercial purposes.
