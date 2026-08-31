# TrackPAR

Automatic person-attribute labelling for CCTV video. You give it an attribute;
it works out how to label it, runs a vision-language model over your footage, and
writes one label file.

Two ways in:

```bash
source config/paths.sh

# label an attribute of your own, on images or video you already have
bash label_attribute.sh --attr holding_item \
    --definition "the person is holding a product in their hand" \
    --images /data/my_frames

# or reproduce the four attributes this was built and measured on
bash run_all.sh --attrs "exposed watched gender age"
```

The four it ships with, and what they scored:

| attribute | answer | how | measured |
|---|---|---|---|
| gender | one per **track** | Qwen3.5-9B + identity LoRA, K=4 frames per call | 0.9456 accuracy |
| age | one per **track** | same adapter, separate integer prompt, K=4 | MAE 3.63 years |
| exposed | one per **frame** | base 9B, `eyes` prompt, K=1 | F1 0.689 |
| watched | one per **frame** | base 9B, `svfd` prompt, K=1 | F1 0.740 |

Identity figures are on 349 held-out tracks; the age one is worth reading against
the best single constant guess, which scores MAE 10.46. Momentary figures cover
all 5,168 annotated instances. Full tables with confidence intervals are
[below](#results).

The pipeline is not restricted to those four: it decides how to label whatever
attribute you hand it. [Labelling a new
attribute](#labelling-a-new-attribute) is the section to read for that. It needs
neither the corpus nor the adapter, and it needs SAM 3 only if your frames hold
several people and you have no boxes for them.

---

## How the pipeline is shaped

Everything follows from one question asked once per attribute: **is this a
property of the person, or a property of the frame?**

```
                    ┌─────────────────────────────┐
   attribute ──────▶│ 0. route   (VLM, cached)    │
                    └──────────────┬──────────────┘
                      identity     │    momentary
               ┌───────────────────┴───────────────────┐
               ▼                                       ▼
   ┌───────────────────────┐             ┌─────────────────────────────┐
   │ 1. SAM 3 tracking     │             │ no tracking: the boxes      │
   │ 2. fragments          │             │ come straight from the      │
   │ 3. gender   K=4       │             │ annotation file             │
   │ 4. age      K=4       │             │                             │
   │                       │             │ facial / non-facial         │
   │                       │             │   selects the prompt        │
   │ one answer per TRACK  │             │ 5. one call per FRAME       │
   │                       │             │    one answer per FRAME     │
   └───────────┬───────────┘             └─────────────┬───────────────┘
               └───────────────────┬───────────────────┘
                                   ▼
                        6. merge → labels.json
```

**Identity attributes are tracked.** Tracking exists so that K frames of the same
person can go into one call, and that is worth its cost. Same model, same 62
held-out tracks, changing only how the frames are used:

| model | frames used | gender | age MAE |
|---|---|---|---|
| base-4B | one at a time (K=1) | 0.8710 | 11.17 |
| | per frame, then majority vote | 0.9194 | 9.74 |
| | **K frames in one call** | **0.9839** | **8.69** |
| base-9B, no fine-tune | one at a time | 0.8548 | 11.51 |
| | per frame, then majority vote | 0.8871 | 9.45 |
| | **K frames in one call** | **0.9516** | 9.71 |
| 9B, fine-tuned single-image | one at a time | 0.8548 | 8.46 |
| | per frame, then majority vote | 0.9032 | **6.65** |
| | K frames in one call | 0.9000 | 7.70 |
| 9B, fine-tuned multi-image | one at a time | 0.9032 | 7.12 |
| | per frame, then majority vote | 0.9677 | 5.98 |
| | **K frames in one call** | **0.9677** | **5.37** |

Reading down each model:

| model | gender | age MAE |
|---|---|---|
| base-4B | 0.8710 → 0.9839 (**+0.113**) | 11.17 → 8.69 (**−2.48**) |
| base-9B, no fine-tune | 0.8548 → 0.9516 (**+0.097**) | 11.51 → 9.71 (**−1.80**) |
| 9B, FT single-image | 0.8548 → 0.9000 (+0.045) | 8.46 → 7.70 (−0.76) |
| 9B, FT multi-image | 0.9032 → 0.9677 (**+0.065**) | 7.12 → 5.37 (**−1.75**) |

Three things follow. **Grouping beats voting** — majority vote over per-frame
answers already removes noise, but handing K frames to the model at once is
better in every row but one. **The gain is largest where the model is weakest**,
so tracking partly substitutes for model capacity. And **the training format has
to match**: the one row where grouping hurts age (6.65 → 7.70) is the model
fine-tuned on *single* images, for which four frames are off-distribution. The
shipped adapter was fine-tuned multi-image for that reason.

These are 62 tracks from an earlier development set, so read the direction and
size of the effect rather than the third decimal.

**Momentary attributes are not tracked.** The answer is per frame, so knowing
which frames belong to the same person buys nothing. Skipping stages 1–2 also
removes SAM 3 from the requirements — if you only want momentary attributes, you
need neither the tracker nor its gated weights.

**Stage 0 is cached.** An attribute is routed once and stored in
`out/attr_routing.json`. Re-running never re-asks.

### What stage 0 does with a new attribute

```
route ──▶ identity ──▶ K frames of one subject in a single call
      │                  gender and age use the identity adapter;
      │                  any other identity attribute runs on the base model
      │
      └─▶ momentary ──▶ is there a prompt written for this attribute?
                          │
                    yes ──┴──▶ reuse it        (config/prompt_registry.json)
                     no  ─────▶ generate one, from exemplars
                                  facial     → eyes, svfd, combined
                                  non-facial → the PADQ template
```

The two exemplar families also differ in **what the model is shown**, which is
the part that is easy to miss:

| | crop family (`prompts/crop/`) | PADQ (`prompts/padq/`) |
|---|---|---|
| images per call | 2 | 1 |
| what they are | the full scene with the target in a red box, **and** a crop of that person with the background removed | the full scene only |
| how the target is named | visually, by the box and the crop | in text, as `{bboxes}` coordinates |
| people per call | 1 | all `{n}` of them |
| attributes per call | several observation fields | 1 |

Both were measured under `--repr full_mask`, which is what produces that pair of
images, and it is also the deployment default. So the crop family never sees a
crop on its own; it sees the scene as well. The directory name is shorter than
the truth.

Whether an existing prompt is reused comes from an explicit table, not a model
judgement. Two of the four shipped attributes have prompts written specifically
for them; anything else goes to generation.

All of this is what [`label_attribute.sh`](#labelling-a-new-attribute) runs for
you, and that is the normal way in. The stages are also separately callable,
which is useful for inspecting a routing decision before committing to a run:

```bash
# route an attribute, or several
python pipeline/route_attributes.py --attrs "holding_item: the person is holding a product"

# see what has been routed so far (no GPU, no model)
python pipeline/route_attributes.py --show
```

There is also `pipeline/make_prompt.py`, which writes a prompt and stops there.
It predates `label_attribute.py` and leaves the harder half undone: the prompt it
writes names its own output field, and the deployment parsers read the two fixed
names `exposed` and `watched`, so an attribute under any other name parses
cleanly and comes back empty. Use it to look at what a generated prompt contains;
use `label_attribute.sh` to actually label something.

---

## Setup

Two conda environments, because SAM 3 and the VLM stack disagree on
`transformers`: SAM 3 needs 5.8.x, the VLM side is pinned to 5.3.0. **You only
need the second one unless you are labelling identity attributes.**

```bash
git clone https://github.com/Junghyun-James-Park/TrackPAR.git
cd TrackPAR

conda create -n trackpar python=3.11 -y
conda activate trackpar
pip install -r requirements.txt

source config/paths.sh
```

That is enough to run [Labelling a new
attribute](#labelling-a-new-attribute) on person crops, or on frames you already
have boxes for. The three sections below cover the rest:

- **The corpus** — only to reproduce the numbers in [Results](#results). Your own
  data does not go through it.
- **The identity adapter** — only for `gender` and `age`. It was trained on those
  two, so a new attribute gains nothing from it and runs on the base model.
- **SAM 3** — builds tracks for identity attributes, and doubles as the detector
  on raw video. Needed when your frames hold several people and you have no
  boxes for them; not needed for a crop dataset, or when your own detector or
  annotation already supplies boxes.

### Check the parser

Costs nothing and needs no GPU, no model download and no data:

```bash
bash label_attribute.sh --self-test
```

It runs the answer parser against a set of replies, including the shapes that
used to fail silently, and prints `11/11 passed`. Worth doing the moment the
environment is built, because it separates "my environment is wrong" from "the
model answered badly" before either can be confused for the other.

### The corpus

Frames plus per-frame person boxes. Not redistributable here.

```bash
$EDITOR config/paths.sh
```
```
LOTTE_IMAGES   frames, one directory per recording session
LOTTE_ANNOT    per-frame person instances (JSON) — this is where boxes come from
LOTTE_CSV      ground truth; only needed to SCORE, never to label
```

### The identity adapter — 1.2 GB

Needed for `gender` and `age`. Momentary attributes run on the base model.

```bash
pip install gdown
bash setup/fetch_weights.sh --gdrive \
  https://drive.google.com/file/d/1uPuaeyGZKUWyHm7AHU21E4LRdcHYnsrT/view?usp=sharing
```

The script unpacks it and prints the sha256 of both weight files. Check them:

```
63dc9e9ec6df2b4ee100f84e3c5cdcbaef21952efc65d649c4e526cefb8af5c0  adapter_model.safetensors   330.3 MB
408e9eeb88df3985532cac345cd4970e0f700e785ba140c9df7cde1eec562ad3  non_lora_state_dict.bin     869.9 MB
```

A truncated download is otherwise indistinguishable from a complete one.

> **Both files matter.** `PeftModel.from_pretrained` loads
> `adapter_model.safetensors` and **silently ignores** `non_lora_state_dict.bin`,
> which holds the trained vision tower and merger. Nothing raises. A copy missing
> the second file evaluates a base vision tower underneath a fine-tuned adapter —
> a model that never existed — and it scores plausibly enough to look correct.
> `setup/check_env.py` refuses to run without it.

Without the adapter, unset `IDENTITY_ADAPTER` to run identity on the base model.
The pipeline still works; the numbers drop.

### SAM 3 — for tracking, and for boxes you do not have

Two jobs, and you may need neither:

- In the shipped pipeline it builds the **tracks** that let K frames of one
  person go into a single call, which only identity attributes use. Momentary
  attributes take their boxes from the annotation file, so `--attrs "exposed
  watched"` never loads it.
- On your own raw video it is also the **detector**. `label_attribute.py` treats
  a whole image as one subject unless you pass `--boxes`, so full frames with
  several people need boxes from somewhere, whatever the attribute is. A crop
  dataset needs none of this.

Public, but the weights are gated.

```bash
git clone https://github.com/facebookresearch/sam3
conda create -n sam3 python=3.12 -y && conda activate sam3
pip install -r requirements-sam3.txt
pip install -e /path/to/sam3

huggingface-cli login     # after accepting the terms at
                          # https://huggingface.co/facebook/sam3
```

Then point `SAM3_ENV` and `SAM3_SRC` in `config/paths.sh` at them. If you already
have tracks, or only want momentary attributes, skip all of this.

### Check before running

```bash
conda activate trackpar
source config/paths.sh
python setup/check_env.py
```

Twelve checks. Each one corresponds to a mistake that cost GPU time here — an
unwritable `HF_HOME` that reads like a HuggingFace outage, a missing
`non_lora_state_dict.bin`, a pipeline module that only fails once the run reaches
it.

---

## Labelling a new attribute

One command. Give it a name, a sentence of definition, and a folder of images or
a video:

```bash
bash label_attribute.sh \
    --attr holding_item \
    --definition "the person is holding a product in their hand" \
    --images /data/my_frames
```

It writes `out/holding_item.json` and `out/holding_item.csv`, with the column
named after your attribute. Nothing is renamed by hand.

```
subject,image,n_frames,holding_item,status
frame_0001.jpg,frame_0001.jpg,1,True,ok
frame_0002.jpg,frame_0002.jpg,1,False,ok
```

What runs, and in what order:

**0. route** — identity or momentary, and if momentary, facial or not. Cached in
`out/attr_routing.json`, so an attribute is asked about once.

**1. prompt** — already in `config/prompt_registry.json`? Reuse the measured one.
Otherwise write one from exemplars.

**2. infer** — momentary runs one call per image at K=1 with no tracking;
identity puts K frames of one subject into a single call.

**3. write** — `.json` carries the routing, the prompt source and the raw text of
every answer that failed. `.csv` carries just the labels.

### Check it before spending GPU time

```bash
bash label_attribute.sh --self-test
```

Runs the parser against a set of answers, including the shapes that used to fail
silently. No GPU, no model download, a second or two.

### Your data

```bash
# a folder of person crops — each image is one subject
--images /data/crops

# full frames plus boxes — each box is one subject
--images /data/frames --boxes boxes.json

# a video — frames are extracted with ffmpeg first
--video site.mp4 --fps 1
```

`boxes.json` is either `{"frame1.jpg": [[x1,y1,x2,y2], ...], ...}` or a list of
`{"image": ..., "box": [...], "track_id": ...}`.

For an **identity** attribute the answer is one per subject, so the frames of one
person have to be grouped. Either give `track_id` in the boxes file, or pull it
out of the filename:

```bash
--track-regex "tarid([0-9]+)"      # capture group 1 is the subject id
```

Without either, every image becomes its own subject and the run tells you so.

### Try it small first

```bash
--limit 50
```

Processes the first 50 units after a stable sort, so repeated runs pick the same
50. The run prints that the result is a subset; a positive rate measured on it is
not the rate for your corpus.

### Reading the output

Two numbers are printed, and the second is the one that matters:

```
parsed as JSON : 24/24
usable answers : 24/24   <-- the number that matters
```

They come apart. A prompt whose schema the model ignores still returns valid
JSON: one arm in this project parsed 799 of 800 answers while only 536 carried
the requested field. The `status` column separates `no-json`, `no-field` and
`bad-value`, and the JSON keeps the raw text of every answer that was not usable.

Then compare `predicted positive` against the rate you expect. A prompt far off
that rate is usually mis-thresholded rather than blind, which is a different
repair — see [docs/RESULTS.md](docs/RESULTS.md).

### What to expect from a generated prompt

**It has not been measured.** No generated prompt in this repository has been
scored against ground truth. Both exemplar sets were written for face-visibility
attributes, and the non-facial set is the PADQ template, which assumes a full
scene with the target given as a box. Treat generated labels as a first pass, and
check a sample by hand before building on them.

Two things are guaranteed rather than hoped for. The output contract — the last
paragraph of every generated prompt, fixing the field name — is written by the
runner, not by the model, so the answer cannot come back under a name the reader
is not looking for. And a generated prompt is validated before use; if it comes
back as commentary rather than instructions, the runner retries once and then
falls back to a plain template built from your definition, saying so.

For an **identity** attribute there is nothing to generate from: both exemplar
sets are momentary. The runner uses a built-in multi-frame template instead and
prints that it did.

Prompts land in `prompts/generated/<attr>.txt`. Edit one and re-run; it is reused
unless you pass `--regenerate`.

**A different model writes the prompt.** `PROMPT_MODEL` in `config/paths.sh`
defaults to `Qwen/Qwen3.5-27B` while labelling and routing stay on
`BASE_MODEL`. Routing asks a two-way question that 9B answers; a prompt is read
on every one of thousands of calls afterwards, so it is worth more there. The
writer is loaded only when there is something to generate and freed straight
after, so the two never share the cards. Set `PROMPT_MODEL=$BASE_MODEL` to use
one model for everything.

### Attribute names to avoid

`eyes`, `nose`, `mouth`, `gaze`, `frames`, `gender`, `age`, `bbox_2d` are field
names the exposed/watched parser reads as a schema signal. The runner refuses
them rather than guessing — use `eyes_closed` rather than `eyes`.

`exposed` and `watched` are accepted and take a different path: they have
measured prompts in the registry, and the runner defers to the one those numbers
came from.

---

## Running the shipped pipeline

```bash
source config/paths.sh

bash run_all.sh                                   # the four shipped attributes
bash run_all.sh --attrs "exposed watched"         # momentary only — no tracking
bash run_all.sh --from fragments                  # reuse existing tracks
bash run_all.sh --score                           # and grade afterwards
```

Every stage skips itself when its output exists, so an interrupted run resumes.

| stage | script | output | cost |
|---|---|---|---|
| 0 | `route_attributes.py` | `out/attr_routing.json` | seconds, cached |
| 1 | `track_sam3_chunked.py` | `out/track_sam3/` | hours, **sam3 env** |
| 2 | `phase1_build_all_fragments.py` | `out/phase1_fragments.json` | minutes, CPU |
| 3 | `multiimg_eval.py` | `out/identity.json` | ~2.5 h |
| 4 | `age_eval.py` | `out/age.json` | ~40 min |
| 5 | `momentary_k1_control.py` | `out/momentary_*.json` | ~6 h per attribute, two GPUs |
| 6 | `merge_labels.py` | `out/labels.json` | seconds |

Set `TRACKPAR_GPUS="0,1"` for the two cards to use. Stage 5 shards across both.

### Changing which prompt an attribute uses

Prompts are plain text files, and each one is paired with the parser that reads
its answers back. Both live in `config/paths.sh`:

```bash
export EXPOSED_PROMPT="$TRACKPAR_ROOT/prompts/crop/combined.txt"
export EXPOSED_STYLE=meta      # parser: observation fields
export WATCHED_PROMPT="$TRACKPAR_ROOT/prompts/crop/svfd.txt"
export WATCHED_STYLE=svfd      # parser: svfd's own schema
```

`STYLE` is the part that is easy to get wrong. Each style is a matched pair of
prompt text and the reader for the answers that text asks for:

- **`meta`** — the only style that reads `PROMPT` from a file. It expects the
  observation fields (`eyes`, `nose`, `mouth`, `gaze`) that the
  `prompts/crop/*` family emits, and applies the rule in code afterwards. This
  is the style to use for a prompt of your own.
- **`svfd`** — builds its own text and reads its own schema, the graded
  visibility tier. `PROMPT` is not read.
- **`plain`**, **`trueonly`** — likewise self-contained; `PROMPT` is not read.
- **`padq`** — self-contained, full scene with the target given as a box, one
  attribute per call. `PROMPT` is not read.

So there are two ways to get it wrong, and neither raises:

- Pointing `PROMPT` at a new file while leaving `STYLE` on a self-contained
  style. Your file is never opened and the old prompt runs.
- Setting `STYLE=meta` with a prompt that answers in some other schema. The
  answer parses as valid JSON, the fields the reader wants are absent, and the
  attribute comes back empty or always-false.

`setup/check_env.py` rejects an unknown `STYLE` and says plainly when `PROMPT`
is not the file being read. Run it after any swap, then watch the derive rate
below.

The runner substitutes `{K}` for the frame count in `prompts/crop/*`, and `{n}` /
`{bboxes}` in `prompts/padq/*`.

To try a prompt before committing to a full run:

```bash
python pipeline/momentary_targeted_eval.py \
    --natural --session <SESSION_ID> --target watched \
    --n-pos 400 --n-neg 400 --repr full_mask \
    --prompt-file prompts/crop/eyes.txt --prefill '{"frames": [{"eyes": "' \
    --out out/try_eyes.json
```

Watch the **derive rate**, not the parse rate. A prompt whose schema the model
ignores still returns valid JSON: one arm here read "parse-valid 799/800" while
only 536 answers carried the requested field. The runner prints
`derive <attr>: N/M answers usable` and flags anything between 0% and 90%.

---

## The prompts, and the ideas behind them

Twelve prompts, all of which were scored. They fall into families, and the
families are the interesting part.

### `prompts/crop/` — one person per call, scene plus crop

| prompt | idea |
|---|---|
| `plain` | Ask for the attribute directly, with a one-line definition. The control. |
| `trueonly` | Sparse output for a sparse event: list only the frames where it is true, so an empty list is the normal answer. |
| `svfd` | Do not ask for a verdict; ask for a graded **visibility tier** (`clearly_visible` / `partially_visible` / `not_visible`) plus an explicit rarity prior, then apply the rule in code. |
| `eyes` | Push that further: ask only what is **observable** — are the eyes resolvable, which way is the gaze — and derive the attribute outside the model. |
| `features` | Same move, split three ways: eyes, nose, mouth reported separately. |
| `combined` | All four observations in one call, so several decision rules can be compared afterwards from one run. |
| `subattr` | The 40-attribute identity schema, with the momentary attribute derived from it. |
| `metav4` | Machine-optimised (OPRO) over earlier prompts. |

The thread running through `svfd → eyes → features → combined` is **moving the
decision out of the model**. The model is asked what it can see; the rule that
turns observations into a label lives in code, where it can be changed without
re-running anything.

That move is what rescued `watched`. It did **not** help `exposed`: six different
rules applied to `combined`'s stored answers span only 0.005, because 99% of
answers set eyes, nose and mouth to the same value. The model makes one
visibility judgement and repeats it three times, so there is nothing to combine.

### `prompts/padq/` — full scene, one attribute per call

The people are already detected; the prompt names them by bounding box and asks a
single true/false question about each. `exposed_v3` and `watched_v2` differ only
in the attribute name and its definition — **the rest is a template**, which is
why this family is what stage 0 generates from for a new non-facial attribute.

It also needs no crop and no track: a frame and its boxes are enough.

---

<a id="results"></a>

## Results

Measured over **every annotated instance of the four annotated sessions**: 5,168
instances, 1,353 `exposed` positives, 201 `watched` positives. Intervals are 95%
paired bootstrap over instances, 2,000 resamples, both arms scored on the same
resample each time. Reproduce with `python eval/full_grid.py`.

### exposed — true rate 26.2%

| prompt | F1 | 95% CI | bAcc | P | R | predicted+ |
|---|---|---|---|---|---|---|
| combined | **0.697** | [0.678, 0.716] | 0.810 | 0.626 | 0.788 | 33.0% |
| eyes | 0.689 | [0.670, 0.708] | 0.799 | 0.637 | 0.751 | 30.9% |
| subattr | 0.677 | [0.658, 0.696] | 0.790 | 0.628 | 0.735 | 30.6% |
| PADQ `exposed_v3` * | 0.664 | [0.646, 0.682] | 0.797 | 0.550 | 0.837 | 39.9% |
| features | 0.659 | [0.640, 0.677] | 0.789 | 0.559 | 0.802 | 37.6% |
| svfd | 0.618 | [0.600, 0.636] | 0.766 | 0.482 | 0.859 | 46.6% |
| metav4 | 0.607 | [0.589, 0.625] | 0.760 | 0.462 | 0.886 | 50.2% |
| plain | 0.515 | [0.497, 0.531] | 0.666 | 0.356 | 0.927 | 68.1% |
| trueonly | 0.453 | [0.438, 0.469] | 0.574 | 0.295 | 0.985 | 87.6% |

**The top four are a tie** — overlapping intervals. 5,168 instances cannot
separate them, and `exposed` looks capped near 0.70 at these crop sizes.

### watched — true rate 3.9%

| prompt | F1 | 95% CI | bAcc | P | R | predicted+ |
|---|---|---|---|---|---|---|
| svfd | **0.740** | [0.689, 0.784] | 0.888 | 0.694 | 0.791 | 4.4% |
| PADQ `watched_v2` * | 0.585 | [0.536, 0.628] | 0.920 | 0.436 | 0.886 | 7.9% |
| eyes | 0.508 | [0.459, 0.555] | 0.882 | 0.367 | 0.821 | 8.7% |
| combined | 0.324 | [0.249, 0.395] | 0.609 | 0.584 | 0.224 | 1.5% |
| trueonly | 0.312 | [0.243, 0.377] | 0.615 | 0.434 | 0.244 | 2.2% |
| plain | 0.259 | [0.183, 0.330] | 0.580 | 0.611 | 0.164 | 1.0% |
| metav4 | 0.211 | [0.186, 0.237] | 0.831 | 0.119 | 0.945 | 30.9% |
| subattr | 0.165 | [0.102, 0.230] | 0.550 | 0.333 | 0.109 | 1.3% |

**`svfd` is separated from every other arm.** The one place a prompt choice buys
something unambiguous.

\* PADQ sees the full scene with the target named by a bounding box rather than a
crop, and answers one attribute per call. A gap against the other rows is prompt
**and** representation together.

Note the `predicted+` column. The worst arms are not failing to *see* the
attribute — `trueonly` has recall 0.985 on exposed — they are calling almost
everything positive. Distance from the true rate orders the ranking.

### Identity

| | value | baseline |
|---|---|---|
| gender | 0.9456 | — |
| age MAE | 3.63 years | best constant guess 10.46 |

On the 349 held-out tracks. **Do not quote whole-corpus identity scores**: run
over every track, age reports MAE 1.16, but 924 of the 1,262 scored tracks were
in the adapter's training data.

More, including the comparison against RAP v2 which the model has never seen:
[docs/RESULTS.md](docs/RESULTS.md) and [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md).

---

## Layout

```
config/paths.sh              every absolute path, in one file
config/prompt_registry.json  which attributes already have a prompt
setup/check_env.py           pre-flight; refuses to run on a broken environment
setup/fetch_weights.sh       install the adapter, local or from Drive
pipeline/route_attributes.py stage 0
label_attribute.sh           label ONE attribute on your own data
pipeline/label_attribute.py  the end-to-end runner behind it
pipeline/make_prompt.py      write a prompt for a new attribute
pipeline/                    the rest of the run
eval/                        scoring, with confidence intervals
prompts/crop/                8 prompts, one person per call (scene + crop)
prompts/padq/                4 prompts, full scene + boxes, one attribute per call
run_all.sh                   all stages, resumable
docs/FINAL_REPORT.md         the delivered system
docs/RESULTS.md              every measured number and how to read it
```

---

## Scoring caveat

Momentary scoring is restricted to the sessions that are actually annotated. In
the corpus this was built on, 4 of 11 carry usable `exposed`/`watched` labels; the
other 7 record an explicit `False` on every row, and that value is wrong — one of
them is the same camera as an annotated session reading 13.7% exposed, with faces
plainly visible in its crops. The four annotated sessions hold 99.8% of exposed
positives and 100% of watched positives, so the restriction costs almost nothing.

## Data terms

The CCTV corpus is not redistributable through this repository. If you use RAP v2
via `pipeline/rap2_eval.py`, its terms apply: do not redistribute the dataset and
do not use it for commercial purposes.
