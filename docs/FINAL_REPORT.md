# TrackPAR — Final Report

Person-attribute auto-labelling for overhead retail CCTV. This document describes
the delivered system: the pipeline, the model, the data, and the evaluation
methodology. It does not narrate how the system was arrived at.

---

## 1. The pipeline

Six stages. SAM 3 finds and tracks every person; a vision-language model then
labels four attributes.

```
1  SAM 3 tracking          video frames  ->  per-person tracks
2  fragment building       tracks        ->  usable tracks with boxes and images
3  gender                  K=4 frames per call, identity LoRA
4  age                     K=4 frames per call, identity LoRA, integer output
5a exposed                 K=8 frames per call, base model, `eyes` prompt
5b watched                 K=1 frame per call, base model, `svfd` prompt
6  merge                   one label file
```

### The one structural decision

**Attributes are split by kind, and the split determines the shape of the
answer.**

| kind | attributes | answer | why |
|---|---|---|---|
| identity | gender, age | one per **track** | stable across the track; more views help |
| momentary | exposed, watched | one per **frame** | changes within the track |

This is not a stylistic choice. Measured on tracks holding at least one positive,
`exposed` changes between consecutive frames **71.8%** of the time and `watched`
**85.7%**. A single value per track for those two would not be a coarse answer;
it would be the wrong kind of answer.

### Why tracking helps identity

Same model, same 62 held-out tracks, changing only how frames are used:

| model | frames used | gender | age MAE |
|---|---|---|---|
| base-4B | one at a time | 0.8710 | 11.17 |
| | majority vote over frames | 0.9194 | 9.74 |
| | **K frames in one call** | **0.9839** | **8.69** |
| base-9B (no fine-tune) | one at a time | 0.8548 | 11.51 |
| | **K frames in one call** | **0.9516** | 9.71 |
| 9B, fine-tuned single-image | one at a time | 0.8548 | 8.46 |
| | K frames in one call | 0.9000 | 7.70 |
| 9B, fine-tuned multi-image | one at a time | 0.9032 | 7.12 |
| | **K frames in one call** | **0.9677** | **5.37** |

Gain from tracking: gender **+0.045 to +0.113**, age MAE **−0.76 to −2.48**.

Three properties follow.

**Grouping beats voting.** Majority vote over per-frame answers already removes
noise, but passing K frames in one call is better in every row but one — the
model uses several views jointly rather than averaging independent guesses.

**The gain is largest where the model is weakest.** base-4B gains +0.113 gender;
the strongest arm gains +0.065. Tracking partly substitutes for model capacity.

**Fine-tuning and inference format must match.** The single row where grouping
hurts age (6.65 → 7.70) is the model fine-tuned on *single* images; four frames
are off-distribution for it. The shipped adapter was fine-tuned on multi-image
inputs for this reason and is the only arm best at both.

### Why exposed and watched use different prompts and different K

They were measured under one protocol and they do not agree on a setting.

| | best prompt | at K | F1 |
|---|---|---|---|
| exposed | `eyes` | 8 | 0.728 |
| watched | `svfd` | **1** | 0.740 |

`svfd` at K=8 scores watched **0.000**. Packed into a multi-frame call, the model
applies the prompt's stated rarity prior to the whole batch and answers no to
every frame. At K=1 it judges each frame and reaches 0.740. `exposed` shows no
such rule — the K effect there is prompt-dependent.

The cost of this is concentrated in stage 5b: K=1 means one call per frame,
~4.4 s each over ~9,500 frames, about 11 h on two GPUs. Stage 5a covers eight
frames per call and finishes in ~3 h.

---

## 2. The model

| | |
|---|---|
| backbone | Qwen3.5-9B (vision-language) |
| identity adapter | LoRA r=64, α=128, plus a trained vision tower and merger |
| momentary | the **base** model, no adapter |

**Identity and momentary deliberately use different models.** Fine-tuning for
identity destroys momentary prompt-following: the same prompt scores exposed F1
0.730 on base 9B and 0.074 on a 9B fine-tuned for age. The two passes therefore
load different weights.

### Fine-tuning

| | |
|---|---|
| method | LoRA on the language tower, with the vision tower and merger also trained |
| data | UPAR 18,000 single-image rows + 9,240 in-domain multi-image rows with GT |
| in-domain share | 33.9% |
| precision | bf16 |
| parallelism | DeepSpeed ZeRO-2 (ZeRO-3 deadlocks on this setup; NF4 + ZeRO-2 works for 27B) |

**Deployment note.** The trained vision tower is saved separately as
`non_lora_state_dict.bin`. `PeftModel.from_pretrained` loads only
`adapter_model.safetensors` and ignores it silently — no error is raised. Loading
without it evaluates a base vision tower under a fine-tuned adapter, a model that
never existed, and the scores look plausible. `setup/check_env.py` refuses to run
if that file is absent.

---

## 3. The data

### Target corpus

11 CCTV sessions from a single day, one retail store, ceiling-mounted cameras.

| | count |
|---|---|
| frames | 4,351 |
| annotated person instances | 15,909 |
| SAM 3 tracks | 2,438 |
| tracks matched to a GT identity | 1,273 |
| held-out tracks for identity scoring | 349 |

### Annotation coverage — a constraint on every momentary number

| session (start time) | group | instances | exposed+ | watched+ | annotated |
|---|---|---|---|---|---|
| 08:06 | 1-image | 4,592 | 3 | 0 | no |
| 08:14 | 1-image | 1,104 | 0 | 0 | no |
| 08:22 | 1-image | 978 | 0 | 0 | no |
| 08:36 | 1-image | 1,672 | 0 | 0 | no |
| 08:44 | 1-image | 1,149 | 0 | 0 | no |
| 09:04 | 2-image | 737 | 240 (33%) | 9 (1.2%) | **yes** |
| 09:12 | 2-image | 1,055 | 320 (30%) | 26 (2.5%) | **yes** |
| 09:20 | 2-image | 1,246 | 501 (40%) | 158 (12.7%) | **yes** |
| 09:32 | 3-image | 2,130 | 292 (14%) | 8 (0.4%) | **yes** |
| 09:48 | 3-image | 926 | 0 | 0 | no |
| 09:56 | 3-image | 320 | 0 | 0 | no |

Only 4 of 11 sessions carry usable momentary labels. The other seven record an
explicit `False` on every row rather than a blank, so nothing downstream can
distinguish "checked and negative" from "never checked" — **and that value is
wrong**. Session 09:48 uses the same freezer-aisle camera as the annotated 09:32
(which reads 13.7% exposed) and faces are plainly visible in its crops.
Annotation appears to have been done in batches aligned with the image-group
column: all three `2-image` sessions are annotated.

Consequences, applied throughout:

1. Momentary scoring is restricted to the four annotated sessions. They hold
   99.8% of exposed positives and 100% of watched positives, so almost no signal
   is lost.
2. The unannotated sessions cannot serve as a specificity test. An arm predicting
   many positives there may simply be correct.

`watched` is the binding constraint on the whole evaluation: **201 positives
corpus-wide, 158 of them (79%) in one session.**

### External data

| dataset | role | seen in training? |
|---|---|---|
| UPAR (PA100K + PETA + Market1501) | fine-tuning data | **yes** |
| RAP v2 | external test | **no** — UPAR excludes it by licence |

Only RAP v2 supports a generalisation claim. Any UPAR number is a training-set
score and is labelled as such.

---

## 4. Delivered output

`labels_final.json`, covering all 2,438 tracks.

| attribute | coverage | distribution |
|---|---|---|
| gender | 2,438 tracks (100%) | male 1,264 / female 1,174 |
| age | 2,427 tracks (99.5%) | median 47, range 1–65 |
| exposed | 18,007 frame rows | 25.4% positive |
| watched | 9,541 frame rows | 1.05% positive |

`exposed` and `watched` cover different frame counts because they draw from
different frame indices — exposed from every tracked frame, watched from the
frames whose image resolved. Each column carries values only where its source
covered.

### Held-out accuracy

| | value | baseline |
|---|---|---|
| gender | 0.9456 | — |
| age MAE | 3.63 years | best constant guess 10.46 |
| exposed F1 | 0.728 | always-negative bAcc 0.500 |
| watched F1 | 0.740 | always-negative bAcc 0.500 |

Identity figures are on the 349 held-out tracks. Momentary figures are over all
5,168 annotated instances.

**Do not quote whole-corpus identity scores.** Running age over all tracks
reports MAE 1.16, but 924 of the 1,262 scored tracks were in the adapter's
training data; held-out tracks give 3.63 and training tracks give 0.25. A gender
figure of 0.9811 was withdrawn earlier for the same reason.

---

## 5. Evaluation methodology

Two techniques below are not standard practice in this project's prior work and
are the part most reusable elsewhere.

### 5.1 Confidence intervals: deciding whether a difference is real

**The problem.** Comparing prompts by a single score invites reading noise as a
ranking. Prompt A at 0.697 and prompt B at 0.689 looks like an ordering. It may
not be one: with a different sample of instances the order could flip.

**What we do.** Resample the evaluation set with replacement 2,000 times. Each
resample scores **both** arms on the **same** instances, so the comparison is
paired and only the arms differ. The 2.5th and 97.5th percentiles of the 2,000
scores give a 95% interval. For binary per-instance correctness we also run an
exact McNemar test.

**What it changes.** On exposed, the top four arms have overlapping intervals:

```
combined  0.697 [0.678, 0.716]
eyes      0.689 [0.670, 0.708]
subattr   0.677 [0.658, 0.696]
PADQ      0.664 [0.646, 0.682]
```

5,168 instances cannot separate these. The correct report is "four arms tied",
not "combined wins". On watched the same procedure gives the opposite verdict —
`svfd` at 0.740 [0.689, 0.784] does not overlap the runner-up at 0.585 — so that
choice is real and worth acting on.

**Why this matters commercially.** Prompt and model changes cost engineering time
and re-labelling budget. This method separates changes worth deploying from
changes that will not survive the next batch of data. It also prevents the
opposite failure: on watched, an eyeball comparison of 0.740 against 0.585 might
be dismissed as within noise, when the intervals show it is not.

One caution to state plainly: **bootstrap intervals assume the labels are
correct.** They quantify sampling uncertainty, not annotation error. Where labels
are unreliable the intervals will be narrow and confidently wrong, which is why
scoring here is restricted to the sessions that are actually annotated.

### 5.2 Predicted positive rate: separating two failure modes

**The problem.** Accuracy alone conflates a model that cannot see an attribute
with a model that sees it but applies the wrong threshold. They need different
fixes — more capacity or better images in the first case, prompt calibration in
the second — so telling them apart matters.

**What we do.** Report each arm's **predicted positive rate** beside the true
prevalence, and compare across populations with different prevalence. Two
sessions here have true exposed rates of 39.5% and 13.9%, a 2.8× difference,
which makes the check possible.

**What it shows.**

| prompt | predicted+ (39.5% true) | predicted+ (13.9% true) | reading |
|---|---|---|---|
| plain | 69% | 69% | does not move — threshold comes from the prompt, not the image |
| trueonly | 86% | 88% | moves the wrong way |
| features | 50% | 24% | tracks the truth |

Over the full evaluation set the diagnostic orders the ranking monotonically:
distance from the true rate predicts performance, while the worst arms have the
**highest** recall (`trueonly` reaches 0.985). They are not failing to see the
attribute. They are calling almost everything positive.

**Why this matters commercially.** For an auto-labeller, over-prediction is the
expensive failure: every false positive becomes a human review. This diagnostic
identifies which arms will generate that cost before deployment, and it does so
using aggregate counts rather than per-instance correctness, so it degrades
gracefully when labels are imperfect.

**It is necessary, not sufficient.** `subattr` had the closest predicted rates of
any arm on both sessions (43% / 15% against 39.5% / 13.9%) and still scored worst
on the harder one, with precision 0.171 and recall 0.189. It counted correctly
and picked the wrong instances. Calibration and discrimination must be read
separately.

### 5.3 Schema adherence, not parse rate

A prompt whose output schema the model ignores still returns valid JSON. One arm
here reported "parse-valid 799/800" while only 536 of those answers contained the
requested field — the model had replied with a free-form description. Parse rate
cannot see this. The evaluation therefore reports the **derivation rate**: the
share of answers from which the target attribute could actually be extracted, and
flags anything between 0% and 90%.

---

## 6. How this compares to a public PAR benchmark

The pipeline was built for one deployment, so the natural question is whether any
of it transfers. Two datasets were used, and they support different claims.

| dataset | relationship to training | supports |
|---|---|---|
| **RAP v2** | never seen — UPAR excludes it by licence | generalisation |
| **UPAR** | the adapter's training distribution; scored on the rows held out of it | a weaker, in-distribution claim |

### 6.1 Fine-tuning transfers to an unseen dataset

RAP v2 test, 2,000 samples, mA (mean per-attribute balanced accuracy), the
UPAR-challenge metric. Parse rate 100% on both arms.

| arm | mA | vs always-negative (0.500) |
|---|---|---|
| base Qwen3.5-9B | 0.7727 | +0.273 |
| **+ our adapter** | **0.7968** | +0.297 |
| | **Δ +0.0241** | |

The gain is broad rather than concentrated: **22 of 36 attributes improve, 10
regress, 4 are unchanged.**

| attribute group | n | base | ours | Δ |
|---|---|---|---|---|
| Hair | 3 | 0.839 | 0.897 | **+0.058** |
| Accessory | 4 | 0.886 | 0.932 | **+0.046** |
| Age | 3 | 0.734 | 0.768 | +0.034 |
| UpperBody | 13 | 0.797 | 0.818 | +0.021 |
| LowerBody | 12 | 0.685 | 0.696 | +0.011 |
| Gender | 1 | 0.969 | 0.979 | +0.010 |

**Where the gain comes from.** The largest improvements are all rare attributes:
Hair-Bald +0.149 (61 positives), UpperBody-Orange +0.136 (51), Age-Old +0.130
(34), LowerBody-Other +0.125 (110). Fine-tuning filled in classes the base model
handled poorly.

**Where it costs.** Regressions cluster in LowerBody — Length-Short −0.076,
Green −0.073, Blue −0.064. This is consistent with the training domain: a
ceiling-mounted camera frequently truncates the lower body, so those attributes
are under-represented in what the adapter learned from. **Age-Young also drops
(0.849 → 0.787)**, which matches the target corpus being 96.4% adult.

Gender was already at 0.969 and had almost no headroom (+0.010).

### 6.2 UPAR, for contrast

UPAR's `val_task1` is the challenge submission list and carries no labels, so the
pool is `train.csv` minus the rows the adapter trained on.

| arm | mA |
|---|---|
| base | 0.8046 |
| + our adapter | 0.8209 |
| | Δ +0.0163 |

**The in-distribution gain is smaller than the out-of-distribution one**
(+0.016 against +0.024), mostly because the base model already starts higher
here. Read this number with care: the balanced training subset absorbed most of
the rare positives, so the remaining pool is depleted of exactly the classes
where fine-tuning helped on RAP v2.

### 6.3 Age: why this pipeline predicts an integer

PAR benchmarks label age as three binary columns. The pipeline predicts an
integer, so comparing requires binning. The cut below is RAP v2's **own**
definition (`age_le16`, `age_gt60`), not one fitted to the results; the other
rows show how much the choice moves things.

| cut (young ≤, old >) | accuracy | mA(3) | young | adult | old |
|---|---|---|---|---|---|
| **16 / 60** (RAP v2's own) | **0.9645** | 0.6913 | 0.806 | 0.726 | 0.542 |
| 18 / 60 | 0.8935 | 0.6697 | 0.775 | 0.692 | 0.542 |
| 25 / 60 | 0.8785 | 0.6982 | 0.827 | 0.725 | 0.542 |
| 30 / 60 | 0.8710 | 0.6956 | 0.823 | 0.721 | 0.542 |
| 30 / 55 | 0.8575 | **0.7218** | 0.823 | 0.735 | 0.607 |
| 35 / 65 | 0.7740 | 0.6464 | 0.777 | 0.662 | 0.500 |
| *always "adult"* | *0.9430* | *0.500* | — | — | — |

**The two metrics point in opposite directions**, and that is the finding.
Accuracy is maximised by the widest cut, which sends almost everyone to "adult";
mA is maximised by a narrower one that catches more young and old at the cost of
accuracy. This happens because **RAP v2 is 94.3% adult** — the same degeneracy as
the target corpus at 96.4%.

The consequence: the best 3-class accuracy, 0.9645, beats always-answering-adult
by only **0.0215**. The scale cannot express what the model knows. The same
predictions scored as integers give **MAE 3.63 against a best-constant baseline
of 10.46** on the held-out set. This is why the deliverable emits an integer and
bins only when a benchmark demands it.

(`old` never exceeds bAcc 0.542 under any cut, but with 34 positives that
number is too unstable to interpret.)

### 6.4 The 40 attributes do not include what this pipeline adds

Asking the model to sort the 40 UPAR attributes along two separate axes:

| axis | count |
|---|---|
| the property itself changes during a track | **5** |
| the property can be unobservable in a given frame | **36** |

The five that change are exactly the removable ones: backpack, bag, glasses
(normal and sun), hat. The other 31 are fixed properties — a black shirt stays
black — that can nonetheless be unreadable from a given viewpoint.

**The schema annotates neither axis.** It records what the person *is*, once per
image, with no field for whether the attribute was legible in that image. Our
`exposed` and `watched` are precisely that missing axis, which is why they have
no counterpart in the benchmark to be compared against.

A first version of this probe asked one merged question and the model answered
33 of 40 "momentary", reasoning about occlusion ("visible colour can change due
to lighting") rather than about change. That conflation was the prompt's fault,
but it is also evidence that observability is a natural axis to read into these
attributes — one the benchmark simply does not carry.

---

## 7. Summary of what is claimed

| claim | evidence | strength |
|---|---|---|
| gender 0.9456, age MAE 3.63 on the target corpus | 349 held-out tracks | strong |
| tracking beats per-frame classification for identity | 4 models, same 62 tracks, +0.045 to +0.113 gender | strong |
| exposed and watched need different prompts and different K | 5,168 instances, non-overlapping intervals on watched | strong |
| exposed is capped near 0.70 at these crop sizes | four arms tie within overlapping intervals | moderate |
| fine-tuning transfers to an unseen public dataset | RAP v2, +0.0241 mA, 22/36 attributes improve | moderate |
| 3-class age cannot express integer-age ability | 0.9645 against a 0.9430 majority baseline | strong |
| the PAR schema has no field for observability | 5 changeable / 36 occludable / 0 annotated | supporting |

Not claimed: that this generalises to other stores, other camera geometries, or
other populations. Only one deployment was annotated, and 7 of its 11 sessions
carry momentary labels that are demonstrably wrong.

---

## 8. Reproduction

See [README.md](../README.md). Verified: gender reproduces at 0.9456 from a clean
checkout, matching the figure above exactly.
