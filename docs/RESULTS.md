# Measured results, and how to read them

All momentary numbers below come from **every annotated instance of the four
annotated sessions**: 5,168 instances, 1,353 exposed positives, 201 watched
positives. Intervals are 95% paired bootstrap over instances, 2,000 resamples,
both arms scored on the same resample each time.

Reproduce with `python eval/full_grid.py`.

## exposed — true rate 26.2%

| prompt | F1 | 95% CI | bAcc | P | R | predicted+ |
|---|---|---|---|---|---|---|
| combined | **0.697** | [0.678, 0.716] | 0.810 | 0.626 | 0.788 | 33.0% |
| eyes | 0.689 | [0.670, 0.708] | 0.799 | 0.637 | 0.751 | 30.9% |
| subattr | 0.677 | [0.658, 0.696] | 0.790 | 0.628 | 0.735 | 30.6% |
| PADQ exposed_v3 * | 0.664 | [0.646, 0.682] | 0.797 | 0.550 | 0.837 | 39.9% |
| features | 0.659 | [0.640, 0.677] | 0.789 | 0.559 | 0.802 | 37.6% |
| svfd | 0.618 | [0.600, 0.636] | 0.766 | 0.482 | 0.859 | 46.6% |
| metav4 | 0.607 | [0.589, 0.625] | 0.760 | 0.462 | 0.886 | 50.2% |
| plain | 0.515 | [0.497, 0.531] | 0.666 | 0.356 | 0.927 | 68.1% |
| trueonly | 0.453 | [0.438, 0.469] | 0.574 | 0.295 | 0.985 | 87.6% |

**The top four are a tie.** combined, eyes, subattr and PADQ have overlapping
intervals. 5,168 instances cannot separate them, and more of the same data
probably will not either — exposed appears to be capped near 0.70 at these crop
sizes. Supporting evidence: on the `combined` prompt, **99% of answers set eyes,
nose and mouth to the same value** (437 all-no, 357 all-yes, 6 mixed out of 800).
The model makes one visibility judgement and repeats it three times, so the
three-way rule has nothing to combine. Six different decision rules applied to
those stored answers span only 0.005.

## watched — true rate 3.9%

| prompt | F1 | 95% CI | bAcc | P | R | predicted+ |
|---|---|---|---|---|---|---|
| svfd | **0.740** | [0.689, 0.784] | 0.888 | 0.694 | 0.791 | 4.4% |
| PADQ watched_v2 * | 0.585 | [0.536, 0.628] | 0.920 | 0.436 | 0.886 | 7.9% |
| eyes | 0.508 | [0.459, 0.555] | 0.882 | 0.367 | 0.821 | 8.7% |
| combined | 0.324 | [0.249, 0.395] | 0.609 | 0.584 | 0.224 | 1.5% |
| trueonly | 0.312 | [0.243, 0.377] | 0.615 | 0.434 | 0.244 | 2.2% |
| plain | 0.259 | [0.183, 0.330] | 0.580 | 0.611 | 0.164 | 1.0% |
| metav4 | 0.211 | [0.186, 0.237] | 0.831 | 0.119 | 0.945 | 30.9% |
| subattr | 0.165 | [0.102, 0.230] | 0.550 | 0.333 | 0.109 | 1.3% |

**svfd is separated from every other arm.** This is the one place where the
prompt choice buys something unambiguous.

Note `subattr` finishing 3rd on exposed and last on watched. The 40-attribute
schema suits stable identity attributes and cannot carry a rare per-frame event —
that is the quantitative form of "the identity route cannot supply momentary".

\* PADQ rows see the full scene with the target named by a bounding box rather
than a crop, and answer one attribute per call. A gap against the other rows is
prompt **and** representation together.

## K matters as much as the prompt

Same 426 trusted-region frames, same representation, same parser; only K differs.
Reproduce with `python eval/k1_grid.py`.

| prompt | exp K=1 | exp K=8 | Δ | wat K=1 | wat K=8 | Δ |
|---|---|---|---|---|---|---|
| svfd | 0.719 | 0.500 | +0.219 | **0.667** | 0.000 | **+0.667** |
| eyes | 0.714 | 0.728 | −0.014 | 0.468 | 0.359 | +0.109 |
| combined | 0.674 | 0.540 | +0.135 | 0.526 | 0.455 | +0.072 |
| metav4 * | 0.647 | 0.611 | +0.036 | 0.415 | 0.462 | −0.046 |
| trueonly | 0.632 | 0.606 | +0.026 | 0.408 | 0.242 | +0.166 |
| features | 0.702 | 0.659 | +0.043 | — | — | — |
| plain | 0.648 | 0.667 | −0.019 | 0.000 | 0.000 | 0.000 |

\* An earlier run of `metav4` scored 0.730 at K=8, and that figure is **not
reproducible**: the prompt file was overwritten after the run, and the file now
in the repository scores 0.611. The table carries what `eval/k1_grid.py` prints
today. Treat any 0.730 in older notes as belonging to a prompt that no longer
exists.

**Watched improves at K=1 for every prompt that can do it at all.** Packed into a
multi-frame call, the model applies the prompt's stated rarity prior to the whole
batch and answers no to everything: svfd predicts positive on 0.0% of frames at
K=8 against 1.9% at K=1, with a true rate of 3.1%.

**Exposed has no such rule** — the deltas point both ways, because each prompt
happens to be calibrated at a different K. Any claim about K has to name the
attribute and the prompt.

That is why the pipeline runs momentary attributes at **K=1 with no tracker**.
Grouping frames only helps when several views describe one fixed fact, which is
the identity case; a momentary attribute needs its own answer per frame either
way. On exposed, 5 of 7 prompts prefer K=1 outright and the two that do not
(−0.014, −0.019) are far inside what this 426-frame sample can resolve — a
bootstrap over the K=1 arms puts a single arm's 95% interval at about ±0.05 — so
there is no K=8 advantage to give up.

The cost is one call per frame: ~4.4 s each, roughly 9,500 frames, about 6 h
across two GPUs per attribute.

## The predicted-positive rate is a diagnostic

Reading the `predicted+` column against the true rate separates two failure modes
that F1 alone conflates.

The bottom arms are not failing to *see* the attribute. `trueonly` has recall
0.985 on exposed — it finds essentially every positive — while calling 87.6% of
instances positive against a true 26.2%. That is a **threshold** failure, not a
perception failure. Across all nine arms the distance from the true rate orders
the ranking monotonically.

The check also works across sessions with different prevalence. Between two
sessions whose true exposed rate is 39.5% and 13.9%, `plain` predicted 69% in
both — a threshold taken from the prompt rather than the image — while `features`
moved 50% → 24%.

**It is necessary, not sufficient.** `subattr` had the closest predicted rates of
any arm on both sessions (43% / 15% against 39.5% / 13.9%) and still scored worst
on the harder one, with precision 0.171 and recall 0.189. It counted correctly
and picked the wrong instances. Calibration and discrimination have to be read
separately.

## Identity

| | value | baseline |
|---|---|---|
| gender accuracy | 0.9456 | — |
| age MAE | 3.63 years | best constant guess 10.46 |

Both on the **held-out 349 tracks**. Quote nothing else.

Running the same age pass over the whole corpus reports MAE 1.16, and that number
must not be used: 924 of the 1,262 scored tracks were in the adapter's training
data. Split apart, held-out tracks give 3.63 and training tracks give 0.25. The
same trap earlier produced a gender figure of 0.9811 that was withdrawn for the
same reason.

## Annotation coverage

| session (time of day) | group | instances | exposed+ | watched+ | annotated |
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

All eleven sessions are from one day; the label is the recording start time.

The seven unannotated sessions store an explicit `False` on every row rather than
a blank, so nothing downstream can distinguish "checked and negative" from "never
checked". **That value is wrong.** Session 09:48 is the same freezer-aisle camera
as the annotated 09:32, which reads 13.7% exposed, and faces are plainly visible
in 09:48's crops. Annotation appears to have been done in batches aligned with
the image-group column — all three `2-image` sessions are annotated.

Two consequences:

1. Scoring is restricted to the four annotated sessions everywhere in `eval/`.
2. **Do not use the unannotated sessions as a specificity test.** An arm that
   predicts many positives there may well be correct.

Watched is the binding constraint on everything above: 201 positives corpus-wide,
158 of them (79%) in a single session.
