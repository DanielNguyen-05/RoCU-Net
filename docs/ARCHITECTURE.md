# RoCU-Net technical specification

## Problem addressed

The original occupancy block increases a one-channel occupancy probability while using
a shallow encoder guide to choose the four child values. It guarantees local
mass consistency, but repeated probability-only transitions form a semantic
bottleneck. It also applies the iterative allocation solver to every parent
cell, although confident foreground/background interiors do not need a complex
sub-pixel allocation.

RoCU adds one paper-facing module, the **Confidence-Routed Semantic Occupancy
block**, with three inseparable functions:

1. propagate a thin semantic carrier;
2. predict a boundary-aware four-child allocation;
3. route confident cells through a solver-free conservative identity path.

## Exact conservative mixture

Let `p` be one parent occupancy and `s_j` the four allocation scores. The refined
path solves a scalar bias `b` by monotone bisection:

```text
q_ref_j = sigmoid(s_j + b)
(1/4) sum_j q_ref_j = p
```

The identity path is `q_base_j = p`. Parent uncertainty is

```text
u = 4 p (1-p),        0 <= u <= 1.
```

A semantic gate predictor produces `g_hat`; routing uses `g = u * g_hat`. The
final children are

```text
q_j = (1-g) q_base_j + g q_ref_j.
```

Because `g` is shared by all four children,

```text
(1/4) sum_j q_j = (1-g)p + gp = p.
```

Therefore confidence routing preserves the local occupancy invariant. At
deployment, if `u < tau`, the implementation skips bisection and sets all four
refined children to `p`; this is numerically conservative and removes solver
iterations from confident cells.

## Semantic path

For parent carrier `Z_l` and higher-resolution encoder feature `E_(l-1)`:

```text
Z_up       = PixelShuffle(Conv1x1(Z_l))
E_project  = Conv1x1(E_(l-1))
Z_(l-1)    = LiteMBConv(DSConv(Z_up + E_project))
```

`Z_(l-1)` drives boundary, routing and allocation heads and is passed into the
next RoCU stage. Setting `use_semantic_carrier=false` zeros `Z_up` while
retaining the current encoder guide. The dataset configs enable the carrier.

## Training objective

The configured objective is

```text
L = L_BCE+Dice(full)
  + 0.40 L_BCE+Dice(half)
  + 0.20 L_BCE+Dice(quarter)
  + 0.50 L_structure
  + 0.25 L_boundary
  + 0.05 L_routing.
```

`L_structure` is weighted BCE plus weighted IoU around regions whose local mask
average differs from the binary target. `L_boundary` supervises the two
auxiliary contour predictions. `L_routing` teaches parent gates to activate for
cells intersecting the morphological target boundary.

## Main claim and required evidence

The intended claim is not merely that another attention block improves Dice.
It is:

> Exact occupancy conservation can coexist with semantic high-resolution
> decoding and conditional allocation, improving boundary quality while
> avoiding iterative refinement in confident regions.

The selected model enables the semantic carrier, confidence routing and
boundary-aware objective. The selected ColonDB run initializes from the Kvasir
checkpoint, uses corrected rotations during training, and applies four-view
flip TTA at inference. Its profiling results include all four forward passes.

Report Dice, IoU, boundary F1, Params, MACs, latency, FPS, memory, active-cell
ratio and conservation error. A result is not sufficient if it improves Dice
but loses the local conservation invariant or has worse measured latency.
