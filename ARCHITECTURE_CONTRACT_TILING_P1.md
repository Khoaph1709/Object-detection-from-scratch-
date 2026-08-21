# Architecture Contract: Tiling, P1 Refinement, and Ensemble

## Existing baseline

The safe baseline is ConvNeXt-Small + BiFPN P2-P7 + shared FCOS head with strides `{p2:4,p3:8,p4:16,p5:32,p6:64,p7:128}`. Existing checkpoints record backbone, fpn type, and BiFPN layers. The baseline must remain loadable without new flags.

## Tile coordinate contract

A tile is represented in original-image coordinates as `(top, left, height, width)`. The model receives a resized tile and produces boxes in resized-tile coordinates. Every tile prediction must first be inverse-scaled using the resized tile dimensions, then translated by `(left, top)`, then clipped to the original image dimensions. Tile GT retention uses visible fraction; default training threshold is 0.70.

## Tile inference contract

Tile inference is inference-only unless explicitly enabled by a tile-training config. Full-image inference remains available as a control. Tile outputs are grouped by original image id and merged class-wise after mapping back to original coordinates. No tile output may be evaluated before inverse mapping.

## Tile training contract

Tile-aware fine-tuning samples object-centered tiles only with an explicit probability. It never applies a second crop in the same transform call. It retains the selected object and drops other boxes whose visible fraction is below threshold. The transform remains disabled for validation/inference.

## P1/refined-P2 contract

The current backbone exposes C2-C5 only and BiFPN is hardcoded to P2-P7. A deadline-safe high-resolution branch must therefore be synthesized from the existing P2 feature, not pretend that a native C1 exists. The optional branch adds a learned stride-2 refinement from P2 and a small-object head, with model strides including P1=2 only when the flag is enabled. Existing checkpoints with the flag disabled must remain compatible. P1-enabled checkpoints must record the architecture flag and reject loading into a baseline model.

## Ensemble contract

Ensemble/WBF accepts prediction JSONs already mapped to original image coordinates. It performs class-aware weighted box fusion and preserves the evaluator schema `{image_id, boxes:[{class,confidence,bbox}]}`. It does not run model inference and cannot fix errors if all input models make the same mistake.

## Validation gates

1. Synthetic tile forward and inverse mapping must recover known boxes within tolerance.
2. Full-image prediction with tiling disabled must remain unchanged.
3. P1-disabled state dict must load exactly as before.
4. P1-enabled model must pass one forward/backward smoke test and record architecture metadata.
5. WBF must be class-isolated and not fuse different classes.
6. Compileall, unit tests, config parsing, synthetic forward, and package checks must pass before commit.
