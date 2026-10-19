# Interpretability hook contract

Any model analysed by `interpret.py` implements one method, `register_interpretability_hooks(self, storage_dict) -> list[str]`
(see `ViT` in `model.py` for the reference implementation). It is the ONE place where model-specific details and quirks live:
the rig never reads model code and never slices tokens by position. It assumes a transformer that passes token sequences
between layers. The contract below is enforced by `assert_hook_interface` in `interpret.py`; when you add a check there,
document the new property here.

## What the method does

Registers forward hooks on the milestones (layers) the rig should analyse and returns their string keys in true
forward-pass order, e.g. `["block_0", ..., "block_11"]`. The rig empties `storage_dict` in place (`storage_dict.clear()`)
before every forward pass, so hooks must keep writing into the same dict object they were given.

## What every hook must write

`storage_dict[layer_name] = {group_name: tensor}` where:

- Every tensor is detached with shape `(B, T, C)`: batch, tokens, hidden dim. `C` is identical across the groups of a
  layer, and no group may be empty.
- `"patch"` `(B, N, C)` is required. `N` must be a perfect square and the tokens in row-major order on the square grid.
- `"cls"` `(B, 1, C)` is reserved and optional (the global summary token). Any other name (e.g. `"register"`) is a
  free-form group that the rig ignores until a method uses it.
- Groups are disjoint (a token is in exactly one group). Group names and per-group token counts are identical across
  all layers and batches.
- Tensors are written as-is: no pooling or flattening, no dtype casts, no device moves (the rig handles those).
- Hooks only observe: they must not alter the forward pass or its outputs.

## Milestone semantics

The rig attaches no meaning to milestones: pre- or post-norm, tuple unpacking, etc. are the implementer's choice.
Cross-model depth comparison uses the relative position in the returned list. Recommended (not enforced): make the last
milestone the representation your downstream heads consume.

## Not checkable by the rig

Disjoint groups, row-major patch order and forward-pass order of the returned list are promises by the implementer.

## Your to-do list to analyse a new model

- [ ] Implement `register_interpretability_hooks` on your model, following the contract above.
- [ ] If your model is not a Nanopath checkpoint, update `load_model` in `interpret.py` so it builds your model.

Nothing else in the rig needs to change.
