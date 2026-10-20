# Interpretability rig

`python -m interpret.interpret configs/main.yaml` (from the repo root) reads the `interpret:` block of the config: it
extracts the hooked activations of a seeded, stratified sample of a dataset into chunk files, then runs the configured
`methods` on them and saves their results. Everything runs on one device, the GPU: `main()` is the only place that
moves models or tensors (the loaded chunks go to the GPU, so methods compute on GPU tensors). A run needs a GPU, a
Nanopath checkpoint (it carries its training `config`, its `step` and the `model_ema`/`model` weights) and
`probe.dataset_roots` pointing at the data.

## Config (`interpret:` in `configs/main.yaml`)

- `checkpoint_path`, `model_weights` (the checkpoint key: `model_ema` for the teacher, `model` for the student): the model to analyse.
- `dataset` + `datasets.<id>.stratification_dimension`: one dataset per run (classes in `datasets.py`). Available:
  `pathorob_camelyon` (PathoROB Camelyon tiles, root `probe.dataset_roots.pathorob`), with the metadata fields
  `biological_class`, `medical_center` and `slide_id` to stratify by. The sampler is sklearn's `resample` with
  `replace=False` and `stratify=`: a seeded subsample that keeps the proportions of the stratification dimension.
  `num_samples` must not exceed the dataset size.
- `extract.{seed, num_samples, batch_size, num_workers, chunk_size, layer_stride}`: `layer_stride` keeps every n-th
  layer counted from the last (the last is always kept); `num_workers` only parallelizes image decoding.
- `methods` + one sub-block per method (e.g. `effective_rank`).

Storage per sample is `kept_layers x 257 tokens x C x 4 bytes` (float32) plus the 0.6 MB input image; for ViT-S (C=384)
that is about 3.4 MB at stride 2 and 2.2 MB at stride 4, so a 500-sample chunk is 1.7 GB / 1.1 GB.

## Output

```
output_dir/                      # from the config
  data/
    <hash>/                      # hash of everything that determines the chunks (checkpoint + step, weights, dataset + its config, seed, num_samples, layer_stride)
      chunk_N.pt                 # the extracted samples
      metadata.json              # settings, git commit, chunk size, kept layers, sampled dataset indices; written last
  results/
    <hash>_<methods>.json        # one per extraction and set of methods, e.g. 8707bab631_effective_rank.json
```

Each `chunk_N.pt` is a list of `chunk_size` samples (the last chunk may hold fewer), each a dict `{"image": (3, H, W)
model input, "metadata": the dataset's metadata dict, "layers": {layer: {"patch": (N, C), "cls": (1, C)}}}`, all tensors
float32. Methods read them through `ChunkedFeatureDataset`, which returns `(image, metadata, layers)` for one sample on the GPU.

An existing `data/<hash>/` directory is reused without re-extracting. If a run died, its `metadata.json` is missing and
loading crashes: delete that `data/<hash>/` directory and rerun.

## Methods

- `effective_rank`: per layer, the effective rank (exp of the entropy of the normalized singular values) of the centered
  GAP (mean patch token) and CLS vectors over all samples; `singular_value_eps` drops numerically zero singular values.
  Result: `{layer: {"gap_effective_rank": ..., "cls_effective_rank": ...}}`.

## Hook contract

Any model analysed by the rig implements one method, `register_interpretability_hooks(self, storage_dict) -> list[str]`
(see `ViT` in `model.py` for the reference implementation). It is the ONE place where model-specific details and quirks live:
the rig never reads model code and never slices tokens by position. It assumes a transformer that passes token sequences
between layers. The contract below is enforced by `assert_hook_interface` in `interpret.py`; when you add a check there,
document the new property here.

The method registers forward hooks on the milestones (layers) the rig should analyse and returns their string keys in true
forward-pass order, e.g. `["block_0", ..., "block_11", "norm"]`. The rig empties `storage_dict` in place
(`storage_dict.clear()`) before every forward pass, so hooks must keep writing into the same dict object they were given.

Each hook writes `storage_dict[layer_name] = {group_name: tensor}` with two REQUIRED groups (`REQUIRED_GROUPS` in
`interpret.py`), the objects downstream methods consume:

- `"patch"` `(B, N, C)`: the patch tokens. `N` must be a perfect square and the tokens in row-major order on the square grid.
- `"cls"` `(B, 1, C)`: the global summary token.
- Each required tensor is detached and finite with shape `(B, T, C)`: `B` is the batch size, `T` the tokens, `C` the hidden
  dim. `C` is identical across the two groups of a layer, and neither may be empty.
- Groups are disjoint (a token is in exactly one group). Token counts are identical across all layers and batches.
- Any other group you write (e.g. `"register"`) is ignored: it is neither validated nor stored. Write it or not.
- Tensors are written as-is: no pooling or flattening, no dtype casts, no device moves (the rig handles those).
- Hooks only observe: they must not alter the forward pass or its outputs.

The rig attaches no meaning to milestones: pre- or post-norm, tuple unpacking, etc. are the implementer's choice.
Cross-model depth comparison uses the relative position in the returned list. Recommended (not enforced): make the last
milestone the representation your downstream heads consume. Disjoint groups, row-major patch order and forward-pass order
of the returned list cannot be checked by the rig: they are promises by the implementer.

## Your to-do list to analyse a new model

- [ ] Implement `register_interpretability_hooks` on your model, following the contract above.
- [ ] If your model is not a Nanopath checkpoint, update `load_model` in `interpret.py` so it loads and builds your model
  and returns `(model, train_cfg, step)`: `train_cfg` is the config its preprocessing reads (`data.mean`, `data.std`,
  the probe transform policy) and `step` goes into the cache hash.

Nothing else in the rig needs to change.

## Extending the rig

- New dataset: a class in `datasets.py` that takes its root folder as an argument (`load_dataset` passes it from
  `probe.dataset_roots`; `probe.py`'s loaders are the precedent), returns `(PIL image, metadata dict)` and exposes
  `records`, a list of `(image, metadata)` per sample (the sampler stratifies over `metadata[stratification_dimension]`);
  add it to `load_dataset` and a block under `datasets:` in the config.
- New method: a class in its own file, `Method(chunked_dataset, method_cfg).run() -> results dict`; add it to the dispatch in `main()`
  and a sub-block (and an entry in `methods`) in the config.

## Notes

The model input uses probe.py's eval transform plus the normalization that probe.py applies at its model-call sites
(`evaluation_preprocess` in `interpret.py`). probe.py is locked, so that normalization is duplicated: keep the two in sync.
