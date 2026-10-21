# Interpretability rig

`python -m interpret.interpret configs/main.yaml` (from the repo root) reads the `interpret:` block of the config: it
extracts the hooked activations of a seeded, stratified sample of a dataset into chunk files, then runs the configured
`methods` on them and saves their results and plots. Everything runs on one device, the GPU: `main()` is the only place
that moves models or tensors (the loaded chunks go to the GPU, so methods compute on GPU tensors). A run needs a GPU, a
Nanopath checkpoint (it carries its training `config`, its `step` and the `model_ema`/`model` weights),
and `probe.dataset_roots` pointing at the data.

## Methods

Each method below has its own subsection: the papers behind it, the questions its results can answer and the author's
notes on the implementation. What a method computes is in its code (`interpret/<method>.py`).

### effective_rank

Papers:
- [Roy & Vetterli 2007](https://infoscience.epfl.ch/record/110188): introduced the effective rank (ER) definition.
- [RankMe, Garrido et al. 2022](https://arxiv.org/abs/2210.02885): applied it in the context of modern SSL.

Questions this method answers:
1. How many of its dimensions does each layer effectively use, and how does that change with depth (separately for the mean-over-patch-tokens and the CLS vector)?
2. [From the RankMe paper, not yet validated in our context] Is ER a label-free proxy for downstream performance
   (i.e. we could try to correlate it with our obtained downstream scores in an effort to test the 2022 paper's premise in our context of ViT in DP)?

Notes on the implementation:
- Rank of embeddings collected for a certain layer across datapoints tells us "how many of its dimensions is this layer actually occupying".
  It will very rarely be the case that one embedding is an exact linear combination of other collected embeddings (analogously it will very rarely
  be the case that a singular value of the embedding matrix will be exactly zero), i.e. the rank of a real world embedding matrix will almost always
  be full even if some dimensions are very close to being unused. Effective rank aims to be more robust: it normalizes singular values to a
  probability distribution and then computes its exponentiated entropy (how uncertain are we about a draw from this distribution), i.e. if one
  singular value dominates (even if the others are non-zero) entropy is close to 0 (low uncertainty) (ER=1, rank would have been full unless the
  other sing vals are exactly 0); if sing vals are approx. uniformly distributed entropy is close to log(k) (ER=k).
- Centering: the embedding cloud is centered (inspired by PCA, not necessarily the linked papers). Without it a strong offset of the whole cloud
  from the origin gives one very large singular value, which can dominate the normalized distribution and can push the entropy (and ER) down even
  if the variation inside the cloud is spread evenly. After centering, the singular values only describe the variation within the cloud.

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
    <hash>_<methods>/            # one folder per extraction and set of methods, e.g. 8707bab631_effective_rank
      results.json               # the results of all methods
      <method>/                  # the plots that method defines, e.g. effective_rank/patch_mean_effective_rank.png
```

Each `chunk_N.pt` is a list of `chunk_size` samples (the last chunk may hold fewer), each a dict `{"image": (3, H, W)
model input, "metadata": the dataset's metadata dict, "layers": {layer: {"patch": (N, C), "cls": (1, C)}}}`, all tensors
float32. Methods read them through `ChunkedFeatureDataset`, which returns `(image, metadata, layers)` for one sample on the GPU.

An existing `data/<hash>/` directory is reused without re-extracting. If a run died, its `metadata.json` is missing and
loading crashes: delete that `data/<hash>/` directory and rerun.

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
- New method: a class in its own file, `Method(chunked_dataset, method_cfg)` with `run()`, which stores a nested dict of
  plain numbers in `self.results`, and `plot()`, which returns `{file name: matplotlib Figure}` built from `self.results`
  (`main()` saves the figures); add it to the dispatch in `main()` and a sub-block (and an entry in `methods`) in the config, and a subsection under
  Methods here (the papers behind it, the questions it answers, notes on the implementation).

## Notes

The model input uses probe.py's eval transform plus the normalization that probe.py applies at its model-call sites
(`evaluation_preprocess` in `interpret.py`). probe.py is locked, so that normalization is duplicated: keep the two in sync.
