# Interpretability rig

`python -m interpret.interpret configs/main.yaml` (from the repo root) reads the `interpret:` block of the config: it
extracts the hooked activations of a seeded, stratified sample of a dataset into files (one `ActivationDataset` per layer, see Output), then runs the configured
`methods` on them and saves their results and plots. Everything runs on one device, the GPU: `main()` picks it, moves the model and gives it to the datasets, whose `get_block` and the `DeviceLoader` (both in `data.py`) are the only places that move data to it,
so the methods never move data themselves. A run needs a GPU, a
Nanopath checkpoint (it carries its training `config`, its `step` and the `model_ema`/`model` weights),
and `probe.dataset_roots` pointing at the data.

## Methods

Each method below has its own subsection: the papers behind it, the questions its results can answer and the author's
notes on the implementation. What a method computes is in its code (`interpret/methods/<method>.py`).

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

### cka

Papers:
- [Kornblith et al. 2019](https://arxiv.org/abs/1905.00414): centered kernel alignment (CKA) for comparing layers of neural networks (Section 3: the definition, Section 2: why these invariances, Section 6.1: sanity check).

Question this method answers: 
- given layer A and layer B, how similar is the *structure* of their embedding spaces, i.e. do the two layers organize the examples in the same way?

What the paper improves on: earlier layer-comparison indexes (CCA and relatives) are invariant to any invertible linear transformation, which the paper argues makes them unable to measure meaningful similarity once a layer is at least as wide as the number of examples (and ignores the scale of directions, which is consistent across networks); CKA is invariant only to rotations and uniform scaling. In the paper's sanity check on CNNs it finds the matching layers of networks trained from different initializations, unlike the other indexes; on Transformers (equal widths, like our ViT) all indexes pass it, but CCA misses feature-scale differences between attention and feed-forward sublayers.

Notes on the implementation:

| Derivation steps for CKA (done in the paper) | Matrix notation |
|---|---|
| **Step 0: notation**<br>we compare layer A and layer B on the same n examples; each layer turns every example into one vector of C numbers. | `X` (n × C) holds layer A's activations, one row `xᵢ` per example; `Y` holds layer B's, same n examples. |
| **Step 1: what should a similarity index be invariant to?** (Section 2)<br>orthogonal transformation (which includes permutations of channels, a symmetry of neural networks) and isotropic scaling, but not invertible linear transformation (an index with that invariance cannot tell any two full-rank layers apart if both are at least as wide as the number of examples, and training is sensitive to the scale of directions). | `s(X, Y) = s(XU, YV)` for `UᵀU = I`, `VᵀV = I` and `s(X, Y) = s(αX, βY)`, but not `s(XA, YB)` for arbitrary invertible `A`, `B`. |
| **Step 2: how do we want to describe the space that a layer produces (so we can compare these across layers)** (Section 3, key insight; the symmetry argument is in Section 2.2)<br>*Naive:* compare the raw embeddings of the two layers directly. *Problem:* the coordinates of two layers are not comparable. A network may permute, rotate or mix its channels from layer to layer, so channel *i* of one layer need not mean anything for channel *i* of another. *Fix:* go up one level of abstraction and describe a layer by its *similarity table*: an n × n table with one entry per pair of examples, the dot product of their vectors, saying how similar the layer considers them (called a representational similarity matrix in the paper). The table does not depend on the axes, so two layers that arrange the examples the same way produce the same table even when their raw numbers look unrelated. The same n examples go through both layers, so the tables can be compared entry by entry. | `K = XXᵀ`, `L = YYᵀ`. Entry (i, j) of `K` is row i of `X` times column j of `Xᵀ`, i.e. the dot product `xᵢ·xⱼ`. |
| **Step 3: how to compare the two tables** (Eq. 1)<br>use the inner product of the tables flattened into 1D. The paper shows that this inner product also equals the sum of the squared dot products between every pair of features of `X` and `Y` (one feature of each, with the dot product taken over the examples), so the agreement between the two tables can also be computed from the features. | `⟨vec(XXᵀ), vec(YYᵀ)⟩ = tr(XXᵀYYᵀ) = ‖YᵀX‖²_F` (`vec` flattens a matrix, the trace `tr(M)` is the sum of the diagonal and for symmetric `A`, `B` `tr(AB) = Σᵢⱼ Aᵢⱼ Bᵢⱼ`, the entry-by-entry product summed up; `‖M‖_F = sqrt(Σᵢⱼ Mᵢⱼ²)` is the length of the flattened matrix). |
| **Step 4: centered data gives a covariance reading** (Eq. 2)<br>for centered `X` and `Y` (every feature has mean 0 across the examples; the paper states this as a condition and does not comment on it here) Eq. 1 is the squared size of the cross-covariance between the features of the two layers, up to the covariance normalization (the Bessel correction `n−1` of a covariance, squared here because the covariance is squared: `1/(n−1)²`). | Start from Eq. 1: `tr(XXᵀYYᵀ) = ‖YᵀX‖²_F`. Take `X`, `Y` centered (every column has mean 0). The cross-covariance of their features, with the Bessel correction `n−1`, is `cov(Xᵀ, Yᵀ) = XᵀY/(n−1)`. Divide Eq. 1 by `(n−1)²`: `tr(XXᵀYYᵀ)/(n−1)² = ‖YᵀX‖²_F/(n−1)² = ‖XᵀY/(n−1)‖²_F = ‖cov(Xᵀ, Yᵀ)‖²_F` (Eq. 2; transposing a matrix does not change its Frobenius norm). |
| **Step 5: HSIC** (Eq. 3)<br>Eq. 2 is the formula for the agreement between the two tables when the similarity is the dot product and the features are centered. Raw activations are not centered, so the formula has to do the centering itself: it is applied to `HX` and `HY` (for data that is already centered, `HX = X` and `H` changes nothing). The Hilbert-Schmidt Independence Criterion (Gretton et al. 2005) is the established general version for any kernel (= a similarity function), with the centering done on the similarity tables themselves (`HKH`) instead of on the features. The paper introduces the centering matrix `H` here. For the linear kernel its value equals the value of Eq. 2, so nothing changes for us - I think the reason the paper talks about HSIC at all is s.t. they introduce the form that accepts an arbitrary kernel (the paper does swap kernels in results section, we are not planning to use kernels other than the linear one). (our intuition why centering is beneficial: without centering, a constant offset shared by all examples, e.g. the color signature of one scanner, adds a potentially large component to every table entry; adding the same offset to two vectors pulls their angle towards 0 (and the normalized version of the step 3 measure, introduced in step 6, is a cosine, i.e. a measure of angle), e.g. (1, 0) and (0, 1) shifted by (10, 10) are only about 5° apart, so all layers would look similar under the normalized measure from step 6.) | `H = I − (1/n)11ᵀ` is the centering matrix (`1` is a column of n ones): `(1/n)1ᵀX` is the mean example `μᵀ`, so `HX = X − 1μᵀ` is every example minus the mean example. Start from Eq. 2 for the centered examples `HX`, `HY`: `tr((HX)(HX)ᵀ · (HY)(HY)ᵀ) / (n−1)²`. The table of the centered examples is `(HX)(HX)ᵀ = HXXᵀH = HKH` (H is symmetric), likewise `HLH`. So the expression is `tr(HKH·HLH) / (n−1)²`. Since `HH = H` (centering twice is the same as once) and the trace lets one H move around, `tr(HKH·HLH) = tr(KHLH)`. End point: `tr(KHLH) / (n−1)² = HSIC(K, L)`, the paper's Eq. 3. |
| **Step 6: normalize** (Eq. 4)<br>*Problem:* HSIC is not invariant to isotropic scaling. *Fix:* normalize it, which gives centered kernel alignment (CKA; Cortes et al. 2012, Cristianini et al. 2002). | Start from Eq. 3, `HSIC(K, L) = tr(KHLH)/(n−1)²`, and divide by the square root of the product of the HSICs of each table with itself, `HSIC(K, K) = tr(KHKH)/(n−1)²` and `HSIC(L, L) = tr(LHLH)/(n−1)²`: `CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K)·HSIC(L, L)) = [tr(KHLH)/(n−1)²] / [sqrt(tr(KHKH)·tr(LHLH))/(n−1)²] = tr(KHLH) / sqrt(tr(KHKH)·tr(LHLH))` (Eq. 4): the `(n−1)²` cancel. Since `‖M‖²_F = tr(MM)` for symmetric `M`, `tr(KHKH) = ‖HKH‖²_F` is the squared length of the flattened centered table, so CKA is the inner product of the two centered tables divided by their two lengths. |

*TLDR.* The key points of the above derivation: 
- Compare similarity structures, not features: the coordinates of two layers are not comparable (channels can be permuted, rotated or mixed), but how each layer arranges the examples is. This is the key reframe.
- Flatten the two similarity tables and take their inner product.
- Center first, so a constant offset shared by all examples cannot dominate the tables (our intuition; the paper only states the condition and defines `H`).
- Normalize, which adds the invariance to uniform scaling that the raw inner product lacks.
- HSIC is the paper's link to the general kernel version; we only use the linear kernel.
 
Ultimately, the normalized inner product of the centered similarity tables is **CKA**. It is unchanged by rotating the axes of a layer or scaling it uniformly, and is not invariant to arbitrary invertible linear maps (which the paper argues is desirable).

*sidenote.* The paper also gives a cheaper form for the linear kernel (regroup the sum by feature: `CKA = ‖YcᵀXc‖²_F / (‖XcᵀXc‖_F · ‖YcᵀYc‖_F)` with `Xc = HX`, `Yc = HY`, which needs only C × C matrices instead of n × n tables; Section 3 and Table 1 of the paper). We omit it: at our sample sizes the n × n version is still tractable, and we want to focus on methods, not on computational tricks - unless we have a good reason to read about and implement it (in which case we will do so in the future).

### linear_probe

Questions this method answers:
- At which layers can a linear probe recover high-level labels (e.g. tissue class) and low-level labels (e.g. site/stain), and how does this change with depth?

### layer_inversion

Papers:
- [Dosovitskiy & Brox 2016](https://arxiv.org/abs/1506.02753): a decoder trained per layer to rebuild the input image from that layer's activations.
- [He et al. 2022 (MAE)](https://arxiv.org/abs/2111.06377): the per-patch linear head and the transformer decoder on ViT tokens.
- [Xu et al. 2020](https://arxiv.org/abs/2002.10689): why a reconstruction measures the information a given decoder family can extract, and not all the information of the layer.
- The survey of the field is in `layer_inversion_literature.md` and the settled design in `layer_inversion_spec.md`.

Questions this method answers:
- How much of the tile (colour, layout, texture) can a decoder rebuild from each layer, and how does this change with depth?
- Is the tile information of a layer local to each patch-token or does it depend on the other patch-tokens? The linear decoder sees only the token of its own patch (the local baseline), the transformer decoder exchanges information between all tokens, so the gap between them is the share that needs context (it also contains the effect of the transformer's nonlinearity).

## Config (`interpret:` in `configs/main.yaml`)

- `checkpoint_path`, `model_weights` (the checkpoint key: `model_ema` for the teacher, `model` for the student): the model to analyse.
- `dataset` + `datasets.<id>.stratification_dimension`: one dataset per run (classes in `datasets.py`). Available:
  `pathorob_camelyon` (PathoROB Camelyon tiles, root `probe.dataset_roots.pathorob`), with the metadata fields
  `biological_class`, `medical_center` and `slide_id` to stratify by. The sampler is sklearn's `resample` with
  `replace=False` and `stratify=`: a seeded subsample that keeps the proportions of the stratification dimension.
  `num_samples` must be smaller than the dataset size.
- `seed`: the one seed of the interpret config. It seeds the extraction sampler, CKA's random subset and, via `set_seeds`, every random number generator plus deterministic GPU mode (an operation without a deterministic implementation raises an error).
- `extract.{num_samples, batch_size, num_workers, layer_stride}`: `layer_stride` keeps every n-th
  layer counted from the last (the last is always kept); `num_workers` only parallelizes image decoding.
- `methods`: the methods to run (`effective_rank`, `cka`, `linear_probe`, `layer_inversion`); `cka.max_samples` caps the examples CKA uses (a seeded random subset, label-free so not exactly stratified), because its similarity tables are n × n (0.1 GB each at 5,000 examples, 0.4 GB at 10,000, 2.5 GB at 25,000); the other methods use all extracted examples.
- `linear_probe`: `targets` (what to predict: the metadata fields and the labels derived from the tile pixels, `brightness`, `tissue_brightness`, `tissue_fraction`, `hematoxylin` and `eosin`, each cut into 4 quantile bins), `lrs` and `weight_decays` (the grid), `epochs`, `batch_size`, `num_workers` (the DataLoader worker processes that fetch the batches), `leakage_group` (the metadata field whose values are never split across train and validation, `slide_id` for PathoROB) and `folds` (a `StratifiedGroupKFold` on that field).
- `layer_inversion`: `decoders` (`linear`: one linear head shared by all patch tokens, `transformer`: transformer blocks over the CLS and patch tokens, then the same head), `transformer` (the settings of that decoder: `depth` and, per `Block`, `head_dim`, `mlp_ratio` and `drop_path`), `lrs` and `weight_decays` (the grid), `epochs`, `batch_size`, `num_workers`, `num_collage_rows` (the rows of the collage), `stratify` (the metadata field the folds are stratified on), `leakage_group` and `folds` (as for the probe). The decoders train on the patch tokens and the images of every layer, so the datasets are opened with `patch_tokens=True, image=True`.

Storage per sample is `kept_layers x 257 tokens x C x 4 bytes` (float32) plus the 0.6 MB input image; for ViT-S (C=384)
that is about 3.4 MB at stride 2 and 2.2 MB at stride 4, so 10,000 samples take about 34 GB / 22 GB on disk. The extraction holds everything in RAM before saving it
(as much as the files take on disk, plus the model and loader), so run it on a high-RAM runtime: it is a one-off operation per extraction.

## Output

```
output_dir/                      # from the config
  data/
    <hash>/                      # hash of everything that determines the files (checkpoint + step, weights, dataset + its config, seed, num_samples, layer_stride)
      activations/
        <layer>_patch_tokens.pt  # (N, 256, C) float32, the patch tokens of that layer: the big files (about 3.9 GB each for 10,000 samples and C=384)
        <layer>_cls.pt           # (N, C) float32, the CLS token of that layer
      precomputed/
        <layer>_patch_mean.pt    # (N, C) float32, the mean over the patch tokens of that layer, computed once during the extraction
      images.pt                  # (N, 3, H, W) float32, the tile as the model saw it (resized, in [0, 1], before the normalization of the model input; about 6 GB for 10,000 samples)
      labels.pt                  # {field: (N,) int64}, every metadata field as integers plus the labels derived from the tile pixels (brightness, tissue_brightness, tissue_fraction, hematoxylin, eosin: quartile bins)
      label_names.json           # {field: names}, the name of integer k is names[k] (sorted unique values of the sampled tiles; "0-25%" .. "75-100%" for the derived labels)
      info.json                  # settings, git commit, kept layers, sampled dataset indices; written last by the extraction (marks it complete)
      config.json                # the full config that produced these files, written right after a fresh extraction
  results/
    <hash>_<methods>/            # one folder per extraction and set of methods (sorted by name), e.g. 8707bab631_cka_effective_rank
      results.json               # the results of all methods
      dashboard.html             # the whole run on one page, to open with a double click (see below)
      config.json                # the full config of the last run that wrote this folder
      <method>/                  # the plots that method defines, e.g. cka/patch_mean_cka.png, effective_rank/patch_mean_effective_rank.png, layer_inversion/error.png and layer_inversion/collage.png
```

At the end of a run, `write_dashboard` (`utils.py`) writes `dashboard.html` into the results folder: the settings of the run, then for every method its plots (embedded in the file, so it can be moved or sent around) and the parts of its subsection of this README (papers, question, what the paper improves on, notes) as toggles. It splits a method's subsection at the lines starting with `Papers`, `Question`, `What the paper improves on` and `Notes on the implementation`.

Row `i` of every file is the `i`-th sampled tile (`sampled_dataset_indices` in `info.json`), so the vectors, the image and the labels of an example sit at the same index by construction.
Methods read one layer through `ActivationDataset(data_dir, layer, device)`, a `torch.utils.data.Dataset`: the constructor sends nothing to the device: the small tensors (`cls`, `patch_mean`, labels) are CPU tensors in memory, the two big files (`patch_tokens`, `image`) are opened only when enabled and memory-mapped (nothing is read until an item is fetched). Data leaves the dataset in two ways and both move it to the device (`__getitem__` itself returns one example as CPU tensors): `get_block(field, key)` returns a whole small block (`field` names a dict of the dataset, `"activations"` or `"labels"`, and `key` picks the block in it: `get_block("activations", "cls")`, `get_block("labels", "medical_center")`; what methods that need a whole layer or fold read; the big blocks are refused, they are memory-mapped so that they are never loaded whole), and `DeviceLoader(dataset, device, indices, batch_size, ...)` from `data.py` (a `DataLoader` over those examples that yields batches already on the device, the only way to read the big blocks, in batches)
returns one example as `{"image", "cls", "patch_tokens", "patch_mean", "labels"}`, always with these keys in this order: `cls`, `patch_mean` (both `(C,)`) and `labels` (`{metadata field: integer}`) are always filled, while
`image` and `patch_tokens` are only opened when enabled in the constructor (`ActivationDataset(data_dir, layer, device, patch_tokens=True, image=True)`) and are an empty tensor `torch.empty(0)` otherwise (empty means "not loaded").
All tensors are private to the dataset. Only two facts are public attributes, so that methods never hard-code them: `.label_names` (`{metadata field: names}`, the name of integer k is names[k]) and `.vector_types` (`{vector type: description}` of the vectors every item holds); the number of examples is `len(dataset)` and the labels are read with `get_block("labels", field)`, or all at once with `get_block("labels", to_device=False)` (`to_device=False` skips the device move, for the few host-side callers like sklearn and printouts). `dataset[i]` is one example as CPU tensors. The loader stacks examples on the CPU, optionally in `num_workers` worker processes, and the main process moves each whole batch to the device once (workers cannot use the GPU), which is far cheaper than moving every example.

An existing `data/<hash>/` directory is reused without re-extracting. If a run died, its `info.json` is missing and
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
- New method: a class in its own file in `interpret/methods/`, `Method(datasets)` (`Method(datasets, device)` if it builds its own modules, like the probe's model; a method that trains models on folds subclasses `Probe` in `probe_base.py` and implements its hooks, like `linear_probe.py` and `layer_inversion.py` (the hooks, their arguments, and what is asserted about them and what is not are documented in `probe_base.py`; not to be confused with the model hooks above); `datasets` is `{layer name: ActivationDataset}`; enable `patch_tokens` or `image` on a dataset if the method needs token-level data or images) with `run()`, which stores a nested dict of
  plain numbers in `self.results`, and `plot()`, which returns `{file name: matplotlib Figure}` built from `self.results`
  (`main()` saves the figures); add it to the dispatch in `main()` and an entry in `methods` in the config (a method with tunables gets its own config sub-block, passed to its constructor in the dispatch), and a subsection under
  Methods here (the papers behind it, the questions it answers, notes on the implementation).

## Notes

The model input uses probe.py's eval transform plus the normalization that probe.py applies at its model-call sites
(`evaluation_preprocess` in `interpret.py`). probe.py is locked, so that normalization is duplicated: keep the two in sync.
