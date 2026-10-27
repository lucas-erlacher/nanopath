<!-- Settled design specification for the planned layer-inversion method of interpret/: a decoder per kept layer reconstructs the tile image from that layer's activations.
     The reasoning behind the choices is in layer_inversion_literature.md. This is a specification only: nothing here is implemented yet. -->

# Layer inversion: settled specification

**Principles behind it (from the user):**
- The target is as close as possible to what the encoder saw, while staying human-readable.
- Reuse the probing code wherever possible.
- Keep the evaluation simple: the MSE trajectory across layers, not many metrics.

| Axis | Decision | Detail and basis |
|---|---|---|
| Goal | One decoder per kept layer rebuilds the tile | To see what visual information each layer can still reconstruct. It is the linear probe's idea with the whole image as the target. |
| Models per layer | One trained model specialized to each layer | Each layer gets the same decoder family and budget, which theory requires for comparing layers (Xu 2020, Def. 3). It is the literature standard and mirrors our probe. |
| Kept layers | `block_0`, `block_4`, `block_8`, `norm` | As extracted: every 4th layer counted from the last. |
| Data | 10,000 PathoROB Camelyon tiles from 95 slides and 5 centres, extracted once and cached by a hash of the settings | The existing extraction. The hash does not cover the image change below. |
| Decoders | Two: A is a linear baseline, B is transformer blocks plus a linear head. No convolutional or VAE-style decoder for now. | Two are enough for a first feeling of what each layer can reconstruct. The transformer is closer to our pipeline. |
| Swapping | Both take the token grid and give an image, so one can replace the other in the same pipeline | the user's design |
| Head | One head shared by all 256 patch tokens, applied to each token in turn | A separate head per position would be about 256 × 0.23M ≈ 59M parameters on 10k tiles, which invites memorization. MAE and the other papers use a shared head. |
| Decoder A | Each patch token alone goes through the shared linear head to its 14 × 14 × 3 = 588 pixel values. Input: patch tokens only, no CLS. No position embedding. | The image version of our linear probe. About 0.23M parameters. No information is exchanged between tokens, so there is no place for CLS, and a position embedding would have nothing to act on. |
| Decoder B input and output | The patch tokens and the CLS token all go into the transformer, so CLS takes part in the information exchange. The head is applied only to the patch tokens at the output. | CLS holds a summary of the whole tile. The extraction already saves it per layer. |
| Decoder B shape | The tensor keeps its width of 384 through all blocks | No projection to another width, unlike MAE's 512. |
| Decoder B blocks | Reuse `Block(dim, heads, mlp_ratio, drop_path_p)` from `model.py`. Depth is configurable, default 4. Block settings copy the encoder's: 6 heads, MLP ratio 4.0, drop-path 0.0, so `Block(384, 6, 4.0, 0.0)`. | The block takes `(B, N, C)` and has no positional dependence, so it stacks as it is. 4 because MAE found 1 block is the minimum with diminishing returns, and the ViT-token decoders in the survey use about 4 (Lei 2025, Allakhverdov 2025). 8 risks overfitting on 10k tiles. |
| Position encoding (B only) | Fixed 2D sin-cos, added at decoder B's input, as the papers do. CLS gets no position: a zero row, as in MAE. | MAE and RAE add a fixed sin-cos embedding (official code), DRAG a learned one. Drexlin 2026 also states fixed 2D sinusoidal. Rathjens is the exception, and Allakhverdov doesn't mention it. Fixed sin-cos has no parameters. |
| Standardization of the inputs | Per layer, per token position and channel: the mean and std over the training-fold tokens at that position and layer. Statistics from the training fold only. It applies to both decoders. CLS gets its own per-channel statistics, like a 257th position. | **Scales and statistics differ strongly across layers.** Our probe briefing shows an input-vector norm of 3.55 at `block_0`, 4.09 at `block_4`, 11.78 at `block_8` and 36.28 at `norm`. Without standardization, one set of optimizer settings would not work for every layer. It is the finer-grained option, and Drexlin 2026 also standardizes per position and channel. Training-fold statistics only, so no validation statistics leak into training (as in the probe). |
| Reconstruction target | The resized 224 × 224 tile in [0, 1], the model's own view in natural colours | The philosophy: the decoder is trained on the image the encoder saw, as closely as possible, while staying readable. The same tile and size, and the only departure is the centering and scaling, which make the image unreadable. **Leaving out the standardization is not a problem: it is only a simple multiplication and addition (`x * std + mean`), which we can expect the decoder to absorb, i.e. learn.** The model keeps its normalized input, and its activations are unaffected. |
| Target rationale | Dropping the standardization costs the decoder nothing in practice | The normalization is an invertible per-channel shift and scale. Undoing it is `x * std + mean`. Both decoders end in a linear head with a bias, which **can represent that map exactly**, so the decoder can learn the same function with either target. This holds with weight decay 0: weight decay would also penalize the bias, which has to sit near the mean intensity (about 0.8 for pale tiles) in [0, 1] space but near 0 in normalized space. Only the scale of the MSE (about 20 times larger in normalized units) and a nearly uniform channel weighting (`1 / std²` is 19.1, 19.9, 19.8) differ. The gain is one readable image space for the target, the collage, the mask and the MSE units. |
| Image storage | `images.pt` holds that tile, float32, about 6 GB, instead of the normalized model input | The extraction already has this tensor (`rgb_images`). The change is one line in the extraction plus the README description. Delete the old `data/<hash>` and re-extract (about 1 minute), since the hash doesn't cover this change. |
| Where the rationale is written | As a comment where `images.pt` is saved, and in the README's Output description of `images.pt` | The argument must live next to the code that makes the choice. The README line must say "the tile as the model saw it, in [0, 1]", not "the model input". |
| Decoder output | Unbounded linear head, clipped to [0, 1] only for display | No sigmoid, so decoder A stays truly linear. |
| Loss | Pixel MSE over all pixels. The tissue and background masks only split the evaluation. | The loss that matches usable information with a Gaussian likelihood, and it avoids the invented detail that perceptual or adversarial losses add. Blur then reflects what the layer does not pin down. |
| Optimizer | Adam, weight decay 0 | As the probe uses Adam. The weight decay is 0 because of the target rationale above. |
| Augmentation | None | The activations are saved, so tiles can't be flipped or rotated. |
| Evaluated epoch | The final epoch. Validation runs each epoch only for the curves. | Exactly what the probe does. Selecting the best validation epoch would be mildly leaky. |
| Train and validation split | Only a train and a validation split, as in the probe, with no separate test set. The error plots, the mean-image reference line's evaluation and the collage are all computed on the validation tiles. | This is downstream analysis, not foundation-model training. A test set would cost data (10k tiles) and trainings for little gain. The only selection made on the validation set is the best grid point per layer, which is slightly optimistic but negligible with 2 learning rates. |
| Metric | The same per-pixel MSE, on validation tiles | In plain intensity units. No SSIM, LPIPS, PSNR, R² or other metrics. The trajectory across layers is what is wanted. |
| Reference line | The error of the mean image, drawn as a flat line on each error plot. The mean image is the average of the training fold's tiles. | It needs no training, and averaging over the training fold means nothing leaks from validation. |
| Tissue mask | Computed on demand with the repo's existing `tissue_mask`, the same function used for the labels in probing, applied directly to the stored [0, 1] tile | No undoing needed, because the stored target is the tensor the function expects. |
| Averaging the masked MSE | Pool all pixels of the validation tiles: the sum of the errors over the mask divided by the pixel count | For now. To be scrutinized later. A tile with no background or no tissue needs no special case. |
| Leakage prevention and stratification | As in probing, stratified on `biological_class` | `StratifiedGroupKFold`, shuffled, seed 42, grouped by `slide_id`. Every tile of a slide stays on one side of each split, so validation never contains a slide the decoder trained on. `biological_class` is balanced and meaningful, so the train and validation folds get a similar class mix. |
| Number of folds | A config parameter `folds`, as in the probe. The value is decided later, once we see how long a training takes. | The probe uses 4. |
| Hyperparameter grid | Keep the probe's grid option (lists of learning rates and weight decays in the config). Use 2 learning rates and 1 weight decay, which is 0. The plot shows the best grid point per layer by validation MSE, as the probe does. | Tuned later through the config. Slightly optimistic, but negligible with two points. |
| Code | Share as much of the probing code as possible | How to share the fit method is an implementation matter. |
| Plots | The error-across-layers plot in three versions (all pixels, tissue only, background only, one line per decoder, plus the mean-image line), the reconstruction collage, and the debug plots of the training dynamics as in probing (train and validation loss curves, from reused code). No other plots. | The collage shows validation tiles next to their reconstructions from each layer. Background is near-white, so the error there is expected to stay low, with the tissue curve the informative one. |

**Deferred to implementation:** other controls (random-init network, pixel ceiling, shuffled tokens), register tokens (to be integrated into all four methods after inversion is done), fused layers, CLS-only or patch-mean-only decoders, the normalized-pixel variant, centre hold-out, convolutional decoder, collage details, training details other than the weight decay, the fold count value, and how to share the fit method.

# Code rules for the build (from the user)

To be revisited while implementing. Sources: `AGENTS.md`, the collected feedback of the interpretability work, and this design discussion.

| Topic | Rule | In practice |
|---|---|---|
| Size | Flat organization, few files, few lines | A great revision lowers the total line count. Over a dozen new lines needs a justification. Reuse before writing. |
| Files | New files only if truly necessary | One method file under `interpret/methods/` follows the rig's pattern. Any new file starts with a few commented lines explaining its purpose. |
| Helpers | No tiny one-use helpers | Code goes where it is used. A shared function is for real reuse only (like `tissue_mask`, `rgb_to_hed`). |
| Shared repo code | Packaged lines stay identical to the originals | The function lives in the file that uses it, takes the needed objects as arguments, and returns what the last line produced. |
| Footprint | New work stays in `interpret/`, with small scoped edits elsewhere | `probe.py` and `benchmarking/` stay untouched. Changing validated probe code needs your agreement and a re-check that its numbers are unchanged. |
| Concerns | One place per concern | Nothing documented twice, one cache logic, device moves only in the dataset layer. |
| Framework | Native PyTorch, single GPU | No DDP, FSDP, Accelerate or Lightning. |
| Configuration | Tunables in YAML (`interpret:` in `configs/main.yaml`), never argparse | Hard-code rarely tuned values. No env vars, options or fallbacks unless often tuned. Short trailing config comments for non-obvious values. |
| Errors | Fail loudly | No `try`/`except`, no guards, no fallbacks. Shape asserts as in the probe. |
| Docs follow code | Update README, config, comments and scripts when the code makes them wrong | |
| Definitions | No nested functions or classes | Lambdas are fine. Use a private method plus `functools.partial` instead of a closure. |
| Explicitness | Explicit over clever | Plain loops, named intermediates, one assignment per line. No `divmod`, no dict-union one-liners. |
| Steps | One logical step per block, mirroring the discussed steps | A named variable per intermediate, one operation per line, the line where dimensions collapse marked. The reference style is the user's `derive_labels`. |
| Tensors | Every dimension indexed explicitly, with shape comments | `x[:, :, :, 0]`, never `...`. Shape comments like `# (B, H, W)`. Named dimensions (`seq_len`, `channel_dim`) instead of repeated `shape[1]`. |
| Pipeline | Never do something and then undo it | Branch the pipeline so both versions exist. |
| Data facts | No facts about the data hard-coded in methods | The dataset advertises them as fields. |
| Dataset | Fixed item interface, native `Dataset` and `DeviceLoader` | The same keys in the same order, a disabled part is `torch.empty(0)`, no custom collate. Device chosen in `main()`. Small blocks via `get_block`, big blocks streamed through the loader. |
| Speed | No speed-driven special cases, no timing or profiling code | Optimize a measured bottleneck later, in place. |
| State | Minimal class state, no defensive no-ops | Attributes used once go inline. No dtype casts "just in case". |
| Naming | Consistent names, no type hints, `Path` over `os.path` | `icfg`, `extract_cfg`, `dataset_cfg`, `method_cfg`, `train_cfg`. Unused arguments get a leading underscore and a comment. |
| Memory | RAM-efficient analysis code | Operate in place and use views. Comment the non-obvious ones. For us, metrics are accumulated per batch, not kept as outputs. |
| Reference code | A guide for correctness, not for style | Write the direct version our scope allows. |
| Comments | Concise "how and why" for functions and important blocks | The exception to the line-count rule. One or two lines, mostly above the block, in the user's one-line style. Not needed for plotting or logging. |
| Comment content | Why, not what | Never narrate abandoned decisions or rejected options. |
| README | A per-method subsection holds only the question (and papers or derivation if any) | No implementation details. The Output and config sections are updated. |
| This build | The image-format rationale is written next to the code and in the README | A comment where `images.pt` is saved, and the README line "the tile as the model saw it, in [0, 1]". |
| Scope of work | Implement only what is asked | Discuss when asked. Ask before removing or changing something not mentioned. Follow a stated principle literally, mention a cost once, no hybrids. |
| Git | Never commit or push unless told for that change | Commit messages end with the Co-Authored-By line. Work is on `main` of the user's fork. |
| Testing | Test in Colab, not the sandbox | Only `py_compile` and `pyflakes` locally. Say what the Colab run should check. |
| Declined items | Corrections are final | Do not re-flag what was declined. |
| Answers | Short, dense, a table with a detail column | Yes or no first for a yes/no question. |
