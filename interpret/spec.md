Core Integration Strategy
* Single Required Method: register_interpretability_hooks(self, storage_dict: dict) -> list — this is the one place where the developer of the model has to encode and deal with all the model specific details and quirks i.e. he encapsulates all of it in this method, hands us the data we ask from him according to the interface and then no downstream code in our interpretability module needs to know model specific details and quirks.
* Hook Buffer Handling: The rig empties storage_dict in place (storage_dict.clear()) before every forward pass, since the hooks hold a reference to the same dict object.
* Transformer-Native Assumption: The rig explicitly assumes the underlying model processes data as a sequence of spatial patch tokens.
* Canonical Output Structure: The hook callback for each layer must write a dictionary of named token groups to storage_dict[layer_name]. Every group is a 3D tensor of shape (Batch_Size, Num_Tokens, Hidden_Dim) (e.g., (B, T, C)). The “patch” group (B, N, C) is required. The “cls” group (B, 1, C) is a reserved name and optional. Any other named group (e.g., “register”) is free-form and ignored by the rig until a method uses it.
* Implementer-Side Token Separation: The implementer tells the rig which tokens are which. The rig never slices by position and never reads model code. Groups are disjoint (a token appears in exactly one group), the key set is identical across all layers and batches, and patch tokens are in row-major order on a square grid (N is a perfect square).
* Chronological Ordering: The register_interpretability_hooks method must return an ordered list of layer string keys representing the true forward-pass sequence.
* Milestone Semantics: The rig attaches no meaning to milestones. Where hooks attach, how tuple outputs are unpacked, and whether a representation is pre- or post-norm are decided by the implementer. Cross-model depth comparison uses relative position in the returned list.
* Zero Forward Pass Pollution: Hooks attach via standard PyTorch module registration without altering the model’s forward logic.

Data, Precision, and Execution Controls
* Deterministic Execution: The extraction script strictly sets the model to eval() and wraps passes in inference_mode() to eliminate stochastic noise.
* Input Preprocessing: The evaluation preprocessing (deterministic resize, crop, normalize; no random augmentation) must be located first. It is either currently defined on a per-model basis or in one central location. In either case it is refactored into a single shared function that the interpretability module imports and reuses rather than reimplements. The saved input image is exactly the tensor passed to the first model layer, i.e. the output of all preprocessing.
* Storage Precision: Hooks write tensors at native device and dtype. The rig casts all tensors to float16, moves them to CPU and calls .contiguous() before saving to reduce disk and I/O footprint.
* Finite Check: After the float16 cast, the rig runs torch.isfinite on every layer and group, on every batch. By default it raises with the layer name, group name and non-finite fraction (configurable via on_nonfinite = raise | warn).
* Compute Bounding: A num_samples parameter restricts the total number of tiles processed to bound overall compute - these tiles are NOT sequentially sampled but randomly from the full set of tiles (probably good to have some stratification). Selection is deterministic given a seed, uses the same index set across models, and is recorded in the paths file.
* Interface Assertions: The boundary where hook outputs are handed over from the model to the rig is the primary place for assertions, and the rig validates everything the contract promises there before any data is stored or used, failing loudly with the layer name, group name and the violated property. Build this as a dedicated, extensible assertions section, to be extended as new properties are identified. Initial checks include: the returned list has no duplicates; every layer’s output is a dict whose key set contains “patch” and matches across all layers and batches; every group is a 3D tensor (B, T, C) with B equal to the batch size and C identical across groups within a layer; N is a perfect square; tensors are detached; and values are finite (the finite check, run again after the float16 cast).
* Layer Skip Stride: A rig parameter layer_stride (default 1 = keep all) keeps every layer_stride-th entry of the returned layer list, counted from the end so the last layer is always kept. Interface assertions run on all layers before the stride is applied, and the stride is applied before the float16 cast and CPU move. The kept layer keys and the stride value are recorded in the paths file.

Chunked PyTorch Storage Layout
* Chunked File Structure: Preprocessing writes serialized PyTorch files (chunks) containing a configurable number of forward passes (e.g., 500 samples per file).
* Datapoint Bundle Schema: Each chunk stores a dictionary bundling the model-input image, ground-truth label/metadata, and all layer feature tensors (each a dictionary of named token groups) for those samples.
* Dataset Abstraction: A custom PyTorch Dataset utilizes a paths file mapping sample indices to chunk paths, exposing a clean idx -> (image, label, layers) interface while abstracting chunk loading.

Execution Tracks and Just-In-Time Reductions
* Track A (Offline Corpus): Statistical methods query representations sequentially across chunks.
    * Rig-Side Reductions: For methods requiring a single vector per sample (B, C), support both CLS extraction (cls[:, 0, :], when available) and Global Average Pooling (patch.mean(dim=1)). Run both variants and report results separately for comparison. If CLS is unavailable, report it as such and run GAP only. Preserve unreduced token groups for spatial methods; never implicitly reduce other groups.
    * Spatial Operations: Layer Inversion and dense probing utilize the full “patch” group (B, N, C) alongside the saved model-input images.
* Track B (Online Probe): Dynamic methods (like RELAX) bypass saved chunk files entirely to perform active input perturbations, intercepting the raw token-group tensors live at native device and dtype (exempt from the float16/CPU cast) and applying rig-side reductions for cosine delta evaluation.

Detailed Interface Documentation (Code Contract)

python
def register_interpretability_hooks(self, storage_dict: dict) -> list[str]:
    """
    Registers forward hooks on intermediate computational milestones (layers)
    for interpretability analysis.

    This interface contract assumes a Transformer-native architecture where representations
    are passed as token sequences across layers.

    INTERFACE CONTRACT REQUIREMENTS:
    --------------------------------
    1. Target Selection: The model author decides which computational milestones to monitor
       (e.g., the output of each Transformer block).
    2. Output Structure & Shape: Every hook callback MUST write a dictionary of named token
       groups to `storage_dict[layer_name]`. Each group is a detached tensor of shape
       (Batch_Size, Num_Tokens, Hidden_Dim) [i.e. (B, T, C)]. The "patch" group (B, N, C) is
       required; "cls" (B, 1, C) is a reserved optional name; any other group (e.g.,
       "register") is free-form. Groups are disjoint, the key set MUST be identical across
       all layers and batches, and patch tokens MUST be in row-major order on a square grid.
    3. Tensor Precision and Device: Write tensors at their native device and dtype. Do NOT
       cast to float16 or move to CPU; the rig handles this.
    4. Token Separation: You, the implementer, decide which tokens belong to which group. The
       rig never slices by position and never reads model code. If the model has a [CLS] or
       global summary token, put it in "cls"; register tokens or similar can go in their own
       named group.
    5. Pure Raw Sequence: Do NOT apply Global Average Pooling or flattening inside the hook
       callback. Return the un-reduced token sequence of each group.
    6. Non-Disruptive: Hooks must run cleanly without altering the model's standard forward
       pass return values or intermediate states.
    7. Milestone Semantics: The rig attaches no meaning to your chosen milestones (pre- or
       post-norm, tuple unpacking, etc.). Optional, non-binding recommendation: make the last
       milestone the representation your downstream heads consume.

    Args:
        storage_dict (dict): A dictionary handle managed by the extraction pipeline. Hooks write
                             a dictionary of named token-group tensors (native device and
                             dtype) into this dict using their assigned layer keys.

    Returns:
        list[str]: An ordered list of string layer keys (e.g., ["block_1", "block_2", ...])
                   reflecting the exact chronological forward-pass order of execution.
    """
    raise NotImplementedError