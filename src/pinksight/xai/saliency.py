"""P10 saliency generators: 3D Grad-CAM (MRI), SHAP (clinical), cross-attention extraction.

grad_cam_3d hand-rolls forward/backward hooks + a trilinear upsample (ponytail: ~15 lines, fully
deterministic — avoids captum's 3D-interpolate edge cases). clinical_shap uses shap.LinearExplainer
(exact for the additive clinical head). attention_map normalises a cross-attention tensor from P06.
All consume tensors, so the faithfulness suite is model-agnostic at test time.

Claim-ledger: saliency shows "characterisation / localisation", never "early detection".
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F


def grad_cam_3d(model, volume, target_layer, target=None, mode: str = "grad_cam") -> np.ndarray:
    """Grad-CAM / HiResCAM for a 3D CNN. volume: (1,C,D,H,W) tensor. Returns (D,H,W) saliency in [0,1].

    The encoder may be fully frozen (all params `requires_grad=False`), so we enable grad on the
    INPUT volume: that alone gives `score.backward()` a graph to traverse and lets the target-layer
    backward hook fire. Without this the backward dies with "element 0 ... does not require grad"
    (TICKET-004). Grads flow to activations regardless of whether params are trainable.

    mode:
      "grad_cam" (default) — Grad-CAM (Selvaraju 2017): GAP the gradient over spatial dims to get
        per-channel weights, then weighted-sum the activations. Backward-compatible; unchanged.
      "hirescam" — HiResCAM (Draelos & Carin 2020, SPEC-07 §4.1): element-wise (grad × activation)
        FIRST, then sum channels — the gradient is retained per-spatial-position before accumulation,
        which is the faithfulness-provable variant. No GAP.
    """
    if mode not in ("grad_cam", "hirescam"):
        raise ValueError(f"mode must be 'grad_cam' or 'hirescam', got {mode!r}")
    model.eval()
    volume = volume.detach().clone().requires_grad_(True)
    store = {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("g", go[0]))
    try:
        out = model(volume)
        score = out[0, 0] if target is None else out[0, target]
        model.zero_grad()
        score.backward()
    finally:
        h1.remove()
        h2.remove()
    if mode == "hirescam":
        # HiResCAM: element-wise grad × activation, then sum channels (no GAP) — faithful per
        # Draelos & Carin 2020; gradient retained per spatial position before accumulation.
        cam = F.relu((store["g"] * store["a"]).sum(dim=1, keepdim=True))
    else:
        weights = store["g"].mean(dim=(2, 3, 4), keepdim=True)  # GAP over spatial dims
        cam = F.relu((weights * store["a"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=volume.shape[2:], mode="trilinear", align_corners=False)
    cam_np = cam[0, 0].detach().cpu().numpy()
    # Release hooked GPU tensors immediately — they are large (activation maps); without
    # explicit deletion they pin VRAM until the next GC cycle, causing per-patient accumulation.
    del cam, store["a"], store["g"], store
    span = cam_np.max() - cam_np.min()
    return (cam_np - cam_np.min()) / span if span > 0 else np.zeros_like(cam_np)


def counterfactual_grade_map(
    model,
    vol: "torch.Tensor",
    grade_head,
    target_layer,
    target_flip: int = 1,
    n_steps: int = 200,
    lr: float = 0.01,
    lambda_l1: float = 0.05,
) -> dict:
    """Minimal-perturbation counterfactual saliency for grade characterisation (GRADE HEAD ONLY).

    WORDING GUARD (claim-ledger, decisions.md LOCK-1, submission-time):
    All outputs represent 'counterfactual input-sensitivity for grade characterisation'. Do NOT
    frame as 'if enhancement were X%, cancer would become grade Y' — that is a growth-rate / kinetics
    claim, FORBIDDEN. Allowed wording: 'model grade characterisation is sensitive to enhancement
    pattern in region Z'. Use 'characterisation sensitivity', never 'causal intervention'.

    NOT a growth-rate / kinetics claim. NOT a causal intervention. Grade-head only.

    Method: compute a HiResCAM map from the trained model on the grade head, then run Adam on the
    INPUT space to find the minimal (L1-penalised) perturbation that flips the grade-head logit sign
    (NHG1 <-> NHG3). The perturbation delta is the counterfactual saliency — the voxels changed to
    flip the prediction. Spatial agreement with the original HiResCAM is reported (Spearman rho).

    Grade-head-only guard (execute-agent E-6): raises ValueError if `grade_head` is a subtype head —
    the subtype head has negligible imaging signal (H6 Shapley −0.025), so a counterfactual over it
    is ill-defined. Detection is by class name OR a `debug_name`/`_head_name` sentinel containing
    'subtype' (case-insensitive).

    Args:
      model: nn.Module producing a feature map at `target_layer` (the shared encoder trunk).
      vol: (1, C, D, H, W) input volume tensor.
      grade_head: nn.Module mapping the pooled features to the grade logit (NOT a subtype head).
      target_layer: conv layer to hook for HiResCAM.
      target_flip: 0 = flip toward NHG1, 1 = flip toward NHG3.
      n_steps, lr, lambda_l1: perturbation optimiser budget + sparsity penalty.

    Returns dict:
      cam_original: (D,H,W) HiResCAM from the trained model (grade head).
      cam_counterfactual: (D,H,W) |perturbation| delta (voxels changed to flip the prediction).
      flip_achieved: bool — did the perturbation flip the grade logit sign?
      delta_logit: float — grade logit change from the perturbation.
      spearman_rho: float — spatial agreement between cam_original and cam_counterfactual.
    """
    import torch as _torch
    from scipy.stats import spearmanr

    _reject_subtype_head(grade_head)

    model.eval()

    def _grade_logit(v):
        """Forward v through the encoder trunk up to target_layer's pooled features, then grade_head.

        The trunk output is GAP-pooled to a vector so a Linear grade_head consumes it — this mirrors
        the encoder->head wiring used across the fixture/real models."""
        store = {}
        h = target_layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
        try:
            model(v)
        finally:
            h.remove()
        feat = store["a"]
        pooled = feat.mean(dim=(2, 3, 4)) if feat.dim() == 5 else feat.reshape(feat.shape[0], -1)
        return grade_head(pooled).reshape(-1)[0]

    # (1) HiResCAM on the grade head — hook the target layer, backprop the grade logit, then take
    #     element-wise (grad × activation) summed over channels (Draelos & Carin 2020, SPEC-07 §4.1).
    cam_original = _hirescam_grade(model, vol, grade_head, target_layer)

    # (2) Minimal-perturbation counterfactual: Adam on a delta added to the input.
    base = vol.detach().clone()
    delta = _torch.zeros_like(base, requires_grad=True)
    opt = _torch.optim.Adam([delta], lr=lr)
    with _torch.no_grad():
        logit0 = float(_grade_logit(base))
    # Flip target: push the logit toward the OPPOSITE sign of logit0 (or the requested grade band).
    want_positive = (target_flip == 1)
    for _ in range(n_steps):
        opt.zero_grad()
        logit = _grade_logit(base + delta)
        # Hinge toward the target sign + L1 sparsity on the perturbation (minimal change).
        margin = (-logit if want_positive else logit)  # want logit>0 if flip-to-NHG3 else logit<0
        loss = _torch.relu(margin + 1.0) + lambda_l1 * delta.abs().mean()
        loss.backward()
        opt.step()
    with _torch.no_grad():
        logit1 = float(_grade_logit(base + delta))
    flip_achieved = bool((logit1 > 0) == want_positive and (logit1 > 0) != (logit0 > 0))
    delta_logit = float(logit1 - logit0)

    cf = delta.detach().abs()
    cf = cf[0].mean(dim=0) if cf.dim() == 5 else cf.reshape(cf.shape[0], -1)[0]
    cf_np = cf.cpu().numpy()
    span = cf_np.max() - cf_np.min()
    cam_counterfactual = (cf_np - cf_np.min()) / span if span > 0 else np.zeros_like(cf_np)

    # (3) Spatial agreement between the two maps (Spearman rho on flattened volumes).
    if cam_original.shape == cam_counterfactual.shape and cam_counterfactual.std() > 0:
        rho = float(spearmanr(cam_original.flatten(), cam_counterfactual.flatten())[0])
    else:
        rho = 0.0

    return {
        "cam_original": cam_original,
        "cam_counterfactual": cam_counterfactual,
        "flip_achieved": flip_achieved,
        "delta_logit": delta_logit,
        "spearman_rho": rho,
    }


def _hirescam_grade(model, vol, grade_head, target_layer) -> np.ndarray:
    """HiResCAM (D,H,W) in [0,1] for the grade logit: hook target_layer, backprop grade logit,
    element-wise (grad × activation) summed over channels, ReLU, trilinear upsample."""
    model.eval()
    v = vol.detach().clone().requires_grad_(True)
    store = {}

    def _fwd(m, i, o):
        # Capture the activation AND attach a tensor-level grad hook so the backward gradient at this
        # exact tensor is recorded (module full-backward hooks are unreliable when the output is
        # consumed outside the module's normal forward continuation — TICKET-004 idiom, tensor hook).
        store["a"] = o
        o.register_hook(lambda g: store.__setitem__("g", g))

    h1 = target_layer.register_forward_hook(_fwd)
    try:
        model(v)  # forward hook captures the activation + registers the tensor grad hook
        feat = store["a"]
        pooled = feat.mean(dim=(2, 3, 4)) if feat.dim() == 5 else feat.reshape(feat.shape[0], -1)
        logit = grade_head(pooled).reshape(-1)[0]
        model.zero_grad()
        for p in grade_head.parameters():
            p.grad = None
        logit.backward()
    finally:
        h1.remove()
    cam = F.relu((store["g"] * store["a"]).sum(dim=1, keepdim=True))  # HiResCAM element-wise
    cam = F.interpolate(cam, size=v.shape[2:], mode="trilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    span = cam.max() - cam.min()
    return (cam - cam.min()) / span if span > 0 else np.zeros_like(cam)


def _reject_subtype_head(head) -> None:
    """Execute-agent E-6: enforce grade-head-only. Raise if `head` looks like a subtype head.

    Checks the class name and any `debug_name` / `_head_name` sentinel attribute for 'subtype'
    (case-insensitive). The subtype head carries negligible imaging signal (H6), so a counterfactual
    over it is ill-defined — this guard blocks that misuse at the API boundary."""
    name_bits = [type(head).__name__]
    for attr in ("debug_name", "_head_name", "name"):
        val = getattr(head, attr, None)
        if isinstance(val, str):
            name_bits.append(val)
    if any("subtype" in b.lower() for b in name_bits):
        raise ValueError(
            "counterfactual_grade_map is grade-head only — received a subtype head "
            f"(name hints: {name_bits}). Subtype imaging signal is negligible (H6); "
            "counterfactuals over it are ill-defined."
        )


def randomize_weights(model, seed=0):
    """Deep-copy the model with every parameter re-initialised (Adebayo model-randomization)."""
    m = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.empty_like(p).uniform_(-0.5, 0.5, generator=g))
    return m


def torch_clinical_predict_proba(model, n_num):
    """predict_proba shim for a real torch FT-Transformer / fusion clinical head.

    Wraps `model(x_cont, x_cat)` (rtdl FTTransformer: separate float-numeric + long-categorical
    inputs, logit out) into a single-matrix `fn(X)->[n,1]` interface KernelExplainer can call. `X`
    columns follow FIX-1 `background_X` order: `n_num` standardized-numeric columns first, then the
    categorical CODES (stored as float; cast back to long here). Returns positive-class probability.
    """
    import torch

    def predict_proba(X):
        arr = np.atleast_2d(np.asarray(X, dtype=float))
        # Move inputs to the model's device (TICKET-024): KernelExplainer builds plain CPU arrays,
        # so a CUDA-resident FT-Transformer would otherwise hit a device-mismatch. `next(...)` reads
        # the model's actual device; falls back to CPU for a parameter-less model.
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        xc = torch.tensor(arr[:, :n_num], dtype=torch.float32, device=device)
        xq = torch.tensor(np.rint(arr[:, n_num:]).astype(int), dtype=torch.long, device=device)
        model.eval()
        with torch.no_grad():
            logit = model(xc, xq)
        prob = torch.sigmoid(logit).cpu().numpy().reshape(-1, 1)
        return prob

    return predict_proba


def _assert_background_covers_levels(background, n_num, cat_cardinalities):
    """TICKET-016 guard: a ≤100-row background can miss a rare categorical level, so the model's
    true cardinality gets undercounted (max(code)+1 < true cards) → wrong/undershooting explanation.

    When the true `cat_cardinalities` are known, assert every level 0..card-1 of each categorical
    appears at least once in `background`. Otherwise (schema unknown) emit a hard warning so the
    silent undershoot is at least recorded. Codes live in the columns after the `n_num` numeric ones
    (FIX-1 background_X layout: numeric first, then categorical codes).
    """
    import warnings

    bg = np.atleast_2d(np.asarray(background, dtype=float))
    cat_codes = np.rint(bg[:, n_num:]).astype(int)
    if cat_cardinalities is None:
        if len(bg) <= 100:
            warnings.warn(
                f"clinical_shap: categorical cardinality inferred from a {len(bg)}-row background "
                "without a known schema; rare levels may be absent and undercount cards "
                "(TICKET-016). Pass cat_cardinalities to assert full coverage.",
                stacklevel=3,
            )
        return
    if cat_codes.shape[1] != len(cat_cardinalities):
        raise ValueError(
            f"cat_cardinalities has {len(cat_cardinalities)} entries but background has "
            f"{cat_codes.shape[1]} categorical columns"
        )
    for j, card in enumerate(cat_cardinalities):
        present = set(cat_codes[:, j].tolist())
        missing = set(range(card)) - present
        if missing:
            raise ValueError(
                f"TICKET-016: categorical column {j} is missing level(s) {sorted(missing)} "
                f"from the {len(bg)}-row SHAP background (true cardinality {card}); the explanation "
                "would undercount cardinality. Widen the background to cover every level."
            )


def clinical_shap(
    model, X, background, *, torch_model=False, n_num=None, nsamples="auto", cat_cardinalities=None
):
    """SHAP attributions for the clinical head. Returns (values (n,d), base_value).

    Two paths:
      * LinearExplainer (default) — exact & additive; for the sklearn linear fixture (has `.coef_`).
      * KernelExplainer — model-agnostic; for the real torch FT-Transformer. Enabled by
        `torch_model=True` OR auto-selected when `model` is an `nn.Module`. Requires `n_num`
        (numeric-column count) so the FIX-1 `background_X` matrix can be split into the numeric ⊕
        categorical-code inputs the torch forward expects. `background` here is FIX-1 `background_X`.

    TICKET-016: pass `cat_cardinalities` (true per-column level counts from the schema) to assert the
    background covers every categorical level; without it a ≤100-row background can silently
    undercount cardinality and undershoot the explanation.
    """
    import shap

    use_kernel = torch_model or _is_torch_module(model)
    if use_kernel:
        if n_num is None:
            raise ValueError("KernelExplainer path needs n_num (numeric-column count of background_X)")
        _assert_background_covers_levels(background, n_num, cat_cardinalities)
        predict_proba = torch_clinical_predict_proba(model, n_num)
        explainer = shap.KernelExplainer(predict_proba, np.asarray(background, dtype=float))
        values = np.asarray(explainer.shap_values(np.asarray(X, dtype=float), nsamples=nsamples), float)
    else:
        explainer = shap.LinearExplainer(model, background)
        values = np.asarray(explainer.shap_values(X), float)
    if values.ndim == 3:  # some shap versions return (n, d, classes)
        values = values[..., -1]
    base = float(np.ravel(explainer.expected_value)[-1])
    return values, base


def _is_torch_module(model) -> bool:
    """True iff `model` is a torch nn.Module (auto-selects the KernelExplainer path)."""
    try:
        import torch.nn as nn
    except Exception:  # torch not importable — treat as non-torch (fixture path)
        return False
    return isinstance(model, nn.Module)


def attention_map(attn_weights) -> np.ndarray:
    """Normalise a cross-attention weight tensor (from P06 fusion) to [0,1]."""
    a = np.asarray(attn_weights, float)
    mx = a.max()
    return a / mx if mx > 0 else a


def _selfcheck() -> int:
    """Runnable check: (1) LinearExplainer fixture path stays exact-additive; (2) KernelExplainer
    path auto-selects on a real torch FTTransformer and consumes FIX-1 background_X (numeric ⊕
    categorical-code columns). Exits 0 on success."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
    from fixtures.synthetic import clinical_model_fixture  # noqa: E402

    # (1) LinearExplainer path — additivity preserved.
    fx = clinical_model_fixture()
    values, base = clinical_shap(fx["model"], fx["X"], fx["background"])
    assert values.shape == fx["X"].shape, values.shape
    margin = fx["model"].decision_function(fx["X"])
    assert np.allclose(base + values.sum(axis=1), margin, atol=1e-6), "linear additivity broke"

    # (2) KernelExplainer path — a torch nn.Module with the SAME (x_cont, x_cat)->logit signature as
    # the real FTTransformer (rtdl), consuming FIX-1 background_X (numeric ⊕ categorical-code cols).
    # A tiny stand-in module is used here (not rtdl.FTTransformer) so the check has no heavy/optional
    # dependency; the KernelExplainer path is model-agnostic — it only needs an nn.Module + n_num.
    import torch
    import torch.nn as nn

    n_num, cards = 3, [2, 3]
    torch.manual_seed(0)

    class _TinyClinical(nn.Module):  # forward mirrors rtdl FTTransformer: (x_cont, x_cat) -> (N,1)
        def __init__(self):
            super().__init__()
            self.emb = nn.ModuleList([nn.Embedding(c, 4) for c in cards])
            self.lin = nn.Linear(n_num + 4 * len(cards), 1)

        def forward(self, x_cont, x_cat):
            e = torch.cat([m(x_cat[:, i]) for i, m in enumerate(self.emb)], dim=1)
            return self.lin(torch.cat([x_cont, e], dim=1))

    model = _TinyClinical()
    rng = np.random.default_rng(0)
    bg_num = rng.normal(0, 1, (16, n_num))
    bg_cat = np.column_stack([rng.integers(0, c, 16) for c in cards]).astype(float)
    background_X = np.column_stack([bg_num, bg_cat])  # FIX-1 layout: numeric first, then cat codes
    X = background_X[:4]
    vals, kbase = clinical_shap(model, X, background_X, n_num=n_num, nsamples=64)  # auto-KernelExplainer
    assert vals.shape == X.shape, vals.shape
    # SHAP efficiency: base + sum(phi) ≈ predicted proba
    pp = torch_clinical_predict_proba(model, n_num)(X).ravel()
    assert np.allclose(kbase + vals.sum(axis=1), pp, atol=1e-3), "kernel efficiency broke"
    print("saliency selfcheck OK: LinearExplainer additive + KernelExplainer(FTTransformer) efficient")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
