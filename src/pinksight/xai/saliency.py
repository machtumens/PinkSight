
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F


def grad_cam_3d(model, volume, target_layer, target=None, mode: str = "grad_cam") -> np.ndarray:
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
        cam = F.relu((store["g"] * store["a"]).sum(dim=1, keepdim=True))
    else:
        weights = store["g"].mean(dim=(2, 3, 4), keepdim=True)  
        cam = F.relu((weights * store["a"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=volume.shape[2:], mode="trilinear", align_corners=False)
    cam_np = cam[0, 0].detach().cpu().numpy()
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
    import torch as _torch
    from scipy.stats import spearmanr

    _reject_subtype_head(grade_head)

    model.eval()

    def _grade_logit(v):
        store = {}
        h = target_layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
        try:
            model(v)
        finally:
            h.remove()
        feat = store["a"]
        pooled = feat.mean(dim=(2, 3, 4)) if feat.dim() == 5 else feat.reshape(feat.shape[0], -1)
        return grade_head(pooled).reshape(-1)[0]

    cam_original = _hirescam_grade(model, vol, grade_head, target_layer)

    base = vol.detach().clone()
    delta = _torch.zeros_like(base, requires_grad=True)
    opt = _torch.optim.Adam([delta], lr=lr)
    with _torch.no_grad():
        logit0 = float(_grade_logit(base))
    want_positive = (target_flip == 1)
    for _ in range(n_steps):
        opt.zero_grad()
        logit = _grade_logit(base + delta)
        margin = (-logit if want_positive else logit)  
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
    model.eval()
    v = vol.detach().clone().requires_grad_(True)
    store = {}

    def _fwd(m, i, o):
        store["a"] = o
        o.register_hook(lambda g: store.__setitem__("g", g))

    h1 = target_layer.register_forward_hook(_fwd)
    try:
        model(v)  
        feat = store["a"]
        pooled = feat.mean(dim=(2, 3, 4)) if feat.dim() == 5 else feat.reshape(feat.shape[0], -1)
        logit = grade_head(pooled).reshape(-1)[0]
        model.zero_grad()
        for p in grade_head.parameters():
            p.grad = None
        logit.backward()
    finally:
        h1.remove()
    cam = F.relu((store["g"] * store["a"]).sum(dim=1, keepdim=True))  
    cam = F.interpolate(cam, size=v.shape[2:], mode="trilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    span = cam.max() - cam.min()
    return (cam - cam.min()) / span if span > 0 else np.zeros_like(cam)


def _reject_subtype_head(head) -> None:
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
    m = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.empty_like(p).uniform_(-0.5, 0.5, generator=g))
    return m


def torch_clinical_predict_proba(model, n_num):
    import torch

    def predict_proba(X):
        arr = np.atleast_2d(np.asarray(X, dtype=float))
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
    if values.ndim == 3:  
        values = values[..., -1]
    base = float(np.ravel(explainer.expected_value)[-1])
    return values, base


def _is_torch_module(model) -> bool:
    try:
        import torch.nn as nn
    except Exception:  
        return False
    return isinstance(model, nn.Module)


def attention_map(attn_weights) -> np.ndarray:
    a = np.asarray(attn_weights, float)
    mx = a.max()
    return a / mx if mx > 0 else a


def _selfcheck() -> int:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
    from fixtures.synthetic import clinical_model_fixture  

    fx = clinical_model_fixture()
    values, base = clinical_shap(fx["model"], fx["X"], fx["background"])
    assert values.shape == fx["X"].shape, values.shape
    margin = fx["model"].decision_function(fx["X"])
    assert np.allclose(base + values.sum(axis=1), margin, atol=1e-6), "linear additivity broke"

    import torch
    import torch.nn as nn

    n_num, cards = 3, [2, 3]
    torch.manual_seed(0)

    class _TinyClinical(nn.Module):  
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
    background_X = np.column_stack([bg_num, bg_cat])  
    X = background_X[:4]
    vals, kbase = clinical_shap(model, X, background_X, n_num=n_num, nsamples=64)  
    assert vals.shape == X.shape, vals.shape
    pp = torch_clinical_predict_proba(model, n_num)(X).ravel()
    assert np.allclose(kbase + vals.sum(axis=1), pp, atol=1e-3), "kernel efficiency broke"
    print("saliency selfcheck OK: LinearExplainer additive + KernelExplainer(FTTransformer) efficient")  
    return 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
