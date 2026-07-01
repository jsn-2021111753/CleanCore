"""DEEPFOOL boundary-sensitive subset selection baseline."""

from __future__ import annotations

import numpy as np
import torch

from common.interfaces import MethodOutput
from methods.base import MethodContext
from methods.common import classwise_topk
from methods.torch_utils import train_aux_model


def deepfool_l2_single(
    model: torch.nn.Module,
    x0: np.ndarray,
    num_classes: int,
    max_iter: int,
    overshoot: float,
    num_classes_attack: int = 0,
    stability_eps: float = 1e-4,
) -> tuple[float, bool]:
    device = next(model.parameters()).device
    x_start = torch.tensor(x0, dtype=torch.float32, device=device)
    with torch.no_grad():
        initial_logits = model(x_start.unsqueeze(0)).squeeze(0)
        k0 = int(torch.argmax(initial_logits).item())
        if num_classes_attack and num_classes_attack > 0:
            class_order = torch.argsort(initial_logits, descending=True)[: min(int(num_classes_attack), int(num_classes))].cpu().numpy().tolist()
        else:
            class_order = list(range(int(num_classes)))
    label = int(k0)
    pert_image = x_start.detach().clone()
    r_tot = torch.zeros_like(x_start)
    success = False
    for _ in range(max(1, int(max_iter))):
        x = pert_image.detach().requires_grad_(True)
        logits = model(x.unsqueeze(0)).squeeze(0)
        k = int(torch.argmax(logits).item())
        if k != label:
            success = True
            break
        grad_orig = torch.autograd.grad(logits[label], x, retain_graph=True)[0].detach()
        best_pert = None
        best_w = None
        for j in class_order[1:]:
            cur_grad = torch.autograd.grad(logits[int(j)], x, retain_graph=True)[0].detach()
            w_k = cur_grad - grad_orig
            f_k = logits[int(j)] - logits[label]
            denom = torch.norm(w_k, p=2)
            if float(denom.item()) <= 0.0:
                continue
            pert_k = torch.abs(f_k.detach()) / denom
            if best_pert is None or float(pert_k.item()) < best_pert:
                best_pert = float(pert_k.item())
                best_w = w_k
        if best_w is None or best_pert is None:
            break
        norm_w = torch.norm(best_w, p=2)
        r_i = (best_pert + float(stability_eps)) * best_w / norm_w
        r_tot = r_tot + r_i
        pert_image = x_start + (1.0 + float(overshoot)) * r_tot
        with torch.no_grad():
            k_new = int(torch.argmax(model(pert_image.unsqueeze(0)).squeeze(0)).item())
        if k_new != label:
            success = True
            break
    r_tot = (1.0 + float(overshoot)) * r_tot
    return float(torch.norm(r_tot, p=2).item()), bool(success)


def run(ctx: MethodContext) -> MethodOutput:
    warmup_epochs = int(ctx.param("warmup_epochs", ctx.param("score_epochs", 5)))
    max_iter = int(ctx.param("max_iter", 20))
    overshoot = float(ctx.param("overshoot", 0.02))
    stability_eps = float(ctx.param("stability_eps", 1e-4))
    num_classes_attack = int(ctx.param("num_classes_attack", ctx.param("num_classes", 0)))
    max_scored_samples = int(ctx.param("max_scored_samples", 0))
    select_mode = str(ctx.param("select_mode", "small")).lower()
    subset_fraction = float(ctx.param("subset_fraction", 0.10))
    subset_size = int(ctx.param("subset_size", 0))
    min_per_class = int(ctx.param("min_per_class", 1))

    with ctx.timed_phase("deepfool.auxiliary_training"):
        model = train_aux_model(ctx, ctx.X_train, ctx.y_train, max_epochs=max(1, warmup_epochs), seed_offset=500)
        model.eval()
        device = torch.device(str(ctx.training_config.get("device", "cpu")))
        model.to(device)

    scores = np.full(ctx.n_samples, np.nan, dtype=np.float32)
    success = np.zeros(ctx.n_samples, dtype=bool)
    if max_scored_samples and 0 < max_scored_samples < ctx.n_samples:
        rng = np.random.default_rng(ctx.seed)
        score_idx = rng.choice(ctx.n_samples, size=max_scored_samples, replace=False).astype(np.int64)
    else:
        score_idx = np.arange(ctx.n_samples, dtype=np.int64)
    with ctx.timed_phase("deepfool.boundary_scoring"):
        for i in score_idx:
            score, ok = deepfool_l2_single(
                model,
                ctx.X_train[int(i)],
                ctx.num_classes,
                max_iter=max_iter,
                overshoot=overshoot,
                num_classes_attack=num_classes_attack,
                stability_eps=stability_eps,
            )
            scores[int(i)] = np.float32(score)
            success[int(i)] = bool(ok)
    with ctx.timed_phase("deepfool.subset_selection"):
        largest = select_mode == "large"
        selected, weights, report = classwise_topk(
            ctx.y_train,
            scores,
            ctx.num_classes,
            subset_fraction=subset_fraction,
            subset_size=subset_size,
            min_per_class=min_per_class,
            largest=largest,
        )
    return MethodOutput.from_arrays(
        n_samples=ctx.n_samples,
        selected_indices=selected,
        sample_weights=weights,
        metadata={
            "paper": "DeepFool: a simple and accurate method to fool deep neural networks",
            "method_type": "training_assisted_data_processing",
            "warmup_epochs": warmup_epochs,
            "max_iter": max_iter,
            "overshoot": overshoot,
            "stability_eps": stability_eps,
            "num_classes_attack": num_classes_attack,
            "select_mode": select_mode,
            "subset_fraction": subset_fraction,
            "num_scored": int(len(score_idx)),
            "num_success": int(success.sum()),
            **report,
        },
    )
