"""Small explicit token-level clipped GRPO update with supplied traffic advantages.

This intentionally does not use TRL's reward callback, which would regenerate
responses after environment scoring. No value head or learned reward model.
"""

from __future__ import annotations

import math
import random


def clipped_objective(new, old, reference, advantage, epsilon, beta):
    """Per-response token mean; token-level importance ratio, sequence-level A."""
    import torch

    ratio = torch.exp(new - old)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantage
    delta = reference - new
    kl = torch.exp(delta) - delta - 1
    loss = (-torch.minimum(unclipped, clipped) + beta * kl).mean()
    return loss, kl.mean(), ((ratio - 1).abs() > epsilon).float().mean()


class GRPOUpdater:
    def __init__(self, policy, config):
        import torch

        self.p, self.c = policy, config
        self.parameters = [p for p in policy.model.parameters() if p.requires_grad]
        if not self.parameters:
            raise RuntimeError("No trainable LoRA parameters")
        self.optimizer = torch.optim.AdamW(
            self.parameters, lr=config["learning_rate"], weight_decay=0.0
        )
        self.random = random.Random(config["seed"])
        self.optimizer_steps = 0

    def update(self, rows):
        import torch

        if not rows:
            raise ValueError("Empty training window")
        p, c = self.p, self.c
        # Freeze behavior densities BEFORE any updates. Ref is original base via disabled adapter.
        p.model.eval()
        old, reference = [], []
        with torch.no_grad():
            for row in rows:
                old.append(p.token_logps(row).detach().cpu())
                reference.append(
                    p.token_logps(row, reference=True).detach().cpu()
                    if c["kl_beta"]
                    else old[-1]
                )
        metrics = {"loss": [], "kl": [], "clip_fraction": [], "gradient_norm": []}
        initial = [x.detach().cpu().clone() for x in self.parameters]
        # We accumulate per-response losses with explicit final-window size, so a
        # partial accumulation window has exactly the same weight as a full one.
        size = c["micro_batch_size"] * c["gradient_accumulation_steps"]
        p.model.train()
        for _ in range(c["update_epochs"]):
            indices = list(range(len(rows)))
            self.random.shuffle(indices)
            for start in range(0, len(indices), size):
                batch = indices[start : start + size]
                self.optimizer.zero_grad(set_to_none=True)
                for j in batch:
                    new = p.token_logps(rows[j])
                    loss, kl, clipped = clipped_objective(
                        new,
                        old[j].to(p.device),
                        reference[j].to(p.device),
                        rows[j].advantage,
                        c["clip_epsilon"],
                        c["kl_beta"],
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            "Nonfinite GRPO loss; no optimizer step performed"
                        )
                    (loss / len(batch)).backward()
                    metrics["loss"].append(float(loss.detach()))
                    metrics["kl"].append(float(kl.detach()))
                    metrics["clip_fraction"].append(float(clipped.detach()))
                norm = torch.nn.utils.clip_grad_norm_(
                    self.parameters, c["max_grad_norm"], error_if_nonfinite=True
                )
                metrics["gradient_norm"].append(float(norm))
                self.optimizer.step()
                self.optimizer_steps += 1
        p.model.eval()
        delta = max(
            float((x.detach().cpu() - before).abs().max())
            for x, before in zip(self.parameters, initial)
        )
        if not math.isfinite(delta):
            raise FloatingPointError("Nonfinite LoRA update")
        return {
            **{k: sum(v) / len(v) for k, v in metrics.items()},
            "max_lora_delta": delta,
            "optimizer_steps": self.optimizer_steps,
            "responses": len(rows),
            "response_tokens": sum(len(r.completion_ids) for r in rows),
            "nonzero_adv_fraction": sum(abs(r.advantage) > 1e-10 for r in rows)
            / len(rows),
            "truncated_fraction": sum(r.truncated for r in rows) / len(rows),
        }
