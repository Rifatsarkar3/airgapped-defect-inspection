# -*- coding: utf-8 -*-
"""
protocol_validation.py — gradient-flow gate for training-time loss components.

A loss term that is mathematically well-posed but disconnected from the
parameter graph (e.g. computed under torch.no_grad(), or on quantities that
do not depend on the weights) silently contributes zero gradient and cannot
train the network. This module provides the gate that catches that failure
before any experiment runs, and demonstrates it on two SSF formulations:

  * the CORRECTED feature-space SSF used in the paper (Experiment 5)  -> PASS
  * the LEGACY raw-pixel / no_grad SSF that produced artifactual results -> FAIL

Run:  python protocol_validation.py
Writes: Results/protocol_validation_report.txt
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os
import torch
import torch.nn as nn

import exp11_paired_validation_v2 as E
from exp14_broken_mystery import broken_ssf

DEVICE = E.DEVICE


def gradient_flow_gate(component_name, loss_closure, n_params_check=None):
    """Return (passed, grad_L1). loss_closure(model, X, y) -> scalar loss whose
    gradient w.r.t. model parameters is tested for non-zero L1 norm."""
    torch.manual_seed(0)
    model = E.build_model().to(DEVICE)
    X = torch.randn(8, 3, 224, 224, device=DEVICE)
    y = torch.tensor([E.DEF_CLASS, E.OK_CLASS] * 4, device=DEVICE)
    model.zero_grad(set_to_none=True)
    loss = loss_closure(model, X, y)
    if not torch.is_tensor(loss) or not loss.requires_grad:
        return False, 0.0
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in model.parameters()
            if p.grad is not None)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return g > 0.0, g


def corrected_ssf(model, X, y):
    _, feat = E.forward_with_feature(model, X)
    return E.ssf_coupled_loss(feat, y, margin=1.0)   # margin=1 forces hinge active


def legacy_broken_ssf(model, X, y):
    # mirrors the historical term: raw-pixel gradient energy under no_grad,
    # added to a CE term so the closure is a plausible "total loss"
    ce = nn.CrossEntropyLoss()(model(X), y)
    return ce * 0.0 + E.ALPHA_STAR * broken_ssf(X, y, margin=1.0)


def main():
    os.makedirs("Results", exist_ok=True)
    cases = [
        ("CORRECTED feature-space SSF (Eq. 2, Experiment 5)", corrected_ssf, "PASS"),
        ("LEGACY raw-pixel no_grad SSF (superseded)", legacy_broken_ssf, "FAIL"),
    ]
    lines = ["=" * 66,
             "  protocol_validation.py — gradient-flow gate",
             "  A loss component PASSES iff it contributes non-zero parameter",
             "  gradient on a random batch; otherwise the experiment must abort.",
             "=" * 66, ""]
    all_ok = True
    for name, closure, expect in cases:
        passed, gnorm = gradient_flow_gate(name, closure)
        verdict = "PASS" if passed else "FAIL"
        ok = (verdict == expect)
        all_ok &= ok
        lines.append(f"  [{verdict:4s}] {name}")
        lines.append(f"         parameter-gradient L1 = {gnorm:.4e}   "
                     f"(expected {expect})  {'OK' if ok else 'UNEXPECTED'}")
        lines.append("")
    lines.append("  Demonstration: the gate accepts the corrected formulation and")
    lines.append("  rejects the legacy one that produced the artifactual results.")
    lines.append("=" * 66)
    report = "\n".join(lines)
    print(report)
    with open("Results/protocol_validation_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n[saved] Results/protocol_validation_report.txt")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
