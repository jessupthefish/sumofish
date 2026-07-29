"""The yardstick. NOT part of the agent-editable surface.

`train.py` is rewritten by an agent; this is not. The metric lived in `train.py`
below its own `# EDITABLE` marker, with a docstring reading "FROZEN, do not
change" and nothing enforcing that word -- `run.py::check_frozen` only hashes the
region ABOVE the marker. An agent could average over fewer batches, change the
base of the log, or drop the hardest examples, and the harness would report an
improvement and promote the file.

Moving the metric out of the file being edited is what makes "frozen" true rather
than aspirational. `run.py` additionally rejects any `def evaluate` in the
editable region, because Python lets a later definition shadow an import.

Nothing here may change without invalidating every recorded result. If it must
change, that is an epoch boundary: say so, and re-run the incumbent under the new
definition rather than comparing across it.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def evaluate(
    model: nn.Module,
    device: str,
    make_loader,
    val_bag,
    eval_batches: int,
    eval_batch_size: int,
    seed: int,
) -> float:
    """Held-out cross-entropy in bits per move.

    Every parameter is passed in rather than read from the caller's globals, so
    that an editable `train.py` cannot quietly change what "held out" means by
    rebinding a constant. `run.py` checks the constants themselves are frozen.
    """
    model.eval()
    loader = make_loader(
        val_bag, batch_size=eval_batch_size, num_workers=4, seed=seed, infinite=True
    )
    it = iter(loader)
    total, n = 0.0, 0
    for _ in range(eval_batches):
        tokens, actions = next(it)
        tokens, actions = tokens.to(device), actions.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(tokens)
        total += F.cross_entropy(logits.float(), actions, reduction="sum").item()
        n += actions.numel()
    model.train()
    return total / n / math.log(2)
