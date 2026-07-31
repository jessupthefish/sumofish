"""HL-Gauss: turn a scalar regression target into a soft classification target.

The state-value task is "given this position, what is the probability the side
to move wins", a number in [0, 1]. The obvious approach is one output neuron
and MSE. That is measurably not the best approach.

Farebrother et al., *Stop Regressing* (arXiv:2403.03950), find the ordering
**HL-Gauss > C51 > MSE**, and separately that plain two-hot binning is *worse*
than MSE. So the binning scheme matters and picking the wrong one is a
regression, not a wash.

The idea: instead of predicting a number, predict a distribution over K bins
spanning [0, 1], and train with cross-entropy. The target for a true value y is
not a one-hot spike on y's bin, and not the two-hot linear split between the
two nearest bins either. It is the mass a Gaussian centred at y with width
sigma puts into each bin, computed from the normal CDF at the bin edges.

Why that helps, in the terms that matter here: cross-entropy on a smoothed
target gives a better-conditioned gradient than MSE, and it stops the model
having to commit to a razor-thin distinction between 0.61 and 0.62 win
probability, which is noise in the labels anyway. It also hands you a full
distribution at inference, so the engine can ask "how sure am I" rather than
just "what number" -- and that uncertainty is exactly what SumoFish's
personality is meant to be built on.

Parameters, both from the paper and both easy to get wrong:

  sigma = 0.75 * bin_width    The paper's text glosses this as "≈ 0.05" for
                              K=128, which is arithmetically wrong (0.75/128 =
                              0.0059). Trust the formula, not the gloss.

  K = 32 or 64                Their bin ablation is flat above 32: puzzle
                              accuracy 83.0 / 83.0 / 84.4 / 83.8 / 83.7 for
                              K = 16/32/64/128/256. K=128 buys nothing and
                              costs compute.
"""

from __future__ import annotations

import math

import torch


class HLGauss:
    """Scalar in [vmin, vmax] <-> soft distribution over `bins` categories."""

    def __init__(
        self,
        bins: int = 64,
        vmin: float = 0.0,
        vmax: float = 1.0,
        sigma_ratio: float = 0.75,
        device: str | torch.device = "cpu",
    ) -> None:
        if bins < 2:
            raise ValueError("need at least 2 bins")
        self.bins = bins
        self.vmin = vmin
        self.vmax = vmax
        self.bin_width = (vmax - vmin) / bins
        self.sigma = sigma_ratio * self.bin_width

        # K+1 edges bounding K bins.
        self.edges = torch.linspace(vmin, vmax, bins + 1, device=device)
        # Bin centres, used to read a scalar back out of a predicted histogram.
        self.centers = (self.edges[:-1] + self.edges[1:]) / 2

    def to(self, device: str | torch.device) -> "HLGauss":
        self.edges = self.edges.to(device)
        self.centers = self.centers.to(device)
        return self

    def targets(self, values: torch.Tensor) -> torch.Tensor:
        """[B] scalars -> [B, bins] probabilities summing to 1.

        Mass in bin i is Phi((edge_{i+1} - y)/sigma) - Phi((edge_i - y)/sigma).
        Because the support is finite, some Gaussian mass falls outside
        [vmin, vmax]; renormalising by the total inside is what keeps the
        target a proper distribution. Values at the extremes (0.0 and 1.0 are
        both common in this dataset) therefore get a half-Gaussian pushed up
        against the boundary, which is correct.
        """
        v = values.reshape(-1, 1).to(self.edges.dtype)
        # 0.5 * (1 + erf(z / sqrt(2))) is the standard normal CDF.
        cdf = 0.5 * (1.0 + torch.erf((self.edges - v) / (self.sigma * math.sqrt(2.0))))
        mass = cdf[:, 1:] - cdf[:, :-1]
        return mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def expectation(self, logits: torch.Tensor) -> torch.Tensor:
        """[B, bins] logits -> [B] scalar values, the distribution's mean."""
        return (logits.softmax(dim=-1) * self.centers).sum(dim=-1)

    def confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """[B, bins] logits -> [B] standard deviation of the predicted value.

        Not needed for play, but this is the number that makes a searchless
        engine able to say something a search engine cannot: how uncertain it
        is about its own evaluation.
        """
        p = logits.softmax(dim=-1)
        mean = (p * self.centers).sum(dim=-1, keepdim=True)
        var = (p * (self.centers - mean) ** 2).sum(dim=-1)
        return var.sqrt()


def loss(logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against a soft target. Mean over the batch."""
    return -(target_dist * logits.log_softmax(dim=-1)).sum(dim=-1).mean()
