"""Key-number-aware distributions of the final NFL home margin and game total, anchored to the betting market.

Why this exists
---------------
NFL margins are lumpy: about 15% of games end by exactly 3 and about 9% by exactly 7, while 1, 2, 5, 9 are rare.
A normal curve (what v2 used) misprices anything that depends on those numbers: moneyline <-> spread
conversions, push probabilities, and the value of a half point. Totals have milder key numbers (37, 41, 44, 47, 51).

The model
---------
    P(outcome = k | mu)  ∝  Normal(k; mu, sigma) * w(k)

* w are key-number weights learned from past games: how much more/less often a value shows up than a smooth curve
  predicts. Margins use w(|k|) (a 3-point win or loss is equally "sticky"); totals use w(k).
* mu is solved so the distribution agrees with the market. With both prices known, mu is chosen so the de-vigged
  price is matched exactly: a -3 at -125 means the market thinks the favorite covers more often than a -3 at +100,
  and mu moves to reflect that.

Everything is fitted from games strictly before the ones being priced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from betting import devig_two_way


class ScoreDist:
    def __init__(self, kind: str = "margin", sigma: float = 13.0, weights=None):
        if kind not in ("margin", "total"):
            raise ValueError("kind must be 'margin' or 'total'")
        self.kind = kind
        self.K = np.arange(-70, 71) if kind == "margin" else np.arange(0, 131)
        self._widx = np.minimum(np.abs(self.K), 45) if kind == "margin" else self.K
        self.sigma = sigma
        self.w = np.ones(self._widx.max() + 1) if weights is None else np.asarray(weights, dtype=float)

    # ------------------------------------------------------------------ fitting
    def fit(self, outcome, line, smooth: float = 30.0) -> "ScoreDist":
        """Learn sigma and key-number weights from completed games (outcome and closing line)."""
        r, s = np.asarray(outcome, dtype=float), np.asarray(line, dtype=float)
        ok = ~(np.isnan(r) | np.isnan(s))
        r, s = r[ok], s[ok]
        self.sigma = float(np.std(r - s, ddof=1))
        dens = norm.pdf(self.K[None, :], loc=s[:, None], scale=self.sigma)
        dens /= dens.sum(axis=1, keepdims=True)
        n = self._widx.max() + 1
        expected = np.bincount(self._widx, weights=dens.sum(axis=0), minlength=n)
        ri = np.clip(r.astype(int), self.K[0], self.K[-1]) - self.K[0]
        observed = np.bincount(self._widx[ri], minlength=n)
        # shrink toward 1 so values with little data don't get extreme weights
        self.w = (observed + smooth) / (expected + smooth)
        return self

    # --------------------------------------------------------------- core pmf
    def pmf(self, mu) -> np.ndarray:
        """Rows = games, columns = self.K. Probability of each exact outcome."""
        mu = np.atleast_1d(np.asarray(mu, dtype=float))
        p = norm.pdf(self.K[None, :], loc=mu[:, None], scale=self.sigma) * self.w[self._widx][None, :]
        return p / p.sum(axis=1, keepdims=True)

    def _over_share(self, mu, line):
        """P(outcome > line | not a push)."""
        p = self.pmf(mu)
        line = np.asarray(line, dtype=float)[:, None]
        hi, lo = (p * (self.K > line)).sum(1), (p * (self.K < line)).sum(1)
        return hi / (hi + lo)

    # -------------------------------------------------------- market anchoring
    def solve_mu(self, line, over_odds=None, under_odds=None, iters: int = 40) -> np.ndarray:
        """Centre of the distribution implied by the market line (and its juice, when both prices are known).

        For margins pass (spread_line, home_spread_odds, away_spread_odds); for totals (total_line, over, under)."""
        s = np.asarray(line, dtype=float)
        target = np.full(len(s), 0.5)
        if over_odds is not None and under_odds is not None:
            ho, ao = np.asarray(over_odds, dtype=float), np.asarray(under_odds, dtype=float)
            has = ~(np.isnan(ho) | np.isnan(ao))
            if has.any():
                target[has], _ = devig_two_way(ho[has], ao[has])
        lo, hi = s - 10.0, s + 10.0
        for _ in range(iters):                      # bisection: the over share rises with mu
            mid = (lo + hi) / 2
            up = self._over_share(mid, s) < target
            lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
        return (lo + hi) / 2

    # --------------------------------------------------------- probabilities
    def prob_over(self, mu, line):
        """(P(outcome > line), P(outcome == line)) for arbitrary lines, e.g. a book's -2.5 when consensus is -3."""
        p = self.pmf(mu)
        line = np.asarray(line, dtype=float)[:, None]
        return (p * (self.K > line)).sum(1), (p * (self.K == line)).sum(1)

    def fair_line(self, mu) -> np.ndarray:
        """The line where each side wins half the time (interpolated median), comparable to a sportsbook number.

        With key numbers this differs from mu: a mean margin of 4.1 can still mean 'win by 4+' is under 50%."""
        p = self.pmf(mu)
        c = p.cumsum(axis=1)
        i = (c >= 0.5).argmax(axis=1)
        rows = np.arange(len(p))
        below = np.where(i > 0, c[rows, i - 1], 0.0)
        return self.K[i] - 0.5 + (0.5 - below) / p[rows, i]

    def win_probs(self, mu) -> dict:
        """Moneyline view of a margin distribution. Ties refund moneyline bets at most books."""
        p = self.pmf(mu)
        win, tie = (p * (self.K > 0)).sum(1), p[:, self.K == 0].ravel()
        return {"p_home_win": win, "p_tie": tie, "p_home_win_ex_tie": win / (1 - tie)}

    def to_dict(self) -> dict:
        return {"kind": self.kind, "sigma": self.sigma, "weights": self.w.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "ScoreDist":
        return cls(d["kind"], d["sigma"], d["weights"])
