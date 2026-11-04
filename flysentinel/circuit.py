"""Mushroom body circuit, the fly's odor novelty detector, rebuilt as code.

Anatomy follows the Drosophila connectome (FlyWire / hemibrain):
  projection neurons (PNs)  -> one input feature each, gain-normalized like glomeruli
  Kenyon cells (KCs)        -> ~2,000 per hemisphere, each wired to ~6-7 random PNs
  APL neuron                -> one giant inhibitory feedback neuron that keeps ~5% of KCs active
  MBON output + dopamine    -> KC->MBON synapses depress each time a pattern is seen,
                               so MBON drive measures how *unfamiliar* the input is
"""

import numpy as np


class InputNormalizer:
    """Running mean/variance per feature, like glomerular divisive normalization."""

    def __init__(self, n_features):
        self.n = 0
        self.mean = np.zeros(n_features)
        self.m2 = np.zeros(n_features)

    def update(self, X):
        # Batched Welford (Chan et al.) update; stays stable on long streams.
        X = np.asarray(X, dtype=float)
        nb = len(X)
        if nb == 0:
            return
        mb = X.mean(axis=0)
        d = mb - self.mean
        tot = self.n + nb
        self.mean = self.mean + d * nb / tot
        self.m2 = self.m2 + ((X - mb) ** 2).sum(axis=0) + d**2 * self.n * nb / tot
        self.n = tot

    def transform(self, X):
        std = np.sqrt(self.m2 / max(self.n - 1, 1))
        std[std == 0] = 1.0
        return (X - self.mean) / std


class MushroomBody:
    """Sparse random expansion plus winner-take-all, also known as FlyHash."""

    def __init__(self, n_features, n_kc=2000, claws=7, active_frac=0.05, seed=None):
        rng = np.random.default_rng(seed)
        self.n_kc = n_kc
        self.k = max(1, int(round(active_frac * n_kc)))
        claws = min(claws, n_features)
        # Each KC samples `claws` distinct PNs, matching the binary random wiring seen in the connectome.
        self.inputs = np.stack([rng.choice(n_features, claws, replace=False) for _ in range(n_kc)])
        # Signed weights let one feature count against another, as in two-sided PN tuning.
        self.weights = rng.choice([-1.0, 1.0], size=self.inputs.shape)
        self.theta = None  # per-KC firing threshold, set by calibrate()
        # Dense PN->KC matrix (mostly zeros) so the projection is a single matmul.
        self.W = np.zeros((n_features, n_kc))
        np.add.at(self.W, (self.inputs, np.arange(n_kc)[:, None]), self.weights)
        self.normalizer = InputNormalizer(n_features)

    def kc_drive(self, X):
        Z = self.normalizer.transform(np.asarray(X, dtype=float))
        return Z @ self.W

    def calibrate(self, X):
        """Set each KC's spiking threshold so it fires for its strongest `active_frac` of normal inputs."""
        frac = self.k / self.n_kc
        self.theta = np.quantile(self.kc_drive(X), 1 - frac / 4, axis=0)

    def encode(self, X):
        """Binary KC activity: the APL-limited top-k winners, plus any KC driven past its own threshold.

        Top-k alone is scale-invariant (x and 5x give the same tag). Per-KC thresholds let an
        unusually strong input recruit extra KCs, so extreme inputs look unfamiliar.
        """
        drive = self.kc_drive(X)
        top = np.argpartition(-drive, self.k - 1, axis=1)[:, : self.k]
        act = np.zeros(drive.shape, dtype=np.float32)
        np.put_along_axis(act, top, 1.0, axis=1)
        if self.theta is not None:
            act[drive > self.theta] = 1.0
        return act
