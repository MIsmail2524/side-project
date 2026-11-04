"""FlySentinel: a streaming anomaly detector built from an ensemble of fly mushroom bodies."""

import numpy as np

from .circuit import MushroomBody


class FlySentinel:
    """Novelty detection the way a fruit fly spots a new smell.

    Each "fly" hashes input into a sparse Kenyon-cell (KC) tag. The fly keeps a memory of KC
    tags it has seen, which models KC->MBON synapses tuned by dopamine. A new input is familiar
    when its tag overlaps strongly (Jaccard) with stored tags. The novelty score is
    1 - (mean overlap with the `m` closest memories), averaged across flies. Higher means more
    anomalous.

    Parameters
    ----------
    n_flies : independent mushroom bodies; more flies give lower variance
    n_kc, claws, active_frac : circuit anatomy, see MushroomBody
    memory : stored tags per fly (reservoir sample of everything seen)
    m : how many closest memories to compare against
    decay : 0 keeps a uniform reservoir; >0 favors recent samples so the model follows drift
    contamination : expected anomaly rate, used to set the `predict` threshold
    """

    def __init__(self, n_flies=4, n_kc=2000, claws=7, active_frac=0.05, memory=4096, m=5,
                 decay=0.0, contamination=0.01, seed=0):
        self.n_flies = n_flies
        self.n_kc = n_kc
        self.claws = claws
        self.active_frac = active_frac
        self.memory = memory
        self.m = m
        self.decay = decay
        self.contamination = contamination
        self.seed = seed
        self.flies = None

    def _build(self, n_features):
        self.rng_ = np.random.default_rng(self.seed)
        self.flies = [MushroomBody(n_features, self.n_kc, self.claws, self.active_frac,
                                   seed=int(self.rng_.integers(2**31))) for _ in range(self.n_flies)]
        self.mem_ = [np.zeros((0, self.n_kc), dtype=np.float32) for _ in self.flies]
        self.seen_ = 0
        self.n_features_ = n_features

    def _store(self, f, codes):
        """Reservoir sampling, with an optional recency bias (decay) for drifting streams."""
        mem = self.mem_[f]
        room = self.memory - len(mem)
        if room > 0:
            mem = np.vstack([mem, codes[:room]])
            codes = codes[room:]
        if len(codes):
            seen = self.seen_ + np.arange(1, len(codes) + 1)
            p = np.maximum(self.memory / seen, self.decay)
            keep = self.rng_.random(len(codes)) < p
            slots = self.rng_.integers(self.memory, size=int(keep.sum()))
            mem[slots] = codes[keep]
        self.mem_[f] = mem

    def partial_fit(self, X, update_stats=True):
        X = np.asarray(X, dtype=float)
        if self.flies is None:
            self._build(X.shape[1])
        for f, fly in enumerate(self.flies):
            if update_stats:
                fly.normalizer.update(X)
            if fly.theta is None:
                fly.calibrate(X)
            self._store(f, fly.encode(X))
        self.seen_ += len(X)
        return self

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self._build(X.shape[1])
        for fly in self.flies:  # learn normalization and thresholds first so early tags are not skewed
            fly.normalizer.update(X)
            fly.calibrate(X)
        # Hold out a slice the flies never memorize, so the alert threshold reflects unseen normal data.
        idx = self.rng_.permutation(len(X))
        n_cal = min(len(X) // 5, 2000)
        self.partial_fit(X[idx[n_cal:]], update_stats=False)
        self.threshold_ = np.quantile(self.score_samples(X[idx[:n_cal]]), 1 - self.contamination)
        return self

    def _overlaps(self, f, X):
        """Jaccard overlap between each input's KC tag and every stored tag."""
        act = self.flies[f].encode(X)
        mem = self.mem_[f]
        inter = act @ mem.T
        union = act.sum(1)[:, None] + mem.sum(1)[None, :] - inter
        return act, inter / np.maximum(union, 1.0)

    def score_samples(self, X, batch=2048):
        X = np.asarray(X, dtype=float)
        scores = np.zeros(len(X))
        m = self.m
        for i in range(0, len(X), batch):
            xb = X[i:i + batch]
            for f in range(self.n_flies):
                ov = self._overlaps(f, xb)[1]
                top = -np.partition(-ov, min(m, ov.shape[1]) - 1, axis=1)[:, :m]
                scores[i:i + batch] += 1.0 - top.mean(axis=1)
        return scores / self.n_flies

    def predict(self, X):
        """1 for anomaly, 0 for normal."""
        return (self.score_samples(X) > self.threshold_).astype(int)

    def explain(self, x, feature_names=None, top=5):
        """Rank the features that fed the Kenyon cells missing from the closest memories."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        blame = np.zeros(self.n_features_)
        for f, fly in enumerate(self.flies):
            act, ov = self._overlaps(f, x)
            tag = np.flatnonzero(act[0])
            near = self.mem_[f][np.argsort(-ov[0])[: self.m]]
            novelty = 1.0 - near[:, tag].mean(axis=0)  # how often each active KC is unfamiliar
            z = np.abs(fly.normalizer.transform(x)[0])
            for kc, nov in zip(tag, novelty):
                cols = fly.inputs[kc]
                blame[cols] += nov * z[cols]
        names = feature_names or [f"f{i}" for i in range(self.n_features_)]
        order = np.argsort(-blame)[:top]
        total = blame.sum() or 1.0
        return [(names[i], float(blame[i] / total)) for i in order]
