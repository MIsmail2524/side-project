import numpy as np

from flysentinel import FlySentinel, MushroomBody


def test_tag_is_sparse_and_locality_sensitive():
    rng = np.random.default_rng(0)
    mb = MushroomBody(20, n_kc=500, active_frac=0.05, seed=1)
    X = rng.normal(size=(200, 20))
    mb.normalizer.update(X)
    mb.calibrate(X)
    a, near, far = X[0], X[0] + 0.01, rng.normal(size=20)
    ta, tn, tf = (set(np.flatnonzero(mb.encode(v[None])[0])) for v in (a, near, far))
    assert 25 <= len(ta) < 60
    assert len(ta & tn) > len(ta & tf)


def test_outliers_score_higher_than_inliers():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(3000, 10))
    det = FlySentinel(n_flies=4, n_kc=1000, contamination=0.02).fit(train)
    inl = det.score_samples(rng.normal(size=(200, 10))).mean()
    out = det.score_samples(rng.normal(5, 1, size=(200, 10))).mean()
    assert out > inl
    assert det.predict(rng.normal(5, 1, size=(50, 10))).mean() > 0.9


def test_streaming_decay_adapts_to_drift():
    rng = np.random.default_rng(0)
    det = FlySentinel(n_flies=2, n_kc=500, decay=1e-3)
    det.partial_fit(rng.normal(size=(2000, 8)))
    new = rng.normal(3, 1, size=(200, 8))
    before = det.score_samples(new).mean()
    for _ in range(5):
        det.partial_fit(rng.normal(3, 1, size=(1000, 8)))
    assert det.score_samples(new).mean() < before


def test_explain_points_at_the_broken_feature():
    rng = np.random.default_rng(0)
    det = FlySentinel(n_flies=4, n_kc=1000).fit(rng.normal(size=(3000, 6)))
    x = np.zeros(6)
    x[4] = 8.0
    assert det.explain(x, top=1)[0][0] == "f4"
