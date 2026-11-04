# FlySentinel

**Anomaly detection modeled on the fruit fly's mushroom body.**

A fruit fly can tell at once that a smell is new, using a circuit of about 2,000 neurons. The
full wiring diagram of that circuit is now public (FlyWire whole-brain connectome, 2024).
FlySentinel turns that circuit into a streaming anomaly detector for security logs, payments and
patient vitals.

It needs only NumPy, keeps a fixed amount of memory however long the stream runs, adapts to drift,
and explains each alert by naming the features behind it.

```
row,score,alert,why
2,0.75987,1,failed_logins:66%; bytes_in:10%
```

## The circuit

| Fly anatomy (connectome) | In FlySentinel |
|---|---|
| ~50 olfactory glomeruli, gain-normalized | each input feature, z-scored with running statistics |
| ~2,000 Kenyon cells (KCs) per hemisphere | `n_kc=2000` expansion layer |
| each KC wired to ~6-7 random projection neurons | `claws=7` sparse random signed wiring |
| APL neuron: global inhibition, ~5% of KCs fire | top-k winner-take-all, `active_frac=0.05` |
| KC spiking thresholds | per-KC thresholds; extreme inputs recruit extra KCs |
| KC→MBON synapses changed by dopamine | memory of KC tags; familiarity = Jaccard overlap |
| many flies | ensemble of independent random circuits |

The KC tag is a sparse, high-dimensional, locality-sensitive code (FlyHash). Similar inputs share
KCs and different inputs do not. Novelty is how little an input's tag overlaps with the closest
stored memories.

## Use

```bash
pip install -e .
flysentinel train normal_traffic.csv model.pkl --contamination 0.01
flysentinel score model.pkl new_traffic.csv --explain 3
```

```python
from flysentinel import FlySentinel

det = FlySentinel(contamination=0.01).fit(X_normal)
scores = det.score_samples(X_new)      # higher = more anomalous
alerts = det.predict(X_new)            # 1 = anomaly
det.explain(X_new[0], feature_names)   # [("failed_logins", 0.66), ...]

stream = FlySentinel(decay=1e-3)       # favor recent data on drifting streams
for batch in batches:
    stream.partial_fit(batch)
```

Only load model files you created yourself: they are Python pickles.

## Benchmark

`python benchmarks/run.py`. Training uses normal data only; the test set mixes in attacks. The
data is synthetic and built to be hard: attacks include correlation-breaking exfiltration,
account takeover between customer segments, and vitals that leave the normal manifold.

| Dataset | Model | ROC-AUC | PR-AUC | Fit (s) | Score (samples/s) |
|---|---|---|---|---|---|
| network intrusion | **FlySentinel** | 0.846 | 0.694 | 6.52 | 957 |
| network intrusion | IsolationForest | 0.682 | 0.333 | 0.32 | 65,155 |
| network intrusion | LOF (novelty) | 0.903 | 0.784 | 0.15 | 57,217 |
| card fraud | **FlySentinel** | 0.947 | 0.799 | 7.88 | 761 |
| card fraud | IsolationForest | 0.755 | 0.108 | 0.43 | 64,042 |
| card fraud | LOF (novelty) | 0.973 | 0.883 | 0.24 | 40,387 |
| patient vitals | **FlySentinel** | 0.985 | 0.871 | 7.26 | 684 |
| patient vitals | IsolationForest | 0.788 | 0.213 | 0.38 | 57,381 |
| patient vitals | LOF (novelty) | 0.998 | 0.989 | 0.15 | 43,975 |

FlySentinel beats IsolationForest on all three datasets but is behind LOF, and it is slower (pure
NumPy, dense matrices). Its advantages over LOF are a bounded memory (a reservoir of KC tags, not
the whole dataset), online updates with drift handling, and per-alert explanations.

## Roadmap

- Sparse-matrix kernels for much faster scoring
- Wiring sampled from real FlyWire PN→KC connectivity instead of uniform random
- Benchmarks on public datasets (CIC-IDS, credit card fraud, MIMIC vitals)

## References

- Dorkenwald et al., "Neuronal wiring diagram of an adult brain", *Nature* 2024 (FlyWire)
- Dasgupta, Stevens, Navlakha, "A neural algorithm for a fundamental computing problem", *Science* 2017
- Dasgupta et al., "A neural data structure for novelty detection", *PNAS* 2018
- Shiu et al., "A Drosophila computational brain model reveals sensorimotor processing", *Nature* 2024

## License

MIT © Muhammad Ismail
