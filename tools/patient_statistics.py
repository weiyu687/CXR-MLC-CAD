"""Paired patient-cluster bootstrap of study-level AUROC.

Patients, not studies, are resampled. All studies of a sampled patient receive
that patient's multiplicity. Tied scores contribute half a concordant pair.
"""
import numpy as np


def prepare_auc(targets, probabilities):
    y = np.asarray(targets)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 2 or y.shape != p.shape or not len(y):
        raise ValueError('Expected nonempty, aligned study-by-class arrays')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError('AUROC requires binary targets and finite scores')
    prepared = []
    for c in range(y.shape[1]):
        order = np.argsort(p[:, c], kind='stable')
        sorted_p = p[order, c]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_p)) + 1]
        prepared.append((order, starts, y[order, c].astype(np.float64)))
    return prepared


def weighted_auc(prepared, weights):
    """Exact weighted AUROC from cached score ordering, including score ties."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Invalid study weights')
    aucs = []
    for order, starts, target in prepared:
        if len(weights) != len(order):
            raise ValueError('Study weights do not align with prepared scores')
        w = weights[order]
        positive = np.add.reduceat(w * target, starts)
        negative = np.add.reduceat(w * (1-target), starts)
        denominator = positive.sum() * negative.sum()
        concordant = np.dot(positive, np.cumsum(negative) - .5 * negative)
        aucs.append(float(concordant / denominator) if denominator else np.nan)
    return np.asarray(aucs)


def paired_cluster_bootstrap(targets, predictions, patients, repetitions=1000, seed=2037):
    """Compare all candidates to the first on the exact same patient draws.

    Replicates missing either label in any originally evaluable class are
    excluded for every candidate together; changing the class set per draw
    would change the estimand. Intervals are descriptive selection estimates,
    not independent confirmation after model/checkpoint selection.
    """
    if len(predictions) < 2 or repetitions < 1:
        raise ValueError('Provide a reference and at least one comparison')
    names = list(predictions)
    y = np.asarray(targets)
    patients = np.asarray(patients)
    if patients.ndim != 1 or len(patients) != len(y):
        raise ValueError('Patient IDs must align with studies')
    unique, inverse = np.unique(patients, return_inverse=True)
    if len(unique) < 2:
        raise ValueError('At least two patient clusters are required')
    prepared = {name: prepare_auc(y, predictions[name]) for name in names}
    points = {name: weighted_auc(value, np.ones(len(y))) for name, value in prepared.items()}
    valid_classes = np.isfinite(points[names[0]])
    if not valid_classes.any():
        raise ValueError('No class contains both labels')
    draws = {name: [] for name in names}
    rng = np.random.default_rng(seed)
    discarded = 0
    for _ in range(repetitions):
        count = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        weights = count[inverse]
        values = {name: weighted_auc(value, weights)[valid_classes] for name, value in prepared.items()}
        if any(not np.isfinite(value).all() for value in values.values()):
            discarded += 1
            continue
        for name in names:
            draws[name].append(float(values[name].mean()))
    if not draws[names[0]]:
        raise ValueError('No bootstrap replicate retained all evaluable classes')
    results = {}
    for name in names:
        value = np.asarray(draws[name])
        delta = value - np.asarray(draws[names[0]])
        results[name] = {
            'macro_auc': float(points[name][valid_classes].mean()),
            'per_class_auc': [float(v) if np.isfinite(v) else None for v in points[name]],
            'macro_auc_percentile_95': np.quantile(value, [.025, .975]).tolist(),
            'delta_from_reference': float(np.mean(points[name][valid_classes] - points[names[0]][valid_classes])),
            'paired_delta_percentile_95': np.quantile(delta, [.025, .975]).tolist(),
        }
    return {'reference': names[0], 'patients': len(unique), 'studies': len(y),
            'seed': seed, 'bootstrap_requested': repetitions, 'bootstrap_accepted': len(draws[names[0]]),
            'bootstrap_discarded_missing_class': discarded,
            'evaluable_class_indices': np.flatnonzero(valid_classes).tolist(),
            'scope': 'Paired patient-cluster percentile bootstrap; checkpoint/model selection estimates, not hidden-test results',
            'models': results}
