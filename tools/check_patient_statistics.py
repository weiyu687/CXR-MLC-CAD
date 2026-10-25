"""Check tied-score AUROC and whole-patient paired resampling against sklearn."""
import numpy as np
from sklearn.metrics import roc_auc_score
from tools.patient_statistics import prepare_auc, weighted_auc, paired_cluster_bootstrap


def main():
    rng = np.random.default_rng(2037)
    y = rng.integers(2, size=(31, 10))
    # Deliberate score ties and patients contributing different study counts.
    p = rng.integers(5, size=y.shape) / 4
    patients = np.repeat(np.arange(9), [1, 2, 4, 3, 7, 1, 5, 2, 6])
    prepared = prepare_auc(y, p)
    for _ in range(20):
        counts = np.bincount(rng.integers(9, size=9), minlength=9)
        weights = counts[patients]
        actual = weighted_auc(prepared, weights)
        expected = []
        for c in range(10):
            if len(np.unique(y[weights > 0, c])) < 2:
                expected.append(np.nan)
            else:
                expected.append(roc_auc_score(y[:, c], p[:, c], sample_weight=weights))
        np.testing.assert_allclose(actual, expected, atol=1e-14, equal_nan=True)
    result = paired_cluster_bootstrap(y, {'reference': p, 'identical': p.copy(), 'perfect': y},
                                      patients, repetitions=50)
    assert result['models']['identical']['paired_delta_percentile_95'] == [0., 0.]
    assert result['models']['identical']['delta_from_reference'] == 0
    assert result['models']['perfect']['macro_auc'] == 1
    assert result['models']['perfect']['paired_delta_percentile_95'][0] > 0
    assert result == paired_cluster_bootstrap(y, {'reference': p, 'identical': p.copy(), 'perfect': y},
                                            patients, repetitions=50)
    # A disease absent in the original split is excluded consistently.
    constant = np.column_stack([y[:, :9], np.zeros(len(y))])
    result = paired_cluster_bootstrap(constant, {'a': p, 'b': p}, patients, 50)
    assert len(result['evaluable_class_indices']) == 9
    assert result['models']['a']['per_class_auc'][-1] is None
    print('PASS: tied-score weighted AUROC matches sklearn; unequal patient clusters; paired zero delta; deterministic draws; constant-class policy')


if __name__ == '__main__':
    main()
