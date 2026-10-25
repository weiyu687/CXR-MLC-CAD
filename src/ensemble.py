"""Fixed global ensemble weights shared by selection, prediction and timing."""
import numpy as np


def normalized_weights(weights, count):
    if count<1:
        raise ValueError('At least one model is required')
    if weights is None:
        return None
    values=np.asarray(weights,dtype=np.float64)
    if values.shape!=(count,) or not np.isfinite(values).all() or (values<=0).any():
        raise ValueError('Provide one finite, strictly positive weight per loaded model')
    total=values.sum()
    if not np.isfinite(total) or total<=0:
        raise ValueError('Invalid sum of ensemble weights')
    return values.tolist() if abs(total-1.)<=1e-15 else (values/total).tolist()


def combine_probabilities(probabilities, weights=None):
    values=np.asarray(probabilities)
    if values.ndim<2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected finite, aligned member probabilities')
    weights=normalized_weights(weights,len(values))
    # Preserve the original dtype and arithmetic when no weighting is requested,
    # including frozen legacy bundles whose reports have no weights field.
    if weights is None:
        return np.mean(values,axis=0)
    return np.tensordot(np.asarray(weights),values,axes=(0,0))
