"""Fixed global or class-specific weights shared by all inference paths."""
import json
from pathlib import Path
import numpy as np


def normalized_weights(weights, count):
    if count<1:
        raise ValueError('At least one model is required')
    if weights is None:
        return None
    values=np.asarray(weights,dtype=np.float64)
    if values.shape not in [(count,),(count,10)] or not np.isfinite(values).all():
        raise ValueError('Provide finite weights shaped [models] or [models, 10 classes]')
    if (values<0).any() or (values.ndim==1 and (values==0).any()):
        raise ValueError('Global weights must be positive; class-specific weights must be nonnegative')
    total=values.sum(axis=0)
    if not np.isfinite(total).all() or (total<=0).any():
        raise ValueError('Invalid sum of ensemble weights')
    return (values/np.where(np.abs(total-1.)<=1e-15,1.,total)).tolist()


def read_weight_spec(path, checkpoint_hashes):
    payload=json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('schema_version')!=1 or payload.get('checkpoint_sha256')!=checkpoint_hashes:
        raise ValueError('Ensemble weight specification does not match checkpoint contents/order')
    if payload.get('weights') is None:
        raise ValueError('Weight specification is empty')
    weights=normalized_weights(payload['weights'],len(checkpoint_hashes))
    if np.asarray(weights).ndim==2 and payload.get('class_ids')!=list(range(1,11)):
        raise ValueError('Class-specific weights must declare class order 1 through 10')
    return {**payload,'weights':weights}


def combine_probabilities(probabilities, weights=None):
    values=np.asarray(probabilities)
    if values.ndim<2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected finite, aligned member probabilities')
    weights=normalized_weights(weights,len(values))
    # Preserve the original dtype and arithmetic when no weighting is requested,
    # including frozen legacy bundles whose reports have no weights field.
    if weights is None:
        return np.mean(values,axis=0)
    weights=np.asarray(weights)
    if weights.ndim==1:
        return np.tensordot(weights,values,axes=(0,0))
    if values.shape[-1]!=weights.shape[-1]:
        raise ValueError('Prediction and class-weight dimensions differ')
    shape=(len(values),)+(1,)*(values.ndim-2)+(weights.shape[-1],)
    return np.sum(values*weights.reshape(shape),axis=0)
