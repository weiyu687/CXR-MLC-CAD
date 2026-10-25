"""Validate and re-evaluate explicitly frozen ensemble configurations."""
import hashlib
import json
from pathlib import Path
import numpy as np
from src.ensemble import read_weight_spec,combine_probabilities


def read_presets(registry, root):
    payload=json.loads(Path(registry).read_text(encoding='utf-8'))
    if payload.get('schema_version')!=1:raise ValueError('Unsupported ensemble registry')
    valid_hash=hashlib.sha256((Path(root)/'valid.csv').read_bytes()).hexdigest()
    presets=[]
    for entry in payload.get('ensemble_presets',[]):
        if type(entry.get('tta')) is not bool or entry.get('batch_size')!=1:
            raise ValueError('Presets require an explicit TTA flag and serial batch size 1')
        checkpoints=[str(Path(path).resolve()) for path in entry['checkpoints']]
        hashes=[hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in checkpoints]
        if sum(Path(path).stat().st_size for path in checkpoints)>500_000_000:
            raise ValueError('Preset exceeds the model-file size budget')
        if hashes!=entry['checkpoint_sha256'] or len(set(hashes))!=len(hashes):
            raise ValueError('Preset checkpoint contents/order differ or contain duplicate files')
        if entry['validation_csv_sha256']!=valid_hash or entry['aggregation'] not in ['frontal','weighted','mean','max']:
            raise ValueError('Preset validation dataset or aggregation differs')
        weights_path=Path(entry['weights_json'])
        if hashlib.sha256(weights_path.read_bytes()).hexdigest()!=entry['weights_json_sha256']:
            raise ValueError('Preset weight specification changed')
        spec=read_weight_spec(weights_path,hashes)
        score=entry['expected_validation_macro_auc']
        if not np.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid preset score')
        presets.append({**entry,'checkpoints':checkpoints,'ensemble_weights':spec['weights'],
                        'ensemble_weight_spec':spec})
    return presets


def evaluate_presets(presets, exports, root, max_members=0):
    """Use fresh validation exports; never trust an earlier reported score alone."""
    from src.data import frame,aggregate
    from src.metrics import evaluate
    df=frame(root,'valid')
    valid_hash=hashlib.sha256((Path(root)/'valid.csv').read_bytes()).hexdigest()
    candidates=[]
    for preset in presets:
        if max_members and len(preset['checkpoints'])>max_members:
            raise ValueError('Explicit ensemble preset exceeds the configured member budget')
        members=[];reference_ids=reference_y=None;settings=None
        for digest in preset['checkpoint_sha256']:
            key=(digest,preset['tta'])
            if key not in exports:raise ValueError('Preset member missing from fresh validation exports')
            with np.load(exports[key]) as saved:
                metadata=json.loads(str(saved['image_prediction_metadata']))
                if metadata['checkpoint_sha256']!=digest or metadata['validation_csv_sha256']!=valid_hash:
                    raise ValueError('Preset prediction/model/data hashes differ')
                if metadata['tta']!=preset['tta'] or metadata['batch_size']!=preset.get('batch_size',1) or not metadata['deterministic_inference']:
                    raise ValueError('Preset inference settings differ')
                np.testing.assert_array_equal(saved['valid_image_ids'],df.Dicom_id.to_numpy(dtype=str))
                current=(metadata['batch_size'],metadata['tta'],json.dumps(metadata['runtime'],sort_keys=True))
                if settings is not None and current!=settings:raise ValueError('Preset member runtimes differ')
                settings=current
                ids,p,y=aggregate(df,saved['valid_images'],preset['aggregation'])
            if reference_ids is not None:
                np.testing.assert_array_equal(ids,reference_ids);np.testing.assert_array_equal(y,reference_y)
            reference_ids,reference_y=ids,y
            members.append(p)
        metrics=evaluate(reference_y,combine_probabilities(members,preset['ensemble_weights']))
        if abs(metrics['macro_auc']-preset['expected_validation_macro_auc'])>1e-8:
            raise ValueError('Fresh preset validation differs from its frozen selection evidence')
        candidates.append({'checkpoints':preset['checkpoints'],'checkpoint_sha256':preset['checkpoint_sha256'],
            'model_bytes':sum(Path(path).stat().st_size for path in preset['checkpoints']),
            'aggregation':preset['aggregation'],'tta':preset['tta'],'ensemble_weights':preset['ensemble_weights'],
            'ensemble_weight_spec':preset['ensemble_weight_spec'],'metrics':metrics,
            'selection_scope':'Explicit fixed ensemble, freshly re-evaluated; patient-CV evidence is embedded in its weight specification'})
    return candidates


def append_weight_arguments(command,candidate,work):
    weights=candidate.get('ensemble_weights')
    if weights is None:return
    if np.asarray(weights).ndim==1:
        command.extend(['--ensemble-weights',*map(str,weights)])
        return
    hashes=candidate.get('checkpoint_sha256') or [hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in candidate['checkpoints']]
    payload={**(candidate.get('ensemble_weight_spec') or {}),'schema_version':1,
             'checkpoint_sha256':hashes,'class_ids':list(range(1,11)),'weights':weights}
    content=json.dumps(payload,indent=2)
    path=Path(work)/('class_weights_'+hashlib.sha256(content.encode()).hexdigest()[:16]+'.json')
    if path.exists() and path.read_text(encoding='utf-8')!=content:raise ValueError('Conflicting weight artifact')
    if not path.exists():path.write_text(content,encoding='utf-8')
    command.extend(['--ensemble-weights-json',str(path)])
