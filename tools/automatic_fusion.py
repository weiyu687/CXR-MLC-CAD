"""Apply the fixed patient-CV fusion protocol to each TTA mode's best triple."""
import hashlib
from pathlib import Path
import numpy as np
from src.ensemble import normalized_weights
from tools.fit_classwise_fusion import fit_exported_fusion


def prune_unused_members(checkpoints,spec):
    """Remove only members whose fitted weight is exactly zero for every class."""
    checkpoints=[str(Path(path).resolve()) for path in checkpoints]
    hashes=[hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in checkpoints]
    if spec['checkpoint_sha256']!=hashes:raise ValueError('Fitted member/model binding changed')
    if spec.get('class_ids')!=list(range(1,11)):raise ValueError('Fitted class order differs')
    weights=np.asarray(normalized_weights(spec['weights'],len(checkpoints)))
    if weights.ndim!=2:raise ValueError('Classwise matrix required')
    active=np.flatnonzero(np.any(weights>0,axis=1)).tolist()
    protocol=dict(spec['selection_protocol'])
    if len(active)!=len(checkpoints):
        protocol.update(unpruned_checkpoint_sha256=hashes,retained_member_indices=active)
    return ([checkpoints[i] for i in active],
            {**spec,'checkpoint_sha256':[hashes[i] for i in active],
             'weights':weights[active].tolist(),'selection_protocol':protocol},active)


def fit_best_global_triples(candidates,exports,root,out):
    """At most two prospective fits, using fresh validation exports only."""
    out=Path(out)
    if out.exists():raise FileExistsError('Existing classwise selection preserved: '+str(out))
    fitted=[]
    for tta in [False,True]:
        eligible=[candidate for candidate in candidates
                  if len(candidate['checkpoints'])==3 and candidate['tta']==tta
                  and not candidate.get('ensemble_weight_spec')]
        if not eligible:continue
        base=min(eligible,key=lambda c:(-c['metrics']['macro_auc'],-c['metrics']['macro_f1_tuned'],c['model_bytes']))
        hashes=[hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in base['checkpoints']]
        files=[]
        for digest in hashes:
            if (digest,tta) not in exports:raise ValueError('Best global triple lacks a fresh member export')
            files.append(exports[digest,tta])
        analysis=out/('tta' if tta else 'plain')
        result,spec=fit_exported_fusion(base['checkpoints'],files,root,base['aggregation'],analysis,tta)
        # Every fitted global option is already present in the ordinary grid.
        # A CV preference for zero classwise strength adds no new candidate.
        if result['selected_shrinkage']==0:continue
        checkpoints,spec,active=prune_unused_members(base['checkpoints'],spec)
        fitted.append({'checkpoints':checkpoints,'checkpoint_sha256':spec['checkpoint_sha256'],
            'model_bytes':sum(Path(path).stat().st_size for path in checkpoints),
            'aggregation':base['aggregation'],'tta':tta,'metrics':result['refit_validation'],
            'ensemble_weights':spec['weights'],'ensemble_weight_spec':spec,
            'classwise_analysis':str((analysis/'analysis.json').resolve()),
            'global_source_validation_macro_auc':base['metrics']['macro_auc'],
            'retained_member_indices':active,
            'selection_scope':'One fixed five-repeat patient-CV fit of the best global triple in this TTA mode; all source and refit selection uses valid only'})
    return fitted
