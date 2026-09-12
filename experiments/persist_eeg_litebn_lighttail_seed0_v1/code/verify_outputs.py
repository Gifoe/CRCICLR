"""Independent read-only verification of published LightTail audit cardinalities."""
import argparse,csv,json,math
from pathlib import Path
def rows(p):
    with p.open(encoding='utf-8') as h:return list(csv.DictReader(h))
def verify(out):
    meta=json.loads((out/'LIGHTTAIL_METADATA.json').read_text(encoding='utf-8'))
    assert meta['TASK']=='OpenBMI_MI' and meta['SEED']==0 and meta['FOLDS']==5
    assert meta['TAIL_WEIGHT']==.1 and meta['EXACT_HISTORICAL_MANIFEST']==meta['TRAINING_EXPOSURE_MATCHED']=='YES'
    manifest=rows(out/'EXACT_MANIFEST_AUDIT.csv');assert len(manifest)==6300
    assert len({(r['fold'],r['epoch'],r['episode']) for r in manifest})==6300
    for r in manifest:
        assert r['historical_episodes_per_epoch']==r['candidate_episodes_per_epoch']=='21'
        assert r['historical_trials']==r['candidate_trials']=='128'
        assert all(r[k]=='True' for k in ['subject_composition_equal','session_composition_equal','trial_indices_equal','sample_order_equal','episode_order_equal'])
        assert r['historical_manifest_sha256']==r['reconstructed_LF_sha256']
    eq=rows(out/'TAIL0_EQUIVALENCE_AUDIT.csv');assert len(eq)==10 and all(r['status']=='PASS' and float(r['max_abs_parameter_update_diff'])<=1e-6 and float(r['scalar_abs_diff'])<=1e-6 for r in eq)
    trajectory=rows(out/'LIGHTTAIL_TRAINING_TRAJECTORY.csv');assert len(trajectory)==300 and len({(r['fold'],r['epoch']) for r in trajectory})==300
    for r in trajectory:
        assert r['optimizer_update_attempts']=='21' and r['trial_exposure']=='2688'
        assert int(r['successful_optimizer_updates'])+int(r['AMP_skipped_updates'])==21
        assert math.isclose(float(r['total_loss']),.9*float(r['mean_subject_risk'])+.1*float(r['tail_risk']),abs_tol=1e-6)
    pairs=rows(out/'LIGHTTAIL_PAIRED_REPLICATES.csv');assert len(pairs)==110
    assert len({(r['fold'],r['population'],r['subject_id']) for r in pairs})==110
    assert sum(r['population']=='outer' for r in pairs)==40
    assert len(rows(out/'LIGHTTAIL_OUTER_FOLD_RESULTS.csv'))==5 and len(rows(out/'LIGHTTAIL_HELDOUT_SUBJECT_RESULTS.csv'))==14
    assert len(rows(out/'LIGHTTAIL_LOWER_TAIL_ANALYSIS.csv'))==2
    return {'status':'PASS','manifest_episodes':6300,'tail0_tests':10,'training_epochs':300,'paired_evaluation_rows':110,'optimizer_attempts':6300,'decision':meta['decision']}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('outputs',type=Path);a=p.parse_args();print(json.dumps(verify(a.outputs),indent=2))
