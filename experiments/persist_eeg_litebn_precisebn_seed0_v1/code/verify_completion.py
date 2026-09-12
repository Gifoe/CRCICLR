"""Read-only cardinality, pairing, source recovery and state-invariant checks."""
import argparse,csv,json,math
from pathlib import Path
def rows(root,name):
    with (root/name).open(encoding='utf-8') as h:return list(csv.DictReader(h))
def verify(root):
    m=json.loads((root/'PRECISE_BN_METADATA.json').read_text(encoding='utf-8'))
    assert m['status']=='COMPLETE' and m['checkpoints']==20 and m['SEED']==0
    assert all(m[k]=='NO' for k in ['LEARNABLE_WEIGHTS_UPDATED','INNER_VAL_USED_FOR_CALIBRATION','OUTER_USED_FOR_CALIBRATION','CURRENT_HELDOUT_USED_FOR_CALIBRATION','NEW_SEALED_FINAL_DATA_ACCESSED'])
    source=rows(root,'LITEBN_PRECISEBN_SOURCE_MANIFEST.csv');assert len(source)==20 and len({(r['task'],r['fold'],r['seed']) for r in source})==20 and all(r['strict_state_load']=='PASS' and r['seed']=='0' for r in source)
    replay=rows(root,'LITEBN_PRECISEBN_BASELINE_REPLAY.csv');assert len(replay)==411 and all(r['status']=='PASS' and abs(float(r['delta_BA']))<=1e-10 for r in replay)
    assert sum(r['population']=='outer' for r in replay)==151 and sum(r['population']=='heldout' for r in replay)==260
    audit=rows(root,'LITEBN_PRECISEBN_STATE_AUDIT.csv');assert len(audit)==20
    assert all(r['status']=='PASS' and float(r['max_abs_parameter_diff'])==0 and r['parameters_bitwise_identical']=='True' and r['non_BN_buffers_identical']=='True' and r['parameter_hash_before']==r['parameter_hash_after'] and r['non_BN_buffer_hash_before']==r['non_BN_buffer_hash_after'] for r in audit)
    layer=rows(root,'LITEBN_PRECISEBN_LAYER_STATS.csv');assert len(layer)==160 and all(int(r['calibration_count'])>0 and math.isfinite(float(r['variance_shift'])) for r in layer)
    paired=rows(root,'LITEBN_PRECISEBN_PAIRED_SUBJECT_REPLICATES.csv');assert len(paired)==411 and len({(r['task'],r['fold'],r['population'],r['subject_id']) for r in paired})==411
    for r in paired:
        for k in ['BA','macro_F1','accuracy']:
            assert math.isclose(100*(float(r['preciseBN_'+k])-float(r['original_'+k])),float(r['delta_'+k+'_pp']),abs_tol=1e-10)
    assert len(rows(root,'LITEBN_PRECISEBN_OUTER_FOLD_RESULTS.csv'))==20
    assert len(rows(root,'LITEBN_PRECISEBN_HELDOUT_SUBJECT_RESULTS.csv'))==52
    assert len(rows(root,'LITEBN_PRECISEBN_OUTER_TASK_SUMMARY.csv'))==4 and len(rows(root,'LITEBN_PRECISEBN_HELDOUT_TASK_SUMMARY.csv'))==4
    return {'status':'PASS','source_checkpoints':20,'replay_subject_replicates':411,'paired_subject_replicates':411,'BN_layer_records':160,'state_audits':20,'heldout_subject_task_units':52,'decision':m['decision']}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('outputs',type=Path);a=p.parse_args();print(json.dumps(verify(a.outputs),indent=2))
