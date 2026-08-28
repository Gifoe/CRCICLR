from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

import common as c


def main():
    required=['README.md','PROTOCOL.md','REPOSITORY_AUDIT.md','DATA_AUDIT.md','FM_AUDIT.md','FM_INPUT_AUDIT.md','FM_TASK_COMPETENCE.md','FM_TRAINING_LEDGER.md','FM_ITERATION_LEDGER.md','D_VS_I_FM_REPORT.md','SCAA_FM_REPORT.md','SCST_FM_REPORT.md','CONFOUND_AUDIT.md','CLAIM_AUDIT.md','REPRODUCIBILITY.md','FM_RESCUE_FINAL_REPORT.md','FM_RESCUE_FINAL_REPORT.json','protocol/DATA_ACCESS_LOCK.json','protocol/FM_INPUT_PROTOCOL_LOCK.json','protocol/FM_RESCUE_STAGE0_PROTOCOL_LOCK.json','results/FM_TASK_PERFORMANCE.csv','results/FM_D_VS_I.csv','results/FM_D_VS_I_STATISTICS.json','results/FM_SCAA_PER_SUBJECT.csv','results/FM_SCAA_SUMMARY.csv','results/FM_SCST_SUMMARY.csv','results/FM_SCST_PER_FOLD.csv','results/FM_RESCUE_MATRIX.csv','results/FM_CONFOUND_CONTROLS.csv']
    errors=[f'missing {x}' for x in required if not (c.EXP/x).is_file()]
    for stem in ['d_vs_i_fm','scaa_utility_transfer_fm','scaa_harm_coverage_fm','scst_manifold_fm','admissibility_matrix_fm']:
        for ext in ('png','pdf'):
            if not (c.FIGURES/f'{stem}.{ext}').is_file():errors.append(f'missing figure {stem}.{ext}')
    report=c.read_json(c.EXP/'FM_RESCUE_FINAL_REPORT.json') if (c.EXP/'FM_RESCUE_FINAL_REPORT.json').is_file() else {}
    if report.get('branch')!='codex/persist-eeg-fm-rescue-stage0':errors.append('branch mismatch')
    if report.get('sealed_resources_untouched') is not True:errors.append('purity false')
    if report.get('constructive_route_authorized') is not False:errors.append('unexpected authorization')
    if (c.PROTOCOL/'STEEGFORMER_CONFIRMATION_LOCK.json').exists()!=bool(report.get('STEEGFORMER_triggered')):errors.append('ST trigger/lock mismatch')
    if (c.RESULTS/'FM_TASK_PERFORMANCE.csv').is_file() and len(pd.read_csv(c.RESULTS/'FM_TASK_PERFORMANCE.csv'))!=4:errors.append('task rows')
    if (c.RESULTS/'FM_D_VS_I.csv').is_file() and len(pd.read_csv(c.RESULTS/'FM_D_VS_I.csv'))!=4:errors.append('D/I rows')
    if (c.RESULTS/'FM_SCAA_PER_SUBJECT.csv').is_file() and len(pd.read_csv(c.RESULTS/'FM_SCAA_PER_SUBJECT.csv'))!=82:errors.append('SCAA rows')
    if (c.RESULTS/'FM_SCST_PER_FOLD.csv').is_file() and len(pd.read_csv(c.RESULTS/'FM_SCST_PER_FOLD.csv'))!=60:errors.append('SCST units')
    status=subprocess.check_output(['git','status','--porcelain'],cwd=c.REPO,text=True).splitlines();forbidden=[x for x in status if x.startswith(('A ','M ')) and ('runtime/' in x or x.endswith('.pt') or 'FM_INPUT_UV' in x)]
    if forbidden:errors.append(f'large runtime staged: {forbidden}')
    payload={'schema':'FM_RESCUE_VALIDATION_V1','pass':not errors,'errors':errors,'final_terminal':report.get('overall_terminal'),'sealed_resources_untouched':report.get('sealed_resources_untouched'),'STEEGFORMER_triggered':report.get('STEEGFORMER_triggered'),'required_files':len(required)};c.write_json(c.RESULTS/'VALIDATION.json',payload)
    if errors:raise RuntimeError(errors)
    print('FM_RESCUE_VALIDATION_PASS',flush=True)


if __name__=='__main__':main()
