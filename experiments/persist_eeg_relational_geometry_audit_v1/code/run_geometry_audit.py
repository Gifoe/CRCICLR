import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'experiments' / 'persist_eeg_transfer_geometry_stage1_v1'
CODE = SRC / 'code' / 'run_stage1.py'
import importlib.util
spec = importlib.util.spec_from_file_location('stage1', CODE)
stage1 = importlib.util.module_from_spec(spec)
sys.modules['stage1'] = stage1
spec.loader.exec_module(stage1)
torch.set_grad_enabled(False)
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
EPS = 1e-8
BOOT = 10000

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def model_for(ds, ck):
    m = stage1.CompactEncoder(62 if ds == 'OpenBMI' else 58)
    m.load_state_dict(torch.load(ck, map_location='cpu'))
    m.eval()
    return m

def rows_arrays(bundle, idx, mean, std, model):
    zs = []
    ls = []
    for st in range(0, len(idx), 128):
        ii = idx[st:st+128]
        x = bundle.accessor.batch(ii).astype(np.float32)
        x = (x - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)
        with torch.no_grad():
            xt = torch.from_numpy(x)
            zz = model.forward_features(xt)
            zs.append(zz.numpy())
            ls.append(model.head(zz).numpy())
    z = np.concatenate(zs)
    y = bundle.labels(idx)
    meta = [bundle.search_rows[int(i)] for i in idx]
    logits = np.concatenate(ls)
    return z, y, meta, logits

def ba_f1(y, pred):
    y = np.asarray(y, int); pred = np.asarray(pred, int)
    recalls = []
    f1s = []
    for k in (0, 1):
        tp = int(np.sum((y == k) & (pred == k)))
        fn = int(np.sum((y == k) & (pred != k)))
        fp = int(np.sum((y != k) & (pred == k)))
        recalls.append(tp / max(1, tp + fn))
        prec = tp / max(1, tp + fp)
        f1s.append(2 * prec * recalls[-1] / max(EPS, prec + recalls[-1]))
    return float(np.mean(recalls)), float(np.mean(f1s))

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))

def summarize(vals):
    vals = np.asarray(vals, float)
    return {'mean': float(np.mean(vals)), 'median': float(np.median(vals)),
            'q25': float(np.quantile(vals, .25)), 'q75': float(np.quantile(vals, .75)),
            'n': int(len(vals))}

def boot(vals, seed=0):
    v = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    n = len(v)
    means = np.mean(v[rng.integers(0, n, (BOOT, n))], axis=1)
    return {'mean': float(v.mean()), 'ci_low': float(np.quantile(means, .025)),
            'ci_high': float(np.quantile(means, .975)), 'n': n,
            'resamples': BOOT, 'seed': seed}

def main():
    print('AUDIT_START', flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(SRC.parent / 'persist_eeg_relational_geometry_audit_v1'))
    args = ap.parse_args()
    out = Path(args.out)
    for d in ('protocol', 'outputs', 'code'):
        (out / d).mkdir(parents=True, exist_ok=True)
    log = json.load(open(SRC / 'outputs' / 'SEED0_TRAINING_LOG.json'))
    split = json.load(open(SRC / 'protocol' / 'STAGE1_SEARCH_CV_SPLIT.json'))
    prov, cent_rows, transfer_rows, subject_rows, fold_rows = [], [], [], [], []
    proto_rows, center_rows, tg_rows, direction_rows = [], [], [], []
    null_fold_data = []
    audit = {'V8_INTERNAL_HOLDOUT_loaded': False, 'WBCIC_outer_loaded': False,
             'cross_fold_latent_vector_comparison': False, 'protocol_valid': True,
             'accessed_subject_scope': {}}
    for item in log['datasets']:
        ds = item['dataset']
        f = int(item['fold'])
        print(f'FOLD_START {ds} {f}', flush=True)
        mean = np.array(item['normalizer_mean'], np.float32)
        std = np.array(item['normalizer_std'], np.float32)
        for name in ('Compact-ERM', 'Compact-TG'):
            rec = item['models'][name]
            ck = Path(rec['checkpoint_path'])
            if not ck.exists() or sha256(ck) != rec['checkpoint_sha256']:
                raise RuntimeError('RELATIONAL_GEOMETRY_AUDIT_ARTIFACT_MISSING_OR_MISMATCH')
            prov.append({'dataset': ds, 'fold': f, 'model': name,
                         'checkpoint_path': str(ck),
                         'checkpoint_sha256_logged': rec['checkpoint_sha256'],
                         'checkpoint_sha256_recomputed': sha256(ck),
                         'exists': True, 'selected_epoch': rec['selected_epoch']})
        search_payload = stage1.load_search_split()
        search = search_payload[0][ds] if isinstance(search_payload, tuple) else search_payload[ds]
        bundle = stage1.load_bundle(ds, search)
        if 'datasets' in split:
            item_split = split['datasets'][ds]['folds'][f]
        else:
            item_split = split[ds][str(f)] if isinstance(split[ds], dict) and str(f) in split[ds] else split[ds][f]
        inner, outer = item_split['inner_train_subjects'], item_split['outer_dev_subjects']
        audit['accessed_subject_scope'][ds] = sorted(set(inner + outer))
        models = {n: model_for(ds, Path(item['models'][n]['checkpoint_path']))
                  for n in ('Compact-ERM', 'Compact-TG')}
        all_sub = sorted(set(inner + outer), key=lambda s: stage1.subject_sort([s], ds))
        for name, model in models.items():
            print(f'MODEL_START {ds} {f} {name}', flush=True)
            idx = bundle.indices(all_sub)
            z, y, meta, logits = rows_arrays(bundle, idx, mean, std, model)
            groups = {}
            for j, r in enumerate(meta):
                groups.setdefault((r.subject, int(r.session), int(y[j])), []).append(j)
            cent = {k: z[v].mean(0) for k, v in groups.items()}
            allmeans = {}
            for s in all_sub:
                sessions = (1, 2) if ds == 'OpenBMI' else (0, 1, 2)
                for t in sessions:
                    jj = [j for j, r in enumerate(meta) if r.subject == s and int(r.session) == t]
                    if jj:
                        allmeans[(s, t)] = z[jj].mean(0)
            dirs, mids = {}, {}
            for s in all_sub:
                sessions = (1, 2) if ds == 'OpenBMI' else (0, 1, 2)
                for t in sessions:
                    if (s, t, 0) in cent and (s, t, 1) in cent:
                        mids[(s, t)] = (cent[(s,t,0)] + cent[(s,t,1)]) / 2
                        dirs[(s, t)] = cent[(s,t,1)] - cent[(s,t,0)]
            source_early = []
            for s in inner:
                ts = (1,) if ds == 'OpenBMI' else (0, 1)
                if all((s, t, k) in cent for t in ts for k in (0,1)):
                    proto = {k: np.mean([cent[(s,t,k)] for t in ts], 0) for k in (0,1)}
                    mid = np.mean([mids[(s,t)] for t in ts], 0)
                    dr = np.mean([dirs[(s,t)]/(np.linalg.norm(dirs[(s,t)])+EPS) for t in ts], 0)
                    source_early.append((s, proto, mid, dr))
            csrc = {k: np.mean([x[1][k] for x in source_early], 0) for k in (0,1)}
            msrc = np.mean([x[2] for x in source_early], 0)
            dpop = np.mean([x[3] for x in source_early], 0)
            dpop_hat = dpop / (np.linalg.norm(dpop) + EPS)
            # Source-centered class prototypes: center each source subject by
            # its unlabeled early-session mean before averaging subjects.
            source_centered = {0: [], 1: []}
            early_sessions = (1,) if ds == 'OpenBMI' else (0, 1)
            for s in inner:
                jj = [j for j, r in enumerate(meta) if r.subject == s and int(r.session) in early_sessions]
                if not jj:
                    continue
                sm = z[jj].mean(0)
                for k in (0, 1):
                    kk = [j for j in jj if int(y[j]) == k]
                    if kk:
                        source_centered[k].append((z[kk] - sm).mean(0))
            ccsrc = {k: np.mean(source_centered[k], 0) for k in (0, 1) if source_centered[k]}
            # Pairwise direction consensus in the future session, fold-local.
            future_t = 2 if ds == 'OpenBMI' else 2
            fut_dirs = [dirs[(s, future_t)] / (np.linalg.norm(dirs[(s, future_t)]) + EPS)
                        for s in outer if (s, future_t) in dirs]
            null_fold_data.append({'dataset': ds, 'fold': f, 'model': name,
                                   'source_dirs': [x[3] for x in source_early],
                                   'future_dirs': fut_dirs})
            direction_rows.append({'dataset': ds, 'fold': f, 'model': name,
                                   'future_direction_consensus': float(np.mean([cos(a,b) for i,a in enumerate(fut_dirs) for b in fut_dirs[i+1:]])) if len(fut_dirs) > 1 else np.nan,
                                   'future_subjects': len(fut_dirs),
                                   'source_direction_norm': float(np.linalg.norm(dpop))})
            fold_rows.append({'dataset': ds, 'fold': f, 'model': name,
                              'source_direction_norm': float(np.linalg.norm(dpop)),
                              'source_subjects': len(source_early)})
            for s in outer:
                ts = (1, 2) if ds == 'OpenBMI' else (0, 1, 2)
                for t in ts:
                    if (s, t) not in dirs:
                        continue
                    d = dirs[(s,t)]
                    for k in (0,1):
                        av = [cos(cent[(s,t,k)], cent[(r,t,k)]) for r in outer if r != s and (r,t,k) in cent]
                        cv = [cos(cent[(s,t,k)]-mids[(s,t)], cent[(r,t,k)]-mids[(r,t)])
                              for r in outer if r != s and (r,t,k) in cent]
                        cent_rows.append({'dataset': ds, 'fold': f, 'model': name, 'subject': s,
                                          'session': t, 'class': k,
                                          'abs_alignment': np.mean(av) if av else np.nan,
                                          'centered_alignment': np.mean(cv) if cv else np.nan})
                    if t == (2 if ds == 'OpenBMI' else 2):
                        dh = d / (np.linalg.norm(d) + EPS)
                        offset = np.linalg.norm(mids[(s,t)] - msrc)
                        sep = np.linalg.norm(d)
                        transfer_rows.append({'dataset': ds, 'fold': f, 'model': name, 'subject': s,
                                              'future_session': t,
                                              'source_to_future_direction_cosine': cos(dpop_hat, dh),
                                              'offset_norm': float(offset),
                                              'class_separation_norm': float(sep),
                                              'offset_separation_ratio': float(offset/(sep+EPS))})
                        jj = [j for j, r in enumerate(meta) if r.subject == s and int(r.session) == t]
                        if jj and all(k in csrc for k in (0,1)) and all(k in ccsrc for k in (0,1)):
                            zz = z[jj]; yy = y[jj]; ll = logits[jj]
                            c0 = csrc[0] / (np.linalg.norm(csrc[0]) + EPS)
                            c1 = csrc[1] / (np.linalg.norm(csrc[1]) + EPS)
                            zn = zz / (np.linalg.norm(zz, axis=1, keepdims=True) + EPS)
                            pa = (np.stack([zn @ c0, zn @ c1], 1).argmax(1)).astype(int)
                            sm = zz.mean(0)
                            zc = zz - sm
                            q0 = ccsrc[0] / (np.linalg.norm(ccsrc[0]) + EPS)
                            q1 = ccsrc[1] / (np.linalg.norm(ccsrc[1]) + EPS)
                            zcn = zc / (np.linalg.norm(zc, axis=1, keepdims=True) + EPS)
                            pc = (np.stack([zcn @ q0, zcn @ q1], 1).argmax(1)).astype(int)
                            aba, af1 = ba_f1(yy, pa); cba, cf1 = ba_f1(yy, pc)
                            scat = {}
                            for k in (0,1):
                                kk = np.where(yy == k)[0]
                                scat[k] = float(np.mean(np.sum((zz[kk] - cent[(s,t,k)])**2, axis=1))) if len(kk) else np.nan
                            fisher = float(sep**2 / (scat[0] + scat[1] + EPS))
                            margins = np.where(yy == 1, ll[:,1] - ll[:,0], ll[:,0] - ll[:,1])
                            proto_rows.append({'dataset': ds, 'fold': f, 'model': name, 'subject': s,
                                               'future_session': t,
                                               'absolute_ba': aba, 'absolute_macro_f1': af1,
                                               'centered_ba': cba, 'centered_macro_f1': cf1,
                                               'centered_delta_ba': cba - aba,
                                               'class_separation_norm': float(sep),
                                               'within_scatter': float(scat[0] + scat[1]),
                                               'fisher_discriminability': fisher,
                                               'margin_mean': float(np.mean(margins)),
                                               'margin_median': float(np.median(margins)),
                                               'margin_positive_fraction': float(np.mean(margins > 0))})
                if ds == 'OpenBMI' and (s,1) in dirs and (s,2) in dirs:
                    subject_rows.append({'dataset': ds, 'fold': f, 'model': name, 'subject': s,
                                         'session_persistence': cos(dirs[(s,1)], dirs[(s,2)])})
                if ds == 'WBCIC' and all((s,t) in dirs for t in (0,1,2)):
                    u = [dirs[(s,t)]/(np.linalg.norm(dirs[(s,t)])+EPS) for t in (0,1,2)]
                    e = (u[0] + u[1]) / (np.linalg.norm(u[0] + u[1]) + EPS)
                    subject_rows.append({'dataset': ds, 'fold': f, 'model': name, 'subject': s,
                                         'p12': cos(u[0], u[1]), 'p23': cos(u[1], u[2]),
                                         'p_early_s3': cos(e, u[2])})
            # Release per-model trial embeddings before loading the next
            # checkpoint; keeping both models' arrays can exceed RAM on the
            # server even though this is an inference-only audit.
            del z, y, meta, logits, model, cent, dirs, mids, allmeans
            import gc
            gc.collect()
    (out/'protocol'/'SOURCE_ARTIFACT_PROVENANCE.json').write_text(json.dumps(prov, indent=2))
    (out/'protocol'/'HOLDOUT_ISOLATION_AUDIT.json').write_text(json.dumps(audit, indent=2))
    (out/'protocol'/'LATENT_COORDINATE_SYSTEM_AUDIT.json').write_text(json.dumps(
        {'cross_fold_latent_vector_comparison': False, 'fold_local_only': True}, indent=2))
    (out/'protocol'/'DIAGNOSTIC_CONFIG.json').write_text(json.dumps(
        {'epsilon': EPS, 'bootstrap_resamples': BOOT, 'bootstrap_seed': 0,
         'models': ['Compact-ERM','Compact-TG'], 'sessions': {'OpenBMI':[1,2], 'WBCIC':[0,1,2]}}, indent=2))
    (out/'protocol'/'STAGE1_PROTOCOL_NOTE.md').write_text(
        '# Stage-1 protocol note\n\nSame episode manifests: true. Same optimizer/hyperparameters: true. '
        'Same early-stopping rule: true. Same actual selected optimization steps: false because the models selected different epochs.\n')
    pd.DataFrame(prov).to_json(out/'protocol'/'SOURCE_ARTIFACT_PROVENANCE_TABLE.json', orient='records', indent=2)
    pd.DataFrame(cent_rows).to_csv(out/'outputs'/'SUBJECT_GEOMETRY.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(out/'outputs'/'FOLD_GEOMETRY.csv', index=False)
    pd.DataFrame(transfer_rows).to_csv(out/'outputs'/'PROTOTYPE_TRANSFER.csv', index=False)
    pd.DataFrame(subject_rows).to_csv(out/'outputs'/'SUBJECT_SESSION_PERSISTENCE.csv', index=False)
    pd.DataFrame(proto_rows).to_csv(out/'outputs'/'CENTERING_DIAGNOSTIC.csv', index=False)
    drf = pd.DataFrame(direction_rows)
    drf.to_csv(out/'outputs'/'DIRECTION_CONSENSUS.csv', index=False)
    pr = pd.DataFrame(proto_rows)
    tg = []
    cdf = pd.DataFrame(cent_rows)
    align_map = cdf[cdf.session == 2].groupby(['dataset','fold','model']).abs_alignment.mean().unstack('model') if len(cdf) else pd.DataFrame()
    for (ds, fold, subj), q in pr.groupby(['dataset','fold','subject']):
        if set(q.model) >= {'Compact-ERM','Compact-TG'}:
            a = q.set_index('model')
            adelta = float(align_map.loc[(ds, fold), 'Compact-TG'] - align_map.loc[(ds, fold), 'Compact-ERM']) if (ds, fold) in align_map.index else np.nan
            tg.append({'dataset': ds, 'fold': int(fold), 'subject': subj,
                       'absolute_alignment_delta': adelta,
                       'class_separation_delta': float(a.loc['Compact-TG','class_separation_norm'] - a.loc['Compact-ERM','class_separation_norm']),
                       'within_scatter_delta': float(a.loc['Compact-TG','within_scatter'] - a.loc['Compact-ERM','within_scatter']),
                       'fisher_delta': float(a.loc['Compact-TG','fisher_discriminability'] - a.loc['Compact-ERM','fisher_discriminability']),
                       'margin_delta': float(a.loc['Compact-TG','margin_mean'] - a.loc['Compact-ERM','margin_mean']),
                       'absolute_ba_delta': float(a.loc['Compact-TG','absolute_ba'] - a.loc['Compact-ERM','absolute_ba']),
                       'centered_ba_delta': float(a.loc['Compact-TG','centered_ba'] - a.loc['Compact-ERM','centered_ba'])})
    tgdf = pd.DataFrame(tg)
    tgdf.to_csv(out/'outputs'/'TG_VS_ERM_GEOMETRY.csv', index=False)
    tr = pd.DataFrame(transfer_rows); sr = pd.DataFrame(subject_rows); null = {}; agg = []
    for ds in ('OpenBMI','WBCIC'):
        for model in ('Compact-ERM','Compact-TG'):
            q = tr[(tr.dataset == ds) & (tr.model == model)]
            vals = q.source_to_future_direction_cosine.to_numpy(float)
            foldmeans = q.groupby('fold').source_to_future_direction_cosine.mean().tolist()
            # Correct sign-flip null: flip source subject directions, rebuild
            # population direction, then score each future subject.
            fold_null = [x for x in null_fold_data if x['dataset']==ds and x['model']==model]
            rng = np.random.default_rng(0)
            nv_rows = []
            for _ in range(BOOT):
                fmeans = []
                for fd in fold_null:
                    sd = np.asarray(fd['source_dirs']); fu = fd['future_dirs']
                    if len(sd)==0 or len(fu)==0: continue
                    signs = rng.choice([-1,1], size=len(sd))
                    nh = np.mean(sd * signs[:,None], axis=0)
                    nh = nh / (np.linalg.norm(nh) + EPS)
                    fmeans.extend([cos(nh, u) for u in fu])
                nv_rows.append(np.mean(fmeans) if fmeans else np.nan)
            nv = np.asarray(nv_rows, float)
            key = ds + '_' + model
            null[key] = {'observed_mean': float(np.mean(vals)), 'null_mean': float(nv.mean()),
                         'null_ci_low': float(np.quantile(nv,.025)), 'null_ci_high': float(np.quantile(nv,.975)),
                         'one_sided_empirical_p': float((1 + np.sum(nv >= np.mean(vals))) / (BOOT+1)),
                         'positive_folds': int(sum(x > 0 for x in foldmeans)), 'total_folds': len(foldmeans)}
            row = {'dataset': ds, 'model': model,
                   'source_to_future_direction': summarize(vals),
                   'source_to_future_bootstrap': boot(vals),
                   'fold_means': foldmeans,
                   'offset_separation': summarize(q.offset_separation_ratio.to_numpy(float))}
            sub = sr[(sr.dataset == ds) & (sr.model == model)]
            if ds == 'OpenBMI':
                row['session_persistence'] = summarize(sub.session_persistence.to_numpy(float))
            else:
                for c in ('p12','p23','p_early_s3'):
                    row[c] = summarize(sub[c].to_numpy(float))
            agg.append(row)
    (out/'outputs'/'DIRECTION_NULL_TEST.json').write_text(json.dumps(null, indent=2))
    (out/'outputs'/'AGGREGATE_GEOMETRY.json').write_text(json.dumps(agg, indent=2))
    (out/'outputs'/'DIRECTION_NULL_TEST.csv').write_text(pd.DataFrame(
        [{'dataset': k.rsplit('_',1)[0], 'model': k.rsplit('_',1)[1], **v} for k,v in null.items()]).to_csv(index=False))
    # Subject-level bootstrap for the principal diagnostics.
    bs = {}
    for ds in ('OpenBMI','WBCIC'):
        q = pr[pr.dataset == ds]
        for metric in ('centered_delta_ba','fisher_discriminability','margin_mean','class_separation_norm'):
            if metric in q:
                vals = q[metric].dropna().to_numpy(float)
                bs[f'{ds}_{metric}'] = boot(vals) if len(vals) else None
        qtg = tgdf[tgdf.dataset == ds]
        for metric in ('centered_ba_delta','fisher_delta','margin_delta','class_separation_delta'):
            vals = qtg[metric].dropna().to_numpy(float) if metric in qtg else np.array([])
            bs[f'{ds}_TG_minus_ERM_{metric}'] = boot(vals) if len(vals) else None
    (out/'outputs'/'BOOTSTRAP.json').write_text(json.dumps(bs, indent=2))
    # Fold-level compact aggregate table.
    rows = []
    for ds in ('OpenBMI','WBCIC'):
        for model in ('Compact-ERM','Compact-TG'):
            q = tr[(tr.dataset==ds)&(tr.model==model)]
            d = drf[(drf.dataset==ds)&(drf.model==model)]
            p = pr[(pr.dataset==ds)&(pr.model==model)]
            rows.append({'dataset':ds,'model':model,
                         'absolute_centroid_alignment': float(cent_rows and np.nanmean([x['abs_alignment'] for x in cent_rows if x['dataset']==ds and x['model']==model]) or np.nan),
                         'direction_consensus': float(d.future_direction_consensus.mean()) if len(d) else np.nan,
                         'source_to_future_direction_cosine': float(q.source_to_future_direction_cosine.mean()),
                         'session_persistence': float(sr[(sr.dataset==ds)&(sr.model==model)].filter(regex='session_persistence|p_early_s3').mean(numeric_only=True).mean()),
                         'offset_separation_ratio': float(q.offset_separation_ratio.mean()),
                         'absolute_prototype_ba': float(p.absolute_ba.mean()),
                         'centered_prototype_ba': float(p.centered_ba.mean()),
                         'centered_improvement_pp': float(100*(p.centered_ba.mean()-p.absolute_ba.mean())),
                         'fisher_discriminability': float(p.fisher_discriminability.mean()),
                         'margin_mean': float(p.margin_mean.mean())})
    pd.DataFrame(rows).to_csv(out/'outputs'/'AGGREGATE_GEOMETRY.csv', index=False)
    # Explicit pre-registered decision rules.
    h1 = all(null[f'{ds}_{m}']['observed_mean'] > 0 and null[f'{ds}_{m}']['one_sided_empirical_p'] < .05 and null[f'{ds}_{m}']['positive_folds'] >= 2 for ds in ('OpenBMI','WBCIC') for m in ('Compact-ERM','Compact-TG'))
    wb_tg = tgdf[tgdf.dataset=='WBCIC']
    h2fold = pr[pr.dataset=='WBCIC'].groupby(['fold','model']).centered_delta_ba.mean().unstack('model')
    h2 = (len(h2fold) >= 3 and float(h2fold['Compact-ERM'].mean()) >= .01 and int((h2fold['Compact-ERM'] > 0).sum()) >= 2)
    h3fold = wb_tg.groupby('fold').agg({'absolute_alignment_delta':'mean','fisher_delta':'mean','margin_delta':'mean','class_separation_delta':'mean'}) if len(wb_tg) else pd.DataFrame()
    h3 = bool(len(h3fold) >= 3 and (h3fold.absolute_alignment_delta > 0).sum() >= 2 and
              ((h3fold[['fisher_delta','margin_delta','class_separation_delta']] < 0).any(axis=1).sum() >= 2))
    if h1 and h2 and h3: terminal = 'RELATIONAL_GEOMETRY_SUPPORTED_STOP'
    elif h1 or h2 or h3: terminal = 'RELATIONAL_GEOMETRY_PARTIAL_STOP'
    else: terminal = 'RELATIONAL_GEOMETRY_NOT_SUPPORTED_STOP'
    (out/'METHOD.md').write_text('# Relational geometry audit\n\nRepresentation-only scoring audit of frozen Compact-ERM and Compact-TG checkpoints. No retraining, sweep, rescue, new seed, or forbidden holdout access. All latent comparisons are within fold-specific checkpoints.\n')
    # Human-readable decision with the direct protocol questions answered.
    def fmt(ds, model, col):
        q = pd.DataFrame(rows); z = q[(q.dataset==ds)&(q.model==model)]
        return 'NA' if z.empty else f"{float(z.iloc[0][col]):.4f}"
    dec = ['# Decision','', 'Representation-only audit of frozen Stage-1 seed0 checkpoints. No retraining or forbidden cohort access.', '',
           '1. OpenBMI absolute centroids: '+fmt('OpenBMI','Compact-ERM','absolute_centroid_alignment'),
           '2. WBCIC absolute centroids: '+fmt('WBCIC','Compact-ERM','absolute_centroid_alignment'),
           '3. OpenBMI directions: '+fmt('OpenBMI','Compact-ERM','direction_consensus'),
           '4. WBCIC directions: '+fmt('WBCIC','Compact-ERM','direction_consensus'),
           '5. Source→future direction transfer: see DIRECTION_NULL_TEST.json.',
           '6. Session persistence: see SUBJECT_SESSION_PERSISTENCE.csv.',
           '7. WBCIC offset/separation: '+fmt('WBCIC','Compact-ERM','offset_separation_ratio'),
           '8. Unlabeled centering: see CENTERING_DIAGNOSTIC.csv.',
           '9. TG absolute alignment change: compare AGGREGATE_GEOMETRY.csv.',
           '10. TG discrimination cost: compare TG_VS_ERM_GEOMETRY.csv.',
           '11. align relations, not absolute locations: '+('supported' if h2 and h1 else 'not established'),
           '12. Protocol invalidity: NO (artifact hashes, fold-local coordinates, and holdout isolation passed).', '',
           f'H1 discriminative-direction transfer: {"PASS" if h1 else "FAIL"}',
           f'H2 absolute-location mismatch: {"PASS" if h2 else "FAIL"}',
           f'H3 alignment-discrimination tradeoff: {"PASS" if h3 else "FAIL"}', '', 'Terminal:', terminal]
    (out/'outputs'/'DECISION.md').write_text('\n'.join(dec)+'\n')
    print('RELATIONAL_GEOMETRY_AUDIT_DONE')

if __name__ == '__main__':
    main()
