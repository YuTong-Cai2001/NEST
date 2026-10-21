"""Verify run provenance, aggregate three seeds, and redraw only Figure 3."""
import csv
import hashlib
import json
import pickle
import shutil
from pathlib import Path
import numpy as np
import torch
import plot_paper_experiment_figures as plots

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PREFIX = "spad_final_ablation_20261007"
OUT = WORKSPACE / "outputs" / PREFIX
NEW_CSV = ROOT / "results/component_ablation_final_20261007_runs.csv"
TABLE_CSV = ROOT / "results/component_ablation_final_20261007_table3_runs.csv"
VARIANTS = ("full", "svd_alignment", "no_descriptors", "no_residual", "no_self", "no_proto", "dual")
SEEDS = (42, 43, 44)

def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def save_csv(path, rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def main():
    new=read(NEW_CSV)
    assert len(new)==18 and len({r['run_name'] for r in new})==18
    old=read(ROOT / 'results/component_ablation_runs.csv')
    full=[r for r in old if r['run_name'].startswith('spad_ablation_no_reg_seed')]
    assert len(full)==3
    rows=[]; checks=[]
    for variant in VARIANTS:
        source=full if variant=='full' else [r for r in new if r['run_name'].startswith(f'{PREFIX}_{variant}_seed')]
        assert sorted(int(r['seed']) for r in source)==list(SEEDS),variant
        for row in sorted(source,key=lambda r:int(r['seed'])):
            checkpoint=Path(row['checkpoint'])
            artifact=torch.load(checkpoint,map_location='cpu',weights_only=False)
            config=artifact['config']; resolved=artifact['resolved_model_config']
            assert config['num_epochs']==50 and config['batch_size']==1 and config['sequence_length']==5
            assert (config['start_year'],config['end_year'])==(2018,2022)
            assert config['seed']==int(row['seed'])
            assert config.get('lambda_reg',0)==0
            assert config['weight_decay']==1e-5 and config['learning_rate']==.001
            assert config['lambda_self']==(0 if variant=='no_self' else .5)
            assert config['lambda_proto']==(0 if variant=='no_proto' else .3)
            assert config['alignment_mode']==('svd' if variant=='svd_alignment' else 'invariant')
            assert config['use_structural_temporal_descriptors']==(variant!='no_descriptors')
            assert config['prototype_mode']==('dual' if variant=='dual' else 'single')
            assert resolved['use_residual_fusion']==(variant!='no_residual')
            assert resolved['feature_dim']==(16 if variant=='no_descriptors' else 24)
            assert resolved['sequence_length']==5 and resolved['tcn_kernel_size']==3
            assert resolved['tcn_hidden_dims']==[32,64,32] and resolved['gnn_hidden_dims']==[32,64,32]
            assert resolved['projection_dim']==32 and resolved['output_dim']==32
            assert resolved['projection_hidden_dim']==32 and resolved['dropout']==.3
            assert resolved['use_transport_residual'] and resolved['use_local_evolution_score']
            history_path=checkpoint.parent.parent / 'training_history.pkl'
            with history_path.open('rb') as f: history=pickle.load(f)
            epochs=history['train_history']; assert len(epochs)==50
            evaluation=history['final_evaluation']
            assert evaluation['num_samples']==5132
            assert evaluation['excluded_unlabeled_nodes']==187336
            assert np.isclose(float(row['auc']),evaluation.get('dynamic_auc',evaluation.get('auc')),atol=1e-12,rtol=0)
            assert np.isclose(float(row['ap']),evaluation.get('dynamic_ap',evaluation.get('ap')),atol=1e-12,rtol=0)
            max_error=max(abs(e['train_loss']-(e['anomaly_loss']+config['lambda_proto']*e['prototype_loss']+config['lambda_self']*e['consistency_loss'])) for e in epochs)
            assert max_error<2e-6,(row['run_name'],max_error)
            result=dict(row)
            result['variant']=variant
            result['original_run_name']=row['run_name']
            result['provenance']='verified_existing_full_no_reg' if variant=='full' else 'fresh_final_objective_rerun'
            result['lambda_reg']='0'
            rows.append(result)
            checks.append({'run_name':row['run_name'],'checkpoint':str(checkpoint),
                           'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                           'epochs':len(epochs),'max_loss_identity_error':max_error,
                           'config':config,'resolved_model_config':resolved})
    # A consistent union schema also preserves legacy full-model metadata.
    columns=list(dict.fromkeys(k for r in rows for k in r))
    rows=[{k:r.get(k,'') for k in columns} for r in rows]
    save_csv(TABLE_CSV,rows)
    metadata={
       'full':('Full SPAD','None','No','Verified final three-term objective','Reference'),
       'svd_alignment':('SVD coordinate alignment','Replace invariant attribute statistics with SVD','Yes','Independent SVD coordinates; retain eight descriptors','Domain transfer'),
       'no_descriptors':('w/o structural-temporal descriptors','Remove eight structural-temporal descriptors','Yes','Disable descriptor appending in both domains','Dynamic graph evidence'),
       'no_residual':('w/o residual interaction','Zero the residual-interaction input to fusion','Yes','Retain the spatial and input skip branches','Representation design'),
       'no_self':('w/o temporal consistency','Remove temporal-consistency loss','Yes','Set lambda_self = 0','Training objective'),
       'no_proto':('w/o prototype compactness','Remove normal-prototype compactness loss','Yes','Set lambda_proto = 0','Training objective'),
       'dual':('Dual prototypes','Use normal/abnormal prototypes and prototype-similarity inference','Yes','Changes prototype objective and final scoring branch','Bundled design comparison'),
    }
    items=[]; summary=[]
    for variant in VARIANTS:
        group=[r for r in rows if r['variant']==variant]
        auc=[float(r['auc']) for r in group]; ap=[float(r['ap']) for r in group]
        label,mechanism,retraining,scope,role=metadata[variant]
        item=dict(variant=variant,label=label,mechanism=mechanism,retraining=retraining,scope=scope,role=role,
                  status='Verified reference' if variant=='full' else 'Rerun complete',auc=auc,ap=ap,
                  auc_mean=float(np.mean(auc)),auc_std=float(np.std(auc)),ap_mean=float(np.mean(ap)),ap_std=float(np.std(ap)))
        items.append(item)
        summary.append({k:item[k] for k in ('variant','label','auc_mean','auc_std','ap_mean','ap_std')})
    runtime={'torch':torch.__version__,'cuda_build':torch.version.cuda,
             'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (OUT/'component_results.json').write_text(json.dumps({'seeds':SEEDS,'ddof':0,'items':items,'checks':checks,
                                                       'rerun_runtime':runtime},indent=2),encoding='utf-8')
    save_csv(OUT/'component_summary.csv',summary)
    plots.OUTPUT_DIR=OUT
    audit=[]
    _,paths=plots.plot_component_ablation(audit,path=TABLE_CSV,prefix=PREFIX,full_key='full')
    save_csv(OUT/'figure3_statistics.csv',audit)
    manifest={'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (NEW_CSV,TABLE_CSV)],
              'outputs':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
              'statistics':'mean and population standard deviation','ddof':0,'seeds':SEEDS,
              'full_reference':'Existing lambda_reg=0 runs; all 50 epoch loss identities checked.',
              'dual_scope':'Prototype mode changes both training and inference; not prototype-count-only.'}
    (OUT/'figure3_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    # Publish only the requested figure and its component-statistics entries.
    for target in (ROOT/'figures/paper/fig_component_ablation.pdf',
                   WORKSPACE/'latex_figure/fig_component_ablation.pdf',
                   WORKSPACE/'JIIS_submission/Fig3.pdf'):
        shutil.copyfile(paths[0],target)
    shutil.copyfile(paths[1],ROOT/'figures/paper/fig_component_ablation.png')
    statistics_path=ROOT/'figures/paper/paper_plot_statistics_population_std.csv'
    unchanged=[r for r in read(statistics_path) if r['figure']!='component_ablation']
    save_csv(statistics_path,unchanged+audit)
    canonical_path=ROOT/'figures/paper/paper_figure_manifest.json'
    canonical=json.loads(canonical_path.read_text(encoding='utf-8'))
    canonical['component_update_20261007']=manifest
    for entry in canonical['outputs']:
        path=Path(entry['path'])
        if path.name in ('fig_component_ablation.pdf','fig_component_ablation.png','paper_plot_statistics_population_std.csv'):
            entry['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    canonical_path.write_text(json.dumps(canonical,indent=2),encoding='utf-8')
    for s in summary: print(s)

if __name__=='__main__': main()
