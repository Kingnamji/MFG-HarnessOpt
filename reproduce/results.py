"""Recompute saved-population results without API access or model fitting."""
import argparse,csv,gzip,hashlib,json,math,socket,sys
from collections import defaultdict
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
def blocked(*args,**kwargs):raise RuntimeError('Offline reproduction prohibits network access.')
socket.socket.connect=blocked;socket.create_connection=blocked
from analysis.metrics_adapter import aggregate,paired
from comparison_protocol import formal_metrics as fm,formal_export as fe
from experiments.utils.config_utils import load_experiment_configs
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)
def compare(a,b,path='value'):
    if isinstance(a,dict) and isinstance(b,dict):
        assert set(a)==set(b),(path,'keys differ')
        for k in a:compare(a[k],b[k],path+'/'+k)
    elif isinstance(a,list) and isinstance(b,list):
        assert len(a)==len(b),(path,'length differs')
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
    elif isinstance(a,(int,float)) and isinstance(b,(int,float)):
        assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-12),(path,a,b)
    else:assert a==b,(path,a,b)
def subset(s):return {k:s['metric_families'][k] for k in ['M01','M02','M04','M05','M06','M07','M08','M09','M10']}
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,default=ROOT/'artifacts/bundles')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/reproduced')
    parser.add_argument('--bootstrap',action='store_true',help='Recompute every prespecified paired comparison with 10,000 resamples.')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    configs=load_experiment_configs(ROOT);table=[];proof={};family_p=defaultdict(dict)
    for ds in ['AI4I','PdM']:
        coverage=read(ROOT/'manifests'/f'{ds}_coverage.json')
        for asset in coverage['assets']:
            p=args.artifacts/asset['file'];h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            assert h.hexdigest()==asset['sha256'],('asset checksum mismatch',asset['file'])
        expected=read(ROOT/'manifests'/f'{ds}_test.json')['task_ids']
        seal=read(ROOT/'harness_configs'/ds/'analysis_seal.json')
        selections=read(ROOT/'harness_configs'/ds/'selections.json')
        reference=read(ROOT/'results'/ds/'allocation_metrics.json')
        schemas=ROOT/'Dataset/Preprocessed'/ds
        labels=read(schemas/('allowed_schema_ai4i.json' if ds=='AI4I' else 'allowed_schema_azure.json'))['candidate_labels']
        stats={};groups={};cpu={};key=None;batch=[];n=0;missing=0
        def finish(key,batch):
            nonlocal n,missing
            if key is None:return
            kind,g=key;g=tuple(g)
            assert len(batch)==150 and {r['task_id'] for r in batch}==set(expected)
            assert all(r['dataset_key']==ds for r in batch)
            if kind=='CPU':
                view,model=g;sig=batch[0]['training_view_sha256'];wrapper=seal['CPU_confidence_maps'][f'{ds}:{sig}:{model}']
                name='CPU:'+':'.join(map(str,g))
            elif kind=='same_label':
                run,parent,draw=g;wrapper=seal['same_label_confidence_maps'][f'{ds}:{run}:{parent}'];name='same_label:'+':'.join(map(str,g))
            else:
                run,cid,draw=g;wrapper=seal['confidence_maps'].get(f'{ds}:{run}:{cid}',seal['confidence_maps'].get(f'{ds}:1701:{cid}'))
                name=f'{ds}:{run}:{cid}:{draw}'
            computed=aggregate(batch,labels=labels,configs=configs,expected_task_ids=expected,wrapper=wrapper)
            compare(subset(computed),reference[name],name)
            stats[name]=computed['metric_families'];n+=len(batch);missing+=computed['population_coverage']['terminal_missing']
            slim=[dict(task_id=r['task_id'],report=None if r['report'] is None else {'predicted_label':r['report']['predicted_label']},metrics=r['metrics']) for r in batch]
            if kind=='CPU':cpu[g]=slim
            elif kind!='same_label':groups[g]=slim
        for row in lines(args.artifacts/(ds+'-records.jsonl.gz')):
            current=(row['kind'],row['group'])
            if current!=key:finish(key,batch);batch=[];key=current
            batch.append(row['record'])
        finish(key,batch)
        def allocations(arm):
            result={}
            for run in [1701,1702,1703]:
                cid=selections[f'{run}:{arm}']['selected_candidate_id']
                for draw in [1,2,3]:result[run,draw]=groups[1701 if cid=='M3' else run,cid,draw]
            return result
        for method in ['M0','M1','M2','M3','M4','Metric-only','COPRO-style']:
            if method in ['M4','Metric-only','COPRO-style']:
                arm={'M4':'trace','Metric-only':'metric_only','COPRO-style':'copro'}[method];names=[]
                for run in [1701,1702,1703]:
                    cid=selections[f'{run}:{arm}']['selected_candidate_id']
                    names.extend(f'{ds}:{1701 if cid=="M3" else run}:{cid}:{draw}' for draw in [1,2,3])
            else:names=[f'{ds}:1701:{method}:{draw}' for draw in [1,2,3]]
            ms=[stats[x] for x in names]
            table.append(dict(dataset=ds,method=method,Accuracy=sum(x['M01']['accuracy'] for x in ms)/len(ms),MacroF1=sum(x['M01']['macro_f1'] for x in ms)/len(ms),EP=sum(x['M04']['evidence_precision']['value'] for x in ms)/len(ms),ER=sum(x['M04']['evidence_recall']['value'] for x in ms)/len(ms),J=sum(x['M02']['J_mfg'] for x in ms)/len(ms)))
        comparisons={}
        if args.bootstrap:
            expected_stats=read(ROOT/'results'/ds/'statistical_results.json')
            plan=[]
            for arm,family in [('metric_only','optimizer_primary'),('copro','COPRO_secondary')]:plan.append((f'{ds}:M4-{arm}',allocations('trace'),allocations(arm),family,'J'))
            for name in ['M3','RULE' if ds=='AI4I' else 'PRIOR','GBT']:
                right={(1701,draw):groups[1701,'M3',draw] for draw in [1,2,3]} if name=='M3' else {('deterministic',1):cpu['M3',name]}
                plan.append((f'{ds}:M4-{name}',allocations('trace'),right,'diagnosis','Accuracy'))
            for name,left,right in fe.planned_contrasts():
                def cell(c):return {(1701,draw):groups[1701,'M2' if c=='S1E1G1' else 'M2_'+c,draw] for draw in [1,2,3]}
                plan.append((ds+':'+name,cell(left),cell(right),'M2_ablation','Accuracy'))
            for name,left,right,family,outcome in plan:
                value=paired(left,right,dataset=ds,labels=labels,configs=configs,n_resamples=10000)
                compare(value,expected_stats['comparisons'][name],name)
                comparisons[name]=value;family_p[family][name]=value['pooled'][outcome]['p_bootstrap_centered']
                print(ds,name,'PASS',flush=True)
        proof[ds]=dict(status='PASS',allocation_groups=len(stats),saved_records=n,terminal_missing_records=missing,paired_comparisons=len(comparisons))
        (args.output/(ds+'_paired.json')).write_text(json.dumps(comparisons,indent=2),encoding='utf-8')
        print(ds,'ALLOCATION_METRICS_PASS',len(stats),flush=True)
    with (args.output/'table5.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    if args.bootstrap:
        corrected={name:fm.holm(rows) for name,rows in family_p.items()}
        compare(corrected,read(ROOT/'results/holm_reference.json'),'cross_dataset_Holm')
        (args.output/'holm.json').write_text(json.dumps(corrected,indent=2),encoding='utf-8')
    proof.update(network_calls=0,model_calls=0,fits=0,new_report_evaluations=0)
    (args.output/'verification.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    print(json.dumps(proof),flush=True)
if __name__=='__main__':main()
