"""Frozen statistical functions; filesystem exporter excluded from the public adapter."""
import pandas as pd
from .protocol import registry
from nonllm_baselines.views import ObservableView, dump
from nonllm_baselines.predictors import matched_associations

def strata(task,records,guide):
    """Gold may label class strata offline, never define prior conflict."""
    result={'class':str(task['gold_label'])}
    if 'AI4I' in str(task['dataset']):
        from experiments.evaluation.public_conditions import check_public_conditions
        from nonllm_baselines.views import finite
        from experiments.data.load_preprocessed import load_ai4i_preprocessed
        schema=load_ai4i_preprocessed().allowed_schema
        checks=check_public_conditions({},records,schema)
        wear=finite(task.get('tool_wear'));result['wear_200_240']=wear is not None and 200<=wear<=240
        result['wear_200_240_NoFailure']=result['wear_200_240'] and task['gold_label']=='No Failure'
        result.update({c['label']+'_condition':str(c['satisfied']) for c in checks['checks']})
    else:
        from nonllm_baselines.views import visible_associations
        view=ObservableView(dump(dict(records=records,associations=visible_associations(guide))))
        matches=matched_associations(view);tops={r['top_component'] for r in matches}
        result.update(prior_conflict=len(tops)>1,prior_matches=len(matches),prior_match_bin='0' if not matches else '1' if len(matches)==1 else 'multiple',
            fallback=not matches,telemetry_channels=len({r.get('field') for r in records if r.get('source')=='telemetry'}),
            matched_patterns=[r['evidence_key'] for r in matches],prior_tops=sorted(tops))
    return result

def paired_test_summary(left,right,*,dataset,labels,configs,n_resamples=10000):
    def flatten(groups):return [dict(r['metrics'],optimizer_run=run,draw=draw) for (run,draw),records in groups.items() for r in records]
    a=flatten(left);b=flatten(right);ids=set(r['task_id'] for r in a)
    clusters={i:registry(dataset)[i]['machineID'] if dataset=='PdM' else i for i in ids}
    import numpy as np
    from experiments.optimization.selection_metrics import load_metric_weights
    groups=sorted(set(clusters.values()),key=str);rng=np.random.default_rng(20261024);weights=load_metric_weights(configs)
    def sufficient(rows):
        vectors=[]
        for group in groups:
            rs=[r for r in rows if clusters[r['task_id']]==group];f=pd.DataFrame(rs)
            recall=pd.to_numeric(f.evidence_recall,errors='coerce')
            values=[len(rs),f.label_correct.sum(),f.evidence_precision.fillna(0).sum(),recall.fillna(0).sum(),recall.notna().sum(),
                f.evidence_linked_action_ratio.fillna(0).sum(),f.unsupported_claim_rate.sum(),f.unsafe_recommendation_rate.sum(),f.cost_per_report.sum(),f.F10.sum()]
            for label in labels:
                t=f.gold_label.eq(label);p=f.predicted_label.eq(label);values.extend([(t&p).sum(),(~t&p).sum(),(t&~p).sum()])
            vectors.append(values)
        return np.array(vectors,dtype=float)
    def score(v):
        n=v[...,0];accuracy=v[...,1]/n;e=.5*v[...,2]/n+.5*np.divide(v[...,3],v[...,4],out=np.zeros_like(n),where=v[...,4]>0)
        cost=v[...,8]/n/.005
        j=(weights['alpha_label']*accuracy+weights['beta_evidence']*e+weights['gamma_action']*v[...,5]/n-
            weights['lambda_unsupported']*v[...,6]/n-weights['lambda_unsafe']*v[...,7]/n-
            weights['mu_cost']*np.clip(cost,0,1)-weights['rho_repair_failure']*v[...,9]/n)
        cf=v[...,10:].reshape(*v.shape[:-1],len(labels),3);den=2*cf[...,0]+cf[...,1]+cf[...,2]
        f1=np.divide(2*cf[...,0],den,out=np.zeros_like(den),where=den>0).mean(axis=-1)
        return np.stack([accuracy,f1,j],axis=-1)
    def summarize(samples,point):
        result={}
        for i,name in enumerate(('Accuracy','MacroF1','J')):
            lo,hi=np.percentile(samples[:,i],[2.5,97.5]);p=(1+np.count_nonzero(abs(samples[:,i]-point[i])>=abs(point[i])))/(n_resamples+1)
            result[name]=dict(effect=float(point[i]),CI95=[float(lo),float(hi)],p_bootstrap_centered=float(p),n_clusters=len(groups),resamples=n_resamples)
        return result
    counts=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=n_resamples)
    aa,bb=sufficient(a),sufficient(b)
    pooled=summarize(score(counts@aa)-score(counts@bb),score(aa.sum(axis=0))-score(bb.sum(axis=0)))
    run_points={};per_run={};arrays=[]
    for run in sorted({k[0] for k in left},key=str):
        x=sufficient(flatten({k:v for k,v in left.items() if k[0]==run}))
        matched={k:v for k,v in right.items() if k[0]==run};y=sufficient(flatten(matched or right))
        point=score(x.sum(axis=0))-score(y.sum(axis=0));run_points[str(run)]={name:float(point[i]) for i,name in enumerate(('Accuracy','MacroF1','J'))}
        per_run[str(run)]=summarize(score(counts@x)-score(counts@y),point);arrays.append((x,y))
    # Resample runs plus paired task/machine clusters. Shared fixed baseline
    # draws are reused as a fixed reference, never claimed as new observations.
    samples=[]
    for k in range(n_resamples):
        selected=rng.integers(len(arrays),size=len(arrays));delta=[]
        for i in selected:
            x,y=arrays[i];delta.append(score(counts[k]@x)-score(counts[k]@y))
        samples.append(np.mean(delta,axis=0))
    hierarchical=summarize(np.array(samples),np.mean([score(x.sum(axis=0))-score(y.sum(axis=0)) for x,y in arrays],axis=0))
    return dict(pooled=pooled,independent_run_effects=run_points,run_conditional_CI=per_run,hierarchical_run_cluster_sensitivity=hierarchical,unique_tasks=len(ids),
        inference_unit='machine cluster' if dataset=='PdM' else 'task',no_equivalence_claim=True)

def planned_contrasts():
    pairs=[]
    for g in ('0','1'):
        pairs.extend([(f'source_G{g}',f'S1E0G{g}',f'S0E0G{g}'),(f'evidence_G{g}',f'S1E1G{g}',f'S1E0G{g}')])
    for root in ('S0E0','S1E0','S1E1'):pairs.append((f'guide_{root}',root+'G1',root+'G0'))
    return pairs
