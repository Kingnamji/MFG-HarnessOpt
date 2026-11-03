"""Paper objective, shared by search, validation and acceptance.

Generic-recommendation behavior is reported/screened separately; the manuscript
does not include it as an additional term in J_mfg.
"""
from experiments.evaluation.metric_policy import selection_grounding_scores

OBJECTIVE_VERSION = 'paper_j_v1_no_generic'


def load_metric_weights(configs):
    w=configs.metric_weights.get('primary_selection',{})
    return {key:float(w.get(key,default)) for key,default in {
        'alpha_label':1,'beta_evidence':1,'gamma_action':1,'lambda_unsupported':1,
        'lambda_unsafe':1,'mu_cost':1,'rho_repair_failure':1}.items()}


def normalized_cost(cost,configs):
    n=configs.metric_weights.get('normalization',{}).get('cost',{})
    ref=float(n.get('reference_cost_per_report',.005))
    return max(float(n.get('clip_min',0)),min(float(n.get('clip_max',1)),cost/ref)) if ref>0 else cost


def compute_j_mfg_for_group(group,configs):
    w=load_metric_weights(configs)
    e,a=selection_grounding_scores(group)
    return float(w['alpha_label']*group.label_correct.mean()+w['beta_evidence']*e+w['gamma_action']*a
        -w['lambda_unsupported']*group.unsupported_claim_rate.mean()
        -w['lambda_unsafe']*group.unsafe_recommendation_rate.mean()
        -w['mu_cost']*normalized_cost(group.cost_per_report.mean(),configs)
        -w['rho_repair_failure']*(group.F10.mean() if 'F10' in group else 0))
