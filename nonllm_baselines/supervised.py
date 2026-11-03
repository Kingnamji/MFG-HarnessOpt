"""Search-only fitting/CV interfaces. Importing does not fit or run experiments.

Each CV held fold receives a prior fitted solely on the other folds. Numeric
imputation and categorical encoding are also learned inside that fold. No
target-dependent prior features are used on a sample that trained its prior:
inner group cross-fitting supplies training features, then an outer-train prior
supplies held-fold features. Outer validation/test inference uses the frozen
full-search prior, not a newly fitted guide. Raw-source data are never loaded.
"""
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import product
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from .views import feature_dict, dump, visible_associations
from .predictors import Prediction, AzurePriorOnly, pattern_key

SEED = 20261023


def parameter_grid(kind):
    if kind == 'DT':
        return [dict(max_depth=d, min_samples_leaf=n) for d,n in product((2,4,6,None),(1,3,5))]
    if kind == 'GBT':
        return [dict(max_leaf_nodes=n, learning_rate=r, l2_regularization=l)
                for n,r,l in product((7,15,31),(.05,.1),(0,1))]
    raise ValueError('DT or GBT required')


def validate_training(views, y, groups):
    if not views or len(views)!=len(y) or len(y)!=len(groups):
        raise ValueError('Training views, labels and groups must align')
    if any(v.task.get('split') != 'search' for v in views):
        raise ValueError('Only search split can train models/priors')
    from comparison_protocol.protocol import assert_training_views
    assert_training_views(views, purpose='classifier/CV training')
    if len({v.task['task_id'] for v in views}) != len(views):
        raise ValueError('Duplicate training task')
    if len({tuple(v.labels) for v in views}) != 1 or len({v.data['method'] for v in views}) != 1:
        raise ValueError('A fitted model uses one dataset and one matched view')
    if set(y)-set(views[0].labels):
        raise ValueError('Invalid training labels')
    # Require the actual grouping key, not an arbitrary row index for Azure.
    key = 'machineID' if 'TWF' not in views[0].labels else 'udi'
    for v,g in zip(views,groups):
        if key not in v.task or str(v.task[key]) != str(g):
            raise ValueError('Group must match the observable '+key)


def fit_prior(views, y, *, min_count=5):
    """Pure in-memory aggregation. Only visible rows of supplied TRAIN views.

    Replicates the frozen guide's count/rank policy without touching its file.
    Repeated patterns count at most once per task. Maintenance can occupy a
    printed guide row but is excluded from diagnostic scoring.
    """
    from comparison_protocol.protocol import assert_training_views
    assert_training_views(views, purpose='prior fitting')
    if len(views)!=len(y): raise ValueError('Prior labels must align with search views')
    counts = defaultdict(Counter)
    labels = views[0].labels
    for view,label in zip(views,y):
        keys = {pattern_key(r) or (f'maintenance:comp:{r["value"]}' if r['source']=='maintenance' else None) for r in view.records} - {None}
        for key in sorted(keys):
            counts[key][label] += 1
    groups = defaultdict(list)
    for key,c in sorted(counts.items()):
        n = sum(c.values())
        if n < min_count:
            continue
        dist = {k:round(c.get(k,0)/n,4) for k in labels}
        top = max(labels, key=lambda k:dist[k])
        ranked = sorted(dist.values(), reverse=True)
        confidence = 'high' if ranked[0]>=.6 and ranked[0]-ranked[1]>=.2 else 'medium' if ranked[0]>=.45 and ranked[0]-ranked[1]>=.1 else 'low'
        entropy = -sum(p*np.log2(p) for p in dist.values() if p>0)
        groups[key.split(':')[0]].append(dict(evidence_key=key, component_distribution=dist,
            n=n, top_component=top, top_probability=dist[top], confidence=confidence, entropy=entropy))
    items = []
    for source in ('errors','telemetry','maintenance'):
        items.extend(sorted(groups[source], key=lambda r:(r['top_probability'],r['n'],-r['entropy']),reverse=True)[:8])
    items.sort(key=lambda r:(r['confidence']=='high',r['top_probability'],r['n']),reverse=True)
    return dict(dataset='Azure PdM', top_associations=items,
        leakage_control=dict(uses_split='search',uses_validation_labels=False,uses_test_labels=False)), dict(Counter(y))


def with_prior(view, guide):
    data = view.data
    data['associations'] = visible_associations(guide)
    # Text is unused by features; keep it honest for the derived view audit.
    from experiments.llm.prompts import format_azure_association_guide
    data['association_text'] = format_azure_association_guide(guide)
    return replace(view,payload=dump(data))


def independent_group_folds(views, y, groups, n_splits):
    validate_training(views,y,groups)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    assignments = np.empty(len(views),dtype=int)
    for fold,(_,held) in enumerate(splitter.split(np.zeros(len(y)),y,groups)):
        assignments[held] = fold
    return assignments


def crossfit_prior_features(views, y, groups):
    validate_training(views,y,groups)
    n_splits = min(3,len(set(groups)))
    if n_splits < 2:
        raise ValueError('At least two training groups required for prior cross-fitting')
    folds = independent_group_folds(views,y,groups,n_splits)
    features = [None]*len(views)
    audit = []
    for fold in sorted(set(folds)):
        train = np.flatnonzero(folds!=fold).tolist()
        held = np.flatnonzero(folds==fold).tolist()
        guide,counts = fit_prior([views[i] for i in train],[y[i] for i in train])
        prior = AzurePriorOnly(counts)
        for i in held:
            clean = with_prior(views[i],guide)
            features[i] = feature_dict(clean, prior_prediction=prior.predict(clean))
        audit.append(dict(train_ids=[views[i].task['task_id'] for i in train],
            held_ids=[views[i].task['task_id'] for i in held],
            group_overlap=sorted(set(groups[i] for i in train)&set(groups[i] for i in held))))
    return features,audit


class TabularClassifier:
    def __init__(self, kind, params=None, *, use_prior_features=True, confidence_policy=None):
        self.kind, self.params = kind, dict(params or parameter_grid(kind)[0])
        if self.params not in parameter_grid(kind):
            raise ValueError('Parameters outside the prespecified 12-setting grid')
        self.use_prior_features = use_prior_features
        # Set from search OOF only; raw predict_proba is not a calibrated risk.
        self.confidence_policy = confidence_policy

    def fit(self, views, y, *, groups):
        views,y,groups = list(views),list(y),list(groups)
        validate_training(views,y,groups)
        started = time.perf_counter()
        self.labels = views[0].labels
        self.training_class_counts = dict(Counter(y))
        self.view_method = views[0].data['method']
        self.azure = 'TWF' not in self.labels
        self.crossfit_audit = []
        if self.azure and self.use_prior_features:
            rows,self.crossfit_audit = crossfit_prior_features(views,y,groups)
        else:
            rows = [feature_dict(v) for v in views]
        self.vectorizer = DictVectorizer(sparse=False)
        encoded = self.vectorizer.fit_transform(rows)
        self.imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
        matrix = self.imputer.fit_transform(encoded)
        if self.kind == 'DT':
            self.model = DecisionTreeClassifier(criterion='gini', splitter='best', random_state=SEED, **self.params)
        else:
            self.model = HistGradientBoostingClassifier(max_iter=200, min_samples_leaf=5,
                early_stopping=False, categorical_features=None, random_state=SEED, **self.params)
        self.model.fit(matrix,y)
        self.train_ids = [v.task['task_id'] for v in views]
        self.fit_seconds = time.perf_counter()-started
        self.n_features = matrix.shape[1]
        return self

    def predict(self, view, *, training_class_counts=None):
        if view.labels != self.labels or view.data['method'] != self.view_method:
            raise ValueError('Inference must match fitted label set and visible view')
        # For outer validation/test, this uses the unchanged frozen full-search
        # guide already in the view. CV passes a fold-specific replacement.
        prior = AzurePriorOnly(self.training_class_counts if training_class_counts is None else training_class_counts).predict(view) if self.azure and self.use_prior_features else None
        rows = feature_dict(view,prior_prediction=prior)
        matrix = self.imputer.transform(self.vectorizer.transform([rows]))
        probs = self.model.predict_proba(matrix)[0]
        label = str(self.model.classes_[np.argmax(probs)])
        confidence = 'low'
        if self.confidence_policy:
            if self.confidence_policy.get('source') != 'search_oof':
                raise ValueError('Confidence thresholds must come from search OOF')
            p = float(max(probs))
            if p >= self.confidence_policy.get('high_threshold',float('inf')):
                confidence='high'
            elif p >= self.confidence_policy.get('medium_threshold',float('inf')):
                confidence='medium'
        explanation=[]
        if self.kind == 'DT':
            names = self.imputer.get_feature_names_out(self.vectorizer.get_feature_names_out())
            for node in self.model.decision_path(matrix).indices:
                feature = self.model.tree_.feature[node]
                if feature >= 0:
                    explanation.append(dict(feature=str(names[feature]), value=float(matrix[0,feature]),
                        threshold=float(self.model.tree_.threshold[node]),
                        branch='left' if matrix[0,feature]<=self.model.tree_.threshold[node] else 'right'))
        else:
            explanation = [dict(type='ensemble', note='HistGBT prediction is a statistical class score. No causal sensor attribution is inferred.')]
        confidence_note = ('Ordinal confidence uses thresholds selected from search OOF reliability; this is not probability calibration.'
                           if self.confidence_policy else 'Search OOF confidence thresholds have not been fitted; confidence remains low.')
        return Prediction(label,confidence,dict(zip(map(str,self.model.classes_),map(float,probs))),
            ['Classifier scores are not calibrated failure probabilities. '+confidence_note,
             'Tree/ensemble prediction describes statistical discrimination, not physical causation.'],
            dict(method=self.kind, parameters=self.params, fit_seconds=self.fit_seconds,
                training_ids=self.train_ids, n_features=self.n_features, explanation=explanation,
                prior_crossfit=self.crossfit_audit, confidence_policy=self.confidence_policy))


def cross_validate(kind, views, y, *, groups, held_folds, use_prior_features=True):
    """Explicit runner, no automatic execution. Consume the frozen fold manifest.

    Select macro-F1, then accuracy, then simpler settings. All candidate fits
    use exactly the same folds. Both inner prior fitting and outer transforms
    exclude held labels/groups. No validation/test view can enter this call.
    """
    views,y,groups = list(views),list(y),list(groups)
    validate_training(views,y,groups)
    folds = np.asarray(held_folds)
    if len(folds)!=len(views) or len(set(folds))<2:
        raise ValueError('Invalid held-fold manifest')
    for g in set(groups):
        if len(set(folds[i] for i in range(len(groups)) if groups[i]==g))!=1:
            raise ValueError('Equipment/UDI appears in multiple held folds')
    records=[]
    cv_started=time.perf_counter()
    for order,params in enumerate(parameter_grid(kind)):
        oof=[None]*len(y); probabilities=[None]*len(y); metrics=[]; audits=[]
        oof_scores=[None]*len(y);prediction_seconds=[None]*len(y);explanations=[None]*len(y)
        for fold in sorted(set(folds)):
            train=np.flatnonzero(folds!=fold).tolist(); held=np.flatnonzero(folds==fold).tolist()
            if set(y[i] for i in train)!=set(views[0].labels) or set(y[i] for i in held)!=set(views[0].labels):
                raise ValueError('Every CV train and held fold must contain all candidate classes')
            model=TabularClassifier(kind,params,use_prior_features=use_prior_features).fit(
                [views[i] for i in train],[y[i] for i in train],groups=[groups[i] for i in train])
            guide,counts=fit_prior([views[i] for i in train],[y[i] for i in train]) if model.azure else (None,None)
            for i in held:
                view=with_prior(views[i],guide) if model.azure and use_prior_features else views[i]
                prediction_started=time.perf_counter()
                pred=model.predict(view,training_class_counts=counts)
                prediction_seconds[i]=time.perf_counter()-prediction_started
                oof[i]=pred.label; probabilities[i]=max(pred.scores.values())
                oof_scores[i]=pred.scores;explanations[i]=pred.audit['explanation']
            metrics.append(dict(macro_f1=float(f1_score([y[i] for i in held],[oof[i] for i in held],labels=views[0].labels,average='macro',zero_division=0)),
                accuracy=float(accuracy_score([y[i] for i in held],[oof[i] for i in held]))))
            audits.append(dict(fold=int(fold), train_ids=model.train_ids,
                held_ids=[views[i].task['task_id'] for i in held], prior_crossfit=model.crossfit_audit,
                fold_prior_guide=guide,fold_class_counts=counts,fit_seconds=model.fit_seconds))
        complexity = ((params['max_depth'] or float('inf')),-params['min_samples_leaf']) if kind=='DT' else (params['max_leaf_nodes'],-params['l2_regularization'],params['learning_rate'])
        records.append(dict(params=params, mean_macro_f1=float(np.mean([m['macro_f1'] for m in metrics])),
            mean_accuracy=float(np.mean([m['accuracy'] for m in metrics])),complexity=complexity,
            order=order, oof=oof, max_probabilities=probabilities, oof_scores=oof_scores,
            prediction_seconds=prediction_seconds,explanations=explanations,folds=metrics,audits=audits))
    records.sort(key=lambda r:(-r['mean_macro_f1'],-r['mean_accuracy'],r['complexity'],r['order']))
    best=records[0]
    policy=dict(source='search_oof', high_threshold=1.01, medium_threshold=1.01)
    # This is an ordinal reliability rule, not probability calibration.
    for name,precision in [('high_threshold',.8),('medium_threshold',.6)]:
        for threshold in (.5,.6,.7,.8,.9):
            ids=[i for i,p in enumerate(best['max_probabilities']) if p>=threshold]
            if len(ids)>=10 and np.mean([best['oof'][i]==y[i] for i in ids])>=precision:
                policy[name]=threshold
                break
    return dict(selected_parameters=best['params'],confidence_policy=policy,
        candidates=records,cv_seconds=time.perf_counter()-cv_started,
        selection='mean fold macro-F1, accuracy, simpler model; search only')
