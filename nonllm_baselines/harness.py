"""Adapter returning the existing HarnessResult without editing M3/M4 runners."""
import hashlib
from pathlib import Path
import time
from experiments.harness.base import HarnessResult
from experiments.utils.provenance import runtime_provenance
from .views import build_view
from .reports import render_report


class NonLLMReportHarness:
    def __init__(self, predictor, *, method, matched_view='M3', frozen_patch=None, training_class_counts=None):
        self.predictor=predictor
        self.method=method
        self.matched_view=matched_view
        self.frozen_patch=frozen_patch
        self.training_class_counts=training_class_counts

    def run(self, *, task, generation_evidence, allowed_schema, action_ontology,
            action_preconditions=None, generic_action_rules=None, association_guide=None):
        started=time.perf_counter()
        view=build_view(task=task,generation_evidence=generation_evidence,allowed_schema=allowed_schema,
            method=self.matched_view,patch=self.frozen_patch,association_guide=association_guide)
        from .supervised import TabularClassifier
        if isinstance(self.predictor,TabularClassifier):
            prediction=self.predictor.predict(view,training_class_counts=self.training_class_counts)
        else:
            prediction=self.predictor.predict(view)
        output=render_report(view=view,prediction=prediction,allowed_schema=allowed_schema,
            action_ontology=action_ontology,action_preconditions=action_preconditions,generic_action_rules=generic_action_rules)
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
        output.update(protocol=runtime_provenance(),implementation_files=files,
            matched_view=self.matched_view,view_feature_count=len(__import__(
                'nonllm_baselines.views',fromlist=['feature_dict']).feature_dict(view)))
        return HarnessResult(method=self.method,harness_id=self.method+'_'+self.matched_view+'_common_v1',
            task_id=str(view.task['task_id']),dataset=str(view.task.get('dataset','')),
            report=output['report'],accessed_sources=sorted({r['source'] for r in view.records}),
            latency_sec=time.perf_counter()-started,model_name='nonLLM_'+self.method,
            raw_response=output['raw_report'],extra=output)
