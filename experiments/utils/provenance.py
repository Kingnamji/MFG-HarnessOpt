"""Immutable digests for revised runs; credentials are never included."""
from functools import lru_cache
import hashlib
from pathlib import Path
from experiments.data.generation_evidence import PROTOCOL_VERSION
from experiments.evaluation.metric_policy import EVALUATION_POLICY_VERSION
from experiments.llm.report_contract import REPORT_CONTRACT_VERSION
from experiments.evaluation.runtime_verifier import RUNTIME_VERIFIER_VERSION
from experiments.optimization.selection_metrics import OBJECTIVE_VERSION


@lru_cache(maxsize=1)
def runtime_provenance():
    code = Path(__file__).resolve().parents[2]
    files = [*code.glob('experiments/**/*.py'), *code.glob('experiments/configs/**/*.yaml'), *code.glob('experiments/configs/**/*.json'),
             *code.glob('preprocessing_*.ipynb'), *code.glob('Dataset/Preprocessed/**/*.csv'),
             *code.glob('Dataset/Preprocessed/**/*.json')]
    hashes = {p.relative_to(code).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    joined = '\n'.join(f'{name}:{digest}' for name, digest in hashes.items())
    return {'protocol_version': PROTOCOL_VERSION, 'evaluation_policy_version': EVALUATION_POLICY_VERSION,
            'objective_version': OBJECTIVE_VERSION,
            'report_contract_version': REPORT_CONTRACT_VERSION,
            'runtime_verifier_version': RUNTIME_VERIFIER_VERSION,
            'code_data_config_sha256': hashlib.sha256(joined.encode()).hexdigest(), 'files': hashes}
