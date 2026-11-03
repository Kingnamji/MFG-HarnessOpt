"""Narrow, deterministic checks for explicit executable contract conflicts.

This rejects instructions, never rewrites a proposer patch. It deliberately
does not claim to prove arbitrary natural-language patch correctness.
"""
import re


def patch_contract_violations(patch):
    def strings(value, path):
        if isinstance(value, str):
            yield path, value
        elif isinstance(value, dict):
            for k, v in value.items(): yield from strings(v, path+'.'+k)
        elif isinstance(value, list):
            for i, v in enumerate(value): yield from strings(v, f'{path}[{i}]')

    consumed = {
        'source_policy': patch.get('source_policy', {}).get('proposed_changes', []),
        'evidence_table_format': patch.get('evidence_table_format', {}).get('formatting_rules', []),
        'prompt_instructions': patch.get('prompt_instructions', {}),
        'action_filter_rules': patch.get('action_filter_rules', {}).get('rules', []),
        'repair_prompt': patch.get('repair_prompt', {}),
    }
    issues = []
    for path, text in strings(consumed, 'patch'):
        for clause in re.split(r'[;\n]|(?<=[.!?])\s+(?=[A-Z])', text):
            s = clause.lower().replace('`', '').strip()
            output_scope = bool(re.search(r'\b(?:output|report|response)\b', s))
            input_scope = bool(re.search(r'\b(?:input|hidden|source record|observation)\b', s))
            code = None
            if re.search(r'\bsupports\s+(?:is|are)\s+(?:not (?:part of|required|allowed)|forbidden|optional)', s) and output_scope:
                code = 'output_supports_contract_conflict'
            if re.search(r'\b(?:never|do not|must not|should not|cannot)\s+(?:place|put|emit|use|include|write)\b.*\b(?:candidate |diagnostic )?labels?\b.*\bsupports\b', s) and not input_scope:
                code = 'output_supports_contract_conflict'
            deletion = re.search(r'\b(?:omit|remove|exclude|delete)\s+(?:the\s+)?(?:required\s+)?(?:supports|predicted_label|confidence|evidence|limitations|next_steps)\b', s)
            if deletion and output_scope and not re.search(r'\b(?:do not|must not|never|cannot)\s*$', s[:deletion.start()]):
                code = 'required_output_field_removed'
            if output_scope and re.search(r'\b(?:do not|must not|never)\s+(?:include|emit|output|return)\s+(?:the\s+)?supports\b', s):
                code = 'output_supports_contract_conflict'
            # A positive request to require hidden confirmation changes the
            # public sufficient conditions. Explicit prohibitions are allowed.
            demand = re.search(r'\b(?:require|demand)\s+(?:an?\s+)?(?:explicit\s+|confirmed\s+)?(?:failure flag|gold label)', s)
            if demand and not re.search(r'\b(?:do not|must not|never|cannot|should not)\s*$', s[:demand.start()]):
                code = 'hidden_confirmation_required'
            disable = re.search(r'\b(?:disable|ignore|bypass|override)\s+(?:the\s+)?(?:public[ -]condition|diagnostic consistency|label consistency)', s)
            if disable and not re.search(r'\b(?:do not|must not|never|cannot|should not)\s*$', s[:disable.start()]):
                code = 'public_condition_check_disabled'
            threshold = re.search(r'\b(?:change|replace|set|raise|lower|override|modify)\s+(?:the\s+)?(?:pwf|hdf|osf|twf)\s+(?:public\s+)?(?:threshold|cutoff|rule)', s)
            if threshold and not re.search(r'\b(?:do not|must not|never|cannot|should not)\s*$', s[:threshold.start()]):
                code = 'public_threshold_modified'
            # Explicit stochastic-to-deterministic conversion is a contract
            # conflict. Negated instructions and ordinary risk hypotheses pass.
            command = re.search(r'\b(?:always|automatically|unconditionally)\s+(?:predict|choose|select|output|assign)\s+(?:the\s+)?(?:label\s+)?(?:twf|no failure)\b', s)
            if command and re.search(r'\b(?:wear|twf)\b', s) and not re.search(r'\b(?:do not|must not|never|cannot|should not)\s*$', s[:command.start()]):
                code = 'probabilistic_twf_forced_label'
            proof = re.search(r'\b(?:treat|use)\s+(?:high |elevated )?(?:tool )?wear\s+(?:alone\s+)?as\s+(?:a\s+)?(?:deterministic|conclusive)\s+proof\s+of\s+twf', s)
            if proof and not re.search(r'\b(?:do not|must not|never|cannot|should not)\s*$', s[:proof.start()]):
                code = 'probabilistic_twf_forced_label'
            if code:
                issues.append(dict(code=code, location=path,
                    message=f'{code}: {clause.strip()}'))
    return issues
