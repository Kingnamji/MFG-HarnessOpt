"""A persistent, fail-closed request budget shared by proposal and report calls.

Construction and offline tests never read credentials or construct an SDK.
Use the same ledger path for sequential dataset runs in one allocation.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import math
import os
from pathlib import Path
import time


class BudgetLimitError(RuntimeError):
    pass


_ACTIVE = ContextVar('manufacturing_api_budget', default=None)


def current_api_budget():
    return _ACTIVE.get()


class RequestBudget:
    INPUT_PRICE = .75
    CACHED_PRICE = .075
    OUTPUT_PRICE = 4.5
    CONTEXT_BOUND = 400000
    MODELS = {'gpt-5.4-mini', 'gpt-5.4-mini-2026-03-17'}

    def __init__(self, path, *, max_calls, max_usd):
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls <= 0:
            raise ValueError('An explicit positive integer API call limit is required')
        if not math.isfinite(max_usd) or max_usd <= 0:
            raise ValueError('An explicit finite positive USD limit is required')
        self.path = Path(path).resolve()
        self.max_calls, self.max_usd = max_calls, float(max_usd)
        self.rows = []
        self.unknown_usage = False
        self.scope = {}
        self._lock = None

    @property
    def spent(self):
        return sum(r.get('cost_usd_uncached') or 0.0 for r in self.rows)

    def snapshot(self):
        return dict(version='shared_request_budget_v1', max_calls=self.max_calls,
                    max_usd=self.max_usd, calls=len(self.rows),
                    spent_usd_uncached=self.spent,
                    spent_usd_actual=sum(r.get('cost_usd_actual') or 0.0 for r in self.rows),
                    unknown_usage=self.unknown_usage, rows=self.rows)

    def persist(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + '.tmp')
        temp.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')
        # Windows readers/virus scanners can briefly deny replacement even
        # though both files are writable. Retry only this local file operation;
        # never dispatch or retry a model request here.
        for attempt in range(20):
            try:
                os.replace(temp, self.path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(min(.025*(attempt+1), .25))

    @contextmanager
    def activate(self):
        if current_api_budget() is not None:
            raise BudgetLimitError('Nested API budgets are not allowed')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_suffix(self.path.suffix + '.lock')
        try:
            self._lock = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise BudgetLimitError('Budget ledger is locked; concurrent use or interrupted run requires review') from None
        token = None
        try:
            if self.path.exists():
                prior = json.loads(self.path.read_text(encoding='utf-8'))
                if (prior.get('version') != 'shared_request_budget_v1'
                        or prior.get('max_calls') != self.max_calls
                        or prior.get('max_usd') != self.max_usd):
                    raise BudgetLimitError('Existing budget allocation differs; never silently reset usage')
                self.rows = prior['rows']
                self.unknown_usage = prior['unknown_usage'] or any(
                    r.get('status') in {'started', 'transport_error', 'missing_usage'}
                    or r.get('cost_usd_uncached') is None for r in self.rows)
            if self.unknown_usage:
                raise BudgetLimitError('Existing request has unknown usage; reconcile it before further calls')
            self.persist()
            token = _ACTIVE.set(self)
            yield self
        finally:
            if token is not None:
                _ACTIVE.reset(token)
            if self._lock is not None:
                os.close(self._lock)
                self._lock = None
                lock.unlink()

    @contextmanager
    def at(self, **scope):
        prior = self.scope
        self.scope = {**prior, **scope}
        try:
            yield
        finally:
            self.scope = prior

    def require_capacity(self, minimum_calls):
        if self.unknown_usage or len(self.rows) + minimum_calls > self.max_calls:
            raise BudgetLimitError('Remaining calls cannot cover the planned initial requests; no client created')

    def create(self, dispatch, **kwargs):
        if kwargs.get('model') not in self.MODELS:
            raise BudgetLimitError('Budget permits only the fixed GPT-5.4-mini model')
        cap = kwargs.get('max_output_tokens')
        if isinstance(cap, bool) or not isinstance(cap, int) or not 0 < cap <= 128000:
            raise BudgetLimitError('Invalid output token bound')
        if self.unknown_usage:
            raise BudgetLimitError('Unknown prior usage; further requests are blocked')
        if len(self.rows) >= self.max_calls:
            raise BudgetLimitError('API call limit reached')
        # Full model-context input plus the requested output bound is a loose
        # upper bound. Cached discounts never make admission more permissive.
        reserve = (self.CONTEXT_BOUND*self.INPUT_PRICE + cap*self.OUTPUT_PRICE)/1e6
        if self.spent + reserve > self.max_usd:
            raise BudgetLimitError('USD admission limit reached before dispatch')
        number = len(self.rows) + 1
        key = {k:self.scope.get(k) for k in ['dataset','method','task_id','candidate_id','iteration']}
        prior_for_report = sum(all(r.get(k) == v for k,v in key.items()) for r in self.rows)
        phase = 'proposer' if self.scope.get('method') == 'Proposer' else ('initial' if prior_for_report == 0 else 'repair')
        row = dict(self.scope, call_number=number, status='started',
                   phase=phase,reserved_usd=reserve, cost_usd_uncached=None)
        self.rows.append(row)
        self.persist()  # A crash after dispatch leaves a blocking pending entry.
        prefix = self.path.parent / (self.path.stem + '_requests') / f'{number:05d}'
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.with_suffix('.request.json').write_text(json.dumps(kwargs, ensure_ascii=False), encoding='utf-8')
        start = time.perf_counter()
        try:
            response = dispatch(**kwargs)
        except Exception as exc:
            self.unknown_usage = True
            row.update(status='transport_error', failure_type=type(exc).__name__,
                       latency_sec=time.perf_counter()-start)
            self.persist()
            raise BudgetLimitError('Transport failure: usage unknown; automatic retry blocked') from None
        self.unknown_usage = True  # Fail closed on any serialization/accounting error below.
        raw = response.model_dump(mode='json') if hasattr(response, 'model_dump') else response
        if not isinstance(raw, dict):
            self.unknown_usage = True
            row['status'] = 'missing_usage'
            self.persist()
            raise BudgetLimitError('Unserializable response; usage unknown')
        prefix.with_suffix('.response.json').write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
        usage = raw.get('usage') or {}
        inp, out = usage.get('input_tokens'), usage.get('output_tokens')
        cached = (usage.get('input_tokens_details') or {}).get('cached_tokens', 0)
        valid = all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in [inp, out, cached])
        if not valid or cached > inp:
            self.unknown_usage = True
            row['status'] = 'missing_usage'
        else:
            self.unknown_usage = False
            row.update(status=raw.get('status', 'completed'),
                       response_id=raw.get('id'), response_model=raw.get('model'),
                       input_tokens=inp, output_tokens=out, cached_input_tokens=cached,
                       reasoning_tokens=(usage.get('output_tokens_details') or {}).get('reasoning_tokens', 0),
                       cost_usd_uncached=(inp*self.INPUT_PRICE+out*self.OUTPUT_PRICE)/1e6,
                       cost_usd_actual=((inp-cached)*self.INPUT_PRICE+cached*self.CACHED_PRICE+out*self.OUTPUT_PRICE)/1e6)
            if inp > self.CONTEXT_BOUND or out > cap:
                self.unknown_usage = True  # Provider accounting exceeded the declared bounds.
        row['latency_sec'] = time.perf_counter()-start
        self.persist()
        if self.unknown_usage:
            raise BudgetLimitError('Response usage missing or outside declared bounds; further requests blocked')
        if raw.get('status', 'completed') != 'completed':
            raise BudgetLimitError('Non-completed response: usage retained and automatic retry blocked')
        return response
