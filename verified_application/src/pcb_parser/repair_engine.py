"""Deterministic, transactional execution of independent PCB repair rules.

Rules only propose edits against a frozen baseline. The engine arbitrates
conflicts and validates permissions before committing each complete proposal.
No model, OCR, reference labels, file I/O or page-specific knowledge lives here.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Protocol


def members(net):
    return {s.strip() for s in str(net.get('hyperGraph') or '').strip('()').split(',') if s.strip()}


def _read(target, path):
    item = target
    for key in path:
        if not isinstance(item, dict) or key not in item:
            return False, None
        item = item[key]
    return True, item


@dataclass(frozen=True)
class Edit:
    path: tuple
    before_exists: bool
    before: object
    after: object = None
    delete: bool = False

    @classmethod
    def set(cls, target, path, value):
        exists, before = _read(target, path)
        return cls(tuple(path), exists, deepcopy(before), deepcopy(value))

    @classmethod
    def remove(cls, target, path):
        exists, before = _read(target, path)
        return cls(tuple(path), exists, deepcopy(before), delete=True)


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    version: str
    description: str
    capabilities: tuple
    enabled_by_default: bool = True
    maturity: str = 'validated'


@dataclass
class RepairContext:
    target: dict
    evidence: dict
    width: int
    height: int


@dataclass(frozen=True)
class Proposal:
    rule_id: str
    edits: tuple
    evidence: dict = field(default_factory=dict)
    reason: str = ''
    rejection: str = ''

    @property
    def proposal_id(self):
        payload = {'rule': self.rule_id, 'edits': [e.__dict__ for e in self.edits],
                   'evidence': self.evidence, 'reason': self.reason, 'rejection': self.rejection}
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
        return self.rule_id + ':' + hashlib.sha256(raw.encode()).hexdigest()[:20]


class RepairRule(Protocol):
    spec: RuleSpec

    def propose(self, context: RepairContext):
        """Yield proposals; do not mutate the target or other rules' candidates."""
        ...


SAFE_CAPABILITIES = {'add_memberships', 'add_pins', 'add_networks', 'remove_empty_networks'}
NC_NAMES = {'NC', 'N.C.', 'N/C', 'DNC'}


def _overlap(left, right):
    length = min(len(left), len(right))
    return left[:length] == right[:length]


def _apply(target, edits):
    out = deepcopy(target)
    for edit in edits:
        exists, current = _read(out, edit.path)
        if exists != edit.before_exists or current != edit.before:
            raise ValueError('stale_precondition')
        node = out
        for key in edit.path[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError('non_mapping_parent')
        if edit.delete:
            if not exists:
                raise ValueError('delete_missing_value')
            del node[edit.path[-1]]
        else:
            node[edit.path[-1]] = deepcopy(edit.after)
    return out


def _required_capability(edit):
    path = edit.path
    if len(path) == 3 and path[0] == 'pins' and not edit.before_exists and not edit.delete:
        return 'add_pins'
    if len(path) == 2 and path[0] == 'nets':
        if not edit.before_exists and not edit.delete:
            return 'add_networks'
        if edit.delete and edit.before_exists and not edit.before.get('edges'):
            return 'remove_empty_networks'
    if len(path) == 3 and path[0] == 'nets' and path[2] == 'hyperGraph' and not edit.delete:
        old = members({'hyperGraph': edit.before})
        new = members({'hyperGraph': edit.after})
        if edit.before_exists and old <= new:
            return 'add_memberships'
    raise ValueError('unsupported_edit_requires_new_validator')


def _assignments(target):
    return {(ref, nid) for nid, net in target['nets'].items() for ref in members(net)}


def _validate_changes(before, after, context):
    """Validate new references/geometry; legacy observations remain untouched."""
    added = _assignments(after) - _assignments(before)
    assignments_after = _assignments(after)
    for ref, nid in added:
        try:
            owner, number = ref.rsplit('.', 1)
            pin = after['pins'][owner]['pin_' + number]
            after['components'][owner]
        except (ValueError, KeyError):
            raise ValueError('dangling_member') from None
        if str(pin.get('pinname', '')).strip().upper() in NC_NAMES:
            raise ValueError('no_connect_member')
        if any(r == ref and n != nid for r, n in assignments_after):
            raise ValueError('member_already_on_another_network')
    def point_valid(point):
        return (isinstance(point, dict) and all(
            isinstance(point.get(k), (int, float)) and not isinstance(point[k], bool)
            and math.isfinite(point[k]) and 0 <= point[k] < bound
            for k, bound in [('x', context.width), ('y', context.height)]))
    for owner, pins in after['pins'].items():
        for key, pin in pins.items():
            if key not in before['pins'].get(owner, {}):
                if owner not in after['components'] or not key.startswith('pin_') or not key[4:]:
                    raise ValueError('invalid_new_pin_identity')
                if not point_valid(pin.get('point')):
                    raise ValueError('invalid_new_pin_point')
    for nid, net in after['nets'].items():
        if nid in before['nets']:
            continue
        if not isinstance(net, dict) or not members(net) or not net.get('edges'):
            raise ValueError('empty_new_network')
        for edge in net['edges'].values():
            if not isinstance(edge, list) or len(edge) < 2 or not all(point_valid(p) for p in edge):
                raise ValueError('invalid_new_wire_geometry')


class RepairPipeline:
    """A rule registry and conservative all-party conflict arbitration.

    Contradicting proposals are all withheld, regardless of registration order.
    Each accepted proposal commits its pin/net edits together, or not at all.
    """
    def __init__(self, rules, *, extension=None):
        self.rules = tuple(rules)
        self.extension = extension
        ids = [r.spec.rule_id for r in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate_rule_id')
        allowed = SAFE_CAPABILITIES | (set(extension.capabilities) if extension else set())
        if any(not r.spec.capabilities or not set(r.spec.capabilities) <= allowed for r in self.rules):
            raise ValueError('unvalidated_rule_capabilities')

    def run(self, context, *, rule_settings=None, maximum_new_contacts=64, maximum_new_branches=16):
        limits = {'contacts': maximum_new_contacts, 'branches': maximum_new_branches}
        if any(type(v) is not int or v < 0 for v in limits.values()):
            raise ValueError('budgets_must_be_nonnegative_integers')
        settings = {} if rule_settings is None else rule_settings
        if not isinstance(settings, dict) or set(settings) - {r.spec.rule_id for r in self.rules}:
            raise ValueError('unknown_rule_configuration')
        baseline = deepcopy(context.target)
        audit = {'policy': 'structure_repair_pipeline_v1', 'rules': [], 'proposals': [],
                 'accepted': [], 'withheld': [], 'errors': [], 'contact_count': 0, 'branch_count': 0}
        candidates = []
        for rule in sorted(self.rules, key=lambda r: r.spec.rule_id):
            config = settings.get(rule.spec.rule_id, {})
            if not isinstance(config, dict) or set(config) - {'enabled'}:
                raise ValueError('invalid_rule_configuration:' + rule.spec.rule_id)
            enabled = config.get('enabled', rule.spec.enabled_by_default)
            if type(enabled) is not bool:
                raise ValueError('rule_enabled_must_be_boolean')
            spec = deepcopy(rule.spec.__dict__)
            spec['enabled'] = enabled
            audit['rules'].append(spec)
            if not enabled:
                continue
            # Collect the complete rule before admitting any of its proposals.
            # A faulty generator cannot commit partial output or change baseline.
            try:
                local = RepairContext(deepcopy(baseline), deepcopy(context.evidence), context.width, context.height)
                generated = deepcopy(list(rule.propose(local)))
                if local.target != baseline:
                    raise ValueError('rule_mutated_target')
                for proposal in generated:
                    if proposal.rule_id != rule.spec.rule_id:
                        raise ValueError('proposal_rule_identity_mismatch')
                    proposal.proposal_id  # Validate serializable, finite audit evidence.
                candidates.extend((rule.spec, p) for p in generated)
            except Exception as error:
                audit['errors'].append({'rule_id': rule.spec.rule_id,
                                        'reason': f'{type(error).__name__}: {error}'})
        prepared = {}
        for spec, proposal in candidates:
            key = proposal.proposal_id
            if key in prepared:
                continue
            row = {'proposal_id': key, 'rule_id': spec.rule_id, 'version': spec.version,
                   'reason': proposal.reason, 'evidence': deepcopy(proposal.evidence),
                   'edits': [deepcopy(e.__dict__) for e in proposal.edits],
                   'status': 'eligible', 'rejection': proposal.rejection, 'conflicts': []}
            prepared[key] = (proposal, row)
            try:
                if proposal.rejection:
                    raise ValueError(proposal.rejection)
                if not proposal.edits:
                    raise ValueError('no_edits')
                paths = [e.path for e in proposal.edits]
                if any(not p or any(not isinstance(k, str) or not k for k in p) for p in paths):
                    raise ValueError('invalid_edit_path')
                if any(_overlap(a, b) for i, a in enumerate(paths) for b in paths[i+1:]):
                    raise ValueError('overlapping_edits_in_proposal')
                candidate = _apply(baseline, proposal.edits)
                if candidate == baseline:
                    raise ValueError('no_effect')
                for edit in proposal.edits:
                    try:
                        capability = _required_capability(edit)
                    except ValueError:
                        if self.extension is None:raise
                        capability = self.extension.required_capability(edit)
                    if capability not in spec.capabilities:raise ValueError('rule_capability_denied')
                _validate_changes(baseline, candidate, context)
                if self.extension is not None:
                    self.extension.validate(baseline, candidate, context, proposal)
                row['_claims'] = _assignments(candidate) - _assignments(baseline)
            except Exception as error:
                row.update(status='withheld', rejection=str(error))
        eligible = [(p, r) for p, r in prepared.values() if r['status'] == 'eligible']
        for i, (left, lrow) in enumerate(eligible):
            for right, rrow in eligible[i+1:]:
                overlap = any(_overlap(a.path, b.path) for a in left.edits for b in right.edits)
                double_assignment = any(a == b and na != nb for a, na in lrow['_claims'] for b, nb in rrow['_claims'])
                if overlap or double_assignment:
                    lrow['conflicts'].append(rrow['proposal_id'])
                    rrow['conflicts'].append(lrow['proposal_id'])
        out = deepcopy(baseline)
        for key in sorted(prepared):
            proposal, row = prepared[key]
            row.pop('_claims', None)
            row['conflicts'].sort()
            if row['status'] == 'eligible':
                try:
                    if row['conflicts']:
                        raise ValueError('conflicting_proposals')
                    candidate = _apply(out, proposal.edits)
                    _validate_changes(out, candidate, context)
                    if self.extension is not None:
                        self.extension.validate(out, candidate, context, proposal)
                    contacts = len(_assignments(candidate) - _assignments(out))
                    branches = len(set(candidate['nets']) - set(out['nets']))
                    if audit['contact_count'] + contacts > maximum_new_contacts:
                        raise ValueError('contact_budget')
                    if audit['branch_count'] + branches > maximum_new_branches:
                        raise ValueError('branch_budget')
                    out = candidate
                    audit['contact_count'] += contacts
                    audit['branch_count'] += branches
                    row['status'] = 'accepted'
                except Exception as error:
                    row.update(status='withheld', rejection=str(error))
            audit['proposals'].append(row)
            audit['accepted' if row['status'] == 'accepted' else 'withheld'].append(key)
        audit['status'] = 'applied' if audit['accepted'] else 'no_qualified_proposal'
        return out, audit
