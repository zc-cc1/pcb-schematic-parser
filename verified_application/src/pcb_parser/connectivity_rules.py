"""Validated structural patterns, expressed as proposals rather than mutations."""
from copy import deepcopy

from .connectivity_evidence import _members, _vertex
from .repair_engine import Edit, Proposal, RuleSpec


def _new_pin_edits(target, contacts):
    edits = []
    for contact in contacts:
        if contact['new_pin']:
            owner, number = contact['member'].rsplit('.', 1)
            edits.append(Edit.set(target, ('pins', owner, 'pin_' + number), contact['pin']))
    return edits


def _contact_evidence(data, refs):
    return [deepcopy(p) for p in data['audit']['proposals'] if p['member'] in refs]


class KnownContactMembershipRule:
    spec = RuleSpec('known_contact_membership', '1.0',
                    '将有唯一导线及原图证据的已知接点补入既有网络',
                    ('add_memberships', 'add_pins'))

    def propose(self, context):
        data, target = context.evidence, context.target
        for pid, selected in sorted(data['grouped'].items()):
            nid = data['destinations'][pid]
            if nid not in target['nets']:
                continue
            refs = {c['member'] for c in selected}
            before = _members(target['nets'][nid])
            value = '(' + ','.join(sorted(before | refs)) + ')'
            change = {'kind': 'attach_existing', 'network_id': nid, 'physical_id': pid,
                      'added': sorted(refs - before)}
            edits = [Edit.set(target, ('nets', nid, 'hyperGraph'), value)]
            edits.extend(_new_pin_edits(target, selected))
            yield Proposal(self.spec.rule_id, tuple(edits),
                           {'physical_id': pid, 'members': sorted(refs), 'change': change,
                            'contacts': _contact_evidence(data, refs)},
                           'unique_model_wire_and_continuous_image_support')


class ConfirmedBranchRule:
    spec = RuleSpec('confirmed_physical_branch', '1.0',
                    '恢复具有两个不同电气顶点且交叉关系明确的模型支路',
                    ('add_networks', 'add_pins'))

    def propose(self, context):
        data, target = context.evidence, context.target
        for pid, selected in sorted(data['grouped'].items()):
            nid = data['destinations'][pid]
            if nid in target['nets']:
                continue
            refs = {c['member'] for c in selected}
            net = data['physical'][pid]
            evidence = {'physical_id': pid, 'members': sorted(refs),
                        'contacts': _contact_evidence(data, refs)}
            reason = ''
            if net.get('unresolved_junction_ids'):
                reason = 'unresolved_crossings'
            elif len(refs) < 2 or len({_vertex(ref, target['components']) for ref in refs}) < 2:
                reason = 'incomplete_branch_without_second_electrical_vertex'
            if reason:
                yield Proposal(self.spec.rule_id, (), evidence, rejection=reason)
                continue
            wire = {}
            for a, b, eid in data['segments'][pid]:
                wire['edge_' + str(len(wire) + 1)] = [
                    {'x': float(p[0]), 'y': float(context.height - p[1])} for p in (a, b)]
            branch = {'hyperGraph': '(' + ','.join(sorted(refs)) + ')', 'edges': wire}
            if data['cleanup'] is not None:
                data['cleanup']({'nets': {nid: branch}})
            if not branch['edges']:
                yield Proposal(self.spec.rule_id, (), evidence, rejection='cleanup_removed_all_edges')
                continue
            edits = [Edit.set(target, ('nets', nid), branch)]
            edits.extend(_new_pin_edits(target, selected))
            evidence['change'] = {'kind': 'restore_branch', 'network_id': nid,
                                  'physical_id': pid, 'added': sorted(refs),
                                  'edge_count': len(branch['edges'])}
            yield Proposal(self.spec.rule_id, tuple(edits), evidence,
                           'two_electrical_vertices_and_resolved_model_branch')


class ZeroWireFragmentRule:
    spec = RuleSpec('zero_wire_fragment', '1.0',
                    '检查无导线的孤立网碎片；只有显式开启时才执行删除',
                    ('remove_empty_networks',))

    def propose(self, context):
        target, green = context.target, context.evidence['green_mask']
        height, width = context.height, context.width
        for nid, net in sorted(target['nets'].items()):
            refs = _members(net)
            owners = {r.rsplit('.', 1)[0] for r in refs}
            if net.get('edges') or len(owners) != 1:
                continue
            owner = next(iter(owners))
            comp = target['components'][owner]
            points = [target['pins'][owner]['pin_' + r.rsplit('.', 1)[1]]['point'] for r in refs]
            points = [[p['x'], height - p['y']] for p in points]
            a, b, c, d = comp['bbox']
            no_green = all(not green[max(0, round(y)-7):min(height, round(y)+8),
                                     max(0, round(x)-7):min(width, round(x)+8)].any() for x, y in points)
            inside = all(a-3 <= x <= c+3 and height-d-3 <= y <= height-b+3 for x, y in points)
            reason = ('isolated_ic_contact_without_external_wire'
                      if comp['type'] in {'box', 'ic'} and len(refs) == 1 and no_green
                      else 'passive_duplicate_tips_inside_body'
                      if comp['type'] in {'r', 'c', 'l'} and inside else None)
            if not reason:
                continue
            evidence = {'network_id': nid, 'members': sorted(refs), 'fragment_reason': reason}
            rejection = '' if context.evidence['config']['prune_zero_wire_fragments'] else 'cleanup_not_opted_in'
            yield Proposal(self.spec.rule_id, (Edit.remove(target, ('nets', nid)),),
                           evidence, reason, rejection)


def default_rules():
    return (KnownContactMembershipRule(), ConfirmedBranchRule(), ZeroWireFragmentRule())
