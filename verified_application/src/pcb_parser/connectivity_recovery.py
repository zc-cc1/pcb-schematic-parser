"""Compatibility entrypoint for the extensible structural repair rule library."""
from copy import deepcopy
import math

from .connectivity_evidence import collect_evidence
from .connectivity_rules import default_rules
from .repair_engine import RepairContext, RepairPipeline

DEFAULT = {
    'enabled': True, 'maximum_distance_px': 3.0,
    'ambiguity_margin_px': 3.0, 'crossing_guard_px': 6.0,
    'minimum_colored_wire_pixels': 200, 'minimum_red_pixels': 100,
    'minimum_power_confidence': .9, 'restore_clear_branches': True,
    'maximum_new_contacts': 64, 'maximum_new_branches': 16,
    'prune_zero_wire_fragments': False, 'rules': {},
    'structural': {'enabled': False},
}


def recover_connectivity(target, graph, pin_document, rgb, *, cleanup=None, cfg=None,
                         records_document=None, ocr_document=None, wire_cfg=None, caption_reader=None):
    """Collect same-image evidence, propose repairs and commit conflict-free edits."""
    config = {**deepcopy(DEFAULT), **deepcopy(cfg or {})}
    if type(config['enabled']) is not bool:
        raise ValueError('enabled must be boolean')
    audit = {'enabled': config['enabled'], 'policy': 'structural_connectivity_rules_v1',
             'changes': [], 'proposals': [], 'withheld_branches': [],
             'new_contact_count': 0, 'new_branch_count': 0, 'inference_calls': 0,
             'fragment_candidates': [], 'pruned_networks': []}
    if not config['enabled']:
        audit['status'] = 'disabled'
        return deepcopy(target), audit
    for key in ('restore_clear_branches', 'prune_zero_wire_fragments'):
        if type(config[key]) is not bool:
            raise ValueError(key + ' must be boolean')
    for key in ('minimum_colored_wire_pixels', 'minimum_red_pixels',
                'maximum_new_contacts', 'maximum_new_branches'):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(key + ' must be a nonnegative integer')
    confidence = config['minimum_power_confidence']
    if (type(confidence) not in (int, float) or not math.isfinite(confidence)
            or not 0 <= confidence <= 1):
        raise ValueError('minimum_power_confidence must be in [0,1]')
    if not isinstance(config['rules'], dict):
        raise ValueError('rules must be a mapping')
    data = collect_evidence(target, graph, pin_document, rgb, config, audit, cleanup)
    if not data['ready']:
        return deepcopy(target), audit
    settings = deepcopy(config['rules'])
    if not config['restore_clear_branches']:
        settings['confirmed_physical_branch'] = {'enabled': False}
    width, height = graph['size']
    context = RepairContext(target, data, width, height)
    result, execution = RepairPipeline(default_rules()).run(context, rule_settings=settings,
        maximum_new_contacts=config['maximum_new_contacts'],
        maximum_new_branches=config['maximum_new_branches'])
    audit['rule_execution'] = execution
    for proposal in execution['proposals']:
        evidence = proposal['evidence']
        if proposal['rule_id'] == 'zero_wire_fragment':
            audit['fragment_candidates'].append({'network_id': evidence['network_id'],
                'members': evidence['members'], 'reason': evidence['fragment_reason']})
        if proposal['status'] == 'accepted':
            if 'change' in evidence:
                audit['changes'].append(evidence['change'])
            elif proposal['rule_id'] == 'zero_wire_fragment':
                audit['pruned_networks'].append(evidence['network_id'])
        elif 'physical_id' in evidence:
            audit['withheld_branches'].append({'physical_id': evidence['physical_id'],
                'reason': proposal['rejection'], 'members': evidence['members']})
    audit['new_contact_count'] = execution['contact_count']
    audit['new_branch_count'] = execution['branch_count']
    audit['status'] = execution['status']
    structural=config['structural']
    if not isinstance(structural,dict) or type(structural.get('enabled',False)) is not bool:
        raise ValueError('invalid structural configuration')
    if structural.get('enabled'):
        if records_document is None or ocr_document is None or wire_cfg is None:
            audit['structural']={'enabled':True,'status':'missing_structural_evidence'}
        else:
            from .structural_recovery import recover_image_structures
            result, structural_audit=recover_image_structures(result,rgb,records_document,
                pin_document,ocr_document,wire_cfg,cfg=structural,cleanup=cleanup,reader=caption_reader)
            audit['structural']=structural_audit
            if structural_audit.get('accepted'):audit['status']='applied'
            # Corrected visible terminal positions can expose additional model
            # attachments. Evaluate those before exporting, rather than making
            # the result depend on whether the caller runs this stage twice.
            result,post=recover_connectivity(result,graph,pin_document,rgb,cleanup=cleanup,
                cfg={**config,'structural':{'enabled':False}})
            audit['post_structural_memberships']=post
            audit['new_contact_count']+=post['new_contact_count']
            audit['new_branch_count']+=post['new_branch_count']
            audit['changes'].extend(post['changes'])
            if post['changes']:audit['status']='applied'
    return result, audit
