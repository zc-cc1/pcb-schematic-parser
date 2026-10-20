from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
s=(WORK/'pin_pairing_implementation_20261020/verify_entrypoint.py').read_text(encoding='utf-8')
s=s.replace('original_pairing=agent.finalize_pin_pairing',
            'original_pairing=agent.finalize_pin_pairing\noriginal_evidence=agent.finalize_pin_evidence_repair')
s=s.replace("cache_roots=[ROOT/'ocr_cache'", "cache_roots=[ROOT/'ocr_cache',WORK/'pin_pairing_implementation_20261020/ocr_cache'")
s=s.replace("for case in ['1','2','4']:","for case in ['1','2','4','5']:")
s=s.replace('pin_relocation_implementation_20261020/deployed','pin_pairing_implementation_20261020/deployed')
s=s.replace("    def pairing_hook(*args):\n        result,audit=checked_call(original_pairing,*args);calls.append(audit);return result,audit", 
'''    def pairing_hook(*args):return checked_call(original_pairing,*args)
    def evidence_hook(*args):
        before=copy.deepcopy(args[0])
        result,audit=checked_call(original_evidence,*args)
        assert result['nets']==before['nets']
        calls.append(audit);return result,audit''')
s=s.replace("side_effect=pairing_hook):", "side_effect=pairing_hook),patch.object(agent,'finalize_pin_evidence_repair',side_effect=evidence_hook):")
s=s.replace("'restored_networks':len(verified['restored_networks']),", '')
s=s.replace("row['pin_pairing']", "row['pin_evidence_repair']")
s=s.replace("logs/{case}_pin_pairing.json", "logs/{case}_pin_evidence_repair.json")
s=s.replace("'pairing_log_written'", "'evidence_log_written'")
s=s.replace("    for restored in verified['restored_networks']:",
            "    prior_pairing=read(WORK/f'pin_pairing_implementation_20261020/deployed/{case}_audit.json')\n    for restored in prior_pairing['restored_networks']:")
s=s.replace("    for change in verified['network_changes']:", "    for change in prior_pairing['network_changes']+verified['network_changes']:")
(ROOT/'verify_entrypoint.py').write_text(s,encoding='utf-8')
print('Prepared four-page live-entrypoint smoke check.')
