from pathlib import Path

# 1. allow a pure-digit token to be a pin NAME (it is how a header prints its pins)
pin_text = Path(r"D:\pcb5_submission\src\pcb_parser\pin_text.py")
text = pin_text.read_text(encoding="utf-8")
old = '''    else:
        if component['type'] not in cfg['name_component_types'] or not re.search(r'[A-Za-z]',text):
            return None'''
new = '''    else:
        # A header prints its pins as bare digits inside the body ("1 2 3"), and the reference
        # uses those digits as both the name and the number.  The letter requirement would drop
        # them, so a pure 1-3 digit token is allowed as a name when that is switched on, with a
        # tighter inward band so it cannot pick up a value such as "1.5uH" elsewhere.
        digit_name = bool(cfg.get('allow_digit_pin_names')) and re.fullmatch(r'\\d{1,3}', text)
        if component['type'] not in cfg['name_component_types'] or not (re.search(r'[A-Za-z]',text) or digit_name):
            return None
        if digit_name:
            cfg = {**cfg, 'name_max_inward_px': float(cfg.get('digit_name_max_inward_px', 40))}'''
assert old in text
text = text.replace(old, new, 1)
pin_text.write_text(text, encoding="utf-8")
print("digit names allowed in pin_text.candidate_cost")

# 2. a digit name becomes the number when the pin has none
stage = Path(r"D:\pcb5_submission\scripts\run_engineering_networks.py")
body = stage.read_text(encoding="utf-8")
anchor = """    data, g, audit = complete_engineering_fields(pins, graph, records, cfg)"""
insert = '''    # A pin whose only printed text is a bare digit (a header) uses that digit as its number
    # too; without it the pin cannot be expressed at all.
    if os.environ.get('PCB_DIGIT_NAME_AS_NUMBER', '1').strip().lower() not in {'0', 'off', 'false'}:
        digit_filled = 0
        for pin in pins.get('pin_candidates') or []:
            if pin.get('owner_status') != 'assigned':
                continue
            if str(pin.get('number') or '').strip():
                continue
            name = str(pin.get('pinname') or '').strip()
            if not name.isdigit() or len(name) > 3:
                continue
            pin['number'] = name
            pin['number_status'] = 'digit_pin_name'
            pin['number_source'] = 'printed_digit_used_as_number'
            digit_filled += 1
            for attachment in (graph.get('pin_attachments') or []):
                if attachment.get('candidate_id') == pin.get('candidate_id'):
                    attachment['number'] = name
                    break
        if digit_filled:
            print(f'DIGIT-NAME-AS-NUMBER {digit_filled}', flush=True)
    data, g, audit = complete_engineering_fields(pins, graph, records, cfg)'''
assert anchor in body
body = body.replace(anchor, insert, 1)
stage.write_text(body, encoding="utf-8")
print("digit name -> number pass added")

import json

config = Path(r"D:\pcb5_submission\configs\pin_text_v2.json")
payload = json.loads(config.read_text(encoding="utf-8"))
payload["allow_digit_pin_names"] = True
payload["digit_name_max_inward_px"] = 40
config.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("config updated")
