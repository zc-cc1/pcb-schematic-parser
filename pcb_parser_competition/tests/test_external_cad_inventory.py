"""Native inventories must not confuse symbol definitions with placed parts."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from collect_external_schematics import cad_inventory


class NativeInventoryTests(unittest.TestCase):
    def test_modern_kicad_embedded_library_and_multiline_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"sample.kicad_sch"
            path.write_text('(kicad_sch (lib_symbols (symbol "Crystal")) '
                '(symbol (lib_id "Device:Crystal") (at 10 20 90) '
                '(property "Reference" "Y1") (property "Value" "12MHz\nclock")))',encoding="utf-8")
            result=cad_inventory(path)
            self.assertEqual(result["component_instance_count"],1)
            self.assertEqual(result["components"][0]["reference"],"Y1")
            self.assertEqual(result["target_candidates"][0]["hints"],["clock_symbol_candidate"])
            self.assertIsNone(result["target_candidates"][0]["verified_class"])

    def test_eagle_keeps_electrical_pin_references(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"sample.sch"
            path.write_text('<?xml version="1.0"?><eagle><drawing><schematic>'
                '<parts><part name="Q1" library="semi" deviceset="MOSFET_PCH" value="model"/></parts>'
                '<sheets><sheet><nets><net name="GATE"><segment><pinref part="Q1" gate="G$1" pin="G"/>'
                '</segment></net></nets></sheet></sheets></schematic></drawing></eagle>',encoding="utf-8")
            result=cad_inventory(path)
            self.assertEqual(result["nets_from_native_xml"][0]["pin_references"][0]["pin"],"G")
            self.assertEqual(result["target_candidates"][0]["hints"],["mosfet_candidate"])

    def test_binary_cad_has_short_actionable_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"binary.sch";path.write_bytes(b"\xff"*10000)
            with self.assertRaises(ValueError) as error: cad_inventory(path)
            self.assertIn("requires a CAD converter",str(error.exception))
            self.assertLess(len(str(error.exception)),200)


if __name__=="__main__": unittest.main()
