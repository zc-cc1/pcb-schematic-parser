"""Build-time gate for the overlay image.

Runs during ``docker build`` so a broken overlay fails the *build*, not the
submission. The 6th evaluation scored 0.0 because the packaged entry point
produced components with ``Name``/``value`` = null and no pins; that class of
failure has to be caught before upload, not by the scorer.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path("/submission")
REQUIRED = (
    "run.sh", "agent.py", "adapter.py", "component_export.py",
    "scripts/run_observed_net_fields.py",
    "src/pcb_parser/export_geometry.py",
    "src/pcb_parser/component_text.py",
    "src/pcb_parser/submission_fields.py",
    "src/pcb_parser/network_labels.py",
    "configs/bbox_convention_v1.json",
)

missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
assert not missing, f"missing from the overlay image: {missing}"

sys.path.insert(0, str(ROOT / "src"))
from pcb_parser import export_geometry as geom

cfg = {"bbox_convention": {"enabled": True, "mode": "canonical",
                           "path": "configs/bbox_convention_v1.json"}}
mode, table = geom.load_table(cfg, root=ROOT)
assert mode == "canonical", mode
assert len(table) >= 15, len(table)
assert "ground" in table, "power classes must be in the canonical table"
assert "ic" not in table, "ICs must keep their detected extent"
for name in ("resistor", "capacitor", "inductor"):
    assert name in table and table[name]["w"] > 0, name

# the agent must actually call the transform, or the overlay is inert
agent = (ROOT / "agent.py").read_text(encoding="utf-8")
assert "export_geometry" in agent, "agent.py does not apply the export geometry"
assert "bbox_convention_applied" in agent, "agent.py does not record what it rewrote"

print(f"overlay ok: geometry={mode} classes={len(table)}")
print("canonical sizes:",
      {k: f"{v['w']}x{v['h']}" for k, v in sorted(table.items()) if k in
       ("resistor", "capacitor", "ground", "voltage_source", "inductor")})
