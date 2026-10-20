# Observed digit-named connector rows

The final `agent.finalize_pin_numeric_connector` hook repairs missing digit-named
connector rows after existing pin/library repairs. Disable it with
`PCB_PIN_NUMERIC_CONNECTOR=0`. The wrapper verifies current-image hash, size,
top-left coordinate origin and prediction-only provenance for the original pin,
graph and OCR documents. All inputs remain unchanged.

Only box components with at least two existing pins whose observed name equals
their numeric key qualify. A real red frame and leads must occur on one side.
Existing independently observed digit names calibrate the interior text column,
row offset and external lead length. Inconsistent anchors cause abstention.
Each missing row needs its own unique assigned original model candidate with
both fields absent, a unique numeric text supported by two original views, and
two fresh >=0.98 local glyph readings agreeing with that text. Number order is
never inferred. Tall OCR strings spanning multiple rows are not split into IDs.
New points use the existing same-side lead length and the actual missing row;
they must remain on the visible red lead and close to the original candidate.
Existing records, component identities and wire geometry are preserved.

A new pin may also expose an omitted supply symbol. Only a continuous narrow red
stem ending in a symmetric red cap, without stem continuation beyond the cap,
qualifies. Two fresh local OCR views of the nearby blue text must independently
agree on a supported power label. Existing nearby power symbols are protected.
The symbol bounding box and contact are measured from the image, with a stable
key derived from the actual candidate ID. No library template is used.

A unique original attachment can update an existing exported network only with
the exact source point, candidate membership, original nearby model segment,
visible green contact and no unresolved junction. Otherwise a directly observed
T symbol and pin may form a two-contact local logical network with empty edges.
This records only the physical shared contact; it does not reconstruct upstream
wires or join a nearby broken model route. That limitation is retained in audit.
No existing wire is changed and no wire segment is invented.

Fresh identity confirmation commits atomically per connector. Supply recovery
is optional and cannot prevent a valid pin-only fix. OCR work is bounded and
idempotent. Tests cover independent/nonconsecutive identities, contradictory
views, false candidates, budgets, power geometry, route ambiguity, provenance and
preservation. Verification freezes predictions before reading scoring references.
