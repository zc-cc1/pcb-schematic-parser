# Joint pairing of shared pin numerals

Enabled by default after pin relocation in the target export. Set
`PCB_PIN_PAIRING=0` to disable. Configuration: `configs/pin_pairing_v1.json`.
Audit output: `logs/{case}_pin_pairing.json`.

Trigger only when an assigned candidate without a number was blocked by
`text_used_by_other_pin`, and exactly one same-owner assigned numbered candidate
uses the same reliable numeral token. Explicit duplicate hypotheses abstain.
The digit must lie above the missing row and below the numbered row, outside the
left edge of a supported red rectangular frame. This version handles left-facing
rows separated by at least 18 pixels. Both model attachments must be exact,
unique, attached to their source networks, and free of unresolved junctions.

The current wrong-number record must still have the original name and geometry,
and its sole exported network membership must match the original model attachment.
Each row must have a unique visible short lead near its candidate. A recovered
digit requires two isolated-glyph OCR views (3x/5x), and each name requires two
whole-component views (1x/3x, 0/90 degrees), all at confidence at least 0.98.
Both corrected rows must form one complete transaction, without reserved keys or
network references. No numerical ordering, device pin library or reference labels
determine the corrected numbers.

Only after all checks pass, the old record is rekeyed while its coordinate/name
are preserved, its existing network reference is renamed, and the missing pin is
created at the observed lead endpoint. The old number then belongs to the newly
confirmed row. Inputs are deep-copied; failures before commit leave outputs intact.

If the missing pin's proven model network is already exported, add its reference
there. Otherwise rerun the existing power-branch exporter against the corrected
candidate-number copy, and copy only a qualified two-contact physical power branch
containing the recovered candidate. Other restored networks/contacts are ignored.
Wire geometry comes only from existing model segments; existing wires are preserved.
Existing IC memberships cannot be stolen. Missing/ambiguous branch evidence is
recorded while retaining the proven pin correction.

Defaults: two candidate pairs and 16 local OCR calls per image. OCR loads lazily;
pages without a qualifying conflict make zero OCR calls. Conflicting OCR, occupied
keys, stale attachments, altered records, duplicate memberships, absent frames,
uncertain geometry or exhausted budgets cause abstention. Wrapper verifies current
image hash, prediction-only provenance, size and coordinate system. Exceptions keep
the prior target export and record `pin_pairing_error`. No external OCR API is used.

Regression is performed on retained same-image model outputs, with all predictions
frozen before evaluation labels are read. Public-page improvement is local evidence,
not an official platform submission or proof of unseen-page generalization.
