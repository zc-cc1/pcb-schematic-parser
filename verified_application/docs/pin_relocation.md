# Unsupported pin relocation

The target export runs this step after pin geometry recovery. It is enabled by
default; set `PCB_PIN_RELOCATION=0` to disable it. Audit output is written to
`logs/{case}_pin_relocation.json`. Configuration is in `configs/pin_relocation_v1.json`.

This step handles existing coordinates supported only by library elimination.
Both number and name must have that provenance, with no observed text evidence,
candidate confidence below 0.8 and model wire probability below 0.2. The current
pin must still match the model coordinate and name, have no exported network
membership, and have exactly one same-point model attachment explicitly reporting
`no_wire_within_radius` with no network ID. Directly observed or attached pins are
protected. Reliable alternative text tokens or contradictory model network
membership also block relocation. Duplicate numbered candidates and explicit
duplicate hypotheses abstain.

The image must support a red rectangular frame and no green wire at the old point.
This version handles left-facing rows only. At least three other pins must share
that side, with exactly one unoccupied actual short lead and a displacement of at
least 25 pixels. Number column and row offset come from observed same-side digits.
An isolated neutral-color numeral at 3x/5x must confirm the existing number, and
whole-component OCR at 2x/3x and 0/90 degrees must independently confirm the existing
name in the new row. Each field requires two distinct views at confidence 0.98.
Conflicting readings cause abstention. No numbering-order guesses or GT are used.

Only the existing pin point is replaced; the key and name are preserved. The old
model candidate attachment is not reused at the new point. Network synchronization
requires visible green wire and a unique already-exported model network with no
unresolved junctions. Missing exported networks are recorded and withheld. No new
network or wire is created, and no existing network membership is moved.

Defaults cap attempts at two candidates and 12 local OCR calls per image. OCR loads
lazily and uses local ONNX models. Image hashes, dimensions and coordinate systems
are verified before processing. The input is not mutated; exceptions in the entry
hook retain the previous target export and record `pin_relocation_error`.

Regression uses retained model outputs to avoid full inference cost. Reference
labels are loaded only after all generated predictions have been saved. Current
public-page results are local evidence; unseen-page generalization is not assured.
