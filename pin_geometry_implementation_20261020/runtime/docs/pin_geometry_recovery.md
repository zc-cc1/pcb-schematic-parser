# Image-supported IC pin recovery

The target export runs this step after pin field recovery. It is enabled by default;
set `PCB_PIN_GEOMETRY_RECOVERY=0` to disable it. Configuration is in
`configs/pin_geometry_recovery_v1.json`. Audit output is written to
`logs/{case}_pin_geometry_recovery.json`; the batch summary records changes and OCR calls.

The step verifies the current image hash, dimensions and coordinate system before
using model artifacts. It automatically locates supported red rectangular component
frames and visible short horizontal red leads. Missing rows require both a printed
number and an alphabetic name supported by at least two distinct OCR views at 0.98
confidence. Existing observed numbers establish the number column and text offset.
Whole-component OCR uses the original image at 1x/3x and 0/90 degrees. A missing
number may use an isolated neutral-color glyph at 3x/5x. Numeral sequence, component
identity, evaluation labels, and case-specific coordinates are never used as evidence.

An ambiguous model candidate can be reused only when its owner eligibility and
visible lead agree uniquely. Otherwise a new pin uses the observed red endpoint.
Existing keys and names are preserved. A sole top/bottom pin can be moved to an
existing collinear ground contact only when a continuous red lead supports that
unique contact. Components and wire geometry are preserved.

Network synchronization requires a unique already-exported network, exact model
candidate attachment/membership or visible green wire evidence, and no unresolved
junctions. NC fields without model attachment are withheld. It never creates a
network or wire. Insufficient connection evidence is recorded in the audit while
the recovered pin remains available. Existing memberships are never moved.

Limits default to 6 OCR components, 32 proposals and 96 local OCR calls per image.
OCR models load lazily; no external OCR API is used. Unsupported colors, small or
ambiguous frames, conflicting text, missing evidence and exhausted budgets cause
abstention. If the hook raises, the existing export is retained and the summary
records `pin_geometry_recovery_error`.

The release was validated by replaying retained model outputs for all 13 public
pages, reading reference labels only after all predictions were frozen. This is
local regression evidence and not a new official platform submission. Synthetic
tests cover image evidence, repeated-view rejection, owner ambiguity, budgets,
network safety and the disable switch. Further unseen-page validation remains useful.
