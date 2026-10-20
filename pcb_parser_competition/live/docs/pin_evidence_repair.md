# Corroborated printed pin evidence

Enabled by default after joint pin pairing. Set `PCB_PIN_EVIDENCE_REPAIR=0` to
disable. Configuration: `configs/pin_evidence_repair_v1.json`. Logs:
`logs/{case}_pin_evidence_repair.json`. OCR is local and lazy-loaded.

## Full printed labels

A primary assigned numeric pin whose name came from `library_pin_name` may retain
only one token of a printed parenthesized label. The current name must still match
the model name and its point must be within 25 pixels of the source candidate.
Require a unique primary numbered candidate and a supported red rectangular frame.
Only actual whole-token forms `HEAD(ALIAS)` containing the short name as HEAD or
ALIAS qualify; arbitrary substrings and invented parentheses do not.

At least two existing independent OCR views must locate the full spelling in this
row on the near interior half of the component. Fresh original-image component
crops at 1x/3x and 0/90 degrees must corroborate exactly that spelling, with at
least two views and confidence 0.98. Conflicting words cause abstention. The actual
observed full spelling replaces the library-shortened name; numbers, coordinates,
network members and wires are preserved. No device or expected label is encoded.

## Isolated collision points

A primary candidate with a library-derived number and confidence below 0.8 is
eligible only when its current point collides within 3 pixels with exactly one
different, stronger same-owner pin (source confidence at least 0.9). The name must
still match, its source point/row must agree, and its sole exported net must contain
only this pin and no edges. The source model net and exact attached candidate must
also be singleton and edgeless, with no unresolved junction. Active connections,
other members and contradictory attachments are protected.

Reuse the existing image-lead recovery algorithm on this component as an evidence
probe. It must find exactly one unoccupied actual lead on the opposite side, with
both existing number and name corroborated by two views at 0.98. The displacement
must span at least the component width. The probe output is discarded; only this
confirmed coordinate replaces the collided point. Other pins and all networks and
wires remain unchanged. The old edgeless singleton is kept isolated, and connection
to the counterpart is explicitly withheld until network evidence is proven.

## Scope and budgets

Defaults: at most 3 components and 24 OCR calls. Component OCR observations are
reused within the label step. Unknown colors, missing frames, repeated views,
unreliable or conflicting text, unsupported layouts and exhausted budgets abstain.
Input artifacts are never mutated. Current-image hash, prediction-only provenance,
dimensions and coordinate system are checked in the entry wrapper. Exceptions keep
the prior target and record `pin_evidence_repair_error`.

This is an evidence correction, not a device-name canonicalizer or a general
cross-body pin mover. Public-page cache regression is local validation; unseen-page
generalization and unresolved network attachment remain separate work.
