# Closed bottom-row number permutation

Enabled by default after observed-evidence repair. Set `PCB_PIN_NUMBER_CYCLE=0`
to disable. Configuration: `configs/pin_number_cycle_v1.json`. Logs:
`logs/{case}_pin_number_cycle.json`.

This step validates exactly three primary assigned alphabetic-name pins on the
bottom of a supported red rectangular frame, with at least one library-derived
number and candidate confidence at least 0.9. Source numbers must be unique,
spacing at least 16 pixels, and current records must still match the original
names and pin columns. Each current point must lie on a visible bottom lead.

A directly observed numeral with two independent source views establishes the
digit-column offset and provides an orientation anchor. Isolated neutral-color
glyphs at 3x/5x are tested at 90/270 degrees. The orientation must be uniquely
supported by the anchor and consistent across usable anchors. Component crops
at 1x/3x and 0/90 degrees corroborate the name assigned to each physical column.
Each number and name needs two views at confidence at least 0.98. Names and
physical points are never replaced by a library or inferred numbering order.

The three observed numbers must form an exact permutation of the three existing
numbers, with no duplicates, missing values or changed observed anchors. Otherwise
the entire component abstains. Correctly numbered control components are verified
but remain unchanged. Current-record mismatch makes repeat runs idempotent.

Every candidate must have an exact unique attached model point and a sole existing
exported membership. An existing direct model network is accepted, or an existing
RESTORED physical network whose candidate, physical ID and edge provenance match
the model's logical source. No new net or connection is inferred. Existing direct
networks may contain upstream unresolved junctions: these are recorded and left
unchanged, because this operation relabels identities on already-exported geometry.
It does not resolve those junctions or claim their topology is correct.

All pin records are copied from one immutable snapshot to their observed keys;
all network references are substituted simultaneously from their original sets.
This prevents a 1→2→1 sequential substitution from collapsing two identities.
Pin counts, names at physical positions, coordinate sets, network IDs, member
counts and all wire segments are preserved. Only the number/identity assignment
and corresponding references change.

Defaults cap processing at 2 components and 24 local OCR calls. Missing provenance,
ambiguous orientation, uncertain text, stale attachments, duplicate memberships,
non-closed mappings, absent leads and budget exhaustion cause abstention. OCR
loads lazily. Inputs are never mutated; wrapper errors retain the prior target and
record `pin_number_cycle_error`. Image hash, prediction-only flag, dimensions and
coordinate system are checked before processing. No external OCR API is used.

Regression uses retained same-image model outputs, with all predictions frozen
before reference labels are read. This version covers this three-column bottom
layout; other orientations or arbitrary multi-pin permutations need separate
validation. Local scoring is not a new official platform evaluation.
