# Observed vertical contact repair

`agent.finalize_pin_vertical_repair` runs before final target export, after the
existing NC recovery hook. Disable it with `PCB_PIN_VERTICAL_REPAIR=0`; the
disabled wrapper reads no artifacts and does not initialize OCR. Image hash,
dimensions, top-left origin and prediction-only provenance are checked first.

This rule targets a three-record IC with a library-assigned field and physically
duplicated output positions. It requires three actual long red vertical leads
on opposite sides (one plus two), consistent measured lengths and separated
columns. Existing identities must each have one exact attached model candidate
and one matching exported membership. Each measured endpoint must reach one
model network and have a visible green continuation.

Original multi-view number tokens establish the digit offset from the frame.
Each number crop is read at 3x and 5x in both 90 and 270 degree orientations;
orientation must be uniquely corroborated and consistent for all three leads.
Names have multiple original source views and new raw/neutral tight-crop OCR.
At least two distinct fresh views must exceed confidence 0.98. Case-only OCR
variation can corroborate a name, while the original observed spelling is kept.
The complete observed number set must equal the existing set, and strong name
conflicts reject the entire component. No device name, page ID, numbering order
or reference answer is encoded.

All three records and their memberships commit as one transaction. The old
duplicate identity is removed from its wrong exported network. Existing network
segments and unrelated members are preserved; existing junction ambiguity is
retained in audit, never declared resolved. A missing network can be restored
only as a two-contact, one-path ground branch with exact logical and physical
provenance, unique contact attachments, unambiguous existing ground symbol,
matching model endpoints and at least 90% visible green support. Exported edges
are copied verbatim from the model polyline after coordinate conversion. No wire
is drawn between guessed endpoints. Existing ground contacts are not stolen.

The default budget reserves 24 local OCR calls per complete transaction, with
at most two components and 48 calls per image. Missing evidence, conflicting
routes, unsupported branches or budget shortfall leave the input unchanged.
Once corrected, the records no longer collide and a repeat call does no OCR.
Validation uses synthetic boundary cases, real fresh local OCR, retained-image
regression and the actual export entrypoint. Reference labels are loaded only
after predictions are frozen for independent evaluation.
