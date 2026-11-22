# Observed switch and one-contact terminal refinement

`agent.finalize_component_kind_repair` runs after existing pin repairs. Disable
using `PCB_COMPONENT_KIND_REPAIR=0`. The wrapper checks the same image hash,
dimensions, top-left origin and prediction-only status of pin, graph and OCR
documents. No reference labels, known page IDs or component-library templates
are used by generation. Original detector outputs remain available in audit.

Only box/ic components with placeholder names are considered. A designator must
be independently visible in at least two >=0.98 OCR views. Prefixes only select
candidates: they never decide a class without matching image structure.

One-contact terminals require an H/J/P designator, a real small empty rectangular
body, exactly one actual external horizontal lead, one exported digit-named 1
pin and its original assigned model candidate. Two cached views of the internal
1 plus two fresh local readings must agree. The class becomes pin; the body and
lead endpoint are measured from the image. Existing network identities and all
wire segments remain untouched. Missing routes are audited rather than guessed.
Multi-contact headers, real model names, broken frames and ambiguous candidates
are protected.

Switches require an SW/S designator, a real frame with external leads on all
four sides and at least two rows of repeated small closed contact shapes inside
the body. Every measured lead must have its own assigned original model candidate.
The dominant measured lead length corrects extensions inflated by attached dots
or ground glyphs; it must agree with at least 60% of observed leads. Existing
physical pin records must all map uniquely to independently confirmed rows.

At least three source-supported printed numbers determine a unique OCR reading
direction. Each row is then read separately at two scales. A neutral crop, a
binary fallback, and only for top labels overlapping the measured frame, a
three-pixel closing restricted to that observed occlusion band, are bounded
fallbacks. No expected number, sequence, remaining-number elimination or glyph
template is supplied to OCR. Confident conflicts and repeated identities abort.
At least 70% of actual contacts must be independently readable; unknown rows are
kept in unresolved_contacts and do not get guessed IDs.

Physical old identities are rekeyed in a single network-membership pass. A new
pin with the same number as an old wrong pin does not inherit the old physical
pin's route. All original wire coordinates are preserved. New identities whose
routes are unverified remain unconnected and audited. Classification and the
confirmed switch rows commit together; no existing physical record is discarded.

The default budgets are four candidate components and 96 local OCR calls.
Repeating on classified outputs changes nothing and requires no OCR. Validation
covers synthetic nonsequential IDs, real geometry, source conflicts, headers,
physical rekeys, unknown contacts, budgets, provenance and cached export replay.
Reference scoring is performed only after all image-based predictions are frozen.
