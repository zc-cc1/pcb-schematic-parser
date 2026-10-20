# Observed rows and exact-model library completion

The final `agent.finalize_pin_row_library` hook repairs missing horizontal row
fields and weak incorrect numbers after all existing repairs. Disable it using
`PCB_PIN_ROW_LIBRARY=0`; no artifacts, OCR or library are read in that mode.
The wrapper verifies image provenance, size and coordinate origin, and pins the
unchanged bundled `data/component_library/BoxICLibrary.json` by SHA-256.

Only box-like components whose model is still a designator placeholder qualify.
Actual red frames and leads supply physical rows. Four local component OCR views
must identify one exact normalized library model outside and near that frame,
with at least two >=0.98 views and a library pin count equal to the visible lead
count. Lookup never reads reference answers or uses a page/component ID. Raw
library aliases and conflicts are preserved rather than picking the first name.

Number tokens must be outside the same side of the frame, names on that side's
interior half, and both close to the actual row. Multi-character identities such
as A5 and B5 remain intact. Weak fields can use whole neutral text crops at 3x
and 5x, then a grouped crop retaining all glyphs and excluding adjacent-row
fragments. A single high-confidence contradiction withholds the result. Current
strong observed numbers, existing fields, physical records and occupied keys
are protected. A weak wrong existing number can be rekeyed only when fresh
number/name OCR and the unique library name-to-number mapping agree. Its full
physical record is preserved and every existing network reference follows it.

A fully unnumbered component may map each observed name to a number only when
that exact model has a unique mapping. Exactly one blank contact may then be
filled from exactly one remaining library record, provided >=4 other real rows
are uniquely named and mapped, the remaining library name is unambiguous, and
the single blank contact has its own real candidate on the opposite side. Two
blank rows, unreadable text, competing model evidence or nonunique names cause
abstention. This is constrained exact-model completion, not numbering by order.

Green no-connect X marks and floating contacts stay unconnected. A new member
can join only one already-exported source model network with exact candidate
attachment, supporting visible green contact, nearby original wire geometry,
and no unresolved junctions. No network or wire segment is created. Missing
exported routes are recorded as withheld. Network audit chains are consolidated
while retaining each intermediate proof step. Successful pin repair also adopts
the strongly observed exact library model name; other component fields stay put.

Each component commits atomically. The default budget is 4 matched components,
64 local OCR calls, with 4 component views and at most 12 additional row reads
reserved per attempt. Once the model and rows are recovered, repeats do no OCR.
Validation covers synthetic boundary cases, actual local OCR, all-image cached
regression and the actual export entrypoint. Reference labels are loaded only
after all predictions are frozen for evaluation.
