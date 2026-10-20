# Crossed NC lead number recovery

`agent.finalize_pin_nc_recovery` runs after the existing number-cycle repair and
before final target export. It is enabled by default; `PCB_PIN_NC_RECOVERY=0`
disables it without reading artifacts or initializing OCR. The wrapper validates
the current image hash, dimensions, prediction-only provenance and coordinate
origin of retained pin and graph artifacts.

The rule requires an actual red component frame and short lead, both diagonals
of a green no-connect X within a small endpoint tolerance, no outgoing green
wire, adequate row separation, and at least two existing digit identities
corroborated by multiple original source views on the same side. Existing pin
records are protected. Numbering is never inferred from device type, page ID,
unused numbers or sequence order.

Each pending row uses fresh local detection OCR at 3x and 5x, a complete neutral
numeral group retaining disconnected leading-digit parts and gray antialias
pixels, and an isolated interior name crop. Numeric groups are read as a whole;
single connected components are never treated as the complete number. Only
exact NC names and numbers with at least two distinct views at confidence >=
0.98 qualify. Any strong conflicting number rejects the row. Duplicate proposed
identities, existing keys or network references, conflicting model identities,
clipped glyphs and unsupported geometry are withheld.

Only new pin records are added at observed short-lead endpoints. The module
does not modify network membership, wire geometry, components or existing pins.
NC records intentionally remain unconnected. A row has a reserved six-call OCR
budget; an insufficient remaining budget never commits a partial observation.
The default limits are 3 components, 8 rows, and 48 OCR calls per image. Repeated
execution has no effect once those physical rows are represented. Exceptions
preserve the caller's input and the export entrypoint keeps the prior result.

Validation combines boundary tests, fresh actual local OCR, all-page retained
model replay, and actual export-hook execution. Reference labels are used only
after predictions are frozen for evaluation; runtime does not read labels.
