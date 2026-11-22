# Wire false-positive cleanup

After network reconciliation, this stage examines the current image and predicted wire geometry. It removes long diagonals with insufficient colored-ink support, short red resistor-symbol leads lacking a confirmed multi-arm junction, tiny fragments, and geometry already covered within or across predicted networks. At a confirmed junction it removes only a longer red resistor lead. It also extends a long wire to one collinear, 20–30 pixel distant observed junction when the intervening pixels continuously contain colored ink.

The stage keeps components, pins, network identities, and network membership strings unchanged. Every removal or endpoint extension is recorded in the audit. Evidence is bound to the current image hash and the retained model graph. A change budget restores the original prediction if exceeded. No reference annotations, page IDs, or external scores are read by the runtime.

Validation used 13 frozen model/OCR outputs, then replayed the staged full entrypoint with those same observations. Page 13 wire score changed from TP 46 / FP 13 / FN 3 (F1 0.8519) to TP 47 / FP 0 / FN 2 (F1 0.9792). Pages 1, 11, and 12 also lost false positives; the other pages were unchanged. No page lost a true-positive wire. This is a result on the available public set, not a guarantee for unseen diagrams.
