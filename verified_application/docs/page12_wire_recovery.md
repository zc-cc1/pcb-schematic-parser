# Image-supported passive and LED wire recovery

Four generic changes address the available page-12 failures:

* A measured capacitor pin can clip a wire's red symbol-lead overrun under the same red/green pixel checks used for box pins.
* A misplaced degree-two orthogonal corner can move by 6–40 pixels when both corrected arms have continuous ink, the original diagonal is unsupported, no red junction is moved, and no component body is crossed.
* A detected, named resistor with no pins can recover two contacts from one rectangular body, two observed stems ending at a junction/color transition/bend, and stem lengths agreeing within 4 pixels. Symmetric resistor contacts use the ordinary left/top-to-right/bottom 1/2 convention. The rule executor independently repeats the image proof.
* An unresolved model LED can be identified from one upright filled triangle, two nearby emission marks, two model pin candidates on opposite sides, and two isolated-crop OCR readings of the same LED designator at confidence >=0.98. Blue-pixel word spacing isolates the label from an adjacent component value. The source pin numbering is preserved. The executor independently verifies geometry and caption evidence.

The runtime reads the current image and retained model/OCR artifacts; it does not read page IDs, reference labels, or scores. All predictions are frozen before independent scoring. On the public page 12, wire TP/FP/FN improved from 19/4/9 (F1 0.7451) to 28/0/0 (F1 1.0). The thirteen-page cached comparison showed no score regressions. Actual bundled local OCR and the full export entrypoint are checked separately. This result describes the available public set.
