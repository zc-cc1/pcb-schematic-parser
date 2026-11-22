# Final export semantics

Default: PCB_EXPORT_SEMANTICS=observed. Normalization runs after topology recovery and before final JSON checkpoint.

- Directional red flag type follows the polygon tip relative to the sole electrical terminal, including rotated/mirrored flags. Proven port captions remain in Name/key; value becomes null. Unproven/ambiguous ports retain existing fields.
- LED, switch, IC, box and connector Name no longer falls back to its own designator. Actual model names and descriptions remain. Resistor/capacitor/inductor/pin naming is unchanged.
- Late switches may use a unique nearby multi-view OCR model in the same text stack as their printed designator; claimed or conflicting glyphs are rejected.
- Pins, nets, component identities and bounding boxes are unchanged. Each change is logged in logs/<case>_export_semantics.json. Failures keep the previous validated prediction.

For old label conventions, set PCB_EXPORT_SEMANTICS=legacy globally before launching the agent. This bypasses the new normalizer. It is never selected per image or by looking at ground truth.

Validation and scores are in the companion export_semantics_optimization_20261106 report. These 13 images were used during development; scores do not establish performance on unseen diagrams.
