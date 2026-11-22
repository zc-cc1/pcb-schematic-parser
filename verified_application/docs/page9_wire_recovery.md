# Guarded supply contact and box-pin wire trimming

The structural recovery stage accepts a named voltage supply with two OCR views at confidence at least 0.85, including one at least 0.93, only when the same image proves a T-shaped supply symbol and a continuous outward conductor. The existing three-view rule remains. The rule executor independently recomputes the OCR and image proof before committing the supply pin and its network membership. It does not use a page identifier, component designator, reference annotation, or network score.

After wire cleanup, a conductor that extends beyond a measured box or IC pin into its red symbol lead is clipped to the pin. The edge must pass within 3 pixels of that pin, run 8–45 pixels toward the body beyond it, have red pixels in the overrun, and have green pixels before the pin. This preserves the electrical membership and rejects an unsupported endpoint guess.

On frozen retained model/OCR predictions, page 9 moved from TP 28 / FP 10 / FN 11 (wire F1 0.7273) to TP 38 / FP 0 / FN 1 (F1 0.9870). The remaining reference segment is 2 pixels. Page 5 moved to wire F1 1.0 and page 11 to 0.8364; no page regressed. The staged 13-page entrypoint and a page-9 actual local-OCR finalization reproduced the results. This is validation on the available public set, not a guarantee for unseen diagrams.
