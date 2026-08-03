"""Create a sample PDF used by the end-to-end verification run."""

from __future__ import annotations

import fitz  # PyMuPDF

OUT = "data/sample_photosynthesis.pdf"

CONTENT = """Photosynthesis: An Overview

Photosynthesis is the process by which green plants, algae, and certain
bacteria convert light energy, usually from the sun, into chemical energy.
The overall chemical equation for photosynthesis is:

6 CO2 + 6 H2O + light energy -> C6H12O6 + 6 O2

Photosynthesis takes place in the chloroplasts of plant cells. Chlorophyll,
the green pigment inside chloroplasts, absorbs light most strongly in the
blue and red parts of the spectrum and reflects green light.

The process has two main stages: the light-dependent reactions and the
Calvin cycle (light-independent reactions). The light-dependent reactions
occur in the thylakoid membranes and produce ATP and NADPH. The Calvin cycle
occurs in the stroma and uses ATP and NADPH to fix carbon dioxide into
glucose.

Key facts:
- The primary photosynthetic pigment is chlorophyll a.
- The wavelength of maximum absorption for chlorophyll a is around 430 nm
  (blue) and 662 nm (red).
- Photolysis of water releases oxygen as a by-product.
- The Calvin cycle was discovered by Melvin Calvin.
- The rate of photosynthesis is affected by light intensity, carbon dioxide
  concentration, and temperature.

Photosynthesis is essential for life on Earth because it produces oxygen and
forms the base of most food chains. Agricultural productivity depends heavily
on optimising the conditions for photosynthesis in crops.
"""


def main() -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Photosynthesis Notes", fontsize=20)
    y = 120
    for line in CONTENT.splitlines():
        page.insert_text((72, y), line, fontsize=11)
        y += 16
        if y > 780:
            page.new_page()
            y = 72
    doc.save(OUT)
    doc.close()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
