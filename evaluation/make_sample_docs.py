"""Create the ground-truth document corpus used by the RAG evaluation metrics.

Writes two short note files into ``evaluation/data/``. Each is small enough to
become a handful of 600-char chunks under the default chunking, and each has a
distinct topic so retrieval queries have a well-defined "expected source".

Run once before the benchmarks::

    python evaluation/make_sample_docs.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

PHOTOSYNTHESIS = """Photosynthesis Notes

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

NEWTON = """Newton's Laws of Motion Notes

Classical mechanics rests on three laws stated by Isaac Newton in 1687.

Newton's first law (law of inertia): an object at rest stays at rest and an
object in uniform straight-line motion stays in motion unless acted on by an
unbalanced external force.

Newton's second law: the net force acting on an object equals the product of
its mass and its acceleration, written F = m * a. For a fixed mass, doubling
the applied force doubles the acceleration; doubling the mass halves the
acceleration for the same force. The SI unit of force is the newton (N),
where one newton is the force needed to accelerate one kilogram by one metre
per second squared.

Newton's third law: for every action there is an equal and opposite reaction.
When a rocket pushes hot gas downward, the gas pushes the rocket upward with
the same strength.

Key facts:
- Force is measured in newtons (N); mass in kilograms (kg); acceleration in
  metres per second squared (m/s^2).
- Weight is the gravitational force on a mass: W = m * g, with g about 9.8
  m/s^2 on Earth.
- Friction is a force opposing motion between surfaces in contact.
"""

DOCS = {
    "notes_photosynthesis.txt": PHOTOSYNTHESIS,
    "notes_newton.txt": NEWTON,
}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in DOCS.items():
        path = DATA_DIR / name
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {path} ({len(content)} chars)")


if __name__ == "__main__":
    main()
