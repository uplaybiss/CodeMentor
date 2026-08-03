"""Generate the experiment corpus (substantial notes on 5 topics).

Each note is deliberately long (~3000-4000 chars ≈ 500-700 tokens) so that every
chunking strategy produces several chunks *per document*. That makes precision@4,
recall@4, hit@1 and MRR discriminating instead of saturating: a query targets a
specific sub-topic living in one chunk, and the store holds enough distractors
that only a good embedding model places the right chunk on top.

Run once::

    python experiments/make_corpus.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

DOCS = {
    "notes_photosynthesis.txt": """Photosynthesis Notes

Photosynthesis is the process by which green plants, algae, and certain
bacteria convert light energy, usually from the sun, into chemical energy. It
is the fundamental source of the organic molecules that feed almost all life on
Earth. The overall chemical equation for photosynthesis is:

6 CO2 + 6 H2O + light energy -> C6H12O6 + 6 O2

Photosynthesis takes place inside the chloroplasts of plant cells.
Chlorophyll, the green pigment inside chloroplasts, absorbs light most strongly
in the blue and red parts of the visible spectrum and reflects green light,
which is why leaves appear green.

The process has two main stages. The light-dependent reactions occur in the
thylakoid membranes of the chloroplast. Here, light energy is captured by
chlorophyll and used to split water molecules (photolysis), releasing oxygen
as a by-product and producing the energy carriers ATP and NADPH.

The second stage is the Calvin cycle, sometimes called the light-independent
reactions. The Calvin cycle occurs in the stroma, the fluid that surrounds the
thylakoids. It uses the ATP and NADPH produced in the light reactions to fix
carbon dioxide from the air into three-carbon sugar molecules, which the plant
combines to make glucose.

The absorption spectrum of chlorophyll explains the action spectrum of
photosynthesis. Chlorophyll a absorbs most strongly at wavelengths around 430
nanometres in the blue region and 662 nanometres in the red region. It absorbs
green and yellow light only weakly, so the rate of photosynthesis is highest
in blue and red light.

Several environmental factors limit the rate of photosynthesis. Light
intensity sets an upper bound on how fast the light reactions can run. Carbon
dioxide concentration limits the Calvin cycle. Temperature affects the speed
of the enzyme-driven reactions, with very high temperatures damaging the
enzymes. A limiting factor is the one that is in shortest supply.

Key facts:
- The primary photosynthetic pigment is chlorophyll a.
- Photolysis of water releases oxygen as a by-product of the light reactions.
- The Calvin cycle was discovered by Melvin Calvin and uses ATP and NADPH.
- Agricultural productivity depends on optimising light, CO2 and temperature.
""",
    "notes_newton.txt": """Newton's Laws of Motion Notes

Classical mechanics rests on three laws first stated by Isaac Newton in his
1687 work Principia Mathematica. Together they describe how forces change the
motion of objects.

Newton's first law, the law of inertia, states that an object at rest stays at
rest and an object in uniform straight-line motion stays in that motion unless
acted on by an unbalanced external force. Inertia is the tendency of a body to
resist changes to its velocity; a body with more mass has more inertia.

Newton's second law quantifies the relationship between force, mass and
acceleration. The net force acting on an object equals the product of its mass
and its acceleration, written F = m * a. For a fixed mass, doubling the applied
force doubles the acceleration; doubling the mass halves the acceleration for
the same force. The SI unit of force is the newton (N), defined as the force
needed to accelerate a mass of one kilogram by one metre per second squared.

Newton's third law states that for every action there is an equal and opposite
reaction. The two forces always act on different objects. When a rocket pushes
hot gas downward, the gas pushes the rocket upward with the same strength; when
you push on a wall, the wall pushes back on you equally.

Weight is a specific kind of force: the gravitational force exerted on a mass.
It is calculated from the second law as W = m * g, where g is the gravitational
field strength, about 9.8 m/s^2 on the surface of the Earth. On the Moon, where
g is about 1.6 m/s^2, the same mass has much less weight.

Forces in contact with surfaces generate friction. Friction opposes the
relative motion between two surfaces in contact and acts parallel to the
surfaces. Static friction prevents an object from starting to slide, while
kinetic friction acts on a sliding object. Friction depends on the surfaces and
the normal force pressing them together, not on the contact area.

Key facts:
- Force is measured in newtons (N); mass in kilograms (kg); acceleration in
  metres per second squared (m/s^2).
- Weight W = m * g with g about 9.8 m/s^2 on Earth.
- Momentum is mass times velocity; in a closed system it is conserved.
- Free-body diagrams show all forces acting on a single object.
""",
    "notes_cells.txt": """Cell Biology Notes

All living organisms are made of cells, the basic structural and functional
units of life. Cells were first described by Robert Hooke in 1665, and the cell
theory states that all organisms are composed of cells and all cells arise from
pre-existing cells.

There are two fundamental types of cells. Prokaryotic cells, such as bacteria,
have no nucleus and no membrane-bound organelles; their DNA floats freely in
the cytoplasm. Eukaryotic cells, such as animal and plant cells, have a nucleus
enclosing their genetic material and a variety of membrane-bound organelles
that specialise in different tasks.

The cell membrane is a phospholipid bilayer studded with proteins. It is
selectively permeable: it controls which substances enter and leave the cell.
Small non-polar molecules diffuse across freely, while larger or charged
molecules need protein channels or carriers.

The nucleus is the control centre of the eukaryotic cell. It contains
chromosomes made of DNA, which carry the genes that code for proteins. The
nucleolus inside the nucleus assembles ribosomes. Ribosomes, found free in the
cytoplasm or attached to the rough endoplasmic reticulum, are the site of
protein synthesis, joining amino acids into polypeptide chains.

Mitochondria are the powerhouses of the cell. Their inner membrane is folded
into cristae, increasing the surface area for cellular respiration, where
glucose and oxygen are converted into ATP, the energy currency of the cell.
Cells that need lots of energy, such as muscle cells, contain many
mitochondria.

Plant cells have structures that animal cells lack: a rigid cell wall made of
cellulose that provides support, chloroplasts where photosynthesis occurs, and
a large central vacuole that stores water and maintains turgor pressure.

Substances move across membranes by passive and active transport. Diffusion
and osmosis are passive: they move substances down their concentration gradient
without using energy. Osmosis is the diffusion of water through a selectively
permeable membrane. Active transport moves substances against their
concentration gradient using energy from ATP.

Cell division produces new cells. Mitosis produces two genetically identical
daughter cells for growth and repair. Meiosis produces four genetically
different gametes, each with half the chromosome number, for sexual
reproduction.

Key facts:
- Prokaryotes lack a nucleus; eukaryotes have one.
- Mitochondria perform cellular respiration producing ATP.
- Mitosis = identical cells; meiosis = genetically different gametes.
""",
    "notes_electricity.txt": """Electricity and Circuits Notes

Electricity is the flow of electric charge. Electric current is the rate of
flow of charge through a conductor, measured in amperes (A). By convention,
conventional current flows from the positive terminal to the negative terminal
of a battery, opposite to the flow of electrons.

Potential difference, commonly called voltage, is the energy transferred per
unit charge as charge moves between two points. It is measured in volts (V).
A battery of 12 volts provides 12 joules of energy per coulomb of charge that
flows.

Resistance is a measure of how strongly a component opposes the flow of
current. It is measured in ohms. A resistor converts electrical energy into
heat. The resistance of a wire increases with its length, decreases with its
cross-sectional area, and depends on the material's resistivity.

Ohm's law states that for many conductors at constant temperature, the current
through a conductor is directly proportional to the voltage across it. It is
written V = I * R, where V is voltage in volts, I is current in amperes and R
is resistance in ohms. A graph of current against voltage is a straight line
through the origin for an ohmic conductor.

Circuits can be arranged in series or in parallel. In a series circuit the
same current flows through every component, and the total resistance is the sum
of the individual resistances. If one component fails, the circuit breaks. In a
parallel circuit the voltage across each branch is the same, and the total
current is the sum of the branch currents. Each branch operates independently,
so lamps in parallel stay lit even if one is removed.

Electrical power is the rate at which electrical energy is transferred,
calculated as P = V * I, measured in watts. The energy used by an appliance is
the power multiplied by the time it operates, measured in kilowatt-hours by
domestic electricity meters.

Safety devices protect circuits and people. A fuse is a thin wire that melts
when the current through it exceeds its rated value, breaking the circuit
before the wiring overheats. Earth wires and double insulation protect users
from electric shock by carrying fault current safely away.

Alternating current (AC) reverses direction periodically, typically 50 or 60
times per second, and is used for mains power because it is easy to transform
between voltages. Direct current (DC) flows in one direction only and is
produced by batteries.

Key facts:
- Current is measured in amperes, voltage in volts, resistance in ohms.
- Ohm's law: V = I * R; power P = V * I.
- Series: same current, resistances add. Parallel: same voltage, currents add.
""",
    "notes_ww1.txt": """World War One Notes

The First World War, also called the Great War, lasted from 1914 to 1918 and
involved the major powers of Europe in one of the deadliest conflicts in
history.

The immediate trigger was the assassination of Archduke Franz Ferdinand, heir
to the Austro-Hungarian throne, by a Serbian nationalist in Sarajevo on 28 June
1914. The deeper causes were the alliances, militarism, imperialism and
nationalism that had built up across Europe over the previous decades.

The major alliances divided Europe into two hostile blocs. The Triple Entente
brought together Britain, France and Russia. The Central Powers comprised
Germany, Austria-Hungary and the Ottoman Empire. When Austria-Hungary declared
war on Serbia, the alliance system drew the great powers into war within weeks.

Fighting on the Western Front quickly bogged down into trench warfare. Millions
of soldiers faced artillery bombardment, machine-gun fire and appalling
conditions in a static line of trenches that stretched from the North Sea to
Switzerland. Major battles such as the Somme in 1916 and Verdun produced
enormous casualties for very little territorial gain.

On the Eastern Front, the war was more mobile. Germany defeated Russia at the
Battle of Tannenberg in 1914, and Russia's collapse was sealed by revolution in
1917. The Russian Revolution brought the Bolsheviks to power and led Russia to
sign the Treaty of Brest-Litovsk, leaving the war early.

The entry of the United States in April 1917, partly in response to German
unrestricted submarine warfare, shifted the balance of power decisively toward
the Entente. American troops and supplies helped the Allies push back German
offensives in 1918.

The war ended with the Armistice of 11 November 1918, agreed as Germany faced
military collapse and revolution at home. Peace was settled by the Treaty of
Versailles, signed in 1919, which imposed heavy reparations and territorial
losses on Germany and created the League of Nations.

The human cost was staggering: about 17 million people died, military and
civilian. The war also redrew the map of Europe and the Middle East, ended four
empires (German, Austro-Hungarian, Russian and Ottoman), and helped create the
political conditions that led to the Second World War.

Key facts:
- Trigger: assassination of Franz Ferdinand, Sarajevo, June 1914.
- Triple Entente: Britain, France, Russia. Central Powers: Germany,
  Austria-Hungary, Ottoman Empire.
- USA entered in 1917; armistice 11 November 1918; Treaty of Versailles 1919.
""",
}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in DOCS.items():
        path = DATA_DIR / name
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {name} ({len(content)} chars)")


if __name__ == "__main__":
    main()
