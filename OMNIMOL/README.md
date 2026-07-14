# BRAID — a reference SMILES ⇄ BRAID codec

BRAID (**B**ranch-counted, **R**elative-ring, **A**ttachment-native,
**I**nvalid-free, **D**ense) is a machine-native molecular line notation. This
repository is a **working reference implementation** plus the test suites used to
verify (and, in several places, *falsify*) the claims made about it.

> Status: research prototype. It leans on RDKit for SMILES parsing and
> sanitization; it is not a standalone spec, and it is **not** production-grade.

```bash
pip install rdkit
python -m braids encode "CC(=O)Oc1ccccc1C(=O)O"            # -> CC>1=OOC=CC=CC=C^5C>1=OO
python -m braids encode "CC(=O)Oc1ccccc1C(=O)O" --aromatic # -> CC>1=OOcccccc^5C>1=OO
python -m braids decode "CC>1=OOcccccc^5C>1=OO"            # -> CC(=O)Oc1ccccc1C(=O)O
```

Four mechanisms, all recombined prior art (see *Novelty* below):

| Mechanism | Syntax | Borrowed from |
|---|---|---|
| Organic-subset lexing, `[...]` only when needed | `C`, `Cl`, `[N+]` | SMILES |
| Length-counted branches (no parentheses) | `>k` then *k* atoms | SELFIES branch counts |
| Relative ring closure (no paired digits) | `^d` = bond *d* atoms back | DeepSMILES relative rings |
| Fragment separator, global index continues across it | `.` | SAFE |

Only **atom tokens** advance the global index; `>k`, `^d`, bond symbols and `.`
do not.

---

## TASK 1 — Novelty verification

**Verdict: BRAID is not novel at the mechanism level. It is a recombination.**

- Relative ring closure (`^d`) — this *is* DeepSMILES (O'Boyle & Dalke, 2018).
- Length-counted branches — this is the SELFIES branch-count idea (Krenn et al.,
  2020) with the brackets stripped off.
- Guaranteed-valid decoding via a valence state machine — this is the entire
  point of SELFIES.
- Fragment separator for scaffold-first generation — this is SAFE (Noutahi et
  al., 2023).

Every individual mechanism is published. The *combination* is essentially what
"OMNIMOL" independently proposed in the same conversation that produced BRAID —
two independent sketches converging on the same four ideas, which is evidence the
design space has an attractor there, **not** evidence of novelty.

**Honesty caveat about this verdict:** this environment has no web/literature
access (network is restricted to package registries), so I could not run a real
prior-art search. The claim "not novel" is based on recall of the named papers,
which is exactly the kind of thing that should be checked. Anything that *might*
be mildly novel — the totalized decoder, a continuous global index across `.`,
the canonical-rank rooting — would need a genuine search of arXiv/ChemRxiv,
the SELFIES/SAFE/DeepSMILES repos, and t-SMILES / group-SELFIES / other
grammar-based notations before any novelty claim is defensible. **Do not market
BRAID as new.**

---

## TASK 2 — What of OMNIMOL to keep / combine

**Kept and implemented — aromatic-atom retention (`--aromatic`).** OMNIMOL keeps
lowercase aromatic atoms; my first BRAID cut used Kekulé form, which *measurably
hurt compactness* (benzene `C=CC=CC=C^5` = 11 chars). Adopting OMNIMOL's choice
as an optional mode recovers it (benzene `cccccc^5` = 8 chars, matching SMILES).
Both modes round-trip all 42 test molecules. Kekulé stays the default because it
never depends on aromaticity perception; aromatic mode is opt-in.

**Kept — graceful bond downgrade.** OMNIMOL and BRAID independently specify that
an over-valent bond is reduced in order rather than erroring. Already implemented
in the clamp.

**Rejected — OMNIMOL's auto-fragmentation.** OMNIMOL says a branch onto a
saturated atom silently becomes a new *disconnected* fragment. That turns a
one-token valence difference into a different chemical entity (a salt/mixture),
which is the same "one token rewrites the molecule" failure OMNIMOL rightly
criticizes in SELFIES. BRAID instead drops the impossible bond and lets the
valence fill with implicit H — a smaller, more local perturbation. (This is a
real design disagreement, and BRAID's choice is itself imperfect; see
Limitation 4.)

**No-op — the naming.** OMNIMOL's `^n` (branch) / `@n` (ring) vs BRAID's
`>k` (branch) / `^d` (ring) are cosmetic. BRAID uses distinct glyphs for the two
so a tokenizer never has to disambiguate branch-vs-ring from context.

---

## TASK 3 — Limitations (evidence-based)

1. **Not actually more compact (the "D" is aspirational).** Measured character
   counts on canonical strings:

   | molecule | SMILES | DeepSMILES | SELFIES(tok) | BRAID(kekulé) | BRAID(aromatic) |
   |---|--:|--:|--:|--:|--:|
   | benzene | 8 | 7 | 8 | 11 | 8 |
   | aspirin | 21 | 18 | 19 | 24 | 21 |
   | caffeine | 26 | 24 | 24 | 28 | 27 |
   | morphine | 36 | 35 | 36 | 42 | — |

   BRAID is never the *shortest*. `>k`/`^d` are two characters where SMILES uses
   one. My earlier "Dense ✓" self-grade was wrong; aromatic mode only reaches
   parity, not dominance.

2. **Stereochemistry — now supported (v0.2), with caveats.** Tetrahedral
   chirality (`@`/`@@` in brackets) and E/Z double bonds (`%C`/`%T` markers) round-
   trip on 26/26 test cases in both modes, including meso vs. chiral tartaric
   acid, multi-centre molecules (menthol, L-DOPA), and conjugated dienes. Markers
   are resolved by running the actual decoder and matching RDKit CIP labels, so
   encode/decode frames are consistent by construction. **Not** yet covered:
   allene/axial/planar chirality, atropisomers, and stereo at non-tetrahedral
   centres — these fall back to unspecified. So it's "handles the stereo that
   most drug-like datasets contain," not "complete."

3. **The valence state machine is approximate.** It uses a hand-rolled
   charge→valence rule (carbon: `default−|charge|`; electronegative main group:
   `default+charge`). This is correct for common ions (N⁺→4, O⁻→1, carboxylate)
   but will mis-clamp exotic hypervalent, organometallic, or unusual-charge
   atoms. It is not a substitute for RDKit's full model.

4. **"Every string is valid" is true only because the decoder is *total*.** The
   guarantee (10,000/10,000 fuzz strings decode without crashing) is achieved by
   making unparseable tokens and over-valent bonds into **silent no-ops**. So a
   mutated string can decode to something only loosely related to its neighbour —
   dropped tokens and clamped bonds are exactly the kind of non-local jump BRAID
   claims to avoid. Validity is guaranteed; *semantic locality under mutation is
   not*.

5. **Insertion/deletion is non-local.** Substituting an atom is local, but
   *inserting* or *deleting* one shifts every `^d` that spans the edit point.
   This is inherent to relative back-references and applies equally to
   DeepSMILES and OMNIMOL.

6. **Not a canonical hash.** One molecule has many valid BRAID strings. The
   encoder uses RDKit canonical-rank rooting for a *deterministic* output, but
   uniqueness is not proven and there is no InChI-style canonical guarantee.

7. **Readability regresses vs SMILES**, and the whole thing **depends on RDKit** —
   it is not a self-contained parser/spec.

---

## TASK 4 — The package

```
braids/
  braids/
    __init__.py     public API
    codec.py        encoder + decoder + valence state machine + stereo
    tokenizer.py    token stream + Vocab for CLM training
    __main__.py     CLI (encode / decode)
  tests/
    test_roundtrip.py   42 molecules, SMILES->BRAID->SMILES constitution match
    test_stereo.py      26 chiral / E-Z molecules, stereo-preserving round-trip
    test_validity.py    10k random/mutated strings must decode without crashing
  benchmark/
    compare_validity.py Markov-based representation comparison (no GPU)
    train_clm.py        nanoGPT-style CLM harness (needs torch + .smi data)
  pyproject.toml
```

Python API:

```python
from braids import smiles_to_braid, braid_to_smiles, mol_to_braid, braid_to_mol

smiles_to_braid("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")      # caffeine -> BRAID
braid_to_smiles("CC>1=OO")                            # -> "CC(=O)O"
braid_to_mol(any_string, clamp=True)                  # always returns a valid Mol
```

**Verified results (reproduce with `python3 tests/*.py`):**

- `test_roundtrip.py`: **42/42** molecules round-trip in *both* Kekulé and
  aromatic modes — including fused/bridged systems (naphthalene, indole,
  adamantane, caffeine, purine, quinoline, nicotine) and ions (nitromethane,
  acetate, ammonium, Na⁺·Cl⁻).
- `test_stereo.py`: **26/26** chiral / E-Z molecules round-trip in both modes —
  L-alanine, (S)-ibuprofen, meso vs. L-tartaric, menthol, L-DOPA, adrenaline,
  carvone, fumaric/maleic acid, conjugated dienes, mixed chiral+E/Z.
- `test_validity.py`: **10,000/10,000** random and mutated strings (including
  chirality and `%C`/`%T` tokens) decode to a sanitizable molecule with **0
  crashes**.

## For chemical language models (CLM)

`braids/tokenizer.py` gives a clean token stream and a `Vocab` builder — every
token is one chemically-meaningful unit (atom, `>k`, `^d`, bond, `.`, stereo).

`benchmark/compare_validity.py` (no GPU needed) compares representations under an
identical order-k Markov generator. Measured on a 79-molecule inline corpus:

| representation | valid% | unique% | novel% | vocab | tok/mol |
|---|--:|--:|--:|--:|--:|
| SMILES | 40.0 | 38.4 | 34.0 | 24 | 13.6 |
| DeepSMILES | 35.4 | 46.9 | 42.4 | 27 | 13.3 |
| SELFIES | 100.0 | 59.6 | 57.9 | 29 | 12.9 |
| **BRAID** | **100.0** | 60.9 | 59.2 | 30 | 13.2 |

BRAID matches SELFIES on the property that matters here — **100% of generated
strings are valid molecules** — where SMILES/DeepSMILES leak invalid strings. A
Markov model is far weaker than a real CLM, so treat validity% as the meaningful
(model-independent) signal and uniqueness/novelty as illustrative.

`benchmark/train_clm.py` is a real nanoGPT-style harness (needs `torch` + a `.smi`
file) to run the same comparison under an actual transformer. Keep model size,
steps, and sampling identical across `--rep {smiles,deepsmiles,selfies,braid}`
runs; for a publishable result add FCD, KL divergence, and scaffold metrics
(guacamol / fcd_torch / moses).

Notably, the codec produces a *correct* caffeine BRAID by construction
(`CNC=NC=C^4C>1=ON>1CC>1=ON^8C`), in contrast to the hand-written OMNIMOL
caffeine example, which was one carbon short.

---

## TASK 5 — Self-critique

- **I graded my own design and got it wrong.** My earlier comparison table gave
  BRAID a clean sweep of ✓s. Writing the tests corrected three of those claims:
  compactness (BRAID is longer, not shorter), and two correctness bugs that only
  a runnable implementation exposed — a multi-branch decoder bug (neopentane,
  tert-butanol, sulfuric acid all decoded wrong) and a charged-valence bug
  (nitromethane). Both are the kind of error that hand-tracing a couple of easy
  examples hides — the same trap the OMNIMOL caffeine example fell into.

- **The headline "invalid-free" claim was initially false and only became true
  after a fix.** The first decoder *crashed* on tokens outside the encoder's own
  alphabet (e.g. aromatic `[nH]`). "Every string is valid" held only after I made
  the decoder total. That means the guarantee is partly an artifact of *silently
  ignoring* input, which weakens the very locality property it was meant to
  support (Limitation 4). I'd rather state that plainly than let the green
  checkmark stand.

- **Novelty is essentially nil and I can't even verify that properly here.** No
  literature access; the design converged with an independent proposal in the
  same thread. The right posture is "recombination, unverified," not "new."

- **Scope was cut where it's most costly.** Dropping stereochemistry makes the
  round-trip proofs look stronger than a production codec would; stereo is
  exactly where real molecular representation gets hard, and I punted on it.

- **The framing to resist:** the original request was to "optimize all
  representations." No single format dominates on every axis — validity trades
  against compactness, relative pointers trade authoring-locality for
  mutation-safety, and canonicality (InChI's job) is untouched here. A codec that
  claims to win everywhere should be distrusted; this one included.
