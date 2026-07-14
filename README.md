# BRAID

**B**ranch-counted, **R**elative-ring, **A**ttachment-native, **I**nvalid-free, **D**ense —
a machine-native molecular line notation in which every string decodes to a valid molecule.

This repository is a **working reference implementation** plus the test suites used to
verify (and, in several places, *falsify*) the claims made about it.

> **Status: research prototype.** It leans on RDKit for SMILES parsing and sanitization; it
> is not a standalone spec, and it is not production-grade. BRAID is a *recombination* of
> published mechanisms.

The pretrained model, **BRAIDBERTa-v9**, lives on the Hub:
[aakothari/BRAIDBERTa-v9](https://huggingface.co/aakothari/BRAIDBERTa-v9).

---

## Install

```bash
pip install rdkit
pip install git+https://github.com/aakothari/braids.git
```

## Quickstart

```bash
python -m braids encode "CC(=O)Oc1ccccc1C(=O)O"            # -> CC>1=OOC=CC=CC=C^5C>1=OO
python -m braids encode "CC(=O)Oc1ccccc1C(=O)O" --aromatic # -> CC>1=OOcccccc^5C>1=OO
python -m braids decode "CC>1=OOcccccc^5C>1=OO"            # -> CC(=O)Oc1ccccc1C(=O)O
```

```python
from braids import smiles_to_braid, braid_to_smiles, mol_to_braid, braid_to_mol

smiles_to_braid("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")   # caffeine -> BRAID
braid_to_smiles("CC>1=OO")                        # -> "CC(=O)O"
braid_to_mol(any_string, clamp=True)              # always returns a valid Mol
```

---

## How it works

Four mechanisms:

| Mechanism | Syntax | Borrowed from |
|---|---|---|
| Organic-subset lexing, `[...]` only when needed | `C`, `Cl`, `[N+]` | SMILES |
| Length-counted branches (no parentheses) | `>k` then *k* atoms | SELFIES branch counts |
| Relative ring closure (no paired digits) | `^d` = bond *d* atoms back | DeepSMILES relative rings |
| Fragment separator, global index continues across it | `.` | SAFE |

Only **atom tokens** advance the global index; `>k`, `^d`, bond symbols and `.` do not.

Two modes. **Kekulé** (default) never depends on aromaticity perception. **Aromatic**
(`--aromatic`) retains lowercase aromatic atoms and is measurably more compact
(benzene: `cccccc^5` = 8 chars vs `C=CC=CC=C^5` = 11). Both round-trip all test molecules.
Whichever you train on, you must also infer with — the strings are not interchangeable.

---

## Novelty: none claimed

**BRAID is not novel at the mechanism level. It is a recombination.**

- Relative ring closure (`^d`) — this *is* DeepSMILES (O'Boyle & Dalke, 2018).
- Length-counted branches — the SELFIES branch-count idea (Krenn et al., 2020) with the
  brackets stripped off.
- Guaranteed-valid decoding via a valence state machine — the entire point of SELFIES.
- Fragment separator for scaffold-first generation — SAFE (Noutahi et al., 2023).

Every individual mechanism is published. The *combination* converged with an independently
sketched proposal ("OMNIMOL") on all four ideas, which is evidence that the design space has
an attractor there — **not** evidence of novelty.

---

## What is verified

Reproduce with `python -m pytest tests/`:

| Test | Result |
|---|---|
| `test_roundtrip.py` | **42/42** molecules round-trip in *both* modes — including fused/bridged systems (naphthalene, indole, adamantane, caffeine, purine, quinoline, nicotine) and ions (nitromethane, acetate, ammonium, Na⁺·Cl⁻) |
| `test_stereo.py` | **26/26** chiral / E-Z molecules round-trip in both modes — L-alanine, (S)-ibuprofen, meso vs. L-tartaric acid, menthol, L-DOPA, adrenaline, carvone, fumaric/maleic acid, conjugated dienes, mixed chiral + E/Z |
| `test_validity.py` | **10,000/10,000** random and mutated strings decode to a sanitizable molecule, **0 crashes** |

---

## Limitations

**1. Stereochemistry is partial.** Tetrahedral chirality (`@`/`@@`) and E/Z double bonds
(`%C`/`%T`) round-trip on 26/26 cases. Markers are resolved by running the actual decoder
and matching RDKit CIP labels, so encode/decode frames are consistent by construction. Not
covered: allene/axial/planar chirality, atropisomers, non-tetrahedral centres — these fall
back to unspecified. So: "handles the stereo most drug-like datasets contain," not
"complete."

**2. The valence state machine is approximate.** A hand-rolled charge→valence rule (carbon:
`default − |charge|`; electronegative main group: `default + charge`). Correct for common
ions (N⁺→4, O⁻→1, carboxylate); will mis-clamp exotic hypervalent, organometallic, or
unusual-charge atoms. Not a substitute for RDKit's model.

**3. Insertion/deletion is non-local.** Substituting an atom is local, but inserting or
deleting one shifts every `^d` that spans the edit point. Inherent to relative
back-references; applies equally to DeepSMILES.

**4. Not a canonical hash.** One molecule has many valid BRAID strings. The encoder uses
RDKit canonical-rank rooting for a *deterministic* output, but uniqueness is not proven and
there is no InChI-style guarantee.

---

## Chemical language modelling

`braids/tokenizer.py` gives a token stream and a `Vocab` builder — every token is one
chemically-meaningful unit (atom, `>k`, `^d`, bond, `.`, stereo).

### Downstream: BBBP

BRAIDBERTa-v9 (RoBERTa MLM, ZINC 100k, BRAID-encoded) vs. an otherwise-identical SMILES
control. Optuna hyperparameter search on validation, then 5 seeds. Test ROC-AUC:

| Model | Notation | BBBP ROC-AUC |
|---|---|---|
| BRAIDBERTa-v9 | BRAID | 0.725 ± 0.017 |
| SMILES control | SMILES | 0.720 ± 0.010 |

**These are statistically indistinguishable** (Welch's *t* ≈ 0.61, *p* ≈ 0.56; seed spread
exceeds the between-model gap). The result is not "BRAID wins." It is:

> BRAID's guaranteed-validity property comes **at no measurable cost** to downstream
> predictive performance on BBBP.

One dataset, one split, a 100k pretraining corpus, n = 5. Treat accordingly.

### Generative validity

`benchmark/compare_validity.py` compares representations under an identical order-*k* Markov
generator (no GPU needed). It reports 100% valid for both SELFIES and BRAID — but **that is
a property of the notations, not a finding**: both are constructed so that every string
decodes, and the benchmark cannot return any other number. For BRAID it additionally
inherits the caveat in Limitation 4. The SMILES/DeepSMILES figures (40.0% / 35.4%) are the
only *measured* quantities in that table, and the uniqueness/novelty columns are computed
against a 79-molecule corpus, which makes them close to meaningless.

`benchmark/train_clm.py` is a nanoGPT-style harness (needs `torch` + a `.smi` file) to run
the comparison under an actual transformer. Keep model size, steps, and sampling identical
across `--rep {smiles,deepsmiles,selfies,braid}`. **For a publishable generative result you
need this, plus FCD, KL divergence, and scaffold metrics** (guacamol / fcd_torch / moses) —
not the Markov proxy.

---

## Repository layout

```
braids/
  braids/
    __init__.py      public API
    codec.py         encoder + decoder + valence state machine + stereo
    tokenizer.py     token stream + Vocab for CLM training
    __main__.py      CLI (encode / decode)
  tests/
    test_roundtrip.py   42 molecules, SMILES -> BRAID -> SMILES constitution match
    test_stereo.py      26 chiral / E-Z molecules, stereo-preserving round-trip
    test_validity.py    10k random/mutated strings must decode without crashing
  benchmark/
    compare_validity.py Markov-based representation comparison (no GPU)
    train_clm.py        nanoGPT-style CLM harness (needs torch + .smi data)
  braid_tok/         trained BPE tokenizer for BRAIDBERTa
  pyproject.toml
```

Model weights are **not** in this repo — they are on the Hub at
[aakothari/BRAIDBERTa-v9](https://huggingface.co/aakothari/BRAIDBERTa-v9).

## References

- Weininger, D. *SMILES, a chemical language and information system.* J. Chem. Inf. Comput. Sci., 1988.
- O'Boyle, N. & Dalke, A. *DeepSMILES: An Adaptation of SMILES for Use in Machine-Learning of Chemical Structures.* ChemRxiv, 2018.
- Krenn, M. et al. *Self-Referencing Embedded Strings (SELFIES): A 100% robust molecular string representation.* Machine Learning: Science and Technology, 2020.
- Noutahi, E. et al. *Gotta be SAFE: A New Framework for Molecular Design.* 2023.

