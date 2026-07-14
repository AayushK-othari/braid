"""Stereochemistry round-trip: tetrahedral (@/@@) and E/Z double bonds."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdkit import Chem
from braids import smiles_to_braid, braid_to_smiles

CASES = {
    # tetrahedral
    "L-alanine": "C[C@@H](N)C(=O)O",
    "D-alanine": "C[C@H](N)C(=O)O",
    "bromochlorofluoromethane": "[C@H](F)(Cl)Br",
    "(S)-ibuprofen": "CC(C)C[C@@H](C)C(=O)O",
    "L-tartaric": "O[C@@H]([C@H](O)C(=O)O)C(=O)O",
    "meso-tartaric": "O[C@@H]([C@@H](O)C(=O)O)C(=O)O",
    "(S)-nicotine": "CN1CCC[C@H]1c1cccnc1",
    "L-lactic-acid": "C[C@H](O)C(=O)O",
    "L-serine": "C([C@@H](C(=O)O)N)O",
    "L-proline": "C1C[C@H](NC1)C(=O)O",
    "(1S,2S)": "C[C@@H](O)[C@@H](C)Cl",
    "trans-4-Me-cyclohexanol": "C[C@H]1CC[C@@H](O)CC1",
    "adrenaline": "CNC[C@@H](O)c1ccc(O)c(O)c1",
    "(R)-carvone": "CC(=C)[C@@H]1CC=C(C)C(=O)C1",
    "L-menthol": "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O",
    "L-DOPA": "C1=CC(=C(C=C1C[C@@H](C(=O)O)N)O)O",
    # E/Z double bonds
    "trans-2-butene": "C/C=C/C",
    "cis-2-butene": "C/C=C\\C",
    "E-difluoroethene": "F/C=C/F",
    "Z-difluoroethene": "F/C=C\\F",
    "trans-trans-diene": "C/C=C/C=C/C",
    "trisub-alkene": "CC/C=C(C)/C(=O)O",
    "fumaric-acid": "OC(=O)/C=C/C(=O)O",
    "maleic-acid": "OC(=O)/C=C\\C(=O)O",
    "retinoic-frag": "C/C=C/C=C(C)/C=C/C",
    # mixed
    "chiral+EZ": "C/C=C/[C@@H](C)O",
}


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def main():
    ok, fails = 0, []
    for name, smi in CASES.items():
        try:
            for aromatic in (False, True):
                br = smiles_to_braid(smi, aromatic=aromatic)
                back = braid_to_smiles(br)
                if canon(smi) != canon(back):
                    raise AssertionError(f"[aromatic={aromatic}] {br} :: "
                                         f"{canon(smi)} != {canon(back)}")
            ok += 1
            print(f"  PASS  {name:26s} {smi:34s} -> {smiles_to_braid(smi)}")
        except Exception as e:
            fails.append(name)
            print(f"  FAIL  {name:26s} {smi}\n        {e}")
    print(f"\n{ok}/{len(CASES)} stereo round-trip (both modes).  Failures: {fails}")
    return len(fails)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
