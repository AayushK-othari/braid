"""Round-trip: SMILES -> BRAID -> SMILES must preserve constitution."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdkit import Chem
from braids import smiles_to_braid, braid_to_smiles

CASES = {
    "methane": "C",
    "ethanol": "CCO",
    "isobutane": "CC(C)C",
    "neopentane": "CC(C)(C)C",
    "benzene": "c1ccccc1",
    "toluene": "Cc1ccccc1",
    "phenol": "Oc1ccccc1",
    "pyridine": "c1ccncc1",
    "pyrrole": "c1cc[nH]c1",
    "imidazole": "c1cnc[nH]1",
    "naphthalene": "c1ccc2ccccc2c1",
    "indole": "c1ccc2[nH]ccc2c1",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "acetic_acid": "CC(=O)O",
    "glycine": "NCC(=O)O",
    "diphenylmethane": "c1ccccc1Cc1ccccc1",
    "biphenyl": "c1ccccc1-c1ccccc1",
    "nitromethane": "C[N+](=O)[O-]",
    "acetate_ion": "CC(=O)[O-]",
    "ammonium": "[NH4+]",
    "trimethylamine": "CN(C)C",
    "cyclohexane": "C1CCCCC1",
    "cyclopropane": "C1CC1",
    "adamantane": "C1C2CC3CC1CC(C2)C3",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "acetonitrile": "CC#N",
    "carbon_dioxide": "O=C=O",
    "salt_mixture": "[Na+].[Cl-]",
    "ethylamine_hcl": "CCN.Cl",
    "isopropanol": "CC(C)O",
    "tert_butanol": "CC(C)(C)O",
    "quinoline": "c1ccc2ncccc2c1",
    "purine": "c1ncc2nc[nH]c2n1",
    "sulfuric_acid": "OS(=O)(=O)O",
    "dimethyl_sulfoxide": "CS(=O)C",
    "chloroform": "C(Cl)(Cl)Cl",
    "deuterium_water": "[2H]O[2H]",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "nicotine": "CN1CCCC1c1cccnc1",
}


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


def main():
    passed, failed = 0, 0
    fails = []
    for name, smi in CASES.items():
        try:
            braid = smiles_to_braid(smi)
            back = braid_to_smiles(braid)
            if canon(smi) == canon(back):
                passed += 1
                print(f"  PASS  {name:20s} {smi:35s} -> {braid}")
            else:
                failed += 1
                fails.append(name)
                print(f"  FAIL  {name:20s} {smi}")
                print(f"        braid={braid}")
                print(f"        expected {canon(smi)}  got {canon(back)}")
        except Exception as e:
            failed += 1
            fails.append(name)
            print(f"  ERROR {name:20s} {smi}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{passed+failed} round-tripped.  Failures: {fails}")
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
