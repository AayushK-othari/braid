"""
Stress the 'every lexically-valid BRAID string decodes to a valid molecule'
claim, the way a generative model would break it: random token soup, and
point-mutations of real strings.
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdkit import Chem
from braids import smiles_to_braid, braid_to_mol
from braids.codec import _tokenize

ATOM_TOKS = ["C", "N", "O", "S", "P", "F", "Cl", "Br",
             "[N+]", "[O-]", "[NH4+]", "[nH]", "[2H]"]
OPS = [">1", ">2", ">3", "^1", "^2", "^3", "^5", "=^2", "=", "#", "."]
ALL = ATOM_TOKS + OPS

SEED_SMILES = ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
               "c1ccccc1", "CC(C)(C)O", "C[N+](=O)[O-]", "OS(=O)(=O)O"]


def sanitizable(mol):
    """A mol counts as 'valid' if RDKit sanitize succeeds (or it's empty)."""
    if mol.GetNumAtoms() == 0:
        return True
    m = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(m)
        return True
    except Exception:
        return False


def random_string(rng, n):
    return "".join(rng.choice(ALL) for _ in range(n))


def mutate(rng, braid):
    toks = [t for _, t in _tokenize(braid)]
    if not toks:
        return braid
    op = rng.choice(["sub", "ins", "del", "dup"])
    i = rng.randrange(len(toks))
    if op == "sub":
        toks[i] = rng.choice(ALL)
    elif op == "ins":
        toks.insert(i, rng.choice(ALL))
    elif op == "del":
        toks.pop(i)
    elif op == "dup":
        toks.insert(i, toks[i])
    return "".join(toks)


def main():
    rng = random.Random(0xBEEF)
    n_random, n_mut = 5000, 5000
    bad = []

    # 1) random token soup
    crashes = valid = 0
    for _ in range(n_random):
        s = random_string(rng, rng.randint(1, 25))
        try:
            mol = braid_to_mol(s, clamp=True)
            if sanitizable(mol):
                valid += 1
            else:
                bad.append(("random-invalid", s))
        except Exception as e:
            crashes += 1
            bad.append(("random-crash", s, repr(e)))
    print(f"random soup : {valid}/{n_random} valid, {crashes} crashes")

    # 2) mutations of real molecules
    mvalid = mcrash = 0
    for _ in range(n_mut):
        base = smiles_to_braid(rng.choice(SEED_SMILES))
        for _ in range(rng.randint(1, 4)):
            base = mutate(rng, base)
        try:
            mol = braid_to_mol(base, clamp=True)
            if sanitizable(mol):
                mvalid += 1
            else:
                bad.append(("mut-invalid", base))
        except Exception as e:
            mcrash += 1
            bad.append(("mut-crash", base, repr(e)))
    print(f"mutations   : {mvalid}/{n_mut} valid, {mcrash} crashes")

    print(f"\ntotal invalid/crash cases: {len(bad)}")
    for b in bad[:15]:
        print("   ", b)
    return len(bad)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
