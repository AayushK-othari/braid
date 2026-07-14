"""
Micro-benchmark: how often does a *generator* trained on each representation
emit a VALID molecule?  This is the axis SELFIES/BRAID are built for.

It deliberately uses a cheap order-k Markov model over tokens (no GPU, no torch)
so it runs anywhere. A Markov model is far weaker than a real CLM, so treat the
*distributional* numbers (uniqueness/novelty) as illustrative only. The
*validity* numbers, however, are meaningful: validity-by-construction does not
depend on model quality, which is exactly the point being tested.

For a real comparison, swap the Markov sampler for the CLM in train_clm.py and
keep these same metrics.

Run:  python benchmark/compare_validity.py
"""
import sys, os, random, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from OMNIMOL import smiles_to_braid, braid_to_smiles
from OMNIMOL.tokenizer import tokenize as braid_tokenize

try:
    import selfies as sf
except ImportError:
    sf = None
try:
    import deepsmiles
    _DS = deepsmiles.Converter(rings=True, branches=True)
except ImportError:
    _DS = None

# A small, diverse drug-like corpus (kept inline so the benchmark is standalone).
CORPUS = [
    "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CC(=O)Nc1ccc(O)cc1", "CN1CCC[C@H]1c1cccnc1", "OC(=O)C1=CC=CC=C1O",
    "C1=CC(=C(C=C1CCN)O)O", "CN(C)CCC=C1c2ccccc2CCc2ccccc21", "Cc1ccc(cc1)S(=O)(=O)N",
    "CC(C)NCC(O)COc1ccccc1CC=C", "Clc1ccccc1C(=O)c1ccc(Cl)cc1", "O=C(O)c1ccccc1Nc1ccccc1",
    "CCN(CC)CCNC(=O)c1cc(Cl)c(N)cc1OC", "CN1CCN(CC1)C1=Nc2ccccc2Nc2ccccc21",
    "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1", "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
    "OC1CCCCC1", "c1ccc2ccccc2c1", "c1ccc2[nH]ccc2c1", "c1ccncc1", "c1cc[nH]c1",
    "CC(=O)O", "CCO", "CCN", "CC#N", "O=C=O", "C1CCCCC1", "C1CCOC1",
    "C[C@@H](N)C(=O)O", "C[C@H](O)C(=O)O", "O[C@@H]([C@H](O)C(=O)O)C(=O)O",
    "C/C=C/C", "C/C=C\\C", "OC(=O)/C=C/C(=O)O", "CC(C)C[C@@H](C)C(=O)O",
    "CNC[C@@H](O)c1ccc(O)c(O)c1", "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O",
    "Nc1ccccc1", "Oc1ccccc1", "Cc1ccccc1", "Clc1ccccc1", "Fc1ccccc1",
    "c1ccc(cc1)-c1ccccc1", "c1ccc(cc1)Cc1ccccc1", "O=C1CCCCC1", "O=C1CCCC1",
    "CC(=O)C", "CCCCO", "CCCCCC", "CC(C)CC(C)C", "c1ccsc1", "c1ccoc1",
    "c1cnc2[nH]cnc2c1", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "OCC(O)CO", "NCCO",
    "CC(=O)Nc1ccc(cc1)O", "COc1ccccc1", "CCOC(=O)C", "CC(=O)OCC",
    "c1ccc(cc1)C(=O)O", "c1ccc(cc1)N", "c1ccc(cc1)O", "OCc1ccccc1",
    "C1=CC=C(C=C1)C=O", "CC(=O)c1ccccc1", "CCc1ccccc1", "CCCc1ccccc1",
    "O=S(=O)(O)O", "O=[N+]([O-])c1ccccc1", "C[N+](=O)[O-]", "CC(=O)[O-]",
    "c1ccc2c(c1)cccc2", "c1ccc2c(c1)[nH]c1ccccc12", "O=C(N)c1ccccc1",
    "CC(N)C(=O)O", "CSCCC(N)C(=O)O", "OC(=O)CCC(N)C(=O)O", "NC(CC(=O)O)C(=O)O",
]


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


# ------- representation adapters: SMILES <-> tokens, and validity check ------
def smiles_tokens(smi):
    import re
    pat = re.compile(r"(\[[^\]]+\]|Br|Cl|@@|@|=|#|\(|\)|\.|/|\\|[0-9]|%[0-9]{2}|[A-Za-z])")
    return pat.findall(smi)


def make_rep(kind):
    if kind == "SMILES":
        return (lambda s: smiles_tokens(canon(s)),
                lambda toks: canon("".join(toks)))
    if kind == "DeepSMILES":
        return (lambda s: list(_DS.encode(canon(s))),
                lambda toks: _safe(lambda: canon(_DS.decode("".join(toks)))))
    if kind == "SELFIES":
        return (lambda s: list(sf.split_selfies(sf.encoder(canon(s)))),
                lambda toks: _safe(lambda: canon(sf.decoder("".join(toks)))))
    if kind == "BRAID":
        return (lambda s: braid_tokenize(smiles_to_braid(s)),
                lambda toks: _safe(lambda: canon(braid_to_smiles("".join(toks)))))


def _safe(fn):
    try:
        return fn()
    except Exception:
        return None


# ----------------------------- order-k Markov -------------------------------
class Markov:
    def __init__(self, k=3):
        self.k = k
        self.table = collections.defaultdict(collections.Counter)

    def fit(self, seqs):
        for seq in seqs:
            s = ["<bos>"] * self.k + list(seq) + ["<eos>"]
            for i in range(self.k, len(s)):
                self.table[tuple(s[i - self.k:i])][s[i]] += 1
        return self

    def sample(self, rng, max_len=120):
        ctx = tuple(["<bos>"] * self.k)
        out = []
        for _ in range(max_len):
            counter = self.table.get(ctx)
            if not counter:
                break
            toks, wts = zip(*counter.items())
            nxt = rng.choices(toks, weights=wts)[0]
            if nxt == "<eos>":
                break
            out.append(nxt)
            ctx = tuple((list(ctx) + [nxt])[-self.k:])
        return out


def evaluate(kind, n_samples=2000, k=3, seed=0):
    enc, dec = make_rep(kind)
    train_seqs = [enc(s) for s in CORPUS]
    train_canon = set(canon(s) for s in CORPUS)
    model = Markov(k).fit(train_seqs)
    rng = random.Random(seed)

    valid, mols = 0, []
    for _ in range(n_samples):
        toks = model.sample(rng)
        smi = dec(toks)
        if smi:
            valid += 1
            mols.append(smi)
    uniq = len(set(mols))
    novel = len(set(mols) - train_canon)
    return {
        "valid%": 100 * valid / n_samples,
        "unique% (of valid)": 100 * uniq / max(1, valid),
        "novel% (of valid)": 100 * novel / max(1, valid),
        "vocab": len({t for seq in train_seqs for t in seq}),
        "avg tokens/mol": round(sum(len(s) for s in train_seqs) / len(train_seqs), 1),
    }


def main():
    reps = ["SMILES"]
    if _DS: reps.append("DeepSMILES")
    if sf: reps.append("SELFIES")
    reps.append("BRAID")

    print(f"Corpus: {len(CORPUS)} molecules | order-{3} Markov | 2000 samples each\n")
    header = ["representation", "valid%", "unique%", "novel%", "vocab", "tok/mol"]
    print("  ".join(f"{h:>14s}" for h in header))
    for kind in reps:
        r = evaluate(kind, k=3)
        row = [kind, f"{r['valid%']:.1f}", f"{r['unique% (of valid)']:.1f}",
               f"{r['novel% (of valid)']:.1f}", str(r["vocab"]),
               str(r["avg tokens/mol"])]
        print("  ".join(f"{c:>14s}" for c in row))
    print("\nNote: Markov <<< real CLM. Validity% is the model-independent signal;")
    print("uniqueness/novelty are illustrative and will rise sharply with a real model.")


if __name__ == "__main__":
    main()
