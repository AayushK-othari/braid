"""
Build ONE shared molecule set that every representation can encode, so a CLM
comparison trains all reps on identical data (the rigorous way to compare).

    python benchmark/make_common.py --data 100k_rndm_zinc_drugs_clean.txt --no-stereo

Writes "<data>.common.smi" (canonical SMILES, one per line) containing only the
molecules that smiles + deepsmiles + selfies + braid can all round-trip, and
prints how many each representation individually dropped. Then train everything
on the common file:

    python benchmark/train_clm.py --data 100k_rndm_zinc_drugs_clean.txt.common.smi --rep braid   --no-stereo ...
    python benchmark/train_clm.py --data 100k_rndm_zinc_drugs_clean.txt.common.smi --rep selfies --no-stereo ...
    (etc. -- note: no --no-stereo is needed if the common file is already stripped,
     but pass it anyway so tokenization matches; it's idempotent.)
"""
import argparse, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from train_clm import get_adapter, _canon   # reuse the same adapters


REPS = ["smiles", "deepsmiles", "selfies", "braid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--no-stereo", action="store_true")
    a = ap.parse_args()
    keep = not a.no_stereo

    encoders = {}
    for r in REPS:
        try:
            encoders[r] = get_adapter(r, keep)[0]
        except Exception as e:
            print(f"[skip] {r}: {e!r}  (install it to include it)")

    out_path = a.data + ".common.smi"
    kept = 0
    drops = {r: 0 for r in encoders}
    total = 0
    with open(a.data, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        for line in f:
            s = line.strip().split()[0] if line.strip() else ""
            if not s:
                continue
            total += 1
            cs = _canon(s, keep)
            if cs is None:
                continue
            ok = True
            for r, enc in encoders.items():
                try:
                    if not enc(cs):
                        drops[r] += 1; ok = False
                except Exception:
                    drops[r] += 1; ok = False
            if ok:
                out.write(cs + "\n"); kept += 1
            if total % 5000 == 0:
                print(f"  ...{total} scanned, {kept} common so far")

    print(f"\nScanned {total} molecules.")
    for r in encoders:
        print(f"  {r:11s} could not encode: {drops[r]}")
    print(f"Common set (all reps OK): {kept}  ->  {out_path}")
    if kept < total:
        print("Train every --rep on this .common.smi file for an apples-to-apples run.")


if __name__ == "__main__":
    main()
