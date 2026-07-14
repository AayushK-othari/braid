"""CLI:  python -m braids encode "CC(=O)O"   |   python -m braids decode "CC>1=OO" """
import argparse, sys
from .codec import smiles_to_braid, braid_to_smiles


def main(argv=None):
    p = argparse.ArgumentParser(prog="braids", description="SMILES <-> BRAID")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode", help="SMILES -> BRAID")
    e.add_argument("smiles")
    e.add_argument("--aromatic", action="store_true",
                   help="keep aromatic lowercase atoms (more compact)")
    d = sub.add_parser("decode", help="BRAID -> canonical SMILES")
    d.add_argument("braid")
    d.add_argument("--no-clamp", action="store_true",
                   help="disable the valence state machine")
    a = p.parse_args(argv)
    if a.cmd == "encode":
        print(smiles_to_braid(a.smiles, aromatic=a.aromatic))
    else:
        print(braid_to_smiles(a.braid, clamp=not a.no_clamp))


if __name__ == "__main__":
    main(sys.argv[1:])
