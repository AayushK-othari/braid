"""
CLM comparison harness (fixed). Needs: torch, rdkit (+ selfies/deepsmiles for
those reps). Trains a small GPT on a chosen representation and reports
validity / uniqueness / novelty on sampled molecules.

    python benchmark/train_clm.py --data zinc.smi --rep braid --steps 3000
    python benchmark/train_clm.py --data zinc.smi --rep braid --no-stereo --steps 3000
    python benchmark/train_clm.py --data zinc.smi --rep selfies --steps 3000

Notes
-----
* --rep braid and --rep omnimol are the same codec (this repo).
* The encoded dataset is CACHED next to --data as
  "<data>.<rep>.<stereo|nostereo>.enc" so repeat runs skip re-encoding.
  Delete that file to force a re-encode.
* Keep model size / --steps / sampling identical across --rep runs so the only
  variable is the representation.
"""
import argparse, os, random, sys

# make the sibling package importable no matter what it's named / where run from
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                    # repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # grandparent, just in case

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


def _canon(s, keep_stereo):
    """Canonical SMILES, optionally stripping stereo so --no-stereo is applied
    UNIFORMLY across every representation (fair comparison)."""
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    if not keep_stereo:
        Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


# ------------------------- representation adapters --------------------------
def _import_codec():
    """Find the BRAID/OMNIMOL codec whatever the package folder is called."""
    import importlib
    last = None
    for name in ("braids.codec", "OMNIMOL.codec", "omnimol.codec", "codec"):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as e:
            last = e
    raise ImportError(
        "Could not import the codec. Make sure this file sits in a 'benchmark/' "
        "folder next to your package folder (braids/ or OMNIMOL/), and that the "
        "package folder has an __init__.py.\nLast error: %r" % last)


def get_adapter(rep, keep_stereo=True):
    rep = rep.lower()
    ks = keep_stereo
    if rep == "smiles":
        import re
        pat = re.compile(r"(\[[^\]]+\]|Br|Cl|@@|@|=|#|\(|\)|\.|/|\\|[0-9]|%[0-9]{2}|[A-Za-z])")
        return (lambda s: pat.findall(_canon(s, ks)),
                lambda t: _safe(lambda: _canon("".join(t), ks)))
    if rep == "deepsmiles":
        import deepsmiles
        c = deepsmiles.Converter(rings=True, branches=True)
        return (lambda s: list(c.encode(_canon(s, ks))),
                lambda t: _safe(lambda: _canon(c.decode("".join(t)), ks)))
    if rep == "selfies":
        import selfies as sf
        return (lambda s: list(sf.split_selfies(sf.encoder(_canon(s, ks)))),
                lambda t: _safe(lambda: _canon(sf.decoder("".join(t)), ks)))
    if rep in ("braid", "omnimol"):
        codec = _import_codec()
        stb, bts, tok = codec.smiles_to_braid, codec.braid_to_smiles, codec._tokenize
        return (lambda s: [text for _kind, text in tok(stb(s, stereo=ks))],
                lambda t: _safe(lambda: _canon(bts("".join(t)), ks)))
    raise ValueError("unknown --rep %r" % rep)


def _safe(fn):
    try:
        return fn()
    except Exception:
        return None


# --------------------------- data loading + cache ---------------------------
def load_sequences(data_path, rep, keep_stereo):
    tag = "stereo" if keep_stereo else "nostereo"
    cache = f"{data_path}.{rep.lower()}.{tag}.enc"
    if os.path.exists(cache):
        print(f"[data] using cache {cache}")
        seqs = [line.rstrip("\n").split(" ") for line in open(cache, encoding="utf-8")
                if line.strip()]
        return seqs, None, cache

    enc, _dec = get_adapter(rep, keep_stereo)
    print(f"[data] encoding {data_path} as {rep} ({tag}) -> {cache}")
    seqs, smis, skipped, total = [], [], 0, 0
    tmp = cache + ".tmp"            # write to temp, rename only on success,
    try:                           # so an interrupted run leaves NO usable cache
        with open(data_path, encoding="utf-8") as f, open(tmp, "w", encoding="utf-8") as out:
            for i, line in enumerate(f):
                s = line.strip().split()[0] if line.strip() else ""
                if not s:
                    continue
                total += 1
                try:
                    toks = enc(s)
                    if toks:
                        seqs.append(toks); smis.append(s)
                        out.write(" ".join(toks) + "\n")
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1
                if (i + 1) % 5000 == 0:
                    print(f"  ...{i+1} lines, {len(seqs)} ok, {skipped} skipped")
        os.replace(tmp, cache)     # atomic: cache now exists only if we finished
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    frac = 100 * skipped / max(1, total)
    print(f"[data] done: {len(seqs)} encoded, {skipped} skipped ({frac:.1f}%)")
    if frac > 5:
        print(f"[data] WARNING: {rep} dropped {frac:.1f}% of molecules. Reps that "
              f"drop different molecules are NOT directly comparable — consider "
              f"benchmark/make_common.py to train all reps on one shared set.")
    return seqs, smis, cache


# ------------------------------- metrics ------------------------------------
def internal_diversity(smiles_list, sample=2000, seed=0):
    """1 - mean pairwise Tanimoto over Morgan fingerprints (higher = more
    diverse). Sampled to bound the O(n^2) cost."""
    from rdkit.Chem import AllChem, DataStructs
    rng = random.Random(seed)
    pool = smiles_list if len(smiles_list) <= sample else rng.sample(smiles_list, sample)
    fps = []
    for smi in pool:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))
    if len(fps) < 2:
        return float("nan")
    tot, cnt = 0.0, 0
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        tot += sum(sims); cnt += len(sims)
    return 1.0 - (tot / cnt if cnt else 0.0)


def fcd_score(gen_smiles, ref_smiles):
    """Frechet ChemNet Distance (lower = closer to the real distribution).
    Requires `pip install fcd_torch`. Returns None if unavailable."""
    try:
        from fcd_torch import FCD
    except Exception:
        return None
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return FCD(device=dev, n_jobs=4)(gen_smiles, ref_smiles)
    except Exception as e:
        print("  [fcd] failed:", e)
        return None


def read_reference(data_path, keep, n=10000, seed=0):
    """A canonical-SMILES reference sample drawn from the training file."""
    rng = random.Random(seed)
    lines = [l.strip().split()[0] for l in open(data_path, encoding="utf-8") if l.strip()]
    if len(lines) > n:
        lines = rng.sample(lines, n)
    return list(filter(None, (_canon(s, keep) for s in lines)))


# --------------------------------- model ------------------------------------
def build_and_train(args):
    import torch
    import torch.nn as nn
    from torch.nn import functional as F

    torch.manual_seed(args.seed); random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    seqs, smis, _cache = load_sequences(args.data, args.rep, not args.no_stereo)
    if not seqs:
        print("No sequences loaded — check --data path/format."); return

    seqs = [["<bos>"] + s + ["<eos>"] for s in seqs]
    vocab = ["<pad>"] + sorted({t for s in seqs for t in s})
    stoi = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)
    train = [[stoi[t] for t in s] for s in seqs]
    block = args.block
    lens = [len(s) for s in train]
    lens_sorted = sorted(lens)
    p95 = lens_sorted[int(0.95 * (len(lens_sorted) - 1))]
    n_trunc = sum(1 for L in lens if L > block)
    print(f"[{args.rep}] {len(train)} seqs | vocab {V} | block {block} | device {dev}")
    print(f"[{args.rep}] seq len: mean {sum(lens)/len(lens):.1f}, p95 {p95}, "
          f"max {max(lens)} | truncated at block: {n_trunc} "
          f"({100*n_trunc/len(lens):.2f}%)")
    if n_trunc / len(lens) > 0.01:
        print(f"[{args.rep}] WARNING: >1% of sequences exceed --block {block} and are "
              f"TRUNCATED. This penalizes longer-encoding reps unfairly. Re-run all "
              f"reps with --block {max(256, ((max(lens)+15)//16)*16)}.")

    _enc, dec = get_adapter(args.rep, not args.no_stereo)
    keep = not args.no_stereo
    if smis is not None:
        train_canon = set(_canon(s, keep) for s in smis)
    else:  # loaded from cache: recover training SMILES by decoding sequences
        train_canon = set(filter(None, (dec(s[1:-1]) for s in seqs)))

    def get_batch():
        rows = [random.choice(train) for _ in range(args.batch)]
        xs, ys = [], []
        for r in rows:
            r = r[:block + 1]
            r = r + [0] * (block + 1 - len(r))
            xs.append(r[:block]); ys.append(r[1:block + 1])
        return (torch.tensor(xs, device=dev), torch.tensor(ys, device=dev))

    class GPT(nn.Module):
        def __init__(self, V, d=192, h=6, L=4, block=block):
            super().__init__()
            self.tok = nn.Embedding(V, d)
            self.pos = nn.Embedding(block, d)
            layer = nn.TransformerEncoderLayer(d, h, 4 * d, batch_first=True,
                                               activation="gelu", dropout=0.1)
            self.blocks = nn.TransformerEncoder(layer, L)
            self.ln = nn.LayerNorm(d)
            self.head = nn.Linear(d, V)

        def forward(self, x):
            T = x.size(1)
            h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
            mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
            h = self.blocks(h, mask=mask)
            return self.head(self.ln(h))

    model = GPT(V).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for step in range(args.steps):
        x, y = get_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(f"  step {step:6d}  loss {loss.item():.3f}")

    model.eval()

    @torch.no_grad()
    def sample(n, temp):
        outs = []
        for _ in range(n):
            ids = [stoi["<bos>"]]
            for _ in range(block - 1):
                x = torch.tensor([ids[-block:]], device=dev)
                logits = model(x)[0, -1] / temp
                p = torch.softmax(logits, -1)
                nxt = torch.multinomial(p, 1).item()
                if vocab[nxt] == "<eos>":
                    break
                if vocab[nxt] not in ("<pad>", "<bos>"):
                    ids.append(nxt)
            outs.append([vocab[i] for i in ids[1:]])
        return outs

    samples = sample(args.n_sample, args.temp)
    decoded = [dec(t) for t in samples]
    valid = [d for d in decoded if d]
    uniq = set(valid)
    print(f"\n[{args.rep}] {args.n_sample} samples, temp {args.temp}:")
    print(f"  valid%   = {100*len(valid)/len(samples):.1f}")
    print(f"  unique%  = {100*len(uniq)/max(1,len(valid)):.1f}   (of valid)")
    print(f"  novel%   = {100*len(uniq - train_canon)/max(1,len(valid)):.1f}   (of valid)")
    if valid:
        div = internal_diversity(list(uniq))
        print(f"  int-div  = {div:.3f}   (1 - mean pairwise Tanimoto; higher=diverse)")
    if args.fcd and valid:
        ref = read_reference(args.data, keep, n=args.n_sample)
        fcd = fcd_score(list(uniq), ref)
        if fcd is None:
            print("  FCD      = n/a   (pip install fcd_torch to enable)")
        else:
            print(f"  FCD      = {fcd:.3f}   (lower = closer to real distribution)")
    print("  e.g.    :", ", ".join(list(uniq)[:5]) if uniq else "(none valid)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="a .smi/.txt file, one SMILES per line")
    ap.add_argument("--rep", default="braid",
                    choices=["smiles", "deepsmiles", "selfies", "braid", "omnimol"])
    ap.add_argument("--no-stereo", action="store_true",
                    help="drop stereochemistry (much faster encoding)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-sample", type=int, default=1000)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--fcd", action="store_true",
                    help="also compute Frechet ChemNet Distance (needs fcd_torch)")
    ap.add_argument("--seed", type=int, default=0)
    build_and_train(ap.parse_args())


if __name__ == "__main__":
    main()