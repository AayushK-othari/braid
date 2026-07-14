"""
BRAID <-> SMILES codec.

BRAID (Branch-counted, Relative-ring, Attachment-native, Invalid-free, Dense)
is a machine-native line notation for molecular constitution. This module is a
*reference implementation*, not an optimized or standardized one.

Design (all four mechanisms are recombined prior art -- see NOVELTY in README):
  - Atoms          : SMILES organic-subset symbols; '[...]' only when needed
                     (charge / isotope / explicit-H / non-organic element).
                     Rings are emitted in KEKULE form (no aromatic perception
                     needed to decode) for round-trip safety.
  - Branches       : '>k' then exactly k atoms hang off the current atom, then
                     the spine resumes from that atom. No parentheses to match.
                     (This is the SELFIES length-count idea without brackets.)
  - Rings          : '^d' bonds the current atom to the atom emitted d atom-
                     positions earlier in one global emission order. No paired
                     digits, no digit reuse. (This is the DeepSMILES relative
                     ring-closure idea.)
  - Fragments      : '.' starts a new spine with no bond to the previous atom,
                     but the global atom counter keeps running so '^d' *can*
                     cross a '.' (SAFE-style linking). Plain mixtures never do.

Only atom tokens advance the global index. '>k', '^d', bond symbols and '.'
do not. Encoder and decoder must agree on that; they do.

Scope of this reference implementation:
  - Handles: connectivity, bond orders, charges, isotopes, explicit H,
    radicals, non-organic elements, disconnected fragments.
  - Does NOT handle: stereochemistry (tetrahedral @/@@, cis/trans). It is
    stripped on encode. This is an honest limitation, not an oversight.
"""

from __future__ import annotations
import re
from rdkit import Chem
from rdkit.Chem import rdmolops

ORGANIC_SUBSET = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}

_BOND_TO_SYM = {
    Chem.BondType.SINGLE: "",
    Chem.BondType.DOUBLE: "=",
    Chem.BondType.TRIPLE: "#",
}
_SYM_TO_ORDER = {"": 1.0, "-": 1.0, "=": 2.0, "#": 3.0}


# --------------------------------------------------------------------------- #
# Encoder: SMILES -> BRAID
# --------------------------------------------------------------------------- #
def _atom_token(atom: Chem.Atom) -> str:
    sym = atom.GetSymbol()
    chg = atom.GetFormalCharge()
    iso = atom.GetIsotope()
    rad = atom.GetNumRadicalElectrons()
    hs = atom.GetTotalNumHs()
    aromatic = atom.GetIsAromatic()

    if aromatic and sym in ORGANIC_SUBSET and chg == 0 and iso == 0 and rad == 0:
        # bare lowercase normally; bracket only when H is 'surprising'
        # (e.g. pyrrole [nH]) -- aromatic C-H is the default, so bare 'c'.
        if sym == "C" or hs == 0:
            return sym.lower()
        return f"[{sym.lower()}" + ("H" if hs == 1 else f"H{hs}") + "]"

    plain_ok = (
        sym in ORGANIC_SUBSET
        and chg == 0
        and iso == 0
        and rad == 0
        and not aromatic
    )
    if plain_ok:
        # For organic-subset neutral atoms RDKit re-infers implicit H on decode.
        # Verify that inference reproduces `hs`; if not, fall back to brackets.
        if _implicit_h_matches(atom):
            return sym
    # bracketed atom
    inner = ""
    if iso:
        inner += str(iso)
    inner += sym
    if hs == 1:
        inner += "H"
    elif hs > 1:
        inner += f"H{hs}"
    if chg > 0:
        inner += "+" if chg == 1 else f"+{chg}"
    elif chg < 0:
        inner += "-" if chg == -1 else f"-{abs(chg)}"
    return f"[{inner}]"


def _implicit_h_matches(atom: Chem.Atom) -> bool:
    """Would a bare organic-subset symbol re-infer the same H count?"""
    pt = Chem.GetPeriodicTable()
    sym = atom.GetSymbol()
    # sum of bond orders to heavy neighbours (kekulized -> integer orders)
    bond_sum = 0
    for b in atom.GetBonds():
        bt = b.GetBondType()
        bond_sum += {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
                     Chem.BondType.TRIPLE: 3}.get(bt, 1)
    default_valences = pt.GetValenceList(pt.GetAtomicNumber(sym))
    for v in default_valences:
        if v - bond_sum == atom.GetTotalNumHs():
            return True
    return False


def mol_to_braid(mol: Chem.Mol, aromatic: bool = False, stereo: bool = True) -> str:
    """
    Encode an RDKit Mol to a BRAID string.

    aromatic=False (default): Kekule form -- always round-trips, but expands
        aromatic rings (benzene -> C=CC=CC=C^5).
    aromatic=True: keep lowercase aromatic atoms & aromatic default bonds
        (benzene -> cccccc^5). More compact/readable (the OMNIMOL choice), but
        relies on RDKit aromaticity perception on decode.
    stereo=True (default): encode tetrahedral (@/@@) and E/Z (%C/%T) stereo.
        stereo=False strips it (constitution only).
    """
    mol = Chem.Mol(mol)
    if stereo:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        try:
            Chem.AssignCIPLabels(mol)
        except Exception:
            pass
    else:
        Chem.RemoveStereochemistry(mol)
    if not aromatic:
        Chem.Kekulize(mol, clearAromaticFlags=True)

    n = mol.GetNumAtoms()
    if n == 0:
        return ""

    # deterministic emission order: canonical ranks pick roots & child order
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))

    adj = {i: [] for i in range(n)}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)

    visited = [False] * n
    parent = [-1] * n
    tree_children = {i: [] for i in range(n)}
    back_edges = {i: [] for i in range(n)}  # at deeper endpoint -> [ancestor,...]

    # iterative DFS to build spanning tree + back edges, honouring canonical order
    def dfs(root):
        stack = [(root, -1, iter(sorted(adj[root], key=lambda x: ranks[x])))]
        visited[root] = True
        order_local = [root]
        while stack:
            node, par, it = stack[-1]
            advanced = False
            for nb in it:
                if nb == par:
                    continue
                if not visited[nb]:
                    visited[nb] = True
                    parent[nb] = node
                    tree_children[node].append(nb)
                    stack.append((nb, node,
                                  iter(sorted(adj[nb], key=lambda x: ranks[x]))))
                    order_local.append(nb)
                    advanced = True
                    break
                else:
                    # non-tree edge; record once, at the deeper (current) endpoint
                    if parent[node] != nb and nb not in back_edges[node] \
                            and node not in back_edges.get(nb, []):
                        # ensure we only record when nb was seen earlier
                        back_edges[node].append(nb)
            if not advanced:
                stack.pop()
        return order_local

    roots = sorted(range(n), key=lambda x: ranks[x])
    components = []
    for r in roots:
        if not visited[r]:
            components.append(dfs(r))

    # subtree sizes (tree edges only)
    size = [1] * n

    def compute_size(u):
        for c in tree_children[u]:
            compute_size(c)
            size[u] += size[c]

    for comp in components:
        compute_size(comp[0])

    # assign global emission index via a pre-order walk that matches output order
    global_idx = {}
    out = []
    counter = [0]
    atom_out_pos = {}         # atom idx -> index in `out` of its atom token
    ez_slots = []             # (out_index_of_placeholder, child_atom, parent_atom)

    def bond_sym(a, b):
        bond = mol.GetBondBetweenAtoms(a, b)
        bt = bond.GetBondType()
        if bt == Chem.BondType.AROMATIC:
            return ""
        if (bt == Chem.BondType.SINGLE
                and mol.GetAtomWithIdx(a).GetIsAromatic()
                and mol.GetAtomWithIdx(b).GetIsAromatic()):
            return "-"          # single bond joining two aromatic systems
        return _BOND_TO_SYM.get(bt, "")

    def emit(u, par):
        # bond prefix (to parent on the spine/branch)
        if par is not None:
            out.append(bond_sym(par, u))
        atom_out_pos[u] = len(out)
        out.append(_atom_token(mol.GetAtomWithIdx(u)))
        global_idx[u] = counter[0]
        counter[0] += 1
        # ring closures: back edges from u to already-emitted ancestors
        for anc in back_edges[u]:
            d = global_idx[u] - global_idx[anc]
            bs = bond_sym(u, anc)
            out.append(f"{bs}^{d}")
        # E/Z placeholder slot if bond (par,u) is a stereo-capable double bond
        if stereo and par is not None:
            b = mol.GetBondBetweenAtoms(par, u)
            if b.GetBondType() == Chem.BondType.DOUBLE:
                ez_slots.append((len(out), u, par))
                out.append("")    # filled in during resolution
        # children: all but last are branches
        kids = tree_children[u]
        for i, c in enumerate(kids):
            if i < len(kids) - 1:
                out.append(f">{size[c]}")
                emit(c, u)
            else:
                emit(c, u)

    for ci, comp in enumerate(components):
        if ci > 0:
            out.append(".")
        emit(comp[0], None)

    if not stereo or (not _has_tetra(mol) and not ez_slots):
        return "".join(out)

    # ---- resolve stereo markers by matching the real decoder's CIP labels ----
    # target CIP per emission index
    inv = {global_idx[u]: u for u in global_idx}          # emission e -> atom u
    tgt_atom_cip = {}
    for u in global_idx:
        a = mol.GetAtomWithIdx(u)
        if a.HasProp("_CIPCode"):
            tgt_atom_cip[global_idx[u]] = a.GetProp("_CIPCode")
    # tetra centers = atoms carrying a chiral tag
    tetra_e = [global_idx[u] for u in global_idx
               if mol.GetAtomWithIdx(u).GetChiralTag() in
               (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.ChiralType.CHI_TETRAHEDRAL_CCW)]

    # target E/Z CIP per (child emission, parent emission)
    tgt_bond_cip = {}
    for (_slot, child, par) in ez_slots:
        b = mol.GetBondBetweenAtoms(par, child)
        code = b.GetProp("_CIPCode") if b.HasProp("_CIPCode") else None
        if code in ("E", "Z"):
            tgt_bond_cip[(global_idx[child], global_idx[par])] = code

    # initial guesses
    tetra_mark = {e: "@" for e in tetra_e}                 # '@'=CCW
    ez_mark = {}
    for (_slot, child, par) in ez_slots:
        key = (global_idx[child], global_idx[par])
        if key in tgt_bond_cip:
            ez_mark[_slot] = "%C"

    def render():
        for u in global_idx:
            e = global_idx[u]
            if e in tetra_mark:
                out[atom_out_pos[u]] = _bracket_with_chir(
                    mol.GetAtomWithIdx(u), tetra_mark[e])
        for (slot, child, par) in ez_slots:
            out[slot] = ez_mark.get(slot, "")
        return "".join(out)

    for _ in range(6):
        d = braid_to_mol(render(), clamp=True)
        try:
            Chem.AssignStereochemistry(d, cleanIt=True, force=True)
            Chem.AssignCIPLabels(d)
        except Exception:
            pass
        changed = False
        dorder = _emission_order(d)      # emission index -> atom idx in d
        for e in tetra_e:
            di = dorder[e] if e < len(dorder) else None
            got = (d.GetAtomWithIdx(di).GetProp("_CIPCode")
                   if di is not None and d.GetAtomWithIdx(di).HasProp("_CIPCode")
                   else None)
            want = tgt_atom_cip.get(e)
            if want and got != want:
                tetra_mark[e] = "@@" if tetra_mark[e] == "@" else "@"
                changed = True
        for (slot, child, par) in ez_slots:
            key = (global_idx[child], global_idx[par])
            want = tgt_bond_cip.get(key)
            if not want:
                continue
            dci, dpi = dorder[key[0]], dorder[key[1]]
            b = d.GetBondBetweenAtoms(dpi, dci)
            got = b.GetProp("_CIPCode") if b and b.HasProp("_CIPCode") else None
            if got != want:
                ez_mark[slot] = "%T" if ez_mark.get(slot) == "%C" else "%C"
                changed = True
        if not changed:
            break
    return render()


def _has_tetra(mol):
    return any(a.GetChiralTag() in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                                    Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
               for a in mol.GetAtoms())


def _bracket_with_chir(atom, marker):
    sym = atom.GetSymbol()
    iso = atom.GetIsotope()
    hs = atom.GetTotalNumHs()
    chg = atom.GetFormalCharge()
    s = "["
    if iso:
        s += str(iso)
    s += sym + marker
    if hs == 1:
        s += "H"
    elif hs > 1:
        s += f"H{hs}"
    if chg > 0:
        s += "+" if chg == 1 else f"+{chg}"
    elif chg < 0:
        s += "-" if chg == -1 else f"-{abs(chg)}"
    return s + "]"


def _emission_order(mol):
    # braid_to_mol adds atoms in emission order, so atom idx == emission index
    return list(range(mol.GetNumAtoms()))


def smiles_to_braid(smiles: str, aromatic: bool = False, stereo: bool = True) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles!r}")
    return mol_to_braid(mol, aromatic=aromatic, stereo=stereo)


# --------------------------------------------------------------------------- #
# Decoder: BRAID -> Mol / SMILES  (with valence state machine)
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(
    r"""
    (?P<dot>\.)
  | (?P<ezstereo>%[CT])
  | (?P<branch>>\d+)
  | (?P<ring>[-=#]?\^\d+)
  | (?P<bond>[-=#])
  | (?P<bracket>\[[^\]]*\])
  | (?P<organic>Cl|Br|b|c|n|o|p|s|B|C|N|O|P|S|F|I)
    """,
    re.VERBOSE,
)

_BRACKET_RE = re.compile(
    r"\[(?P<iso>\d+)?(?P<sym>[A-Za-z][a-z]?)(?P<chir>@{1,2})?"
    r"(?P<h>H\d*)?(?P<chg>(?:\+|-)\d*)?\]"
)

_PT = Chem.GetPeriodicTable()


def _tokenize(s: str):
    toks = []
    i = 0
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        m = _TOKEN_RE.match(s, i)
        if not m:
            # Total lexer: unrecognized chars are noise, skipped (keeps the
            # 'every string decodes' guarantee instead of raising).
            i += 1
            continue
        toks.append((m.lastgroup, m.group()))
        i = m.end()
    return toks


def _make_atom(token: str):
    """Return an Atom, or None if the token can't be an atom (-> skipped)."""
    if token.startswith("["):
        m = _BRACKET_RE.match(token)
        if not m:
            return None
        sym = m.group("sym")
        sym = sym[0].upper() + sym[1:]  # accept aromatic lowercase, e.g. [nH]
        if _PT.GetAtomicNumber(sym) == 0:
            return None
        atom = Chem.Atom(sym)
        if m.group("chir"):
            atom.SetChiralTag(
                Chem.ChiralType.CHI_TETRAHEDRAL_CCW if m.group("chir") == "@"
                else Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        if m.group("iso"):
            atom.SetIsotope(int(m.group("iso")))
        if m.group("h"):
            h = m.group("h")
            atom.SetNumExplicitHs(1 if h == "H" else int(h[1:]))
            atom.SetNoImplicit(True)
        if m.group("chg"):
            c = m.group("chg")
            if c in ("+", "-"):
                atom.SetFormalCharge(1 if c == "+" else -1)
            else:
                atom.SetFormalCharge(int(c))
        return atom
    if token.islower():
        atom = Chem.Atom(token[0].upper() + token[1:])
        atom.SetIsAromatic(True)
        return atom
    return Chem.Atom(token)


def _free_valence(rw: Chem.RWMol, idx: int, used: dict) -> float:
    """Remaining valence capacity for atom idx given bonds added so far."""
    atom = rw.GetAtomWithIdx(idx)
    default_vs = _PT.GetValenceList(atom.GetAtomicNum())
    default_v = max(default_vs) if default_vs else 4
    chg = atom.GetFormalCharge()
    # Charge->valence rule (OpenSMILES-style, common cases):
    #   carbon: both cation and anion are trivalent  -> default - |chg|
    #   electronegative main group (N,O,P,S,...):        default + chg
    #     (N+ -> 4, O- -> 1, etc.)
    if atom.GetSymbol() == "C":
        cap = default_v - abs(chg)
    else:
        cap = default_v + chg
    explicit_h = atom.GetNumExplicitHs()  # bracket atoms already spent this
    return cap - used.get(idx, 0) - explicit_h


def braid_to_mol(braid: str, clamp: bool = True) -> Chem.Mol:
    """
    Decode a BRAID string to an RDKit Mol.

    clamp=True implements the SELFIES-style validity state machine: bonds that
    would overflow valence are reduced in order or dropped, ring closures to a
    saturated atom become no-ops. With clamp=True the decoder never raises on a
    lexically valid string; it always returns a sanitizable Mol (possibly empty).
    """
    toks = _tokenize(braid)
    rw = Chem.RWMol()
    used = {}                 # atom idx -> valence consumed by heavy bonds
    order_by_global = []      # global index -> atom idx
    emit_parent = {}          # atom idx -> the atom it first bonded to
    ez_marks = []             # (a, b, 'C'/'T') double-bond stereo, resolved later

    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def add_bond(a, b, order, explicit=True):
        if a is None or b is None or a == b:
            return
        if rw.GetBondBetweenAtoms(a, b) is not None:
            return
        # implicit bond between two aromatic atoms -> aromatic bond
        aromatic = (not explicit
                    and rw.GetAtomWithIdx(a).GetIsAromatic()
                    and rw.GetAtomWithIdx(b).GetIsAromatic())
        o = order
        if clamp:
            room = min(_free_valence(rw, a, used), _free_valence(rw, b, used))
            if room <= 0:
                return          # no-op: keeps string valid
            o = min(order, room)
        if aromatic:
            rw.AddBond(a, b, Chem.BondType.AROMATIC)
            cost = 1
        else:
            bt = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                  3: Chem.BondType.TRIPLE}[int(o)]
            rw.AddBond(a, b, bt)
            cost = int(o)
        used[a] = used.get(a, 0) + cost
        used[b] = used.get(b, 0) + cost

    def parse_span(attach_idx, first_bond_order, budget, first_bond_explicit=False):
        """Consume atoms onto a spine. Returns number of atoms created."""
        created = 0
        prev = attach_idx
        pending = first_bond_order
        pending_explicit = first_bond_explicit
        while pos[0] < len(toks):
            if budget is not None and created >= budget:
                break
            kind, text = peek()
            if kind == "dot":
                if budget is not None:
                    break        # '.' shouldn't appear inside a counted branch
                pos[0] += 1
                prev = None
                pending = 1.0
                pending_explicit = False
                continue
            if kind == "bond":
                pending = _SYM_TO_ORDER[text]
                pending_explicit = True
                pos[0] += 1
                continue
            if kind == "ezstereo":
                pos[0] += 1          # stray marker (e.g. from mutation) -> skip
                continue
            if kind in ("organic", "bracket"):
                pos[0] += 1
                atom = _make_atom(text)
                if atom is None:
                    continue          # unparseable atom token -> skip (no-op)
                idx = rw.AddAtom(atom)
                order_by_global.append(idx)
                if prev is not None:
                    add_bond(prev, idx, pending, explicit=pending_explicit)
                    emit_parent[idx] = prev
                pending = 1.0
                pending_explicit = False
                created += 1
                prev = idx
                # ring closures immediately after the atom
                while peek() and peek()[0] == "ring":
                    _, rtext = peek()
                    pos[0] += 1
                    bpart = rtext[:-len(rtext.lstrip("-=#"))] if rtext[0] in "-=#" else ""
                    # split bond symbol and ^d
                    msym = ""
                    core = rtext
                    if rtext[0] in "-=#":
                        msym = rtext[0]
                        core = rtext[1:]
                    d = int(core[1:])
                    tgt_global = len(order_by_global) - 1 - d
                    if 0 <= tgt_global < len(order_by_global):
                        add_bond(idx, order_by_global[tgt_global],
                                 _SYM_TO_ORDER[msym], explicit=bool(msym))
                    # else: dangling ref -> no-op (validity preserved)
                # E/Z stereo markers attach here (between ring closures and
                # branches) so branch parsing is not desynced.
                while peek() and peek()[0] == "ezstereo":
                    _, etext = peek()
                    pos[0] += 1
                    if idx in emit_parent:
                        ez_marks.append((emit_parent[idx], idx, etext[1]))
                # If this atom completed the span's budget it must be a leaf:
                # its trailing '>k'/continuation belong to the PARENT span.
                if budget is not None and created >= budget:
                    break
                # branches
                while peek() and peek()[0] == "branch":
                    _, btext = peek()
                    pos[0] += 1
                    k = int(btext[1:])
                    # optional explicit bond for the branch's first atom
                    bord = 1.0
                    bexpl = False
                    if peek() and peek()[0] == "bond":
                        bord = _SYM_TO_ORDER[peek()[1]]
                        bexpl = True
                        pos[0] += 1
                    consumed = parse_span(idx, bord, k, bexpl)
                    created += consumed
                # loop continues: next atom attaches to `prev` (== idx)
                continue
            # unknown / stray -> stop this span
            break
        return created

    parse_span(None, 1.0, None)

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # last-resort: partial sanitize so we still return a valid-ish mol
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)

    # Resolve E/Z double-bond stereo using a fixed convention: on each end of
    # the double bond the reference substituent is the neighbour with the
    # smallest emission index. 'C' = those two refs cis, 'T' = trans.
    if ez_marks:
        emidx = {a: e for e, a in enumerate(order_by_global)}

        def ref_of(end, other):
            cands = [nb.GetIdx() for nb in mol.GetAtomWithIdx(end).GetNeighbors()
                     if nb.GetIdx() != other]
            return min(cands, key=lambda x: emidx.get(x, 1 << 30)) if cands else None

        any_set = False
        for a, b, flag in ez_marks:
            bond = mol.GetBondBetweenAtoms(a, b)
            if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            ra, rb = ref_of(a, b), ref_of(b, a)
            if ra is None or rb is None:
                continue
            bond.SetStereoAtoms(ra, rb)
            bond.SetStereo(Chem.BondStereo.STEREOCIS if flag == "C"
                           else Chem.BondStereo.STEREOTRANS)
            any_set = True
        if any_set:
            Chem.SetDoubleBondNeighborDirections(mol)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def braid_to_smiles(braid: str, clamp: bool = True) -> str:
    mol = braid_to_mol(braid, clamp=clamp)
    return Chem.MolToSmiles(mol)
