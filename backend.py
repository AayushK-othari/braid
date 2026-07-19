"""
Conformer backend — optional.

Serves BRAIDBERTa and DeepBERTa from your own machine using the same route
shape the workbench already calls, so nothing in the frontend has to change.

    # converters — light, no GPU, no torch
    pip install "fastapi>=0.110" "uvicorn[standard]" rdkit selfies
    pip install git+https://github.com/AayushK-othari/braid.git

    # encoders — only if you want embeddings and fill-mask
    pip install onnxruntime tokenizers

    python backend.py

onnxruntime and tokenizers are imported lazily, so the BRAID and SELFIES
routes work on a machine with neither installed. The encoder routes return a
clear 501 in that case rather than failing obscurely.

The encoders run from ONNX graphs exported by export_onnx.py, not from the
torch checkpoints. That is a deployment decision, not a modelling one: torch
plus transformers is ~1 GB resident, which does not fit a 512 MB free tier,
while onnxruntime plus tokenizers is ~350 MB, which does. check_parity.py
verifies the graphs agree with the checkpoints to ~3e-6 on hidden states and
are bit-for-bit deterministic across runs.

Expects the exported graphs beside this file:

    onnx/braid/{model.onnx, model.onnx.data, tokenizer.json, ...}
    onnx/deep/ {model.onnx, model.onnx.data, tokenizer.json, ...}

Override with BRAID_ONNX / DEEP_ONNX if they live elsewhere.

Then in Conformer: header -> "Encoders offline" -> Inference endpoint:

    http://127.0.0.1:8000/models

Why you might want this: Hugging Face's serverless inference does not host
every custom architecture, and feature-extraction on a private or unusual
encoder often is not available there. Running locally sidesteps that entirely,
and your structures never leave the machine.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
import subprocess
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("conformer")

MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
MAX_BATCH = int(os.getenv("MAX_BATCH", "64"))      # refuse absurd public requests
HERE = Path(__file__).parent

# The page is looked for beside this file. Accept the obvious names, fall back
# to "the only .html in the folder", and allow an explicit override, because
# "404" is a miserable thing to debug when the file is simply one directory over.
FRONTEND_NAMES = ("conformer.html", "app.html", "index.html")


def find_frontend() -> Path | None:
    override = os.getenv("FRONTEND_PATH")
    if override:
        p = Path(override).expanduser()
        return p if p.is_file() else None
    for name in FRONTEND_NAMES:
        p = HERE / name
        if p.is_file():
            return p
    loose = sorted(HERE.glob("*.html"))
    return loose[0] if len(loose) == 1 else None

# Pin model revisions for reproducibility. Set these to commit SHAs before a
# paper release so a reader gets byte-identical weights, not "whatever main is".
MODELS = {
    "braid": os.getenv("BRAID_REPO", "aakothari/BRAIDBERTa"),
    "deep": os.getenv("DEEP_REPO", "aakothari/DeepBERTa_zinc_base_100k_v4"),
}
REVISIONS = {
    MODELS["braid"]: os.getenv("BRAID_REV", "main"),
    MODELS["deep"]: os.getenv("DEEP_REV", "main"),
}

# Where the exported graphs live. Keyed by repo id so the existing route shape
# — /models/{owner}/{name}/pipeline/... — keeps working unchanged and the
# frontend needs no edit.
ONNX_DIRS = {
    MODELS["braid"]: Path(os.getenv("BRAID_ONNX", str(HERE / "onnx" / "braid"))),
    MODELS["deep"]: Path(os.getenv("DEEP_ONNX", str(HERE / "onnx" / "deep"))),
}

# torch and transformers are imported lazily. The converter routes below are
# useful on their own, and there is no reason to require a 2 GB install to
# turn SMILES into BRAID.
_DEVICE = None


def device() -> str:
    """Reported by /health and logged at startup.

    Always CPU. A 43 M-parameter encoder on sub-30-token sequences does not
    benefit from a GPU — the transfer costs more than the matmul saves.
    """
    global _DEVICE
    if _DEVICE is None:
        try:
            import onnxruntime as ort
            _DEVICE = f"cpu (onnxruntime {ort.__version__})"
        except ImportError:
            _DEVICE = "no-onnxruntime"
    return _DEVICE

@asynccontextmanager
async def lifespan(app: FastAPI):
    """PRELOAD=1 pays the model-loading cost at boot so the first user does not."""
    if os.getenv("PRELOAD") == "1":
        for repo in MODELS.values():
            try:
                load(repo)
                log.info("preloaded %s", repo)
            except Exception as exc:              # noqa: BLE001
                log.warning("preload failed for %s: %s", repo, exc)
    yield


app = FastAPI(title="Conformer backend", lifespan=lifespan)
# A page opened from disk sends "Origin: null" and is cross-origin to
# 127.0.0.1, so the local dev loop needs CORS. The default is therefore "*".
#
# In a same-origin deployment (this app serving conformer.html at /) no
# cross-origin request happens at all, so the Dockerfile sets ALLOWED_ORIGINS=""
# to drop the middleware entirely. Set it to an explicit origin list if you ever
# host the page separately.
_origins = [o for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

_cache: dict[str, tuple[Any, Any]] = {}


class _Tok:
    """Thin adapter over `tokenizers.Tokenizer`.

    Exposes only the handful of methods the routes below used from the
    transformers API — __call__, tokenize, mask_token, mask_token_id,
    convert_ids_to_tokens — so the route bodies stay recognisable and
    transformers (and therefore its huggingface-hub version constraint,
    and therefore torch) drops out of the dependency tree entirely.
    """

    def __init__(self, path: Path):
        from tokenizers import Tokenizer

        self.t = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.path = path

        # Mask token: whatever the tokenizer actually calls it. The two repos
        # disagree (<mask> vs [MASK]) and guessing wrong is a 400 on every
        # fill-mask request, so read it rather than assume.
        self.mask_token = None
        cfg = path / "tokenizer_config.json"
        smap = path / "special_tokens_map.json"
        for f in (smap, cfg):
            if f.is_file():
                d = json.loads(f.read_text(encoding="utf-8"))
                m = d.get("mask_token")
                if isinstance(m, dict):
                    m = m.get("content")
                if m:
                    self.mask_token = m
                    break
        if self.mask_token is None:
            for cand in ("<mask>", "[MASK]"):
                if self.t.token_to_id(cand) is not None:
                    self.mask_token = cand
                    break

        self.mask_token_id = (
            self.t.token_to_id(self.mask_token) if self.mask_token else None
        )
        pad = self.t.token_to_id("<pad>")
        if pad is None:
            pad = self.t.token_to_id("[PAD]")
        self.pad_id = pad if pad is not None else 0

    def __call__(self, texts, padding=True, truncation=True, max_length=None):
        """Return {"input_ids": [[...]], "attention_mask": [[...]]} as lists."""
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)

        self.t.no_padding()
        self.t.no_truncation()
        if truncation:
            self.t.enable_truncation(max_length or MAX_LENGTH)
        if padding and len(batch) > 1:
            self.t.enable_padding(pad_id=self.pad_id, pad_token="<pad>")

        encs = self.t.encode_batch(batch)
        return {
            "input_ids": [e.ids for e in encs],
            "attention_mask": [e.attention_mask for e in encs],
        }

    def tokenize(self, text: str):
        return self.t.encode(text).tokens

    def convert_ids_to_tokens(self, i: int):
        return self.t.id_to_token(int(i))


def resolve_onnx(repo: str) -> Path:
    """Where is this model's exported graph?

    Local folder first, so a development machine with onnx/ beside this file
    never touches the network. Otherwise pull the onnx/ subfolder of the model
    repo on the Hub, at the pinned revision — the same revision /version
    reports, so what a reader downloads is what the server ran.
    """
    d = ONNX_DIRS.get(repo)
    if d is None:
        raise HTTPException(
            404, f"{repo} is not one of the exported models ({', '.join(ONNX_DIRS)})"
        )
    if (d / "model.onnx").is_file():
        return d

    rev = REVISIONS.get(repo, "main")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise HTTPException(
            503,
            f"no graph at {d} and huggingface_hub is not installed. Run "
            f"export_onnx.py, or pip install huggingface-hub to fetch it.",
        ) from exc

    log.info("fetching onnx/ from %s@%s", repo, rev)
    try:
        snap = snapshot_download(
            repo,
            revision=rev,
            allow_patterns=["onnx/*"],
            token=os.getenv("HF_TOKEN") or None,
        )
    except Exception as exc:                      # noqa: BLE001
        raise HTTPException(503, f"could not fetch onnx/ from {repo}: {exc}") from exc

    got = Path(snap) / "onnx"
    if not (got / "model.onnx").is_file():
        raise HTTPException(
            503,
            f"{repo}@{rev} has no onnx/model.onnx. Upload the export with "
            f"`hf upload {repo} onnx/<key> onnx --repo-type model`.",
        )
    return got


def prefetch() -> None:
    """Download every graph without starting a server.

    Call this in a build step so the first request does not pay for ~350 MB
    of downloads, and so a fetch failure breaks the build rather than the
    running service:

        python -c "import backend; backend.prefetch()"
    """
    for repo in MODELS.values():
        try:
            p = resolve_onnx(repo)
            log.info("prefetched %s -> %s", repo, p)
        except Exception as exc:                  # noqa: BLE001
            log.error("prefetch failed for %s: %s", repo, exc)
            raise


def load(repo: str):
    """Load and memoise a tokenizer + ONNX session pair."""
    if repo not in _cache:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise HTTPException(
                501, "pip install onnxruntime tokenizers to serve the encoders"
            ) from exc

        d = resolve_onnx(repo)

        log.info("loading %s from %s on %s", repo, d, device())
        opts = ort.SessionOptions()
        # A free-tier instance has one or two cores and serves one request at
        # a time. Letting ORT spawn a thread per core costs more in contention
        # than it wins in parallelism at this sequence length.
        opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "1"))
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        sess = ort.InferenceSession(
            str(d / "model.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        _cache[repo] = (_Tok(d), sess)
    return _cache[repo]


def _run(sess, tok, texts, max_length=None):
    """Tokenise, pad, and run the graph. Returns (logits, hidden, mask)."""
    import numpy as np

    enc = tok(texts, padding=True, truncation=True,
              max_length=max_length or MAX_LENGTH)
    ids = np.array(enc["input_ids"], dtype=np.int64)
    mask = np.array(enc["attention_mask"], dtype=np.int64)

    wanted = {i.name for i in sess.get_inputs()}
    feed = {"input_ids": ids, "attention_mask": mask}
    feed = {k: v for k, v in feed.items() if k in wanted}

    logits, hidden = sess.run(["logits", "last_hidden_state"], feed)
    return logits, hidden, mask


class Payload(BaseModel):
    inputs: Any = None
    options: dict | None = None


@app.get("/health")
def health():
    """The workbench calls this on connect to discover which converters exist."""
    return {
        "ok": True,
        "device": device(),
        "loaded": list(_cache),
        "converters": {
            "braid": BRAID_OK,
            "selfies": SELFIES_OK,
        },
    }


@app.post("/models/{owner}/{name}/pipeline/feature-extraction")
def feature_extraction(owner: str, name: str, body: Payload):
    """Return [batch][sequence][hidden] — the workbench pools client-side,
    so mean vs CLS stays a choice you make in the interface."""
    repo = f"{owner}/{name}"
    texts = body.inputs if isinstance(body.inputs, list) else [body.inputs]
    texts = [str(t) for t in texts if t is not None]
    if not texts:
        raise HTTPException(400, "no inputs")
    if len(texts) > MAX_BATCH:
        raise HTTPException(413, f"batch of {len(texts)} exceeds MAX_BATCH={MAX_BATCH}")
    try:
        tok, sess = load(repo)
    except HTTPException:
        raise                                     # keep 501 "install ..." as-is
    except Exception as exc:                      # noqa: BLE001
        raise HTTPException(503, f"could not load {repo}: {exc}") from exc

    _, hidden, mask = _run(sess, tok, texts)      # hidden: (B, T, H)
    out = []
    for i in range(hidden.shape[0]):
        keep = hidden[i][mask[i].astype(bool)]    # drop padding so mean-pooling is honest
        out.append(keep.tolist())
    return out


@app.post("/models/{owner}/{name}/pipeline/fill-mask")
def fill_mask(owner: str, name: str, body: Payload):
    repo = f"{owner}/{name}"
    text = body.inputs if isinstance(body.inputs, str) else str(body.inputs)
    try:
        tok, sess = load(repo)
    except HTTPException:
        raise                                     # keep 501 "install ..." as-is
    except Exception as exc:                      # noqa: BLE001
        raise HTTPException(503, f"could not load {repo}: {exc}") from exc

    if tok.mask_token is None:
        raise HTTPException(400, f"{repo} has no mask token")
    # accept whichever mask spelling the user typed
    for alias in ("[MASK]", "<mask>", "<MASK>"):
        text = text.replace(alias, tok.mask_token)
    if tok.mask_token not in text:
        raise HTTPException(400, f"input contains no {tok.mask_token}")

    import numpy as np

    enc = tok(text, padding=False, truncation=True, max_length=MAX_LENGTH)
    ids = np.array(enc["input_ids"], dtype=np.int64)
    mask = np.array(enc["attention_mask"], dtype=np.int64)

    wanted = {i.name for i in sess.get_inputs()}
    feed = {"input_ids": ids, "attention_mask": mask}
    feed = {k: v for k, v in feed.items() if k in wanted}
    logits = sess.run(["logits"], feed)[0][0]     # (T, V)

    hits = np.nonzero(ids[0] == tok.mask_token_id)[0]
    if hits.size == 0:
        raise HTTPException(400, "mask token did not survive tokenisation")
    pos = int(hits[0])

    row = logits[pos].astype(np.float64)
    row -= row.max()                              # stable softmax
    probs = np.exp(row)
    probs /= probs.sum()

    k = min(10, probs.shape[0])
    top = np.argsort(-probs)[:k]
    return [[
        {
            "token": int(idx),
            "token_str": tok.convert_ids_to_tokens(int(idx)),
            "score": float(probs[idx]),
        }
        for idx in top
    ]]


@app.get("/tokenizer/{owner}/{name}")
def tokenizer_json(owner: str, name: str):
    """Hand the browser a tokenizer.json for a repo.

    The page cannot always fetch this itself: the repo may be private, the
    files may not be laid out where a URL guess would find them, or the Hub may
    not send CORS headers for that path. This route sidesteps all three, because
    the server has huggingface_hub, transformers and (optionally) HF_TOKEN.
    """
    repo = f"{owner}/{name}"
    rev = REVISIONS.get(repo, "main")
    tried = []

    # 1. the copy saved beside the exported graph — guaranteed to be the
    #    tokenizer the served model was actually exported with, rather than
    #    whatever main happens to hold today
    try:
        f = resolve_onnx(repo) / "tokenizer.json"
        if f.is_file():
            return json.loads(f.read_text(encoding="utf-8"))
        tried.append(f"{f}: not found")
    except Exception as exc:  # noqa: BLE001
        tried.append(f"onnx dir: {exc}")

    # 2. the Hub, if huggingface_hub happens to be installed. It is not a
    #    dependency of this deployment, so this is a convenience for local
    #    development against a repo that was never exported.
    try:
        from huggingface_hub import hf_hub_download
        token = os.getenv("HF_TOKEN") or None
        path = hf_hub_download(repo, "tokenizer.json", revision=rev, token=token)
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:                      # noqa: BLE001
        tried.append(f"hub: {exc}")

    raise HTTPException(502, f"no tokenizer for {repo} (rev {rev}) — " + " | ".join(tried))


@app.get("/repo-check/{owner}/{name}")
def repo_check(owner: str, name: str):
    """Does this repo exist, and can we see it? Distinguishes 404 from 401."""
    repo = f"{owner}/{name}"
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, token=os.getenv("HF_TOKEN") or None)
        return {
            "repo": repo, "exists": True, "private": info.private,
            "sha": info.sha,
            "files": sorted(f.rfilename for f in info.siblings)[:60],
        }
    except Exception as exc:                      # noqa: BLE001
        return {"repo": repo, "exists": False, "error": str(exc)[:300]}


@app.post("/models/{owner}/{name}/pipeline/tokenize")
def tokenize(owner: str, name: str, body: Payload):
    repo = f"{owner}/{name}"
    texts = body.inputs if isinstance(body.inputs, list) else [body.inputs]
    tok, _ = load(repo)
    return [{"tokens": tok.tokenize(str(t)),
             "ids": tok(str(t), padding=False)["input_ids"][0]}
            for t in texts]


# --------------------------------------------------------------------------- #
# Converters
#
# Both of these call the *reference implementations*, not reimplementations.
# BRAID in particular is not ported to JavaScript anywhere in this project:
# braids/codec.py depends on RWMol, GetPeriodicTable().GetValenceList(),
# AssignCIPLabels and SetDoubleBondNeighborDirections, none of which RDKit's
# WebAssembly build (MinimalLib) exposes. A hand-rolled JS version would
# diverge precisely at valence clamping and CIP-based stereo resolution — the
# parts that are hardest to notice going wrong. So the browser asks this
# server, and this server runs your code.
#
#     pip install git+https://github.com/AayushK-othari/braid.git
#     pip install selfies
# --------------------------------------------------------------------------- #

try:
    from braids import braid_to_smiles, smiles_to_braid
    from braids.tokenizer import tokenize as braid_tokenize
    BRAID_OK = True
except ImportError:                               # pragma: no cover
    BRAID_OK = False
    log.warning("braids not installed — /convert/braid will return 501")

try:
    import selfies as sf
    SELFIES_OK = True
except ImportError:                               # pragma: no cover
    SELFIES_OK = False
    log.warning("selfies not installed — /convert/selfies will return 501")


class ConvertIn(BaseModel):
    smiles: str | None = None
    text: str | None = None                       # for decode
    aromatic: bool = False                        # BRAID Kekule vs aromatic mode
    clamp: bool = True                            # BRAID decoder valence clamping


class BatchIn(BaseModel):
    smiles: list[str] = []
    aromatic: bool = False


def _need(flag: bool, pkg: str, hint: str):
    if not flag:
        raise HTTPException(501, f"{pkg} is not installed — {hint}")


@app.post("/convert/braid")
def convert_braid(body: ConvertIn):
    """SMILES -> BRAID, via braids.smiles_to_braid.

    `aromatic` picks the mode. They are NOT interchangeable: whichever you
    pretrained on is the one you must encode with at inference time.
    """
    _need(BRAID_OK, "braids", "pip install git+https://github.com/AayushK-othari/braid.git")
    smi = body.smiles or body.text
    if not smi:
        raise HTTPException(400, "no smiles supplied")
    try:
        braid = smiles_to_braid(smi, aromatic=body.aromatic)
    except Exception as exc:                      # noqa: BLE001
        raise HTTPException(400, f"could not encode: {exc}") from exc
    toks = braid_tokenize(braid)
    # BRAIDBERTa's tokenizer is WordLevel + WhitespaceSplit over a 151-token
    # vocabulary, so the model input is the SPACE-JOINED token stream, not the
    # raw BRAID string. Feed it the raw string and WhitespaceSplit sees one
    # unknown word and the whole molecule collapses to <unk>.
    return {"result": braid, "tokens": " ".join(toks), "n_tokens": len(toks)}


@app.post("/convert/braid/decode")
def decode_braid(body: ConvertIn):
    """BRAID -> SMILES. With clamp=True every string decodes to something
    sanitizable, which is the whole point of the notation."""
    _need(BRAID_OK, "braids", "pip install git+https://github.com/AayushK-othari/braid.git")
    s = body.text or body.smiles
    if not s:
        raise HTTPException(400, "no braid string supplied")
    try:
        return {"result": braid_to_smiles(s, clamp=body.clamp)}
    except Exception as exc:                      # noqa: BLE001
        raise HTTPException(400, f"could not decode: {exc}") from exc


@app.post("/convert/braid/batch")
def batch_braid(body: BatchIn):
    """One request per set instead of one per molecule."""
    _need(BRAID_OK, "braids", "pip install git+https://github.com/AayushK-othari/braid.git")
    if len(body.smiles) > MAX_BATCH:
        raise HTTPException(413, f"batch exceeds MAX_BATCH={MAX_BATCH}")
    out, toks, errs = [], [], []
    for smi in body.smiles:
        try:
            b = smiles_to_braid(smi, aromatic=body.aromatic)
            out.append(b)
            toks.append(" ".join(braid_tokenize(b)))
            errs.append(None)
        except Exception as exc:                  # noqa: BLE001
            out.append(None)
            toks.append(None)
            errs.append(str(exc))
    return {"results": out, "tokens": toks, "errors": errs}


@app.post("/convert/braid/tokens")
def braid_tokens(body: ConvertIn):
    """BRAID string -> the token stream the model actually consumes."""
    _need(BRAID_OK, "braids", "pip install git+https://github.com/AayushK-othari/braid.git")
    s = body.text or body.smiles
    if not s:
        raise HTTPException(400, "no braid string supplied")
    toks = braid_tokenize(s)
    return {"tokens": toks, "text": " ".join(toks), "n_tokens": len(toks)}


@app.post("/convert/selfies")
def convert_selfies(body: ConvertIn):
    """SMILES -> SELFIES via the reference implementation."""
    _need(SELFIES_OK, "selfies", "pip install selfies")
    smi = body.smiles or body.text
    if not smi:
        raise HTTPException(400, "no smiles supplied")
    try:
        return {"result": sf.encoder(smi)}
    except Exception as exc:                      # noqa: BLE001
        raise HTTPException(400, f"could not encode: {exc}") from exc


@app.post("/convert/selfies/decode")
def decode_selfies(body: ConvertIn):
    _need(SELFIES_OK, "selfies", "pip install selfies")
    s = body.text or body.smiles
    if not s:
        raise HTTPException(400, "no selfies string supplied")
    try:
        return {"result": sf.decoder(s)}
    except Exception as exc:                      # noqa: BLE001
        raise HTTPException(400, f"could not decode: {exc}") from exc


@app.post("/convert/selfies/batch")
def batch_selfies(body: BatchIn):
    _need(SELFIES_OK, "selfies", "pip install selfies")
    if len(body.smiles) > MAX_BATCH:
        raise HTTPException(413, f"batch exceeds MAX_BATCH={MAX_BATCH}")
    out, errs = [], []
    for smi in body.smiles:
        try:
            out.append(sf.encoder(smi))
            errs.append(None)
        except Exception as exc:                  # noqa: BLE001
            out.append(None)
            errs.append(str(exc))
    return {"results": out, "errors": errs}


# --------------------------------------------------------------------------- #
# Static frontend + provenance
#
# Serving the page from this same app means same-origin: no CORS, no separate
# deployment, one URL to cite.
# --------------------------------------------------------------------------- #

@app.get("/", response_class=HTMLResponse)
def index():
    page = find_frontend()
    if page:
        return FileResponse(page, media_type="text/html")

    found = sorted(f.name for f in HERE.glob("*.html"))
    listing = ("<li><code>" + "</code></li><li><code>".join(found) + "</code></li>"
               if found else "<li><em>no .html files here at all</em></li>")
    return HTMLResponse(status_code=404, content=f"""<!doctype html>
<meta charset="utf-8"><title>Frontend not found</title>
<style>
 body{{background:#0B0E14;color:#AAB3C5;font:15px/1.65 system-ui,sans-serif;padding:48px;max-width:760px;margin:auto}}
 h1{{color:#E8ECF4;font-size:21px;margin:0 0 6px}} code{{color:#7DD3A0;font-family:ui-monospace,monospace}}
 .box{{background:#121722;border-left:3px solid #E0A458;padding:16px 20px;border-radius:8px;margin:20px 0}}
 li{{margin:4px 0}} p{{margin:12px 0}}
</style>
<h1>The server is running. The page is missing.</h1>
<p>The backend is fine — it just cannot find the workbench HTML to serve.</p>
<div class="box">
  <p style="margin-top:0">Looked in:<br><code>{HERE}</code></p>
  <p>For a file named <code>conformer.html</code>, <code>app.html</code> or <code>index.html</code>.</p>
  <p style="margin-bottom:0">HTML files actually in that folder:</p>
  <ul>{listing}</ul>
</div>
<p><strong>Fix:</strong> move <code>conformer.html</code> into the folder above, next to this
script, and reload. Or point at it directly:</p>
<p><code>set FRONTEND_PATH=C:\\path\\to\\conformer.html</code> &nbsp;(Windows cmd)<br>
<code>$env:FRONTEND_PATH="C:\\path\\to\\conformer.html"</code> &nbsp;(PowerShell)<br>
<code>FRONTEND_PATH=/path/to/conformer.html</code> &nbsp;(macOS / Linux)</p>
<p>Everything else works meanwhile — <code>/health</code>, <code>/version</code>,
<code>/convert/braid</code> and <code>/docs</code> are all up.</p>""")


def _git_sha() -> str:
    if os.getenv("GIT_SHA"):
        return os.getenv("GIT_SHA")
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:                             # noqa: BLE001
        return "unknown"


@app.get("/version")
def version():
    """Everything a reader needs to reproduce a result from this deployment."""
    import importlib.metadata as md

    def ver(pkg):
        try:
            return md.version(pkg)
        except Exception:                         # noqa: BLE001
            return None

    return {
        "git_sha": _git_sha(),
        "models": MODELS,
        "revisions": REVISIONS,
        "max_length": MAX_LENGTH,
        "max_batch": MAX_BATCH,
        "onnx_dirs": {k: str(v) for k, v in ONNX_DIRS.items()},
        "runtime": device(),
        "packages": {p: ver(p) for p in
                     ("rdkit", "braids-mol", "selfies", "onnxruntime",
                      "tokenizers", "fastapi")},
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = os.getenv("PORT", "8000")
    log.info("device: %s", device())
    page = find_frontend()
    if page:
        log.info("serving %s", page.name)
        log.info("open the workbench at  http://%s:%s/", host, port)
    else:
        log.warning("no conformer.html / app.html / index.html found in %s", HERE)
        log.warning("put the page there, or set FRONTEND_PATH, then reload")
    log.info("(opening conformer.html from disk also works, but going through")
    log.info(" the server is same-origin and avoids CORS entirely)")
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
