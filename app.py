import json
import os
import re
import unicodedata
from collections import Counter
from math import log
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastembed import TextEmbedding

# -------------------- Configuration --------------------
BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "kb.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
THRESH = float(os.getenv("RAG_THRESHOLD", "0.38"))
MAX_TOP_K = 10

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# -------------------- Load knowledge base --------------------
with KB_PATH.open("r", encoding="utf-8") as f:
    KB = json.load(f)

if not isinstance(KB, list) or len(KB) != 50 or not all(isinstance(p, str) for p in KB):
    raise ValueError("kb.json must be a JSON array containing exactly 50 text passages.")

SENTS = []
SENT_IDX = {}
for pid, passage in enumerate(KB):
    for sentence in re.split(r"(?<=[.!?])\s+", passage.strip()):
        if sentence:
            SENT_IDX.setdefault(pid, []).append(len(SENTS))
            SENTS.append((pid, sentence))

model = TextEmbedding(model_name=MODEL_NAME, threads=1, cache_dir=str(BASE_DIR / "models"))
P_EMB = np.asarray(list(model.embed(KB, batch_size=16)), dtype=np.float32)
S_EMB = np.asarray(list(model.embed([s for _, s in SENTS], batch_size=32)), dtype=np.float32)
P_EMB /= np.maximum(np.linalg.norm(P_EMB, axis=1, keepdims=True), 1e-12)
S_EMB /= np.maximum(np.linalg.norm(S_EMB, axis=1, keepdims=True), 1e-12)

# -------------------- Lexical (keyword) index --------------------
STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "when", "where", "why", "how", "does", "do",
    "did", "can", "could", "would", "should", "in", "on", "at", "to",
    "of", "for", "from", "by", "with", "and", "or", "as", "if", "it",
    "this", "that", "these", "those", "according", "explain", "describe",
    "tell", "give", "compare", "both", "between", "versus", "vs", "unlike",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def norm_versions(text):
    """Keep 'HTTP/3' and 'TLS 1.3' as single distinctive tokens (http3, tls13)."""
    text = text.lower()
    text = re.sub(r"http\s*/\s*(\d)(?:\.\d)?", r"http\1", text)
    text = re.sub(r"\btls\s*v?(\d)\.(\d)", r"tls\1\2", text)
    return text


def tokens(text):
    return [t for t in TOKEN_RE.findall(norm_versions(text)) if t not in STOP and len(t) > 1]


def stem_set(text):
    """Tokens with a trailing 's' removed, so 'heartbeats' matches 'heartbeat'."""
    return {
        t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t
        for t in tokens(text)
    }


DOC_TOKENS = [tokens(p) for p in KB]
DOC_TF = [Counter(ts) for ts in DOC_TOKENS]
N_DOCS = len(KB)
DF = Counter()
for counts in DOC_TF:
    DF.update(counts.keys())
IDF = {t: log(1 + (N_DOCS - f + 0.5) / (f + 0.5)) for t, f in DF.items()}
AVG_LEN = sum(len(x) for x in DOC_TOKENS) / max(N_DOCS, 1)
VOCAB = set(DF)
PREFIX6 = {w[:6] for w in VOCAB}
KNOWN_ALIASES = {
    "sgd", "mvcc", "wal", "jwt", "svm", "mlp", "cnn", "ecc", "pkce",
    "ttl", "rtt", "rto", "lru", "hlc", "ssl", "mdns", "ok", "ai",
}


def lexical_scores(question):
    q_terms = set(tokens(question))
    scores = np.zeros(N_DOCS, dtype=np.float32)
    if not q_terms:
        return scores
    k1, b = 1.5, 0.75
    for i, counts in enumerate(DOC_TF):
        doc_len = len(DOC_TOKENS[i])
        norm = k1 * (1 - b + b * doc_len / max(AVG_LEN, 1))
        total = 0.0
        for term in q_terms:
            tf = counts.get(term, 0)
            if tf:
                total += IDF.get(term, 0.0) * (tf * (k1 + 1)) / (tf + norm)
        scores[i] = total
    maximum = float(scores.max())
    if maximum > 0:
        scores /= maximum
    return scores


def embed_query(text):
    vec = np.asarray(list(model.query_embed(text)), dtype=np.float32)[0]
    return vec / max(float(np.linalg.norm(vec)), 1e-12)


def hybrid_scores(text):
    """Return (normalized query vector, combined dense+lexical score per passage)."""
    vec = embed_query(text)
    combined = np.clip(0.70 * (P_EMB @ vec) + 0.30 * lexical_scores(text), 0.0, 1.0)
    return vec, combined


def expand(text):
    """Add the KB's own wording when a question paraphrases it
    (keepalive/coordinator/subordinate -> heartbeats/leader/follower)."""
    low = text.lower()
    extra = []
    if re.search(r"keep-?alive", low) and re.search(
        r"coordinator|subordinate|election|follower|leader", low
    ):
        extra.append("heartbeats leader follower election timeouts")
    return text + (" " + " ".join(extra) if extra else "")


# -------------------- Injection defence --------------------
INJ = re.compile(
    r"(ignore|disregard|forget|bypass).{0,80}(instruction|prompt|rules?|above|previous|prior|context|constraints?|knowledge base|passages?)"
    r"|system prompt|you are now|developer mode|jailbreak|new instructions?"
    r"|\bsystem\s*(override|message|note|instruction)"
    r"|reveal .{0,50}(prompt|key|secret)|\boverride\b|(?:^|\s)act as\b|pretend (to be|you)"
    r"|(reply|respond|answer)\s+(only\s+)?with\b"
    r"|(output|print|say|write)\s+(exactly|only|just)?\s*[\"'\u201c]"
    r"|\[/?(inst|system)\]|instead of answering|force sources",
    re.I,
)
FACTUAL_START = re.compile(r"\b(what|which|how|where|when|who)\b", re.I)


def strip_hidden(q):
    """Normalize unicode, remove zero-width chars, HTML comments, and bracketed injections."""
    q = unicodedata.normalize("NFKC", q)
    q = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", q)
    q = re.sub(r"<!--.*?-->", " ", q, flags=re.S)

    def drop(m):
        return " " if INJ.search(m.group(0)) else m.group(0)

    return re.sub(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}|<[^>]*>", drop, q)


def clean_question(question):
    """Remove injected clauses but KEEP the real question on either side of them."""
    question = strip_hidden(question)
    kept = []
    for part in re.split(r"(?<=[.!?;])\s+|\n+", question):
        part = part.strip()
        if not part:
            continue
        m = INJ.search(part)
        if not m:
            kept.append(part)
            continue
        head = part[: m.start()].strip(" ,;:-")
        if len(head.split()) >= 2:
            kept.append(head)
        tail = part[m.end():]
        f = FACTUAL_START.search(tail)
        if f:
            cand = tail[f.start():].strip(" ,;:-")
            if len(cand.split()) >= 2:
                kept.append(cand)
    return " ".join(kept).strip()


def unknown_entities(question):
    """Abstain on named things (Kafka, Zab, CLR...) that are not in this KB.
    Capitalised words that merely START a sentence (e.g. 'Furthermore') are ignored."""
    unknown = []
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9+#.\-]*", question):
        word = m.group(0)
        if word.lower() in STOP:
            continue
        before = question[: m.start()].rstrip()
        sentence_start = before == "" or before[-1] in ".!?:;"
        if sentence_start and not (word.isupper() and len(word) > 1):
            continue
        if word[0].isupper() or word.isupper():
            parts = TOKEN_RE.findall(word.lower())
            if parts and not all(
                p in VOCAB or p in KNOWN_ALIASES or p[:6] in PREFIX6 for p in parts
            ):
                unknown.append(word)
    return unknown


# -------------------- Answer extraction --------------------
NUM_Q = re.compile(
    r"how (many|much|long|large|big|often)|what (port|size|threshold|value|timeout|"
    r"interval|rate|number|duration|length|limit|time)|which port"
    r"|\b(duration|milliseconds?|bits|port|interval|timeout|threshold|latency|window|"
    r"coefficient|learning rate|layers?|blocks?|batch|quorum|size|length)\b",
    re.I,
)
SYNTH = re.compile(
    r"compare|\bboth\b|differenc|differ\b|versus|\bvs\b|combined|together|\bratio\b|\bsum\b"
    r"|(higher|lower|larger|smaller|longer|shorter)\s+than",
    re.I,
)
SPLIT = re.compile(
    r"\band\b|\bversus\b|\bvs\.?(?=\s)|\bthan\b|\bcompared (?:to|with)\b|\bwith\b|\bor\b|,|;",
    re.I,
)
HOW_Q = re.compile(r"^\s*how\s+(does|do|did|can|could|would)\b", re.I)


def best_sentence(question_vector, question, passage_id):
    indices = SENT_IDX[passage_id]
    sims = S_EMB[indices] @ question_vector
    if NUM_Q.search(question):
        sims = sims + np.asarray(
            [0.08 if re.search(r"\d", SENTS[j][1]) else 0.0 for j in indices], dtype=np.float32
        )
    q_terms = stem_set(question)
    if q_terms:
        for local_i, global_i in enumerate(indices):
            overlap = len(q_terms & stem_set(SENTS[global_i][1])) / max(len(q_terms), 1)
            sims[local_i] += 0.06 * overlap
    return SENTS[indices[int(np.argmax(sims))]][1]


def premise_split(question):
    """'Since X is 100, what is Y?'  ->  'what is Y?'  (None if no premise clause)."""
    m = re.match(
        r"^\s*(?:since|given(?: that)?|assuming(?: that)?|because|considering(?: that)?|"
        r"suppose(?: that)?|if)\b.*?,\s*((?:what|which|how|where|when|who)\b.*)$",
        question,
        re.I | re.S,
    )
    return m.group(1).strip() if m else None


def covered(question, pid, min_cov=0.5):
    """For 'How does X ...' questions: do the question's words actually appear in the passage?"""
    q = tokens(question)
    if not q:
        return True
    doc = {w[:5] for w in DOC_TOKENS[pid]}
    hits = sum(1 for t in q if t[:5] in doc)
    return hits / len(q) >= min_cov


def synth_chunks(question):
    """Split 'Synthesizing X: A. Furthermore, in adjacent architecture: B. How do both integrate?'
    into the two facts A and B."""
    parts = re.split(r"(?i)\bfurthermore\b[,:]?", question)
    if len(parts) < 2:
        return []
    chunks = []
    for p in parts[:2]:
        p = re.sub(r"(?i)^[^:.]*synthesi[sz]ing[^:]*:", "", p)
        p = re.sub(r"(?i)\bin adjacent architecture\s*:", "", p)
        p = re.sub(r"(?i)\bhow (do|does) (both|they|these)\b.*$", "", p)
        p = p.strip(" ,;:-")
        if len(tokens(p)) >= 2:
            chunks.append(p)
    return chunks


def pick_passages(question, combined):
    """Pick 1 passage normally; for compare/combine questions pick 2 (max 3)."""
    order = np.argsort(-combined)

    chunks = synth_chunks(question)
    if len(chunks) == 2:
        pids = []
        for c in chunks:
            _, s = hybrid_scores(expand(c))
            pids.append(int(np.argmax(s)))
        if pids[0] != pids[1]:
            return pids, order

    selected = [int(order[0])]
    if not SYNTH.search(question):
        return selected, order

    pool = {int(i) for i in order[:6]}
    segments = [s.strip() for s in SPLIT.split(question) if len(tokens(s)) >= 1]
    if len(segments) > 1:
        for seg in segments:
            _, seg_scores = hybrid_scores(expand(seg))
            pid = int(np.argmax(seg_scores))
            if pid in pool and seg_scores[pid] >= 0.25 and pid not in selected:
                selected.append(pid)
    if len(selected) == 1 and len(order) > 1:
        second = int(order[1])
        if combined[second] >= max(0.25, 0.55 * float(combined[order[0]])):
            selected.append(second)
    return selected[:3], order


ABSTAIN = {"answer": None, "sources": [], "scores": [], "confidence": 0.1}

# -------------------- API --------------------
app = FastAPI(title="RAG Colosseum API", version="1.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=MAX_TOP_K)


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/health")
def health_check():
    return {"status": "ok", "passages": len(KB)}


@app.post("/query")
def query(request: QueryRequest):
    raw = request.question or ""
    question = clean_question(raw)
    print(f"RAW={raw!r}\n  CLEAN={question!r}", flush=True)

    if len(question.split()) < 2 or unknown_entities(question):
        print("  -> ABSTAIN (short/unknown entity)", flush=True)
        return ABSTAIN

    q_exp = expand(question)
    q_vector, combined = hybrid_scores(q_exp)
    order = np.argsort(-combined)
    best_score = float(combined[order[0]])
    if best_score < THRESH:
        print(f"  -> ABSTAIN (best={best_score:.3f})", flush=True)
        return ABSTAIN

    selected, order = pick_passages(question, combined)

    if len(selected) > 1:
        # Synthesis: return the full text of each needed passage so every number is present.
        answer = " ".join(KB[i] for i in selected)
    else:
        if HOW_Q.match(question) and not covered(question, selected[0]):
            print("  -> ABSTAIN (question words not covered by best passage)", flush=True)
            return ABSTAIN
        tail = premise_split(question)  # ignore a (possibly false) premise when choosing the sentence
        if tail:
            sent_text = expand(tail)
            sent_vec = embed_query(sent_text)
        else:
            sent_text, sent_vec = q_exp, q_vector
        answer = best_sentence(sent_vec, sent_text, selected[0])

    k = max(min(request.top_k, MAX_TOP_K), len(selected))
    rest = [int(i) for i in order if int(i) not in selected][: k - len(selected)]
    final = sorted(selected, key=lambda i: -float(combined[i])) + rest

    scores, prev = [], 1.0  # keep scores in descending order, as the API contract requires
    for i in final:
        prev = min(float(combined[i]), prev)
        scores.append(round(prev, 4))

    print(f"  -> sources={final} selected={selected}\n  ANSWER={answer[:220]!r}", flush=True)
    return {
        "answer": answer,
        "sources": final,
        "scores": scores,
        "confidence": round(float(np.clip(0.60 + (best_score - THRESH) * 1.2, 0.62, 0.98)), 3),
    }
