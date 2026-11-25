import json
import os
import re
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

# Keep ONNX runtime within a small Render instance.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# -------------------- Load and validate the knowledge base --------------------
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

# FastEmbed uses ONNX Runtime and is lighter than loading PyTorch on Render free.
model = TextEmbedding(model_name=MODEL_NAME, threads=1)
P_EMB = np.asarray(list(model.embed(KB, batch_size=16)), dtype=np.float32)
S_EMB = np.asarray(
    list(model.embed([sentence for _, sentence in SENTS], batch_size=32)),
    dtype=np.float32,
)
P_EMB /= np.maximum(np.linalg.norm(P_EMB, axis=1, keepdims=True), 1e-12)
S_EMB /= np.maximum(np.linalg.norm(S_EMB, axis=1, keepdims=True), 1e-12)

# -------------------- Lightweight BM25-style lexical index --------------------
STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "when", "where", "why", "how", "does", "do",
    "did", "can", "could", "would", "should", "in", "on", "at", "to",
    "of", "for", "from", "by", "with", "and", "or", "as", "if", "it",
    "this", "that", "these", "those", "according", "explain", "describe",
    "tell", "give", "compare", "both", "between", "versus", "vs", "unlike",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP and len(t) > 1]


DOC_TOKENS = [tokens(p) for p in KB]
DOC_TF = [Counter(ts) for ts in DOC_TOKENS]
N_DOCS = len(KB)
DF = Counter()
for counts in DOC_TF:
    DF.update(counts.keys())
IDF = {
    term: log(1 + (N_DOCS - freq + 0.5) / (freq + 0.5))
    for term, freq in DF.items()
}
AVG_LEN = sum(len(x) for x in DOC_TOKENS) / max(N_DOCS, 1)
VOCAB = set(DF)
PREFIX6 = {word[:6] for word in VOCAB}
KNOWN_ALIASES = {"sgd", "mvcc", "wal", "jwt", "svm", "mlp", "cnn", "ecc", "pkce"}


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


# -------------------- Query safety and intent helpers --------------------
INJ = re.compile(
    r"(ignore|disregard|forget).{0,80}(instruction|prompt|rules|above|previous)"
    r"|system prompt|you are now|developer mode|jailbreak|new instructions?"
    r"|reveal .{0,50}(prompt|key|secret)|\boverride\b|(?:^|\s)act as\b",
    re.I,
)
FACTUAL_START = re.compile(r"\b(what|which|how|where|when|who)\b", re.I)


def clean_question(question):
    """Remove injection-only clauses while retaining a factual question where possible."""
    kept = []
    # Semicolons/newlines commonly separate an injected command from the real query.
    parts = re.split(r"(?<=[.!?;])\s+|\n+", question)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = INJ.search(part)
        if not match:
            kept.append(part)
            continue

        # If a factual interrogative follows the injected instruction in the same
        # sentence (e.g. "ignore rules and tell me what port Redis uses"), retain it.
        tail = part[match.end():]
        factual = FACTUAL_START.search(tail)
        if factual:
            candidate = tail[factual.start():].strip(" ,;:-")
            if len(candidate.split()) >= 2:
                kept.append(candidate)

    return " ".join(kept).strip()


def unknown_entities(question):
    """Abstain on likely named entities not represented in this fixed KB."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9+#.\-]*", question)
    unknown = []
    for index, word in enumerate(words):
        if word.lower() in STOP:
            continue
        looks_named = (index > 0 and word[0].isupper()) or word.isupper()
        if looks_named:
            parts = TOKEN_RE.findall(word.lower())
            if parts and not all(
                p in VOCAB or p in KNOWN_ALIASES or p[:6] in PREFIX6 for p in parts
            ):
                unknown.append(word)
    return unknown


NUM_Q = re.compile(
    r"how (many|much|long|large|big|often)|what (port|size|threshold|value|timeout|"
    r"interval|rate|number|duration|length|limit|time)|which port",
    re.I,
)
SYNTH = re.compile(
    r"compare|both|difference|differ|versus|\bvs\b|between|than|sum|total|combined|"
    r"together|ratio|higher|lower|larger|smaller|longer|shorter",
    re.I,
)


def best_sentence(question_vector, question, passage_id):
    indices = SENT_IDX[passage_id]
    similarities = S_EMB[indices] @ question_vector

    if NUM_Q.search(question):
        similarities = similarities + np.asarray(
            [0.08 if re.search(r"\d", SENTS[j][1]) else 0.0 for j in indices],
            dtype=np.float32,
        )

    q_terms = set(tokens(question))
    if q_terms:
        for local_i, global_i in enumerate(indices):
            sentence_terms = set(tokens(SENTS[global_i][1]))
            overlap = len(q_terms & sentence_terms) / max(len(q_terms), 1)
            similarities[local_i] += 0.06 * overlap

    return SENTS[indices[int(np.argmax(similarities))]][1]


ABSTAIN = {"answer": None, "sources": [], "scores": [], "confidence": 0.1}

# -------------------- API --------------------
app = FastAPI(title="RAG Colosseum API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    question = clean_question(request.question or "")
    if len(question.split()) < 2 or unknown_entities(question):
        return ABSTAIN

    q_vector = np.asarray(list(model.query_embed(question)), dtype=np.float32)[0]
    q_vector /= max(float(np.linalg.norm(q_vector)), 1e-12)

    dense = P_EMB @ q_vector
    lexical = lexical_scores(question)
    # A slightly stronger lexical component helps distinguish technical confounders.
    combined = np.clip(0.70 * dense + 0.30 * lexical, 0.0, 1.0)

    k = min(request.top_k, MAX_TOP_K)
    order = np.argsort(-combined)[:k]
    best_score = float(combined[order[0]])

    if best_score < THRESH:
        return ABSTAIN

    selected = [int(order[0])]
    # For explicit comparison/synthesis questions, include a second passage when
    # it has meaningful support. This is less restrictive than the previous 0.80.
    if SYNTH.search(question) and len(order) > 1:
        second = int(order[1])
        second_score = float(combined[second])
        if second_score >= max(0.25, 0.55 * best_score):
            selected.append(second)

    answer_sentences = [best_sentence(q_vector, question, pid) for pid in selected]

    return {
        "answer": " ".join(answer_sentences),
        "sources": [int(i) for i in order],
        "scores": [round(float(combined[i]), 4) for i in order],
        "confidence": round(
            float(np.clip(0.60 + (best_score - THRESH) * 1.2, 0.62, 0.98)), 3
        ),
    }
