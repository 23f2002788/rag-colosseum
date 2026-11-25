import json
import math
import os
import re
from collections import Counter

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

THRESH = float(os.getenv("THRESH", "0.40"))

# ---------------- 1. Load and index the knowledge base once ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "kb.json"), "r", encoding="utf-8") as f:
    KB = json.load(f)

if not isinstance(KB, list) or len(KB) != 50 or not all(isinstance(p, str) for p in KB):
    raise ValueError("kb.json must be a JSON array containing exactly 50 passage strings.")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
P_EMB = model.encode(KB, normalize_embeddings=True)

SENTS = []
SENT_IDX = {}
for pid, passage in enumerate(KB):
    for sentence in re.split(r"(?<=[.!?])\s+", passage.strip()):
        if sentence:
            SENT_IDX.setdefault(pid, []).append(len(SENTS))
            SENTS.append((pid, sentence))

S_EMB = model.encode([sentence for _, sentence in SENTS], normalize_embeddings=True)

tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
).fit(KB)
P_TFIDF = tfidf.transform(KB)

STOP = set(ENGLISH_STOP_WORDS) | {
    "what", "which", "how", "does", "do", "used", "use", "using", "value",
    "number", "many", "much", "long", "often", "default", "protocol",
    "system", "mentioned", "passage", "according", "knowledge", "base",
    "kb", "describe", "explain", "give", "tell", "name", "compare",
    "difference", "differ", "versus", "vs", "between", "both", "unlike",
    "need", "needed", "required", "require", "requires", "specified",
    "configured", "typical",
}


def toks(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def stem(token):
    return token[:5] if len(token) > 4 else token


PASS_STEMS = [{stem(token) for token in toks(passage)} for passage in KB]
DF = Counter(stem_token for passage_stems in PASS_STEMS for stem_token in passage_stems)


def content_stems(question):
    result = []
    for token in toks(question):
        token_stem = stem(token)
        if token not in STOP and token_stem not in result:
            result.append(token_stem)
    return result


# ---------------- 2. Prompt-injection and abstention helpers ----------------
INJ = re.compile(
    r"(ignore|disregard|forget).{0,40}(instruction|prompt|rules|above|previous)"
    r"|system prompt|you are now|developer mode|jailbreak|new instructions?"
    r"|reveal .{0,30}(prompt|key|secret)|\boverride\b|(?:^|\s)act as\b",
    re.I,
)


def clean(question):
    kept_parts = []
    for part in re.split(r"(?<=[.!?])\s+|\n+", question):
        match = INJ.search(part)
        if not match:
            kept_parts.append(part)
        else:
            head = part[:match.start()].strip()
            if len(head.split()) >= 3:
                kept_parts.append(head)
    return " ".join(kept_parts).strip()


SKIPCAP = {
    "what", "which", "how", "who", "when", "where", "why", "does", "do",
    "is", "are", "the", "in", "on", "of", "a", "an", "if", "can",
    "according", "compare", "explain", "describe", "give", "tell", "name",
    "unlike", "both",
}


def unknown_entities(question):
    bad = []
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#.\-]*", question)
    for i, token in enumerate(tokens):
        if token.lower() in SKIPCAP or len(token) < 2:
            continue
        if (token[0].isupper() and i > 0) or token.isupper() or any(c.isdigit() for c in token):
            parts = re.findall(r"[a-z0-9]+", token.lower())
            if parts and not all(stem(part) in DF for part in parts):
                bad.append(token)
    return bad


NUMQ = re.compile(
    r"\b(port|timeout|threshold|interval|size|rate|quorum|number|limit|duration|"
    r"length|bits?|retries|window|deadline|lease|ratio|how many|how much|"
    r"how long|how often)\b",
    re.I,
)
SYNTH = re.compile(
    r"compare|both|difference|differ|versus|\bvs\b|between|than|sum|total|"
    r"combined|together|ratio|higher|lower|larger|smaller|longer|shorter|"
    r"\band\b|&|,|\bor\b",
    re.I,
)

N = len(KB)


def key_stems(content, count):
    known = [token_stem for token_stem in content
             if token_stem in DF and DF[token_stem] <= 8]
    return sorted(known, key=lambda token_stem: DF[token_stem])[:count]


def supported(key, passage_id):
    """For a single-topic question, require key terms in the same passage."""
    return all(token_stem in PASS_STEMS[passage_id] for token_stem in key)


def best_sentences(query_embedding, question, passage_id, count=1):
    indices = SENT_IDX[passage_id]
    similarities = S_EMB[indices] @ query_embedding
    if NUMQ.search(question):
        similarities = similarities + np.array([
            0.1 if re.search(r"\d", SENTS[i][1]) else 0
            for i in indices
        ])
    best_indices = sorted(np.argsort(-similarities)[:count])
    return [SENTS[indices[j]][1] for j in best_indices]


ABSTAIN = {
    "answer": None,
    "sources": [],
    "scores": [],
    "confidence": 0.1,
}


def log(**kwargs):
    print("PROBE", json.dumps(kwargs, default=str), flush=True)


# ---------------- 3. FastAPI service ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Req(BaseModel):
    question: str
    top_k: int = 3


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "passages": len(KB)}


@app.post("/query")
def query(req: Req):
    k = max(1, min(int(req.top_k), 10))
    q = clean(req.question or "")
    content = content_stems(q)
    unknown = [token_stem for token_stem in content if token_stem not in DF]
    multi = bool(SYNTH.search(q))
    rare = key_stems(content, 4 if multi else 2)

    if len(q.split()) < 2 or not content:
        log(q=req.question, decision="abstain:empty")
        return ABSTAIN

    entities = unknown_entities(q)
    if entities:
        log(q=req.question, decision="abstain:unknown-entity", ents=entities)
        return ABSTAIN

    if len(unknown) / len(content) >= 0.4:
        log(q=req.question, decision="abstain:off-topic", unknown=unknown)
        return ABSTAIN

    query_embedding = model.encode([q], normalize_embeddings=True)[0]
    dense = P_EMB @ query_embedding
    lexical = (P_TFIDF @ tfidf.transform([q]).T).toarray().ravel()
    score = np.clip(0.75 * dense + 0.25 * np.clip(lexical, 0, 1), 0, 1)
    order = [int(i) for i in np.argsort(-score)]

    top = order[0]
    # Prefer a close-scoring passage that contains the key terms.
    if not supported(rare, top):
        alternatives = [
            p for p in order[:5]
            if supported(rare, p) and score[p] >= 0.7 * score[top]
        ]
        if alternatives and not multi:
            top = alternatives[0]

    chosen = [top]

    if multi:
        covered = {token_stem for token_stem in rare if token_stem in PASS_STEMS[top]}
        for _ in range(2):
            uncovered = [token_stem for token_stem in rare if token_stem not in covered]
            if not uncovered:
                break

            candidates = [p for p in order[:8] if p not in chosen]
            if not candidates:
                break

            candidate = max(
                candidates,
                key=lambda p: (
                    sum(token_stem in PASS_STEMS[p] for token_stem in uncovered),
                    score[p],
                ),
            )
            newly_covered = {
                token_stem for token_stem in uncovered
                if token_stem in PASS_STEMS[candidate]
            }
            if not newly_covered:
                break

            chosen.append(candidate)
            covered |= newly_covered

        ok = (not rare) or len(covered) >= math.ceil(0.75 * len(rare))
    else:
        ok = supported(rare, top)

    best = float(score[top])
    if best < THRESH or not ok:
        log(
            q=req.question,
            clean=q,
            decision="abstain:low-support",
            best=round(best, 3),
            rare=rare,
            ok=ok,
        )
        return ABSTAIN

    pool = order[:max(k, len(chosen))]
    for passage_id in chosen:
        if passage_id not in pool:
            pool.append(passage_id)

    pool.sort(key=lambda p: -float(score[p]))
    chosen_sorted = sorted(chosen, key=lambda p: -float(score[p]))

    sentence_count = 2 if multi else 1
    answer_parts = []
    for passage_id in chosen_sorted:
        answer_parts.extend(
            best_sentences(query_embedding, q, passage_id, sentence_count)
        )

    log(
        q=req.question,
        clean=q,
        decision="answer",
        multi=multi,
        chosen=chosen_sorted,
        top=round(best, 3),
    )

    return {
        "answer": " ".join(answer_parts),
        "sources": pool,
        "scores": [round(float(score[p]), 4) for p in pool],
        "confidence": round(
            float(np.clip(0.6 + (best - THRESH) * 1.2, 0.62, 0.98)),
            3,
        ),
    }
