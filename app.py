
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
