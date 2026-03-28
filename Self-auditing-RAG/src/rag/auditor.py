import re
from dataclasses import dataclass, field

import numpy as np

from . import config
from .embedder import Embedder


@dataclass
class SentenceVerdict:
    sentence: str
    score: float
    supported: bool


@dataclass
class AuditResult:
    faithfulness_score: float
    sentence_verdicts: list[SentenceVerdict] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


class Auditor:
    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def audit(self, answer: str, context_chunks: list[str]) -> AuditResult:
        sentences = _split_sentences(answer)
        if not sentences:
            return AuditResult(faithfulness_score=0.0)

        sentence_embeddings = self._embedder.embed(sentences)
        chunk_embeddings = self._embedder.embed(context_chunks)

        # Normalize for cosine similarity
        s_norm = sentence_embeddings / (
            np.linalg.norm(sentence_embeddings, axis=1, keepdims=True) + 1e-10
        )
        c_norm = chunk_embeddings / (
            np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-10
        )

        # Similarity matrix: (num_sentences, num_chunks)
        sim_matrix = s_norm @ c_norm.T

        verdicts = []
        unsupported = []
        for i, sentence in enumerate(sentences):
            score = float(sim_matrix[i].max())
            supported = score >= config.SENTENCE_SUPPORT_THRESHOLD
            verdicts.append(SentenceVerdict(sentence, score, supported))
            if not supported:
                unsupported.append(sentence)

        overall_score = float(np.mean([v.score for v in verdicts]))

        return AuditResult(
            faithfulness_score=overall_score,
            sentence_verdicts=verdicts,
            unsupported_claims=unsupported,
        )
