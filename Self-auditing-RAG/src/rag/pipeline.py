from dataclasses import dataclass, field
from .auditor import AuditResult, Auditor
from .decision import Decision, decide
from .embedder import Embedder
from .generator import Generator
from .retriever import Retriever


@dataclass
class PipelineResult:
    query: str
    context_chunks: list[str]
    initial_answer: str
    audit: AuditResult
    decision: Decision
    final_answer: str
    was_revised: bool = False


class RAGPipeline:
    def __init__(self):
        self._embedder = Embedder()
        self._retriever = Retriever(self._embedder)
        self._generator = Generator()
        self._auditor = Auditor(self._embedder)

    def ingest(self) -> int:
        return self._retriever.ingest()

    def query(self, user_query: str) -> PipelineResult:
        # Step 1: Retrieve
        chunks = self._retriever.retrieve(user_query)

        # Step 2: Generate
        answer = self._generator.generate(user_query, chunks)

        # Step 3: Audit
        audit_result = self._auditor.audit(answer, chunks)

        # Step 4: Decide
        decision = decide(audit_result)

        # Step 5: Revise if needed
        final_answer = answer
        was_revised = False

        if decision == Decision.REVISE and audit_result.unsupported_claims:
            final_answer = self._generator.revise(
                user_query, chunks, audit_result.unsupported_claims
            )
            was_revised = True
        elif decision == Decision.REJECT:
            final_answer = (
                "The system could not generate a sufficiently faithful answer "
                "based on the available documents. The retrieved context may "
                "not contain relevant information for this query."
            )

        return PipelineResult(
            query=user_query,
            context_chunks=chunks,
            initial_answer=answer,
            audit=audit_result,
            decision=decision,
            final_answer=final_answer,
            was_revised=was_revised,
        )
