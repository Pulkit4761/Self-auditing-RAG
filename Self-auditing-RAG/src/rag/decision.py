from enum import Enum
from . import config
from .auditor import AuditResult

class Decision(Enum):
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"
    REJECT = "REJECT"


def decide(audit_result: AuditResult) -> Decision:
    score = audit_result.faithfulness_score
    if score >= config.ACCEPT_THRESHOLD:
        return Decision.ACCEPT
    if score >= config.REJECT_THRESHOLD:
        return Decision.REVISE
    return Decision.REJECT
