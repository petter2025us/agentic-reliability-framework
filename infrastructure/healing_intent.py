"""
HealingIntent – the immutable advisory output of the OSS layer.

Contains the evaluation results and a recommended action.
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional


class RecommendedAction(str, Enum):
    """Advisory recommendation from the OSS engine."""
    APPROVE = "approve"
    DENY = "deny"
    ESCALATE = "escalate"


class HealingIntent(BaseModel):
    """
    Immutable output of the infrastructure simulator.

    This object is intended to be passed to the enterprise execution layer,
    which will decide whether to act on it.
    """
    intent_summary: str = Field(description="Short description of the requested action")
    cost_projection: Optional[float] = Field(None, description="Estimated monthly cost in USD")
    risk_score: float = Field(ge=0, le=1, description="Overall risk score (0 = safe, 1 = dangerous)")
    policy_violations: List[str] = Field(default_factory=list, description="List of policy violations")
    recommended_action: RecommendedAction
    justification: str = Field(description="Human-readable explanation of the recommendation")
    confidence_score: float = Field(ge=0, le=1, description="Confidence in the evaluation (0-1)")

    class Config:
        frozen = True  # make the object immutable
