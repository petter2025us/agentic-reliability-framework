# agentic_reliability_framework/infrastructure/healing_intent.py
"""
HealingIntent – The advisory output of the OSS layer.

This immutable object encapsulates the result of evaluating an infrastructure intent.
It includes risk score, policy violations, a recommended action, and detailed
justification. It is designed to be consumed by the enterprise execution layer,
which may augment it with enforcement decisions.

The structure supports explainable AI (XAI) by providing both a high-level
recommendation and a detailed breakdown of the reasoning.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class RecommendedAction(str, Enum):
    """Advisory recommendation from the OSS engine."""
    APPROVE = "approve"
    DENY = "deny"
    ESCALATE = "escalate"


class HealingIntent(BaseModel):
    """
    Immutable output of the infrastructure simulator.

    This object is intended to be passed to the enterprise execution layer,
    which will decide whether to act on it. It contains all information needed
    for a human (or an enterprise agent) to make an informed decision.
    """
    intent_id: str = Field(..., description="ID of the original intent")
    intent_summary: str = Field(..., description="Short human-readable description")
    cost_projection: Optional[float] = Field(None, description="Estimated monthly cost in USD")
    risk_score: float = Field(ge=0, le=1, description="Overall risk score (0 = safe, 1 = dangerous)")
    policy_violations: List[str] = Field(default_factory=list, description="List of policy violations")
    recommended_action: RecommendedAction
    justification: str = Field(..., description="Human-readable explanation")
    confidence_score: float = Field(ge=0, le=1, description="Confidence in the evaluation (0-1)")
    evaluation_details: Optional[Dict[str, Any]] = Field(
        None, description="Raw factor contributions and metadata (for debugging/UI)"
    )

    class Config:
        frozen = True  # immutable
        json_schema_extra = {
            "example": {
                "intent_id": "123e4567-e89b-12d3-a456-426614174000",
                "intent_summary": "provision_resource requested by alice",
                "cost_projection": 280.0,
                "risk_score": 0.62,
                "policy_violations": ["Region not allowed"],
                "recommended_action": "deny",
                "justification": "Risk score 0.62. Policy violations: Region not allowed.",
                "confidence_score": 0.9,
                "evaluation_details": {
                    "factor_contributions": {
                        "intent_type": 0.1,
                        "cost": 0.05,
                        "policy_violations": 0.4
                    }
                }
            }
        }
