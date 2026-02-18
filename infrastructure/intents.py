# agentic_reliability_framework/infrastructure/intents.py
"""
Infrastructure Intent Schema – Algebraic Data Types with ARF Extensions.

This module implements ARF's core principle: OSS advisory layer with clean
enterprise upgrade paths. Intents are self-validating, cryptographically
auditable, and carry full provenance including chain-of-thought from agents.
"""

from __future__ import annotations

import uuid
import hashlib
import hmac
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, Literal, Optional, Union, List
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic.functional_validators import AfterValidator

# -----------------------------------------------------------------------------
# ARF-Specific Base Classes
# -----------------------------------------------------------------------------
class ARFBaseModel(BaseModel):
    """Base model with ARF-specific configuration."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
        validate_assignment=False,  # frozen=True already prevents assignment
    )

class SignableMixin:
    """Mixin for cryptographic intent signing."""
    
    def sign(self, secret_key: bytes) -> str:
        """Create HMAC signature of intent content."""
        content = self.model_dump_json(exclude={'signature'}).encode()
        return hmac.new(secret_key, content, hashlib.sha256).hexdigest()
    
    def verify(self, signature: str, secret_key: bytes) -> bool:
        """Verify intent signature."""
        expected = self.sign(secret_key)
        return hmac.compare_digest(expected, signature)

# -----------------------------------------------------------------------------
# Enhanced Domain Primitives
# -----------------------------------------------------------------------------
Region = Annotated[str, AfterValidator(lambda x: x if x in VALID_AZURE_REGIONS else 
                                        ValueError(f"Invalid region: {x}"))]
Size = str  # Can be enhanced with regex patterns
Principal = str
ResourceScope = str
ServiceName = Annotated[str, AfterValidator(lambda x: x if len(x) >= 3 else 
                                           ValueError("Service name too short"))]

# -----------------------------------------------------------------------------
# ARF Intent Base with Full Provenance
# -----------------------------------------------------------------------------
class ARFIntent(ARFBaseModel, SignableMixin):
    """
    ARF Base Intent with full provenance and chain-of-thought tracking.
    
    Implements ARF principles:
    - Traceability: Every intent has unique ID and full provenance
    - Explainability: Includes chain-of-thought from generating agent
    - Auditability: Optional cryptographic signing
    """
    intent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    requester: Principal
    signature: Optional[str] = Field(None, description="HMAC signature for authenticity")
    
    # ARF-Specific Fields
    chain_of_thought: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Step-by-step reasoning from agent that generated this intent"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in this intent (0-1)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (session ID, agent version, etc.)"
    )
    
    def summarize(self) -> str:
        """Human-readable summary for explainability."""
        return f"{self.intent_type} by {self.requester} at {self.timestamp.isoformat()}"

# -----------------------------------------------------------------------------
[Rest of intent classes with ARFIntent as base]
