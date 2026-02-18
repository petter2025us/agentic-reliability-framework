# agentic_reliability_framework/constants.py
"""
OSS HARD LIMITS - Build-time enforced boundaries for OSS edition
Infrastructure-specific constants added for the new module
"""

from typing import Final, Tuple

# === EXECUTION BOUNDARIES ===
OSS_EDITION: Final[str] = "open-source"
OSS_LICENSE: Final[str] = "Apache 2.0"
EXECUTION_ALLOWED: Final[bool] = False
MCP_MODES_ALLOWED: Final[Tuple[str, ...]] = ("advisory",)

# === MEMORY/RAG BOUNDARIES ===
MAX_INCIDENT_NODES: Final[int] = 1_000
MAX_OUTCOME_NODES: Final[int] = 5_000
MAX_SIMILARITY_CACHE: Final[int] = 100
EMBEDDING_DIM: Final[int] = 384
SIMILARITY_THRESHOLD: Final[float] = 0.3

# === INFRASTRUCTURE-SPECIFIC CONSTANTS ===
MAX_POLICY_VIOLATIONS: Final[int] = 100
MAX_RISK_FACTORS: Final[int] = 20
MAX_COST_PROJECTIONS: Final[int] = 10
MAX_DECISION_TREE_DEPTH: Final[int] = 10
MAX_ALTERNATIVE_ACTIONS: Final[int] = 5

# === UPGRADE PATH ===
ENTERPRISE_UPGRADE_URL: Final[str] = "https://arf.dev/enterprise"
