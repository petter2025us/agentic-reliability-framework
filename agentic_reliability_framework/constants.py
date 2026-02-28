# agentic_reliability_framework/constants.py
"""
OSS HARD LIMITS - Build-time enforced boundaries for OSS edition
Infrastructure-specific constants added for the new module
"""

from typing import Final, Tuple
import hashlib
import os

# ============================================================================
# VERSION HELPERS
# ============================================================================

def _get_oss_version() -> str:
    """Get OSS version from package metadata"""
    try:
        from . import __version__
        version = __version__
        if not version.endswith("-oss"):
            version = f"{version}-oss"
        return version
    except ImportError:
        return "1.0.0-oss"

# ============================================================================
# OSS ARCHITECTURAL BOUNDARIES
# ============================================================================

# === EXECUTION BOUNDARIES ===
OSS_EDITION: Final[str] = "open-source"
OSS_LICENSE: Final[str] = "Apache 2.0"
OSS_VERSION: Final[str] = _get_oss_version()
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

# === COMPATIBILITY HASH ===
def _generate_oss_hash() -> str:
    """Generate hash of OSS constants for validation"""
    constants_data = {
        "MAX_INCIDENT_NODES": MAX_INCIDENT_NODES,
        "EXECUTION_ALLOWED": EXECUTION_ALLOWED,
        "MCP_MODES_ALLOWED": MCP_MODES_ALLOWED,
    }
    json_str = str(sorted(constants_data.items()))
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]

OSS_CONSTANTS_HASH: Final[str] = _generate_oss_hash()

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

class OSSBoundaryError(RuntimeError):
    """Error raised when OSS boundaries are violated"""
    pass

def validate_oss_config(config: dict) -> None:
    """Validate configuration against OSS boundaries"""
    violations = []
    
    # Check MCP mode
    if config.get("mcp_mode", "advisory") != "advisory":
        violations.append(f"MCP mode must be 'advisory' in OSS edition")
    
    # Check execution capability
    if config.get("execution_allowed", False):
        violations.append("Execution not allowed in OSS edition")
    
    # Check RAG limits
    if config.get("rag_max_incident_nodes", 0) > MAX_INCIDENT_NODES:
        violations.append(f"rag_max_incident_nodes exceeds OSS limit")
    
    if config.get("rag_max_outcome_nodes", 0) > MAX_OUTCOME_NODES:
        violations.append(f"rag_max_outcome_nodes exceeds OSS limit")
    
    if violations:
        raise OSSBoundaryError(
            f"OSS configuration violations:\n" +
            "\n".join(f"  • {v}" for v in violations)
        )

def get_oss_capabilities() -> dict:
    """Get OSS edition capabilities"""
    return {
        "edition": OSS_EDITION,
        "license": OSS_LICENSE,
        "version": OSS_VERSION,
        "constants_hash": OSS_CONSTANTS_HASH,
        "execution": {
            "allowed": EXECUTION_ALLOWED,
            "modes": list(MCP_MODES_ALLOWED),
        },
        "limits": {
            "max_incident_nodes": MAX_INCIDENT_NODES,
            "max_outcome_nodes": MAX_OUTCOME_NODES,
            "max_similarity_cache": MAX_SIMILARITY_CACHE,
            "embedding_dim": EMBEDDING_DIM,
            "similarity_threshold": SIMILARITY_THRESHOLD,
        },
        "infrastructure_limits": {
            "max_policy_violations": MAX_POLICY_VIOLATIONS,
            "max_risk_factors": MAX_RISK_FACTORS,
            "max_cost_projections": MAX_COST_PROJECTIONS,
            "max_decision_tree_depth": MAX_DECISION_TREE_DEPTH,
            "max_alternative_actions": MAX_ALTERNATIVE_ACTIONS,
        },
        "upgrade_available": True,
        "upgrade_url": ENTERPRISE_UPGRADE_URL,
    }

def check_oss_compliance() -> bool:
    """Check if current environment is OSS compliant"""
    try:
        # Check environment variables
        tier = os.getenv("ARF_TIER", "oss").lower()
        if tier != "oss":
            return False
        
        # Check for enterprise dependencies
        enterprise_deps = ["neo4j", "psycopg2", "sqlalchemy", "sentence_transformers"]
        for dep in enterprise_deps:
            try:
                __import__(dep)
                # If we can import it, check if it's being used
                if dep == "sqlalchemy" and os.getenv("DATABASE_URL"):
                    return False
                if dep in ["neo4j", "psycopg2"]:
                    return False
            except ImportError:
                pass
        
        return True
    except Exception:
        return True

# Export
__all__ = [
    "OSS_EDITION",
    "OSS_LICENSE", 
    "OSS_VERSION",
    "OSS_CONSTANTS_HASH",
    "EXECUTION_ALLOWED",
    "MCP_MODES_ALLOWED",
    "MAX_INCIDENT_NODES",
    "MAX_OUTCOME_NODES",
    "MAX_SIMILARITY_CACHE",
    "EMBEDDING_DIM",
    "SIMILARITY_THRESHOLD",
    "MAX_POLICY_VIOLATIONS",
    "MAX_RISK_FACTORS",
    "MAX_COST_PROJECTIONS",
    "MAX_DECISION_TREE_DEPTH",
    "MAX_ALTERNATIVE_ACTIONS",
    "ENTERPRISE_UPGRADE_URL",
    "OSSBoundaryError",
    "validate_oss_config",
    "get_oss_capabilities",
    "check_oss_compliance",
]
