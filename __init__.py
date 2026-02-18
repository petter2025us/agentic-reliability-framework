# agentic_reliability_framework/__init__.py
"""
Agentic Reliability Framework - OSS Edition
Unified package containing both the original ARF components and the new
infrastructure governance module for Azure.

The original ARF exports are available at the top level for backward compatibility.
New infrastructure components can be accessed via the `.infrastructure` submodule
or via top‑level symbols with an `Infra` prefix (e.g., `InfraHealingIntent`).
"""

__version__ = "1.0.0"

# ============================================================================
# IMPORT ORIGINAL ARF COMPONENTS (for drop‑in compatibility)
# ============================================================================

# 1. Constants and exceptions
from .arf_core.constants import (
    OSS_EDITION,
    OSS_LICENSE,
    EXECUTION_ALLOWED,
    MCP_MODES_ALLOWED,
    MAX_INCIDENT_NODES,
    MAX_OUTCOME_NODES,
    validate_oss_config,
    get_oss_capabilities,
    check_oss_compliance,
    OSSBoundaryError,
)

# 2. HealingIntent and related factories
from .arf_core.models.healing_intent import (
    HealingIntent as OriginalHealingIntent,
    HealingIntentSerializer,
    IntentSource as OriginalIntentSource,
    IntentStatus as OriginalIntentStatus,
    create_rollback_intent,
    create_restart_intent,
    create_scale_out_intent,
    create_oss_advisory_intent,
)

# 3. Core models (ReliabilityEvent, etc.)
from .arf_core.models import (
    ReliabilityEvent,
    EventSeverity,
    create_compatible_event,
)

# 4. OSS MCP client
from .arf_core.engine.oss_mcp_client import (
    OSSMCPClient,
    OSSMCPResponse,
    OSSAnalysisResult,
    create_oss_mcp_client,
)

# 5. Engine factory
from .engine.engine_factory import (
    EngineFactory,
    create_engine,
    get_engine,
    get_oss_engine_capabilities,
)

# 6. Availability flag (computed from old constants)
OSS_AVAILABLE = True

# ============================================================================
# IMPORT NEW INFRASTRUCTURE COMPONENTS
# ============================================================================

from .infrastructure import (
    AzureInfrastructureSimulator,
    ProvisionResourceIntent,
    DeployConfigurationIntent,
    GrantAccessIntent,
    ResourceType,
    Environment,
    PermissionLevel,
    Policy,
    CostEstimator,
    RiskEngine,
    HealingIntent as InfraHealingIntent,
    IntentSource as InfraIntentSource,
    IntentStatus as InfraIntentStatus,
    RecommendedAction,
    create_infrastructure_healing_intent,
    validate_infrastructure_config,
)

# ============================================================================
# TOP‑LEVEL EXPORTS – MIX OF ORIGINAL AND PREFIXED NEW SYMBOLS
# ============================================================================

__all__ = [
    # Version
    "__version__",

    # Original ARF constants
    "OSS_EDITION",
    "OSS_LICENSE",
    "EXECUTION_ALLOWED",
    "MCP_MODES_ALLOWED",
    "MAX_INCIDENT_NODES",
    "MAX_OUTCOME_NODES",
    "validate_oss_config",
    "get_oss_capabilities",
    "check_oss_compliance",
    "OSSBoundaryError",

    # Original ARF models
    "HealingIntent",               # alias to OriginalHealingIntent
    "HealingIntentSerializer",
    "IntentSource",                 # alias to OriginalIntentSource
    "IntentStatus",                 # alias to OriginalIntentStatus
    "create_rollback_intent",
    "create_restart_intent",
    "create_scale_out_intent",
    "create_oss_advisory_intent",

    # Core models
    "ReliabilityEvent",
    "EventSeverity",
    "create_compatible_event",

    # Original ARF engine
    "OSSMCPClient",
    "OSSMCPResponse",
    "OSSAnalysisResult",
    "create_oss_mcp_client",

    # Engine factory
    "EngineFactory",
    "create_engine",
    "get_engine",
    "get_oss_engine_capabilities",

    # Availability
    "OSS_AVAILABLE",

    # New infrastructure components (prefixed to avoid conflicts)
    "InfraHealingIntent",
    "InfraIntentSource",
    "InfraIntentStatus",
    "AzureInfrastructureSimulator",
    "ProvisionResourceIntent",
    "DeployConfigurationIntent",
    "GrantAccessIntent",
    "ResourceType",
    "Environment",
    "PermissionLevel",
    "Policy",
    "CostEstimator",
    "RiskEngine",
    "RecommendedAction",
    "create_infrastructure_healing_intent",
    "validate_infrastructure_config",
]

# ============================================================================
# CREATE ALIASES FOR ORIGINAL SYMBOLS (to keep top‑level names)
# ============================================================================

HealingIntent = OriginalHealingIntent
IntentSource = OriginalIntentSource
IntentStatus = OriginalIntentStatus

# ============================================================================
# EXPOSE THE INFRASTRUCTURE SUBMODULE (so users can also do `from .infrastructure import ...`)
# ============================================================================

from . import infrastructure

# ============================================================================
# LAZY LOADING FOR HEAVY MODULES (optional)
# ============================================================================

# (You can keep lazy loading for any heavy original components if desired)

# ============================================================================
# IMPORT-TIME VALIDATION (kept from original)
# ============================================================================

def _validate_oss_environment():
    import os
    import sys
    enterprise_vars = ["ARF_TIER", "ARF_DEPLOYMENT_TYPE"]
    for var in enterprise_vars:
        if os.getenv(var) and os.getenv(var).lower() != "oss":
            print(f"⚠️  Warning: Non-OSS environment variable: {var}={os.getenv(var)}")
    if sys.version_info < (3, 10):
        print(f"⚠️  Warning: Python {sys.version} detected, 3.10+ recommended")

try:
    _validate_oss_environment()
except Exception:
    pass

# ============================================================================
# MODULE METADATA
# ============================================================================

def get_oss_info():
    """Get OSS edition information (original style)"""
    return {
        "edition": OSS_EDITION,
        "license": OSS_LICENSE,
        "version": __version__,
        "execution_allowed": EXECUTION_ALLOWED,
        "upgrade_url": "https://arf.dev/enterprise",
        "components": [
            "infrastructure_governance",
            "cost_estimation",
            "risk_scoring",
            "policy_enforcement",
        ],
        "limits": {
            "max_incident_nodes": MAX_INCIDENT_NODES,
            "max_outcome_nodes": MAX_OUTCOME_NODES,
            "advisory_only": True,
        },
    }
