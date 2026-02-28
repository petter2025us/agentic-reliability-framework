"""
Agentic Reliability Framework - OSS Edition v4
Unified platform with core governance, runtime agents, and memory systems.

This package provides:
- Core governance engine (intents, policies, risk scoring)
- MCP client for advisory analysis
- HealingIntent for OSS→Enterprise handoff
- RAG memory for similar incident retrieval
- Multi-agent runtime with anomaly detection, root cause analysis, and forecasting
- Hamiltonian Monte Carlo (HMC) probabilistic risk assessment
"""

__version__ = "4.0.0"

# ============================================================================
# CONSTANTS & OSS BOUNDARIES
# ============================================================================

from .core.config.constants import (
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
    ENTERPRISE_UPGRADE_URL,
)

# ============================================================================
# HEALING INTENT (CORE MODEL)
# ============================================================================

from .core.models.healing_intent import (
    HealingIntent,
    HealingIntentSerializer,
    IntentSource,
    IntentStatus,
    create_rollback_intent,
    create_restart_intent,
    create_scale_out_intent,
    create_oss_advisory_intent,
)

# ============================================================================
# OSS MCP CLIENT (ADVISORY MODE)
# ============================================================================

from .core.mcp.oss_client import (
    OSSMCPClient,
    OSSMCPResponse,
    OSSAnalysisResult,
    create_oss_mcp_client,
)

# ============================================================================
# CORE GOVERNANCE (INFRASTRUCTURE SIMULATION)
# ============================================================================

# Import with aliases to avoid confusion with HealingIntent above
from .core.governance import (
    AzureInfrastructureSimulator as InfraAzureSimulator,
    ProvisionResourceIntent as InfraProvisionIntent,
    DeployConfigurationIntent as InfraDeployIntent,
    GrantAccessIntent as InfraGrantIntent,
    ResourceType as InfraResourceType,
    Environment as InfraEnvironment,
    PermissionLevel as InfraPermissionLevel,
    Policy as InfraPolicy,
    CostEstimator as InfraCostEstimator,
    RiskEngine as InfraRiskEngine,
    HealingIntent as InfraHealingIntent,
    IntentSource as InfraIntentSource,
    IntentStatus as InfraIntentStatus,
    RecommendedAction as InfraRecommendedAction,
    create_infrastructure_healing_intent,
    validate_infrastructure_config,
)

# ============================================================================
# RUNTIME MEMORY (RAG GRAPH, FAISS) - MAY BE PARTIALLY AVAILABLE
# ============================================================================

try:
    from .runtime.memory.rag_graph import RAGGraphMemory
    from .runtime.memory.models import (
        IncidentNode,
        OutcomeNode,
        GraphEdge,
        SimilarityResult,
        NodeType,
        EdgeType,
    )
    _MEMORY_AVAILABLE = True
except ImportError:
    RAGGraphMemory = None
    IncidentNode = OutcomeNode = GraphEdge = SimilarityResult = None
    NodeType = EdgeType = None
    _MEMORY_AVAILABLE = False
    import warnings
    warnings.warn(
        "RAG memory components not fully installed. "
        "Some functionality may be limited.",
        ImportWarning,
        stacklevel=2,
    )

# ============================================================================
# NEW RUNTIME COMPONENTS (v4) - Multi-Agent System, Analytics, HMC
# ============================================================================

try:
    from .runtime.engine import EnhancedReliabilityEngine
    from .runtime.analytics import (
        AdvancedAnomalyDetector,
        SimplePredictiveEngine,
        BusinessImpactCalculator,
    )
    from .runtime.agents import (
        AnomalyDetectionAgent,
        RootCauseAgent,
        PredictiveAgent,
    )
    from .runtime.orchestration import OrchestrationManager
    from .runtime.hmc import HMCRiskLearner
    _RUNTIME_AVAILABLE = True
except ImportError as e:
    EnhancedReliabilityEngine = None
    AdvancedAnomalyDetector = None
    SimplePredictiveEngine = None
    BusinessImpactCalculator = None
    AnomalyDetectionAgent = None
    RootCauseAgent = None
    PredictiveAgent = None
    OrchestrationManager = None
    HMCRiskLearner = None
    _RUNTIME_AVAILABLE = False
    import warnings
    warnings.warn(
        f"Runtime components not fully installed: {e}. "
        "Some functionality may be limited.",
        ImportWarning,
        stacklevel=2,
    )

# ============================================================================
# AVAILABILITY FLAGS
# ============================================================================

OSS_AVAILABLE = True
MEMORY_AVAILABLE = _MEMORY_AVAILABLE
RUNTIME_AVAILABLE = _RUNTIME_AVAILABLE

# ============================================================================
# TOP-LEVEL EXPORTS
# ============================================================================

__all__ = [
    # Version
    "__version__",

    # Constants
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
    "ENTERPRISE_UPGRADE_URL",

    # Core models
    "HealingIntent",
    "HealingIntentSerializer",
    "IntentSource",
    "IntentStatus",
    "create_rollback_intent",
    "create_restart_intent",
    "create_scale_out_intent",
    "create_oss_advisory_intent",

    # MCP client
    "OSSMCPClient",
    "OSSMCPResponse",
    "OSSAnalysisResult",
    "create_oss_mcp_client",

    # Infrastructure governance (prefixed to avoid conflicts)
    "InfraAzureSimulator",
    "InfraProvisionIntent",
    "InfraDeployIntent",
    "InfraGrantIntent",
    "InfraResourceType",
    "InfraEnvironment",
    "InfraPermissionLevel",
    "InfraPolicy",
    "InfraCostEstimator",
    "InfraRiskEngine",
    "InfraHealingIntent",
    "InfraIntentSource",
    "InfraIntentStatus",
    "InfraRecommendedAction",
    "create_infrastructure_healing_intent",
    "validate_infrastructure_config",

    # Memory components
    "RAGGraphMemory",
    "IncidentNode",
    "OutcomeNode",
    "GraphEdge",
    "SimilarityResult",
    "NodeType",
    "EdgeType",
    "MEMORY_AVAILABLE",

    # Runtime components
    "EnhancedReliabilityEngine",
    "AdvancedAnomalyDetector",
    "SimplePredictiveEngine",
    "BusinessImpactCalculator",
    "AnomalyDetectionAgent",
    "RootCauseAgent",
    "PredictiveAgent",
    "OrchestrationManager",
    "HMCRiskLearner",
    "RUNTIME_AVAILABLE",

    # Availability
    "OSS_AVAILABLE",
]

# ============================================================================
# IMPORT-TIME VALIDATION
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
    """Get OSS edition information"""
    components = ["core_governance", "mcp_advisory"]
    if MEMORY_AVAILABLE:
        components.append("rag_memory")
    else:
        components.append("rag_memory (optional)")
    if RUNTIME_AVAILABLE:
        components.append("multi_agent_system")
        components.append("hmc_probabilistic_risk")
    else:
        components.append("runtime (optional)")

    return {
        "edition": OSS_EDITION,
        "license": OSS_LICENSE,
        "version": __version__,
        "execution_allowed": EXECUTION_ALLOWED,
        "upgrade_url": ENTERPRISE_UPGRADE_URL,
        "components": components,
        "limits": {
            "max_incident_nodes": MAX_INCIDENT_NODES,
            "max_outcome_nodes": MAX_OUTCOME_NODES,
            "advisory_only": True,
        },
        "runtime_available": RUNTIME_AVAILABLE,
    }
