# agentic_reliability_framework/__init__.py
"""
Agentic Reliability Framework - OSS Edition
Infrastructure Governance Module for Azure
Apache 2.0 Licensed - Enterprise features require commercial license
"""

__version__ = "1.0.0"
__all__ = [
    # Core constants
    "OSS_EDITION",
    "OSS_LICENSE",
    "EXECUTION_ALLOWED",
    "ENTERPRISE_UPGRADE_URL",
    
    # Infrastructure components
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
    "HealingIntent",
    "IntentSource",
    "IntentStatus",
    "RecommendedAction",
    
    # Factory functions
    "create_infrastructure_healing_intent",
    "validate_infrastructure_config",
]

# ============================================================================
# DIRECT IMPORTS - CORE COMPONENTS
# ============================================================================

# Import constants first
from .constants import (
    OSS_EDITION,
    OSS_LICENSE,
    EXECUTION_ALLOWED,
    ENTERPRISE_UPGRADE_URL,
)

# Infrastructure components
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
    HealingIntent,
    IntentSource,
    IntentStatus,
    RecommendedAction,
    create_infrastructure_healing_intent,
)

# ============================================================================
# LAZY LOADING FOR HEALING INTEGRATION
# ============================================================================

_validate_infrastructure_config = None

def _get_validate_infrastructure_config():
    """Lazy load validation function"""
    global _validate_infrastructure_config
    if _validate_infrastructure_config is not None:
        return _validate_infrastructure_config
    
    try:
        from .infrastructure import validate_infrastructure_config
        _validate_infrastructure_config = validate_infrastructure_config
        return _validate_infrastructure_config
    except ImportError:
        # Fallback validation
        def _fallback_validate(config):
            return {"valid": True, "warnings": ["Using fallback validation"]}
        _validate_infrastructure_config = _fallback_validate
        return _validate_infrastructure_config

def __getattr__(name):
    """Lazy loading for validation function"""
    if name == "validate_infrastructure_config":
        return _get_validate_infrastructure_config()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# ============================================================================
# IMPORT-TIME VALIDATION
# ============================================================================

def _validate_oss_environment():
    """Validate OSS environment at import time"""
    import os
    import sys
    
    # Check for enterprise environment variables
    enterprise_vars = ["ARF_TIER", "ARF_DEPLOYMENT_TYPE"]
    for var in enterprise_vars:
        if os.getenv(var) and os.getenv(var).lower() != "oss":
            print(f"⚠️  Warning: Non-OSS environment variable: {var}={os.getenv(var)}")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print(f"⚠️  Warning: Python {sys.version} detected, 3.10+ recommended")

# Run validation silently
try:
    _validate_oss_environment()
except Exception:
    pass

# ============================================================================
# MODULE METADATA
# ============================================================================

def get_oss_info():
    """Get OSS edition information"""
    return {
        "edition": OSS_EDITION,
        "license": OSS_LICENSE,
        "version": __version__,
        "execution_allowed": EXECUTION_ALLOWED,
        "upgrade_url": ENTERPRISE_UPGRADE_URL,
        "components": [
            "infrastructure_governance",
            "cost_estimation",
            "risk_scoring",
            "policy_enforcement",
        ],
        "limits": {
            "max_incident_nodes": 1000,
            "max_outcome_nodes": 5000,
            "advisory_only": True,
        },
    }
