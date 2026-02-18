# agentic_reliability_framework/infrastructure/intents.py
"""
Infrastructure Intent Schema for the Agentic Reliability Framework (OSS).

This module defines the structured representations of infrastructure change requests
that ARF can evaluate. All intents are immutable Pydantic models, designed to be
deterministic and serializable. They serve as the input to the advisory engine.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field


class ResourceType(str, Enum):
    """Azure resource types that can be provisioned."""
    VM = "vm"
    STORAGE_ACCOUNT = "storage_account"
    DATABASE = "database"
    KUBERNETES_CLUSTER = "kubernetes_cluster"
    FUNCTION_APP = "function_app"
    VIRTUAL_NETWORK = "virtual_network"


class ProvisionResourceIntent(BaseModel):
    """
    Intent to provision a new cloud resource.

    This intent represents a request to create a resource of a specified type,
    with given size and configuration, in a particular region/environment.
    """
    intent_type: str = "provision_resource"
    resource_type: ResourceType
    region: str
    size: str = Field(description="e.g., 'Standard_D2s_v3', '50GB', etc.")
    configuration: Dict[str, Any] = Field(default_factory=dict)
    requester: str
    environment: str = Field(description="e.g., 'dev', 'staging', 'prod'")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DeployConfigurationIntent(BaseModel):
    """
    Intent to deploy a configuration change to an existing service.

    This includes rolling out new settings, feature flags, or application configs.
    """
    intent_type: str = "deploy_config"
    service_name: str
    change_scope: str = Field(description="e.g., 'single_instance', 'canary', 'global'")
    deployment_target: str = Field(description="e.g., 'production', 'staging'")
    risk_level_hint: Optional[float] = Field(
        None, ge=0, le=1,
        description="Optional agent‑estimated risk (0 = safe, 1 = dangerous)"
    )
    configuration: Dict[str, Any] = Field(default_factory=dict)
    requester: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class GrantAccessIntent(BaseModel):
    """
    Intent to grant a permission to a principal on a resource scope.

    This models requests to modify IAM policies.
    """
    intent_type: str = "grant_access"
    principal: str = Field(description="User or service principal identifier")
    permission_level: str = Field(description="e.g., 'read', 'write', 'admin'")
    resource_scope: str = Field(description="Azure resource scope (e.g., subscription, resource group, resource ID)")
    justification: Optional[str] = None
    requester: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Union type for all supported intents – used by the simulator.
InfrastructureIntent = Union[
    ProvisionResourceIntent,
    DeployConfigurationIntent,
    GrantAccessIntent,
]

__all__ = [
    "ResourceType",
    "ProvisionResourceIntent",
    "DeployConfigurationIntent",
    "GrantAccessIntent",
    "InfrastructureIntent",
]
