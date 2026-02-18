"""
Infrastructure Intent Schema for the Agentic Reliability Framework (OSS).

This module defines the structured representations of infrastructure change requests.
All intents are immutable Pydantic models with strict validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field, field_validator


class ResourceType(str, Enum):
    """Azure resource types that can be provisioned."""
    VM = "vm"
    STORAGE_ACCOUNT = "storage_account"
    DATABASE = "database"
    KUBERNETES_CLUSTER = "kubernetes_cluster"
    FUNCTION_APP = "function_app"
    VIRTUAL_NETWORK = "virtual_network"


class Environment(str, Enum):
    """Deployment environment."""
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"
    TEST = "test"


class PermissionLevel(str, Enum):
    """Access permission levels."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class ProvisionResourceIntent(BaseModel):
    """Request to provision a new cloud resource."""
    intent_type: str = "provision_resource"
    resource_type: ResourceType
    region: str = Field(..., examples=["eastus", "westeurope"])
    size: str = Field(..., description="e.g., 'Standard_D2s_v3', '50GB'")
    configuration: Dict[str, Any] = Field(default_factory=dict)
    requester: str
    environment: Environment
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator('region')
    def validate_region(cls, v):
        # Simple validation – can be extended with a list of valid Azure regions
        if not v or not isinstance(v, str):
            raise ValueError('region must be a non‑empty string')
        # Optionally check pattern: lower case letters and hyphens
        return v


class DeployConfigurationIntent(BaseModel):
    """Request to deploy a configuration change to an existing service."""
    intent_type: str = "deploy_config"
    service_name: str
    change_scope: str = Field(..., description="e.g., 'single_instance', 'canary', 'global'")
    deployment_target: Environment
    risk_level_hint: Optional[float] = Field(
        None, ge=0, le=1,
        description="Optional agent‑estimated risk (0 = safe, 1 = dangerous)"
    )
    configuration: Dict[str, Any] = Field(default_factory=dict)
    requester: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class GrantAccessIntent(BaseModel):
    """Request to grant a permission to a principal on a resource scope."""
    intent_type: str = "grant_access"
    principal: str = Field(..., description="User or service principal identifier")
    permission_level: PermissionLevel
    resource_scope: str = Field(..., description="Azure resource scope (e.g., subscription, resource group)")
    justification: Optional[str] = None
    requester: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Union type for all supported intents
InfrastructureIntent = Union[
    ProvisionResourceIntent,
    DeployConfigurationIntent,
    GrantAccessIntent,
]

__all__ = [
    "ResourceType",
    "Environment",
    "PermissionLevel",
    "ProvisionResourceIntent",
    "DeployConfigurationIntent",
    "GrantAccessIntent",
    "InfrastructureIntent",
]
