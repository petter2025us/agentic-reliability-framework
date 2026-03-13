from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ReliabilityEvent:
    id: str
    timestamp: datetime
    service_name: str
    event_type: str
    severity: Severity
    metrics: Dict[str, float]          # e.g., latency, error_rate, cpu
    metadata: Dict[str, Any] = field(default_factory=dict)
