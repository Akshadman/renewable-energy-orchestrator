"""Common record for anything the agent learns from unstructured / multimodal input."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

APPLY_THRESHOLD = 0.75   # below this confidence a finding is quarantined (shown, not acted on)


@dataclass
class Evidence:
    """One finding extracted from an image, a document or a satellite scene."""

    modality: str                 # satellite | thermal_image | document
    source: str                   # file name / scene id
    kind: str                     # cloud_bank | panel_defects | maintenance | breakdown | line_limit | demand_change | none
    summary: str                  # one line for humans
    confidence: float             # 0..1
    asset: Optional[str] = None   # asset / consumer id it concerns
    start_tick: Optional[int] = None
    duration_ticks: Optional[int] = None
    value: Optional[float] = None  # severity / factor / derate (meaning depends on kind)
    details: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"       # applied | quarantined | info | rejected
    reason: str = ""              # why quarantined / rejected

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
