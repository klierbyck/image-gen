from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class CapabilityProfile:
    default_aspect_ratio: str
    default_image_size: str
    default_exact_size: Optional[str]
    supported_formats: tuple[str, ...]
    supported_qualities: tuple[str, ...]
    max_input_image_bytes: int
    default_output_format: str = "png"
    default_quality: str = "auto"

    def as_dict(self) -> dict[str, Any]:
        return {
            "default_aspect_ratio": self.default_aspect_ratio,
            "default_image_size": self.default_image_size,
            "default_exact_size": self.default_exact_size,
            "default_output_format": self.default_output_format,
            "default_quality": self.default_quality,
            "supported_formats": list(self.supported_formats),
            "supported_qualities": list(self.supported_qualities),
            "max_input_image_bytes": self.max_input_image_bytes,
        }


@dataclass(frozen=True)
class ProviderResponse:
    image: bytes
    mime_type: str
    endpoint: str
    http_status: int
    request_id: Optional[str] = None
    usage: Optional[dict[str, Any]] = None
