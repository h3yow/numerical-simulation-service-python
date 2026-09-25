"""Distinguishable error / rejection reasons for the simulation service."""

from enum import Enum


class ErrorReason(str, Enum):
    """Machine-distinguishable failure reasons."""

    INVALID_PARAMETERS = "invalid_parameters"
    NUMERICALLY_UNSTABLE = "numerically_unstable"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"
