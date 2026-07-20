from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


@dataclass
class ParamSpec:
    name: str
    type: str  # "number" | "integer" | "string" | "boolean" | "enum"
    required: bool = False
    enum: Optional[list[str]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    unit: Optional[str] = None  # "celsius" | "percent" | "level" | ...
    description: str = ""


@dataclass
class FunctionCard:
    name: str
    domain: str
    description: str
    params: list[ParamSpec] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    utterances: list[str] = field(default_factory=list)
    hard_negatives: list[str] = field(default_factory=list)
    response_template: str = ""

    def param(self, name: str) -> Optional[ParamSpec]:
        return next((p for p in self.params if p.name == name), None)

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self.params]

    @property
    def required_params(self) -> list[str]:
        return [p.name for p in self.params if p.required]


@dataclass
class Candidate:
    function: str
    score: float
    embedding_score: float = 0.0
    signal_scores: dict[str, float] = field(default_factory=dict)
    best_prototype: str = ""


class Band(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Decision:
    band: Band
    chosen: Optional[str]
    candidates: list[Candidate]
    ood_score: float = 0.0
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class ToolCall:
    name: str
    parameters: dict[str, Any]


@dataclass
class ValidationError:
    code: str
    message: str


@dataclass
class PendingState:
    pending_function: str
    known_parameters: dict[str, Any]
    missing_parameters: list[str]


@dataclass
class ClarificationRequest:
    question: str
    pending: Optional[PendingState] = None


@dataclass
class LexFeatures:
    numbers: list[float] = field(default_factory=list)
    temperatures: list[float] = field(default_factory=list)
    percentages: list[float] = field(default_factory=list)
    levels: list[int] = field(default_factory=list)
    positions: list[str] = field(default_factory=list)      # normalized: driver/passenger/rear/left/right/all
    directions: list[str] = field(default_factory=list)      # up/down/left/right/front/back
    on_off: Optional[bool] = None
    operation: Optional[str] = None                          # "increase" | "decrease" | "max" | "min"
    raw: str = ""


@dataclass
class ClauseResult:
    clause: str
    decision: Decision
    tool_call: Optional[ToolCall] = None
    validation_errors: list[ValidationError] = field(default_factory=list)
    clarification: Optional[ClarificationRequest] = None
    response: Optional[str] = None
    needs_llm: bool = False
    latency_ms: float = 0.0


@dataclass
class RouteResult:
    utterance: str
    clauses: list[ClauseResult] = field(default_factory=list)
