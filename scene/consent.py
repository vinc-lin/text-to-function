"""Was that a yes?

The engine asks a question the driver never invited, so whatever they say next may or may not
be an answer. Getting it wrong actuates the car on consent that was never given — the
proactive form of the OOD false-execution this project reports as 0.000.

So consent has exactly one shape: EXACT membership in a closed set, on the normalised
utterance. 好 is a yes; 好像有点热 is not, and a substring test would make it one. Anything
outside both sets drops the pending question and is routed as an ordinary command, which
loses an oblique yes like 开吧 — a cost accepted deliberately, and measured by
`scene_recall`.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from t2f.normalize import normalize
from t2f.types import ToolCall

_AFFIRM = frozenset({"好", "好的", "好吧", "可以", "行", "嗯", "嗯嗯",
                     "是", "是的", "对", "没问题", "麻烦你了"})
_DECLINE = frozenset({"不用", "不要", "不必", "算了", "不了", "没事", "不需要"})

# normalize() folds 。 to . and ！ to !, so terminators are stripped in their ASCII form.
_STRIP = " .,!?;:、"


class Answer(str, Enum):
    YES = "yes"
    NO = "no"
    NOT_AN_ANSWER = "not_an_answer"


@dataclass
class PendingConsent:
    scene: str
    proposal: ToolCall
    asked_at: float
    expires_after: float

    def is_live(self, now: float) -> bool:
        return now <= self.asked_at + self.expires_after


def classify(utterance: str) -> Answer:
    text = normalize(utterance or "").strip(_STRIP)
    if text in _AFFIRM:
        return Answer.YES
    if text in _DECLINE:
        return Answer.NO
    return Answer.NOT_AN_ANSWER
