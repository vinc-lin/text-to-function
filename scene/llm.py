"""The constrained fallback: it decides, it does not act, and it does not write sentences.

Three properties come from the schema rather than from a check that could be forgotten. The
decision vocabulary contains no `execute`. `scene` and `reply_intent` are enums, so the model
selects from what exists. And `no_action` is a legal answer — a constrained decoder always
emits something, which is exactly why REJECT_NAME exists on the tool-call path
(t2f/llm/schema.py:34-41); without a legal way to decline, a model declines by picking
something, which is the mechanism behind the 99°→16° substitution.
"""
from __future__ import annotations
import json
from typing import Optional

UNMATCHED = "unmatched"


def _branch(decision: str, scenes: list, intents: list) -> dict:
    props = {
        "decision": {"const": decision},
        "scene": {"enum": scenes},
        # Bounded in the grammar rather than truncated afterwards: `reason` shares a token
        # budget with the fields that decide the outcome, and a string cut off mid-generation
        # makes the whole object unparseable — so an over-long reason discards the decision.
        "reason": {"type": "string", "maxLength": 80},
    }
    required = ["decision", "scene", "reason"]
    if intents:
        props["reply_intent"] = {"enum": intents}
        required.append("reply_intent")
    return {"type": "object", "properties": props, "required": required,
            "additionalProperties": False}


def scene_decision_schema(rules, speech: dict) -> dict:
    """One branch per decision, each carrying only the scenes and intents that fit it.

    A flat schema let the model return `decision: notify` with an `ask_*` intent — a question
    spoken into the cabin with no pending consent behind it, which the driver can answer into
    the void. That is the same defect the ask branch guards against, arriving through the
    notify door. Keyed branches make it UNGRAMMATICAL rather than merely checked, which is how
    t2f/llm/schema.py already constrains tool calls.

    The branches also encode two things that were previously code: only a rule carrying a
    `proposes` may be asked about, and `no_action` needs no intent because it says nothing.
    """
    askable = sorted(r.id for r in rules if r.proposes is not None)
    notifiable = sorted([r.id for r in rules if r.proposes is None] + [UNMATCHED])
    ask_intents = sorted(k for k in speech if k.startswith("ask_"))
    notify_intents = sorted(k for k in speech if k.startswith("notify_"))
    every_scene = sorted([r.id for r in rules] + [UNMATCHED])

    branches = []
    # An empty enum is not valid JSON schema, so a rule set with nothing askable drops the
    # branch entirely rather than emitting one the decoder cannot satisfy.
    if askable and ask_intents:
        branches.append(_branch("ask", askable, ask_intents))
    if notify_intents:
        branches.append(_branch("notify", notifiable, notify_intents))
    branches.append(_branch("no_action", every_scene, []))
    return {"oneOf": branches}


def build_scene_prompt(snapshot: dict, rules, speech: dict) -> list[dict]:
    """Only what is live right now, never accumulated history.

    'Do not continuously append raw vision text to the LLM prompt' is enforced structurally:
    this function receives a snapshot, so there is nowhere for history to accumulate.
    """
    known = "\n".join(f"- {r.id}: {r.description}" for r in rules)
    return [
        {"role": "system",
         "content": "你是车内场景助手。只能返回 JSON 决策，不能执行任何车辆功能。"
                    f"已知场景：\n{known}\n无法归类时 scene 填 \"{UNMATCHED}\"。"
                    "不确定时选择 no_action。"},
        {"role": "user", "content": json.dumps(snapshot, ensure_ascii=False)},
    ]


class FakeSceneLLM:
    """Scripted decisions, and a call counter so budget tests can assert on it."""

    def __init__(self, script: list):
        self._script = list(script)
        self.calls = 0

    def decide(self, snapshot, rules, speech) -> Optional[dict]:
        self.calls += 1
        return self._script.pop(0) if self._script else None


class TransformersSceneLLM:
    """Qwen3-0.6B under an xgrammar constraint, mirroring TransformersXGrammarClient."""

    def __init__(self, model_id: str = "Qwen/Qwen3-0.6B", max_new_tokens: int = 96,
                 device: str | None = None):
        import sys as _sys
        _sys.modules.setdefault("torchvision", None)
        import torch
        import xgrammar as xgr
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._torch, self._xgr = torch, xgr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        # ADJUST if API differs — same caveat as t2f/llm/client.py:62. xgrammar's HF
        # integration has shifted names across versions, and there are now two copies of this
        # line; if one needs changing, so does the other.
        info = xgr.TokenizerInfo.from_huggingface(self.tok, vocab_size=self.model.config.vocab_size)
        self.compiler = xgr.GrammarCompiler(info)

    def decide(self, snapshot, rules, speech) -> Optional[dict]:
        torch = self._torch
        schema = scene_decision_schema(rules, speech)
        messages = build_scene_prompt(snapshot, rules, speech)
        prompt = self.tok.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        compiled = self.compiler.compile_json_schema(json.dumps(schema))
        processor = self._xgr.contrib.hf.LogitsProcessor(compiled)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, logits_processor=[processor],
                                      pad_token_id=self.tok.eos_token_id)
        raw = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            return json.loads(raw.strip())
        except Exception:
            return None      # unparseable output is silence, not a crash
