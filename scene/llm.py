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


def scene_decision_schema(rules, speech: dict) -> dict:
    return {
        "type": "object",
        "properties": {
            "decision": {"enum": ["notify", "ask", "no_action"]},
            "scene": {"enum": [r.id for r in rules] + [UNMATCHED]},
            "reason": {"type": "string"},
            "reply_intent": {"enum": sorted(speech)},
        },
        "required": ["decision", "scene", "reason", "reply_intent"],
        "additionalProperties": False,
    }


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
