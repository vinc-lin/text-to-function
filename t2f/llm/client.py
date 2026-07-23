from __future__ import annotations
import json
from abc import ABC, abstractmethod
from ..types import FunctionCard, ToolCall, LLMResult
from .prompt import build_prompt
from .schema import candidates_to_json_schema, REJECT_NAME


class LLMClient(ABC):
    @abstractmethod
    def complete_tool_call(self, clause: str, candidate_cards: list[FunctionCard],
                           extracted_params: dict) -> LLMResult: ...


class FakeLLMClient(LLMClient):
    """Deterministic client for tests: maps a clause substring -> a scripted LLMResult."""

    def __init__(self, scripts: dict[str, LLMResult] | None = None, default: LLMResult | None = None):
        self.scripts = scripts or {}
        self.default = default

    def complete_tool_call(self, clause, candidate_cards, extracted_params) -> LLMResult:
        for key, res in self.scripts.items():
            if key in clause:
                return res
        return self.default or LLMResult(error="no_script_match")


def _parse_tool_call(raw: str) -> LLMResult:
    try:
        obj = json.loads(raw)
    except Exception as e:  # pragma: no cover - defensive; grammar should prevent this
        return LLMResult(raw=raw, error=f"json_parse:{e}")
    if not isinstance(obj, dict) or "name" not in obj:
        return LLMResult(raw=raw, error="missing_name")
    if obj["name"] == REJECT_NAME:
        return LLMResult(clarification=REJECT_NAME, raw=raw)
    return LLMResult(tool_call=ToolCall(name=obj["name"], parameters=obj.get("parameters", {})), raw=raw)


class TransformersXGrammarClient(LLMClient):
    """Qwen3-0.6B via transformers, output constrained to the candidate JSON schema via xgrammar.

    NOTE: xgrammar's HF integration API name has shifted across versions. The pattern below targets
    xgrammar's `GrammarCompiler` + `contrib.hf.LogitsProcessor`. If the installed version differs,
    adjust the two marked lines to that version's equivalent (verified by the model-marked test).
    """

    def __init__(self, model_id: str = "Qwen/Qwen3-0.6B", max_new_tokens: int = 128, device: str | None = None):
        import sys as _sys
        _sys.modules.setdefault("torchvision", None)
        import torch
        import xgrammar as xgr
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._torch = torch
        self._xgr = xgr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        tok_info = xgr.TokenizerInfo.from_huggingface(self.tok, vocab_size=self.model.config.vocab_size)  # ADJUST if API differs
        self.compiler = xgr.GrammarCompiler(tok_info)

    def complete_tool_call(self, clause, candidate_cards, extracted_params) -> LLMResult:
        torch = self._torch
        schema = candidates_to_json_schema(candidate_cards, allow_reject=True)
        messages = build_prompt(clause, candidate_cards, extracted_params, allow_reject=True)
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        compiled = self.compiler.compile_json_schema(json.dumps(schema))
        processor = self._xgr.contrib.hf.LogitsProcessor(compiled)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                       do_sample=False, logits_processor=[processor],
                                       pad_token_id=self.tok.eos_token_id)
        raw = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return _parse_tool_call(raw)


class GgufLLMClient(LLMClient):  # Spec 3
    def __init__(self, *a, **k):
        raise NotImplementedError("GGUF/llama.cpp GBNF client is Spec 3")

    def complete_tool_call(self, *a, **k):
        raise NotImplementedError
