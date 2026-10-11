"""Provider-neutral LLM client with tool use.

Picks the first configured provider from the environment:
  ANTHROPIC_API_KEY  → Claude (Anthropic Messages API)
  GROQ_API_KEY       → Groq (OpenAI-compatible, free tier)       default model llama-3.3-70b-versatile
  GEMINI_API_KEY     → Google Gemini (OpenAI-compatible endpoint) default model gemini-3.8-flash
  OPENROUTER_API_KEY → OpenRouter (OpenAI-compatible)
  LLM_BASE_URL       → any OpenAI-compatible server, e.g. Ollama at http://localhost:11434/v1
LLM_MODEL overrides the default model; LLM_PROVIDER forces a provider.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

import requests

OPENAI_COMPAT = {
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "gemini": ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.8-flash"),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct:free"),
}
DEFAULT_CLAUDE = "claude-sonnet-5"
RETRY_STATUS = {429, 500, 502, 503, 504}
RETRY_WAITS_S = (2.0, 5.0)              # two retries, then the optimizer + ladder decide this tick

Handler = Callable[[str, Dict[str, Any]], Dict[str, Any]]


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Run a bounded tool-use conversation; returns the input of ``final_tool`` when the model calls it."""

    def __init__(self, provider: str, model: str, api_key: str = "", base_url: str = "", timeout: float = 60) -> None:
        self.provider, self.model, self.api_key, self.base_url, self.timeout = provider, model, api_key, base_url, timeout
        self._anthropic = None
        if provider == "anthropic":
            import anthropic
            self._anthropic = anthropic.Anthropic(api_key=api_key)

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    # ------------------------------------------------------------ public API
    def run_tools(self, system: str, user: str, tools: List[Dict[str, Any]], handler: Handler, final_tool: str,
                  max_turns: int = 4, force_first: Optional[str] = None) -> Dict[str, Any]:
        """Let the model call ``tools``; ``handler`` executes non-final tools. Returns final tool input."""
        if self.provider == "anthropic":
            return self._run_anthropic(system, user, tools, handler, final_tool, max_turns, force_first)
        return self._run_openai(system, user, tools, handler, final_tool, max_turns, force_first)

    def call_tool(self, system: str, user: str, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Single forced tool call (structured extraction)."""
        return self.run_tools(system, user, [tool], lambda n, a: {}, tool["name"], 1, force_first=tool["name"])

    # ------------------------------------------------------------ Anthropic
    def _run_anthropic(self, system, user, tools, handler, final_tool, max_turns, force_first):
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user}]
        for turn in range(max_turns):
            last = turn == max_turns - 1
            choice = {"type": "tool", "name": final_tool} if last else \
                ({"type": "tool", "name": force_first} if force_first and turn == 0 else {"type": "any"})
            resp = self._anthropic.messages.create(model=self.model, max_tokens=900, system=system, tools=tools,
                                                   tool_choice=choice, messages=messages)
            calls = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
            if not calls:
                raise LLMError("model returned no tool call")
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for c in calls:
                if c.name == final_tool:
                    return dict(c.input)
                results.append({"type": "tool_result", "tool_use_id": c.id, "content": json.dumps(handler(c.name, dict(c.input)))})
            messages.append({"role": "user", "content": results})
        raise LLMError("no final decision within turn limit")

    # ------------------------------------------------------------ OpenAI-compatible (Groq, Gemini, OpenRouter, Ollama)
    def _post(self, url: str, body: Dict[str, Any], headers: Dict[str, str]) -> Any:
        """POST with short retries on transient overload (free tiers often answer 429 / 503 under load)."""
        for wait in RETRY_WAITS_S + (None,):
            try:
                r = requests.post(url, json=body, headers=headers, timeout=self.timeout)
            except requests.ConnectionError:       # dropped connection under load (not timeouts: those already
                if wait is None:                   # cost a full timeout each)
                    raise
                time.sleep(wait)
                continue
            if r.status_code not in RETRY_STATUS or wait is None:
                return r
            try:                                   # honour a short Retry-After, never block a tick for long
                wait = min(float(r.headers.get("Retry-After", wait)), 10.0)
            except (TypeError, ValueError):
                pass
            time.sleep(wait)
        return r

    def _run_openai(self, system, user, tools, handler, final_tool, max_turns, force_first):
        fns = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                 "parameters": t["input_schema"]}} for t in tools]
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        turn, nudged = 0, False
        while turn < max_turns:
            last = turn == max_turns - 1
            name = final_tool if last else (force_first if force_first and turn == 0 else None)
            choice = {"type": "function", "function": {"name": name}} if name else "required"
            # Room for hidden reasoning on "thinking" models (e.g. Gemini 3.x), or the tool call gets cut off.
            body = {"model": self.model, "messages": messages, "tools": fns, "tool_choice": choice, "temperature": 0.2,
                    "max_tokens": 4096}
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            r = self._post(f"{self.base_url.rstrip('/')}/chat/completions", body, headers)
            if r.status_code >= 400:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
            msg = r.json()["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if not calls:
                if nudged:
                    raise LLMError("model returned no tool call")
                nudged = True                      # some models answer in prose despite tool_choice: ask once more
                messages += [{"role": "assistant", "content": msg.get("content") or "(no tool call)"},
                             {"role": "user", "content": f"Reply only with a tool call (finish with {final_tool})."}]
                continue
            turn += 1
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for c in calls:
                fname = c["function"]["name"]
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError as exc:
                    raise LLMError(f"bad tool arguments: {exc}") from exc
                if fname == final_tool:
                    return args
                messages.append({"role": "tool", "tool_call_id": c.get("id", fname), "name": fname,
                                 "content": json.dumps(handler(fname, args))})
        raise LLMError("no final decision within turn limit")


def from_env() -> Optional[LLMClient]:
    """The first configured provider, or None (offline mode)."""
    forced = os.environ.get("LLM_PROVIDER", "").strip().lower()
    model = os.environ.get("LLM_MODEL", "").strip()
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key and forced in ("", "anthropic", "claude"):
        try:
            return LLMClient("anthropic", model or DEFAULT_CLAUDE, key)
        except Exception:  # SDK missing
            pass
    for prov, (env, base, default) in OPENAI_COMPAT.items():
        k = os.environ.get(env, "").strip()
        if k and forced in ("", prov):
            return LLMClient(prov, model or default, k, base)
    base = os.environ.get("LLM_BASE_URL", "").strip()
    if base:
        return LLMClient(forced or "openai-compatible", model or "llama3.1", os.environ.get("LLM_API_KEY", ""), base)
    return None
