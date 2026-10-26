from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from openai import OpenAI


SYSTEM_PROMPT = '''You are a single-file coding agent. You complete the user's task by writing ONE self-contained runnable file, running it, and fixing it until it works.

HARD RULES
- Use ONLY the functions/APIs shown in REFERENCE. Never invent names, libraries, arguments, or behavior. Never assume. If a required API is missing and asking is allowed, ask; otherwise solve using only what you have.
- The deliverable is exactly one file: <OUTPUT>. Temporary test files are allowed but must be deleted before finishing.
- Do not hardcode expected outputs. The file must compute its result at runtime (read/parse real inputs, call the given APIs).
- Before finishing you MUST run the deliverable and see exit=0. If it fails, rewrite and run again. Never finish on an error.
- Reply with ACTION blocks only, no prose or explanation. You may put several ACTION blocks in one reply; they run in order.

ACTION FORMAT
Start every reply with an ACTION line.

Write a file (content MUST be inside a fenced code block):
ACTION: write_file
PATH: <path>
```python
<full file content>
```

Run one shell command:
ACTION: run_shell
```bash
<command>
```

Read a file:
ACTION: read_file
PATH: <path>

List workspace files:
ACTION: list_files

<ASK>
Finish only after the file exists and runs correctly:
ACTION: finish
FILE: <path>
SUMMARY: <one line>

All paths are relative to the workspace: <WORKSPACE>.

REFERENCE (the ONLY APIs you may use; never invent others):
<REFERENCE>'''


ASK_ENABLED = '''Ask the user for a missing API (use only when truly unavailable):
ACTION: ask_user
QUESTION: <question>
'''

ASK_DISABLED = '''Asking is NOT allowed. Use only the APIs in REFERENCE and the standard library. Never invent APIs.'''


class ModelError(RuntimeError):
    pass


@lru_cache(maxsize=8)
def _client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)


class LlamaCppChatModel(BaseChatModel):
    base_url: str = "http://127.0.0.1:8081/v1"
    model: str = "local"
    api_key: str = "sk-no-key"
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout: float = 180.0

    @property
    def _llm_type(self) -> str:
        return "llamacpp-openai"

    def _to_oai(self, m: BaseMessage) -> dict:
        if isinstance(m, SystemMessage):
            role = "system"
        elif isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        else:
            role = "user"
        content = m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
        return {"role": role, "content": content}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        payload = [self._to_oai(m) for m in messages]
        client = _client(self.base_url, self.api_key, self.timeout)
        last = "unknown"
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=payload,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop=list(stop) if stop else None,
                )
                text = (resp.choices[0].message.content or "").strip()
                if text:
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
                last = "empty response"
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
        raise ModelError(f"model call failed: {last}")


ACTION_RE = re.compile(r"(?im)^[ \t]*(?:[#>*`\-]+[ \t]*)*ACTION[ \t]*[\*`]*[ \t]*[:=：][ \t]*[\*`]*[ \t]*([a-z_]+)")
FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")

WRITE_ACTIONS = {"write_file", "write", "save_file", "create_file", "render_file"}
RUN_ACTIONS = {"run_shell", "shell", "run", "bash", "exec", "execute", "run_command"}
READ_ACTIONS = {"read_file", "cat", "open"}
LIST_ACTIONS = {"list_files", "ls", "list", "list_dir"}
ASK_ACTIONS = {"ask_user", "ask"}
FINISH_ACTIONS = {"finish", "done", "complete", "final"}


def _field(text: str, name: str) -> Optional[str]:
    token = re.escape(name)
    m = re.search(rf"(?im)^[ \t]*\**{token}\**[ \t]*[:=：][ \t]*[\*`]*[ \t]*(.+?)[ \t]*[\*`]*[ \t]*$", text)
    if not m:
        return None
    value = m.group(1).strip().strip("`").strip().strip("*").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value


def _first_fence(text: str) -> Optional[str]:
    m = FENCE_RE.search(text)
    return m.group(1).rstrip("\n") if m else None


def _after_marker(text: str, marker: str) -> Optional[str]:
    token = re.escape(marker)
    m = re.search(rf"(?im)^[ \t]*\**{token}\**[ \t]*[:=：][ \t]*\n?(.*)", text, re.DOTALL)
    content = m.group(1) if m else None
    return content.rstrip("\n") if content and content.strip() else None


def _drop_field_line(text: str, *names: str) -> str:
    out = text
    for name in names:
        token = re.escape(name)
        out = re.sub(rf"(?im)^[ \t]*\**{token}\**[ \t]*[:=：].*?$", "", out, count=1)
    return out.strip()


def _balanced_objects(text: str) -> list[str]:
    out: list[str] = []
    depth = 0
    start = None
    in_str = False
    escaped = False
    for i, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            if in_str:
                escaped = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                out.append(text[start : i + 1])
    return out


def _parse_json_action(text: str) -> Optional[dict]:
    candidates: list[str] = []
    m = JSON_FENCE_RE.search(text)
    if m:
        candidates.append(m.group(1))
    candidates.extend(_balanced_objects(text))
    for raw in candidates:
        for cand in (raw, TRAILING_COMMA_RE.sub(r"\1", raw)):
            try:
                obj = json.loads(cand)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            action = obj.get("action") or obj.get("name")
            args = obj.get("args") or obj.get("arguments") or {}
            if isinstance(action, str):
                return {"action": action.strip().lower(), "args": args if isinstance(args, dict) else {}}
    return None


def parse_actions(text: str) -> list[dict]:
    if not text:
        return []
    matches = list(ACTION_RE.finditer(text))
    if not matches:
        one = _parse_json_action(text)
        return [one] if one else []
    out: list[dict] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        one = parse_action(text[start:end])
        if one:
            out.append(one)
    return out


def parse_action(text: str) -> Optional[dict]:
    if not text:
        return None
    m = ACTION_RE.search(text)
    if not m:
        return _parse_json_action(text)
    name = m.group(1).lower()
    rest = text[m.end() :]
    if name in WRITE_ACTIONS:
        content = _first_fence(rest)
        if content is None:
            content = _after_marker(rest, "CONTENT")
        if content is None:
            content = _drop_field_line(rest, "PATH", "FILE")
        path = _field(text, "PATH") or _field(text, "FILE") or ""
        return {"action": "write_file", "args": {"path": path, "content": content or ""}}
    if name in RUN_ACTIONS:
        cmd = _first_fence(rest) or _field(text, "COMMAND")
        if not cmd:
            cmd = _drop_field_line(rest, "COMMAND", "TIMEOUT")
        return {"action": "run_shell", "args": {"command": cmd or "", "timeout": _field(text, "TIMEOUT")}}
    if name in READ_ACTIONS:
        return {"action": "read_file", "args": {"path": _field(text, "PATH") or _field(text, "FILE") or ""}}
    if name in LIST_ACTIONS:
        return {"action": "list_files", "args": {}}
    if name in ASK_ACTIONS:
        return {"action": "ask_user", "args": {"question": _field(text, "QUESTION") or _field(text, "Q") or ""}}
    if name in FINISH_ACTIONS:
        return {
            "action": "finish",
            "args": {
                "file": _field(text, "FILE") or _field(text, "PATH") or "",
                "summary": _field(text, "SUMMARY") or _field(text, "DESCRIPTION") or "",
            },
        }
    return {"action": name, "args": {}}


@dataclass
class AgentConfig:
    workspace: Path
    output: str = "solution.py"
    max_steps: int = 25
    allow_ask: bool = False
    shell_timeout: int = 60
    max_context_chars: int = 28000
    max_observation_chars: int = 4000
    verify: bool = True
    verbose: bool = True
    test_command: Optional[str] = None
    keep_history: bool = False


@dataclass
class AgentResult:
    success: bool
    file: Optional[str] = None
    summary: str = ""
    steps: int = 0
    transcript: Optional[list] = None


class SingleFileAgent:
    def __init__(self, llm: LlamaCppChatModel, config: AgentConfig, reference: str = ""):
        self.llm = llm
        self.cfg = config
        self.ws = Path(config.workspace).resolve()
        self.ws.mkdir(parents=True, exist_ok=True)
        self.reference = reference or (
            "(none provided - use only the Python standard library and standard shell tools; "
            "never invent third-party APIs)"
        )
        self.messages: list = []
        self.reset()

    def reset(self) -> None:
        head = self._system().replace("<REFERENCE>", self.reference)
        self.messages = [SystemMessage(content=head)]

    def _system(self) -> str:
        ask = ASK_ENABLED if self.cfg.allow_ask else ASK_DISABLED
        return (
            SYSTEM_PROMPT.replace("<ASK>", ask)
            .replace("<OUTPUT>", self.cfg.output)
            .replace("<WORKSPACE>", str(self.ws))
        )

    def _safe(self, path: str) -> Optional[Path]:
        if not path:
            return None
        p = Path(path)
        p = p if p.is_absolute() else self.ws / p
        try:
            p = p.resolve()
        except Exception:
            return None
        if p != self.ws and self.ws not in p.parents:
            return None
        return p

    def _clip(self, value: Any, limit: Optional[int] = None) -> str:
        s = "" if value is None else str(value)
        lim = limit or self.cfg.max_observation_chars
        if len(s) <= lim:
            return s
        head = lim // 2
        tail = lim - head
        return s[:head] + f"\n...[truncated {len(s) - lim} chars]...\n" + s[-tail:]

    def _size(self, messages: list) -> int:
        return sum(len(str(m.content)) for m in messages)

    def _trim(self, messages: list) -> None:
        while self._size(messages) > self.cfg.max_context_chars and len(messages) > 3:
            del messages[2:4]

    def _run_cmd(self, cmd: str, timeout: Optional[int] = None) -> tuple[int, str]:
        timeout = timeout or self.cfg.shell_timeout
        try:
            r = subprocess.run(
                cmd,
                shell=True,
                cwd=str(self.ws),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = (r.stdout or "") + (r.stderr or "")
            return r.returncode, out.strip()
        except subprocess.TimeoutExpired as e:
            out = ""
            for part in (e.stdout, e.stderr):
                if isinstance(part, bytes):
                    out += part.decode("utf-8", "replace")
                elif isinstance(part, str):
                    out += part
            return -1, f"TIMEOUT after {timeout}s\n{out}".strip()
        except Exception as e:
            return -1, f"{type(e).__name__}: {e}"

    def _coerce_timeout(self, value: Any) -> Optional[int]:
        try:
            t = int(float(str(value)))
            return max(1, min(t, 600))
        except Exception:
            return None

    def _execute(self, action: str, args: dict) -> str:
        if action == "write_file":
            p = self._safe(str(args.get("path", "")))
            if p is None:
                return "ERROR: invalid or out-of-workspace path."
            content = args.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            if not content.strip():
                return "ERROR: empty file content. Provide a fenced code block."
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"wrote {p.relative_to(self.ws)} ({len(content.encode('utf-8'))} bytes)"
        if action == "read_file":
            p = self._safe(str(args.get("path", "")))
            if p is None or not p.exists() or not p.is_file():
                return "ERROR: file not found."
            return f"contents of {p.relative_to(self.ws)}:\n{self._clip(p.read_text(encoding='utf-8', errors='replace'))}"
        if action == "list_files":
            items = []
            for q in sorted(self.ws.rglob("*")):
                if q.is_file():
                    items.append(str(q.relative_to(self.ws)))
                if len(items) >= 200:
                    break
            return "\n".join(items) if items else "(empty)"
        if action == "run_shell":
            cmd = str(args.get("command", ""))
            if not cmd.strip():
                return "ERROR: empty command."
            rc, out = self._run_cmd(cmd, self._coerce_timeout(args.get("timeout")))
            return f"exit={rc}\n{self._clip(out)}"
        if action == "ask_user":
            if not self.cfg.allow_ask:
                return "ERROR: asking the user is not allowed. Solve with available APIs."
            question = str(args.get("question", "")).strip()
            try:
                answer = input(f"\n[AGENT ASKS] {question}\n> ")
            except (EOFError, KeyboardInterrupt):
                return "ERROR: no interactive input available."
            return f"user answered: {answer}"
        return f"ERROR: unknown action '{action}'."

    def _handle_finish(self, args: dict) -> dict:
        name = str(args.get("file") or self.cfg.output).strip().strip("`'\"")
        p = self._safe(name)
        if p is None:
            return {"ok": False, "observation": "ERROR: invalid file path."}
        if not p.exists() or not p.is_file():
            return {"ok": False, "observation": f"ERROR: file '{name}' does not exist. Write it first."}
        if self.cfg.verify and p.suffix == ".py":
            rc, out = self._run_cmd(f"{sys.executable} {p.name}")
            if rc != 0:
                return {
                    "ok": False,
                    "observation": f"ERROR: final verification of {p.name} failed (exit={rc}):\n{self._clip(out)}",
                }
        if self.cfg.test_command:
            rc, out = self._run_cmd(self.cfg.test_command)
            if rc != 0:
                return {
                    "ok": False,
                    "observation": f"ERROR: acceptance test failed (exit={rc}):\n{self._clip(out)}",
                }
        if p.name != self.cfg.output:
            dst = self.ws / self.cfg.output
            dst.write_text(p.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            p = dst
        return {"ok": True, "file": str(p), "summary": str(args.get("summary", "")).strip()}

    def _brief(self, args: dict) -> str:
        for key in ("path", "command", "file", "question"):
            if args.get(key):
                return str(args[key])[:90]
        return ""

    def run(self, task: str, reference: str = "") -> AgentResult:
        if reference:
            self.reference = reference
        self.reset()
        return self.run_turn(f"TASK:\n{task}")

    def run_turn(self, user_text: str, show_current: bool = True) -> AgentResult:
        messages = self.messages
        content = user_text
        out_path = self.ws / self.cfg.output
        if show_current and out_path.exists() and len(messages) > 1:
            try:
                current = out_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                current = ""
            if current.strip():
                content = (
                    f"{user_text}\n\nCURRENT {self.cfg.output} (modify/complete this file, then finish):\n"
                    f"```python\n{current}\n```"
                )
        messages.append(HumanMessage(content=content))

        last_sig: Optional[str] = None
        repeat = 0
        parse_fail = 0

        for step in range(1, self.cfg.max_steps + 1):
            self._trim(messages)
            try:
                ai = self.llm.invoke(messages)
            except ModelError as e:
                return AgentResult(False, summary=f"model error: {e}", steps=step, transcript=messages)
            text = ai.content if isinstance(ai.content, str) else str(ai.content)
            messages.append(AIMessage(content=text))
            actions = parse_actions(text)

            if not actions:
                parse_fail += 1
                messages.append(
                    HumanMessage(
                        content="RESULT: ERROR: no valid ACTION block found. Reply with exactly one action "
                        "using the required format (see instructions)."
                    )
                )
                if parse_fail >= 4:
                    return AgentResult(False, summary="too many malformed replies", steps=step, transcript=messages)
                continue
            parse_fail = 0

            observations: list[str] = []
            for parsed in actions:
                action = parsed["action"]
                args = parsed.get("args") or {}
                if self.cfg.verbose:
                    print(f"[step {step}] {action} {self._brief(args)}", flush=True)

                if action == "finish":
                    res = self._handle_finish(args)
                    if res.get("ok"):
                        messages.append(
                            HumanMessage(content=f"RESULT: finished. {self.cfg.output} is written and verified. Awaiting next instruction.")
                        )
                        return AgentResult(True, file=res["file"], summary=res.get("summary", ""), steps=step, transcript=messages)
                    observations.append(res["observation"])
                    break

                sig = action + "|" + json.dumps(args, sort_keys=True, ensure_ascii=False)[:500]
                if sig == last_sig:
                    repeat += 1
                else:
                    last_sig = sig
                    repeat = 1
                if repeat > 2:
                    observations.append(
                        "ERROR: the same action was just tried repeatedly without progress. Change approach."
                    )
                    continue

                observation = self._execute(action, args)
                if self.cfg.verbose:
                    print(f"      -> {self._clip(observation, 300).replace(chr(10), ' | ')}", flush=True)
                observations.append(observation)

            if observations:
                messages.append(HumanMessage(content=self._clip("RESULT:\n" + "\n---\n".join(observations))))

        return AgentResult(False, summary="reached max steps without finishing", steps=self.cfg.max_steps, transcript=messages)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-file coding agent backed by a local llama.cpp OpenAI-compatible server.")
    p.add_argument("task", nargs="?", help="task description; if omitted, read from --task-file or stdin")
    p.add_argument("--task-file", help="read the task from a file")
    p.add_argument("--reference", "--material", dest="reference", action="append", default=[],
                   help="file with available APIs/demos (repeatable); the agent may use only these")
    p.add_argument("--workspace", default=".", help="directory where the agent works (default: current dir)")
    p.add_argument("--output", default="solution.py", help="name of the single deliverable file")
    p.add_argument("--max-steps", type=int, default=3, help="max agent steps per user turn (default: 3)")
    p.add_argument("--interactive", "-i", action="store_true",
                   help="start an interactive session; keep chatting to refine the deliverable")
    p.add_argument("--test", dest="test_command",
                   help="acceptance test command; must exit 0 before the agent may finish")
    p.add_argument("--allow-ask", action="store_true", help="allow the agent to ask the user for missing APIs")
    p.add_argument("--no-verify", action="store_true", help="skip the final run-time verification of the deliverable")
    p.add_argument("--base-url", default="http://127.0.0.1:8081/v1")
    p.add_argument("--model", default="local")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=3072)
    p.add_argument("--shell-timeout", type=int, default=60)
    p.add_argument("--max-context-chars", type=int, default=28000)
    p.add_argument("--quiet", action="store_true")
    return p


INTERACTIVE_HELP = """Commands:
  /show             print the deliverable file
  /run              run the deliverable file
  /test             run the acceptance test command (if --test was set)
  /output <name>    change the deliverable filename (resets the conversation)
  /reset            clear the conversation (workspace is kept)
  /help             show this help
  /exit, /quit      leave
Anything else is sent to the agent as an instruction."""


def _print_result(result: AgentResult) -> None:
    if result.success:
        print(f"OK ({result.steps} steps) -> {result.file}")
        if result.summary:
            print(f"    {result.summary}")
    else:
        print(f"INCOMPLETE ({result.steps} steps): {result.summary}")


def _run_deliverable(agent: SingleFileAgent, cfg: AgentConfig) -> tuple[int, str]:
    out = cfg.workspace / cfg.output
    if not out.exists():
        return 1, f"(no deliverable yet: {cfg.output})"
    cmd = f"{sys.executable} {cfg.output}" if out.suffix == ".py" else cfg.output
    return agent._run_cmd(cmd)


def interactive_session(agent: SingleFileAgent, cfg: AgentConfig) -> int:
    print("-" * 60)
    print("Interactive mode. Type /help for commands, /exit to quit.")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        low = line.lower()
        if low in ("/exit", "/quit", "exit", "quit", ":q"):
            break
        if low == "/help":
            print(INTERACTIVE_HELP)
            continue
        if low == "/show":
            out = cfg.workspace / cfg.output
            print(out.read_text(encoding="utf-8", errors="replace") if out.exists() else "(no deliverable yet)")
            continue
        if low == "/run":
            rc, out = _run_deliverable(agent, cfg)
            print(f"exit={rc}\n{out}")
            continue
        if low == "/test":
            if not cfg.test_command:
                print("(no --test command set)")
            else:
                rc, out = agent._run_cmd(cfg.test_command)
                print(f"exit={rc}\n{out}")
            continue
        if low.startswith("/output"):
            parts = line.split()
            if len(parts) == 2:
                cfg.output = parts[1]
                agent.reset()
                print(f"deliverable -> {cfg.output} (conversation reset)")
            else:
                print("usage: /output <name>")
            continue
        if low == "/reset":
            agent.reset()
            print("conversation reset")
            continue
        try:
            result = agent.run_turn(line)
        except KeyboardInterrupt:
            print("\n(turn interrupted)")
            continue
        _print_result(result)
    print("bye")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    refs = []
    for f in args.reference:
        path = Path(f)
        if not path.exists():
            print(f"error: reference file not found: {f}", file=sys.stderr)
            return 2
        refs.append(f"### {path.name}\n{path.read_text(encoding='utf-8', errors='replace')}")
    reference = "\n\n".join(refs)

    llm = LlamaCppChatModel(
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    cfg = AgentConfig(
        workspace=Path(args.workspace),
        output=args.output,
        max_steps=args.max_steps,
        allow_ask=args.allow_ask,
        shell_timeout=args.shell_timeout,
        max_context_chars=args.max_context_chars,
        verify=not args.no_verify,
        verbose=not args.quiet,
        test_command=args.test_command,
    )
    agent = SingleFileAgent(llm, cfg, reference)

    task = args.task
    if args.task_file:
        task = Path(args.task_file).read_text(encoding="utf-8")

    if args.interactive:
        print(f"workspace  : {cfg.workspace.resolve()}")
        print(f"deliverable: {cfg.output}")
        if cfg.test_command:
            print(f"test       : {cfg.test_command}")
        if task:
            print(f"task       : {task.strip()[:200]}")
            try:
                _print_result(agent.run_turn(f"TASK:\n{task}"))
            except KeyboardInterrupt:
                print("\n(interrupted)")
        return interactive_session(agent, cfg)

    if not task:
        task = sys.stdin.read().strip()
    if not task:
        print("error: no task provided (or use --interactive)", file=sys.stderr)
        return 2

    print(f"workspace  : {cfg.workspace.resolve()}")
    print(f"deliverable: {cfg.output}")
    print(f"task       : {task.strip()[:200]}")
    print("-" * 60)
    try:
        result = agent.run(task, reference)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    print("=" * 60)
    if result.success:
        print(f"DONE in {result.steps} steps")
        print(f"file    : {result.file}")
        if result.summary:
            print(f"summary : {result.summary}")
        out = Path(result.file) if result.file else None
        if out and out.exists():
            print("-" * 60)
            print(out.read_text(encoding="utf-8", errors="replace"))
        return 0
    print(f"FAILED after {result.steps} steps: {result.summary}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
