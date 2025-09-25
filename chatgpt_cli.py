#!/usr/bin/env python3
"""
chatgpt_cli.py - Fast CLI for OpenAI Chat with persistent "Conversation Threads".
Requires: pip install prompt_toolkit openai tiktoken
"""
import os, sys, json, uuid, argparse, pathlib, textwrap, re, shutil
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# ========= Context window limits & tokenization =========
MODEL_MAX_TOKENS = {
    "gpt-5": 272000,
    "gpt-4": 128000,
    "gpt-4o": 128000,
    "gpt-4.1": 128000,
    "gpt-3.5": 16385,
}

try:
    import tiktoken
except Exception:
    tiktoken = None

def _normalize_model_name(model: str) -> str:
    m = (model or "").lower()
    if m.startswith("gpt-5"):
        return "gpt-5"
    if m.startswith("gpt-4o"):
        return "gpt-4o"
    if m.startswith("gpt-4.1"):
        return "gpt-4.1"
    if m.startswith("gpt-4"):
        return "gpt-4"
    if m.startswith("gpt-3.5"):
        return "gpt-3.5"
    return m or "gpt-4"

def get_model_limit(model: str) -> int:
    key = _normalize_model_name(model)
    return MODEL_MAX_TOKENS.get(key, 128000)

def num_tokens_from_messages(model: str, messages: List[Dict[str, str]]) -> int:
    """Estimate tokens using tiktoken if available; otherwise fallback; add per-message overhead."""
    overhead_per_msg = 8
    if not tiktoken:
        return sum(overhead_per_msg + len(m.get("content", "").split()) for m in messages)
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    toks = 0
    for m in messages:
        toks += overhead_per_msg
        toks += len(enc.encode(m.get("content", "")))
    return toks

def trim_messages_to_fit(model: str,
                         messages: List[Dict[str,str]],
                         reply_tokens: int,
                         buffer_tokens: int = 1024) -> Tuple[List[Dict[str,str]], int]:
    """Trim oldest messages so (tokens(messages)+reserve) <= limit; returns (trimmed, dropped_count)."""
    limit = get_model_limit(model)
    allowed = max(0, limit - max(0, reply_tokens) - max(0, buffer_tokens))
    trimmed = list(messages)
    dropped = 0
    toks = num_tokens_from_messages(model, trimmed)
    while trimmed and toks > allowed:
        trimmed.pop(0)
        dropped += 1
        toks = num_tokens_from_messages(model, trimmed)
    return trimmed, dropped

# ========= Role gutters & color policy =========
DEFAULT_MY_LABEL  = os.environ.get("CHATGPT_CLI_MY_LABEL",  "YOU")
DEFAULT_LLM_LABEL = os.environ.get("CHATGPT_CLI_LLM_LABEL", "GPT")

def _colors_enabled(mode: str) -> bool:
    """mode: 'auto' (tty only) | 'force' | 'off'"""
    if mode == "off": return False
    if mode == "force": return True
    return sys.stdout.isatty()

PTK_STYLE_MAP = {
    "none": "", "default": "",
    "black": "ansiblack", "red": "ansired", "green": "ansigreen", "yellow": "ansiyellow",
    "blue": "ansiblue", "magenta": "ansimagenta", "cyan": "ansicyan", "white": "ansiwhite",
    "brightblack": "ansibrightblack", "brightred": "ansibrightred", "brightgreen": "ansibrightgreen",
    "brightyellow": "ansibrightyellow", "brightblue": "ansibrightblue", "brightmagenta": "ansibrightmagenta",
    "brightcyan": "ansibrightcyan", "brightwhite": "ansibrightwhite",
    "dim": "dim", "bold": "bold",
}
def style_name(name: Optional[str], enable: bool) -> str:
    return PTK_STYLE_MAP.get((name or "").lower(), "") if enable else ""

# ========= prompt_toolkit =========
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.output.color_depth import ColorDepth

def build_key_bindings() -> KeyBindings:
    kb = KeyBindings()
    @kb.add("enter")
    def _(event): event.app.exit(result=event.app.current_buffer.text)
    @kb.add("c-j")
    def _(event): event.app.current_buffer.insert_text("\n")
    @kb.add("escape","enter")
    def _(event): event.app.current_buffer.insert_text("\n")
    @kb.add("c-c")
    def _(event): event.app.exit(result=None)
    return kb

def make_session() -> PromptSession:
    return PromptSession(
        history=FileHistory(str(HIST_FILE)),
        auto_suggest=AutoSuggestFromHistory(),
        multiline=True,
        key_bindings=build_key_bindings(),
        enable_history_search=True,
        include_default_pygments_style=False,
    )

# ========= OpenAI =========
from openai import OpenAI

# ========= Storage =========
HOME = pathlib.Path.home()
ROOT_DIR = HOME / ".chatgpt_cli"
THREADS_DIR = ROOT_DIR / "threads"
THREADS_DIR.mkdir(parents=True, exist_ok=True)
HIST_FILE = ROOT_DIR / "history"

DEFAULT_MODEL = os.environ.get("CHATGPT_CLI_DEFAULT_MODEL", "gpt-5")
DEFAULT_TEMPERATURE = float(os.environ.get("CHATGPT_CLI_TEMPERATURE", "0.2"))
DEFAULT_TOP_P       = float(os.environ.get("CHATGPT_CLI_TOP_P", "1.0"))
DEFAULT_STREAM      = os.environ.get("CHATGPT_CLI_STREAM", "1") == "1"

MODEL_CAPS: Dict[str, Dict[str, bool]] = {}

def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\-]+", "-", name.strip())
    s = re.sub(r"-+", "-", s).strip("-")
    return s.lower() or uuid.uuid4().hex[:8]

def tpath(tid: str) -> pathlib.Path:
    return THREADS_DIR / f"{tid}.json"

def save_thread(obj: Dict[str, Any]) -> None:
    p = tpath(obj["id"])
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)

def load_thread(key: str) -> Optional[Dict[str, Any]]:
    p = tpath(key)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    for f in THREADS_DIR.glob("*.json"):
        obj = json.loads(f.read_text(encoding="utf-8"))
        if obj.get("name", "").lower() == key.lower():
            return obj
    return None

def new_thread(name: str, system: Optional[str], model: str) -> Dict[str, Any]:
    tid = slugify(name)
    obj = {"id": tid, "name": name, "created_at": now_iso(), "updated_at": now_iso(), "model": model, "messages": []}
    if system:
        obj["messages"].append({"role": "system", "content": system, "ts": now_iso()})
    save_thread(obj)
    return obj

def list_threads() -> List[Dict[str, Any]]:
    return [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(THREADS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)]

def show_thread(obj: Dict[str, Any]) -> str:
    lines = [
        f"Thread: {obj['name']} (id={obj['id']})",
        f"Created: {obj['created_at']}  Updated: {obj['updated_at']}  Model: {obj.get('model','?')}",
        "-" * 80,
    ]
    for m in obj["messages"]:
        lines.append(f"[{m.get('ts','')}] {m.get('role','?').upper()}:\n{textwrap.indent(m.get('content',''), '  ')}\n")
    return "\n".join(lines)

def rename_thread(obj: Dict[str, Any], new_name: str) -> None:
    obj["name"] = new_name; obj["updated_at"] = now_iso(); save_thread(obj)

def delete_thread(obj: Dict[str, Any]) -> None:
    tpath(obj["id"]).unlink(missing_ok=True)

def import_file(obj: Dict[str, Any], file_path: str, role: str) -> None:
    data = pathlib.Path(file_path).read_text(encoding="utf-8", errors="replace")
    obj["messages"].append({"role": role, "content": data, "ts": now_iso()})
    obj["updated_at"] = now_iso(); save_thread(obj)

def export_thread(obj: Dict[str, Any], out_path: str) -> None:
    p = pathlib.Path(out_path)
    if p.suffix.lower() in (".md", ".markdown"):
        md = [f"# {obj['name']}\n", f"_Created: {obj['created_at']} • Updated: {obj['updated_at']} • Model: {obj.get('model','?')}_\n"]
        for m in obj["messages"]:
            md += ["---\n", f"**{m.get('role','').capitalize()}** · `{m.get('ts','')}`\n\n{m.get('content','')}\n"]
        p.write_text("".join(md), encoding="utf-8")
    else:
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def build_messages(obj: Dict[str, Any]) -> List[Dict[str, str]]:
    return [{"role": m["role"], "content": m["content"]} for m in obj["messages"]]

def ensure_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY", file=sys.stderr); sys.exit(2)

# ========= PTK printing (forced 4-bit color in REPL) =========
def pt_print(text: str, style: str = "", end: str = "") -> None:
    ft = FormattedText([(style, text)]) if style else FormattedText([("", text)])
    print_formatted_text(ft, end=end, color_depth=ColorDepth.DEPTH_4_BIT)

def _term_width(default: int = 76) -> int:
    try:
        return max(20, min(200, shutil.get_terminal_size(fallback=(default, 20)).columns))
    except Exception:
        return default

def pt_hr(style: str, char: str = "-") -> None:
    pt_print(char * _term_width(), style=style, end="\n")

def pt_print_user_block(label: str, body: str, style: str) -> None:
    """Blank line + colored HR + colored gutter/text + colored HR + blank line."""
    pt_print("\n")
    pt_hr(style)
    pt_print(f"{label}> ", style=style)
    if body:
        for line in body.splitlines():
            pt_print(line + "\n", style=style)
    pt_hr(style)
    pt_print("\n")

def pt_print_llm_block(label: str, body: str, label_style: str, body_style: str) -> None:
    pt_print(f"{label}> ", style=label_style)
    if body:
        for line in body.splitlines():
            pt_print(line + "\n", style=body_style)

# ========= Capability + completions =========
def _cap_init(model: str) -> Dict[str, bool]:
    if model not in MODEL_CAPS:
        MODEL_CAPS[model] = {"temperature": True, "top_p": True, "max_tokens": True, "stream": True}
    return MODEL_CAPS[model]

def _apply_supported_kwargs(model: str, temperature: Optional[float], top_p: Optional[float], max_tokens: Optional[int]) -> Dict[str, Any]:
    caps = _cap_init(model); kw: Dict[str, Any] = {}
    if caps.get("temperature", True) and (temperature is not None): kw["temperature"] = temperature
    if caps.get("top_p", True)       and (top_p is not None)      : kw["top_p"] = top_p
    if caps.get("max_tokens", True)  and (max_tokens is not None) : kw["max_tokens"] = int(max_tokens)
    return kw

def _parse_bad_request(e) -> Tuple[Optional[str], Optional[str], str]:
    try:
        resp = getattr(e, "response", None)
        if resp is not None:
            data = resp.json(); err = data.get("error", {})
            return err.get("param"), err.get("code"), err.get("message", "")
    except Exception:
        pass
    s = str(e)
    for p in ("temperature","top_p","max_tokens","stream"):
        if p in s: return p, None, s
    return None, None, s

def complete_once(client: OpenAI, model: str, messages: List[Dict[str,str]],
                  temperature: Optional[float], top_p: Optional[float],
                  max_tokens: Optional[int], stream: bool,
                  llm_label: str, llm_style: str) -> Tuple[str, bool, str]:
    # >>> Context window protection <<<
    reply_tokens = max(0, (max_tokens or 4096))  # reserve for reply
    buffer_tokens = 1024                         # safety
    messages, _ = trim_messages_to_fit(model, messages, reply_tokens, buffer_tokens)

    base = {"model": model, "messages": messages}
    attempt = 0
    while True:
        attempt += 1
        caps = _cap_init(model)
        extra = _apply_supported_kwargs(model, temperature, top_p, max_tokens)
        want_stream = stream and caps.get("stream", True)
        try:
            if want_stream:
                collected: List[str] = []
                pt_print_llm_block(llm_label, "", llm_style, llm_style)
                for ev in client.chat.completions.create(stream=True, **base, **extra):
                    delta = ev.choices[0].delta
                    if delta and (txt := delta.get("content")):
                        collected.append(txt)
                        pt_print(txt, style=llm_style, end="")
                pt_print("\n")
                return "".join(collected), True, model
            else:
                r = client.chat.completions.create(**base, **extra)
                text = (r.choices[0].message.content or "")
                model_used = getattr(r, "model", model)
                return text, False, model_used
        except Exception as e:
            param, _, msg = _parse_bad_request(e)
            if "context_length_exceeded" in (msg or "").lower() or (param == "messages" and "exceed" in (msg or "").lower()):
                # drop 10% more messages and retry (up to 5x)
                if attempt <= 5 and messages:
                    dropn = max(1, len(messages)//10)
                    messages = messages[dropn:]
                    base["messages"] = messages
                    print(f"[Notice] Additional trim: dropped {dropn} more messages after API rejection.", file=sys.stderr)
                    continue
            if param in ("temperature","top_p","max_tokens","stream") and attempt <= 3:
                MODEL_CAPS[model][param] = False
                continue
            raise

# ========= prompt_toolkit session =========
def build_key_bindings() -> KeyBindings:
    kb = KeyBindings()
    @kb.add("enter")          # Enter = submit
    def _(event): event.app.exit(result=event.app.current_buffer.text)
    @kb.add("c-j")            # Ctrl+J = newline
    def _(event): event.app.current_buffer.insert_text("\n")
    @kb.add("escape","enter") # Alt+Enter (Esc,Enter) = newline
    def _(event): event.app.current_buffer.insert_text("\n")
    @kb.add("c-c")            # Ctrl+C = cancel line
    def _(event): event.app.exit(result=None)
    return kb

def make_session() -> PromptSession:
    return PromptSession(
        history=FileHistory(str(HIST_FILE)),
        auto_suggest=AutoSuggestFromHistory(),
        multiline=True,
        key_bindings=build_key_bindings(),
        enable_history_search=True,
        include_default_pygments_style=False,
    )

# ========= Command handlers =========
def cmd_new(args):
    model = args.model or DEFAULT_MODEL
    obj = new_thread(args.name, args.system, model)
    print(f"Created thread '{obj['name']}' id={obj['id']} (model={model})")

def cmd_list(_args):
    items = list_threads()
    if not items: print("No threads yet."); return
    for o in items: print(f"{o['id']}\t{o['name']}\tupdated:{o['updated_at']}\tmodel:{o.get('model','?')}")

def cmd_show(args):
    obj = load_thread(args.thread)
    if not obj: sys.exit(f"Thread not found: {args.thread}")
    print(show_thread(obj))

def cmd_rename(args):
    obj = load_thread(args.thread)
    if not obj: sys.exit(f"Thread not found: {args.thread}")
    rename_thread(obj, args.name); print("Renamed.")

def cmd_delete(args):
    obj = load_thread(args.thread)
    if not obj: sys.exit(f"Thread not found: {args.thread}")
    delete_thread(obj); print("Deleted.")

def cmd_import(args):
    obj = load_thread(args.thread)
    if not obj:
        if args.create:
            obj = new_thread(args.thread, None, args.model or DEFAULT_MODEL)
            print(f"(created) thread '{obj['name']}' id={obj['id']}")
        else:
            sys.exit(f"Thread not found: {args.thread}")
    role = args.role.lower()
    if role not in ("user","system","assistant"): sys.exit("Role must be one of: user, system, assistant")
    import_file(obj, args.file, role); print(f"Imported {args.file} as {role}")

def pt_print_user_block(label: str, body: str, style: str) -> None:
    pt_print("\n"); pt_hr(style); pt_print(f"{label}> ", style=style)
    if body:
        for line in body.splitlines(): pt_print(line + "\n", style=style)
    pt_hr(style); pt_print("\n")

def pt_print_llm_block(label: str, body: str, label_style: str, body_style: str) -> None:
    pt_print(f"{label}> ", style=label_style)
    if body:
        for line in body.splitlines(): pt_print(line + "\n", style=body_style)

def _echo_user_line(line: str, my_label: str, my_style: str):
    pt_print_user_block(my_label, line, my_style)

def cmd_send(args):
    ensure_api_key()
    obj = load_thread(args.thread)
    if not obj: sys.exit(f"Thread not found: {args.thread}")
    model  = args.model or obj.get("model") or DEFAULT_MODEL
    client = OpenAI()

    enable_color = _colors_enabled(args.color.lower())
    my_style  = style_name(args.my_color,  enable_color)
    llm_style = style_name(args.llm_color, enable_color)
    dim_style = style_name("dim",          enable_color)

    my_label  = args.my_label  or DEFAULT_MY_LABEL
    llm_label = args.llm_label or DEFAULT_LLM_LABEL

    msgs = build_messages(obj) + [{"role":"user","content":args.message}]
    _echo_user_line(args.message, my_label, my_style)
    text, streamed, model_used = complete_once(
        client, model, msgs,
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
        stream=(not args.no_stream), llm_label=llm_label, llm_style=llm_style
    )
    if not streamed:
        pt_print(f"[Model: {model_used}]\n", style=dim_style)
        pt_print_llm_block(llm_label, text + "\n", llm_style, llm_style)
    obj["messages"].append({"role":"user","content":args.message,"ts":now_iso()})
    obj["messages"].append({"role":"assistant","content":text,"ts":now_iso()})
    obj["updated_at"] = now_iso()
    if args.model: obj["model"] = model
    save_thread(obj)

def cmd_chat(args):
    ensure_api_key()
    obj = load_thread(args.thread)
    if not obj:
        if args.create:
            obj = new_thread(args.thread, args.system, args.model or DEFAULT_MODEL)
            print(f"(created) thread '{obj['name']}' id={obj['id']}")
        else:
            sys.exit(f"Thread not found: {args.thread}")
    if args.system:
        obj["messages"].append({"role":"system","content":args.system,"ts":now_iso()})
        obj["updated_at"] = now_iso(); save_thread(obj)

    model  = args.model or obj.get("model") or DEFAULT_MODEL
    session = make_session()

    enable_color = _colors_enabled(args.color.lower())
    my_style  = style_name(args.my_color,  enable_color)
    llm_style = style_name(args.llm_color, enable_color)
    dim_style = style_name("dim",          enable_color)

    my_label  = args.my_label  or DEFAULT_MY_LABEL
    llm_label = args.llm_label or DEFAULT_LLM_LABEL

    print(f"[Chat] Thread='{obj['name']}'  Model={model}  (Enter=send, Ctrl+J=NL, Alt+Enter=NL, Ctrl+C=cancel)")
    print("-"*80)

    client = OpenAI()
    with patch_stdout():
        while True:
            try:
                prompt = session.prompt("> ")
            except (KeyboardInterrupt, EOFError):
                print("\nBye."); break
            if prompt is None: print(); continue
            prompt = prompt.strip()
            if not prompt: continue
            if prompt in ("/quit","/exit"): break
            if prompt == "/show": print(show_thread(obj)); continue
            if prompt.startswith("/setmodel "):
                model = prompt.split(" ",1)[1].strip()
                print(f"Model set to {model}")
                obj["model"] = model; obj["updated_at"]=now_iso(); save_thread(obj); continue

            _echo_user_line(prompt, my_label, my_style)

            msgs = build_messages(obj) + [{"role":"user","content":prompt}]
            text, streamed, model_used = complete_once(
                client, model, msgs,
                temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                stream=DEFAULT_STREAM, llm_label=llm_label, llm_style=llm_style
            )
            if not streamed:
                pt_print(f"[Model: {model_used}]\n", style=dim_style)
                pt_print_llm_block(llm_label, text + "\n", llm_style, llm_style)
            obj["messages"].append({"role":"user","content":prompt,"ts":now_iso()})
            obj["messages"].append({"role":"assistant","content":text,"ts":now_iso()})
            obj["updated_at"] = now_iso(); save_thread(obj)

def cmd_export(args):
    obj = load_thread(args.thread)
    if not obj: sys.exit(f"Thread not found: {args.thread}")
    export_thread(obj, args.out); print(f"Exported to {args.out}")

def cmd_search(args):
    q = args.query.lower(); hits = []
    for o in list_threads():
        s = 0
        if q in o["name"].lower(): s += 2
        for m in o["messages"]:
            if q in m.get("content","").lower(): s += 1; break
        if s: hits.append((s,o))
    if not hits: print("No matches."); return
    hits.sort(key=lambda x: (-x[0], x[1]["updated_at"]))
    for _, o in hits: print(f"{o['id']}\t{o['name']}\tupdated:{o['updated_at']}")

# ========= Parser =========
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ChatGPT CLI with persistent threads")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("new", help="Create a new thread")
    sp.add_argument("--name", required=True); sp.add_argument("--system"); sp.add_argument("--model")
    sp.set_defaults(func=cmd_new)

    sub.add_parser("list", help="List threads").set_defaults(func=cmd_list)

    sp = sub.add_parser("show", help="Show a thread transcript")
    sp.add_argument("--thread", required=True); sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("rename", help="Rename a thread")
    sp.add_argument("--thread", required=True); sp.add_argument("--name", required=True)
    sp.set_defaults(func=cmd_rename)

    sp = sub.add_parser("delete", help="Delete a thread")
    sp.add_argument("--thread", required=True); sp.set_defaults(func=cmd_delete)

    sp = sub.add_parser("import", help="Import a text file into a thread")
    sp.add_argument("--thread", required=True); sp.add_argument("--file", required=True)
    sp.add_argument("--role", default="user"); sp.add_argument("--create", action="store_true")
    sp.add_argument("--model"); sp.set_defaults(func=cmd_import)

    sp = sub.add_parser("export", help="Export a thread to .json or .md")
    sp.add_argument("--thread", required=True); sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("send", help="Send a one-off message")
    sp.add_argument("--thread", required=True); sp.add_argument("-m","--message", required=True)
    sp.add_argument("--model"); sp.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    sp.add_argument("--top_p", type=float, default=DEFAULT_TOP_P); sp.add_argument("--max_tokens", type=int)
    sp.add_argument("--no-stream", action="store_true")
    sp.add_argument("--my-color",   default="magenta",    help="Color for your prompts (palette name)")
    sp.add_argument("--llm-color",  default="default",    help="Color for assistant replies (palette name)")
    sp.add_argument("--my-label",   default=DEFAULT_MY_LABEL,  help="Label for your prompts (default: YOU)")
    sp.add_argument("--llm-label",  default=DEFAULT_LLM_LABEL, help="Label for assistant replies (default: GPT)")
    sp.add_argument("--color",      default=os.environ.get("CHATGPT_CLI_COLOR","auto"),
                    help="Color mode: auto|force|off (default: env CHATGPT_CLI_COLOR or 'auto')")
    sp.set_defaults(func=cmd_send)

    sp = sub.add_parser("chat", help="Interactive REPL (prompt_toolkit)")
    sp.add_argument("--thread", required=True); sp.add_argument("--create", action="store_true")
    sp.add_argument("--system"); sp.add_argument("--model")
    sp.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    sp.add_argument("--top_p", type=float, default=DEFAULT_TOP_P); sp.add_argument("--max_tokens", type=int)
    sp.add_argument("--my-color",   default="magenta",    help="Color for your prompts (palette name)")
    sp.add_argument("--llm-color",  default="default",    help="Color for assistant replies (palette name)")
    sp.add_argument("--my-label",   default=DEFAULT_MY_LABEL,  help="Label for your prompts (default: YOU)")
    sp.add_argument("--llm-label",  default=DEFAULT_LLM_LABEL, help="Label for assistant replies (default: GPT)")
    sp.add_argument("--color",      default=os.environ.get("CHATGPT_CLI_COLOR","auto"),
                    help="Color mode: auto|force|off (default: env CHATGPT_CLI_COLOR or 'auto')")
    sp.set_defaults(func=cmd_chat)

    sp = sub.add_parser("search", help="Search threads")
    sp.add_argument("--query", required=True); sp.set_defaults(func=cmd_search)
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
