#!/usr/bin/env python3
# import_chatgpt_export.py - Import ChatGPT web export into chatgpt_cli threads.
#
# Usage:
#   python import_chatgpt_export.py --export /path/to/export.zip --conv "Conversation Title" --thread "New Thread Name" --model gpt-5
#   python import_chatgpt_export.py --export /path/to/conversations.json --list
#   python import_chatgpt_export.py --export export.zip --conv 0 --thread "Imported Chat" --model gpt-5
#   python import_chatgpt_export.py --export export.zip --conv "partial title" --append
#
# Notes:
# - This script does NOT call the OpenAI API. It only writes JSON files into
#   ~/.chatgpt_cli/threads/ matching the format used by chatgpt_cli.py.
# - It handles both .zip (ChatGPT "Export data") and a direct conversations.json.
# - It tolerates both the old 'mapping' format and newer 'messages' lists.

import argparse, json, zipfile, pathlib, re, sys
from datetime import datetime

HOME = pathlib.Path.home()
THREADS_DIR = HOME / ".chatgpt_cli" / "threads"
THREADS_DIR.mkdir(parents=True, exist_ok=True)

def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\-]+", "-", name.strip())
    s = re.sub(r"-+", "-", s).strip("-")
    return s.lower() or "thread"

def load_export(path: str):
    p = pathlib.Path(path)
    if not p.exists():
        sys.exit(f"Export file not found: {p}")
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as z:
            # Prefer conversations.json; fallback to messages.json
            target = None
            for name in z.namelist():
                base = name.rsplit("/", 1)[-1].lower()
                if base == "conversations.json":
                    target = name; break
            if target is None:
                for name in z.namelist():
                    base = name.rsplit("/", 1)[-1].lower()
                    if base == "messages.json":
                        target = name; break
            if target is None:
                sys.exit("Could not find conversations.json or messages.json in the ZIP.")
            data = z.read(target).decode("utf-8", errors="replace")
            return json.loads(data)
    else:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))

def extract_messages_from_conversation(conv) -> list:
    # Return a list[{role, content, ts}] in chronological order.
    # Supports both 'mapping' dict and 'messages' list variants.
    msgs = []

    if isinstance(conv, dict) and "messages" in conv and isinstance(conv["messages"], list):
        for m in conv["messages"]:
            role = (m.get("author", {}) or {}).get("role") or m.get("role") or ""
            content = ""
            if isinstance(m.get("content"), dict):
                ct = m["content"].get("content_type")
                if ct == "text":
                    parts = m["content"].get("parts") or []
                    content = "\n".join([str(x) for x in parts])
                else:
                    content = json.dumps(m["content"], ensure_ascii=False)
            elif isinstance(m.get("content"), list):
                content = "\n".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in m["content"])
            elif isinstance(m.get("content"), str):
                content = m["content"]
            ts = m.get("create_time") or m.get("update_time") or ""
            if role in ("user","assistant","system") and content:
                msgs.append({"role": role, "content": content, "ts": ts or now_iso()})
        # Attempt to sort by ts if numeric
        def _key(x):
            try:
                return float(x["ts"])
            except Exception:
                return 0.0
        msgs.sort(key=_key)
        return msgs

    # Older export style with 'mapping'
    mapping = conv.get("mapping") if isinstance(conv, dict) else None
    if isinstance(mapping, dict):
        tmp = []
        for node_id, node in mapping.items():
            message = node.get("message") if isinstance(node, dict) else None
            if not message:
                continue
            role = (message.get("author") or {}).get("role") or ""
            if role not in ("user","assistant","system"):
                continue
            content = ""
            cont = message.get("content")
            if isinstance(cont, dict):
                if cont.get("content_type") == "text":
                    parts = cont.get("parts") or []
                    content = "\n".join([str(x) for x in parts])
                else:
                    content = json.dumps(cont, ensure_ascii=False)
            elif isinstance(cont, list):
                content = "\n".join(str(x) for x in cont)
            elif isinstance(cont, str):
                content = cont
            ts = message.get("create_time") or node.get("create_time") or ""
            if content:
                tmp.append({"role": role, "content": content, "ts": ts or now_iso()})
        # sort by numeric ts if available
        def _key2(x):
            try:
                return float(x["ts"])
            except Exception:
                return 0.0
        tmp.sort(key=_key2)
        return tmp

    # Fallback: nothing recognized
    return msgs

def list_conversations(export_data):
    rows = []
    if isinstance(export_data, list):
        for idx, conv in enumerate(export_data):
            title = conv.get("title") or conv.get("conversation_title") or f"Conversation {idx}"
            rows.append((idx, title))
    elif isinstance(export_data, dict) and "conversations" in export_data:
        for idx, conv in enumerate(export_data["conversations"]):
            title = conv.get("title") or conv.get("conversation_title") or f"Conversation {idx}"
            rows.append((idx, title))
    return rows

def pick_conversation(export_data, conv_selector):
    convs = export_data
    if isinstance(export_data, dict) and "conversations" in export_data:
        convs = export_data["conversations"]

    if isinstance(conv_selector, int):
        try:
            return convs[conv_selector]
        except Exception:
            sys.exit(f"Conversation index out of range: {conv_selector}")
    # string selector: title substring match (case-insensitive)
    key = str(conv_selector).lower()
    for conv in convs:
        title = (conv.get("title") or conv.get("conversation_title") or "").lower()
        if key in title:
            return conv
    sys.exit(f"Conversation not found by selector: {conv_selector}")

def load_or_create_thread(thread_name: str, model: str, append: bool):
    tid = slugify(thread_name)
    path = THREADS_DIR / f"{tid}.json"
    if path.exists() and append:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj, path
    # new thread
    obj = {
        "id": tid,
        "name": thread_name,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "model": model,
        "messages": []
    }
    return obj, path

def save_thread_file(obj, path: pathlib.Path):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def main():
    ap = argparse.ArgumentParser(description="Import ChatGPT export into chatgpt_cli threads")
    ap.add_argument("--export", required=True, help="Path to ChatGPT export zip or conversations.json")
    ap.add_argument("--list", action="store_true", help="List conversations in export and exit")
    ap.add_argument("--conv", help="Conversation selector: index (int) or substring of title")
    ap.add_argument("--thread", help="Target thread name (defaults to conversation title)")
    ap.add_argument("--model", default="gpt-5", help="Model to tag the thread with (default: gpt-5)")
    ap.add_argument("--append", action="store_true", help="Append to existing thread if it exists")
    ap.add_argument("--max-msgs", type=int, help="Limit number of messages imported (from start)")
    args = ap.parse_args()

    data = load_export(args.export)

    if args.list:
        rows = list_conversations(data)
        if not rows:
            print("No conversations found in export."); return
        for idx, title in rows:
            print(f"{idx}\t{title}")
        return

    if args.conv is None:
        sys.exit("Please provide --conv (index or title substring), or --list to see options.")

    # integer index or substring
    try:
        conv_sel = int(args.conv)
    except Exception:
        conv_sel = args.conv

    conv = pick_conversation(data, conv_sel)
    title = conv.get("title") or conv.get("conversation_title") or "Imported Chat"
    thread_name = args.thread or title

    msgs = extract_messages_from_conversation(conv)
    if args.max_msgs is not None:
        msgs = msgs[: args.max_msgs]

    thread_obj, path = load_or_create_thread(thread_name, args.model, append=args.append)

    # Append messages into thread
    count = 0
    for m in msgs:
        role = m["role"]
        content = m["content"]
        ts = m.get("ts") or now_iso()
        # normalize role
        if role not in ("user","assistant","system"):
            continue
        thread_obj["messages"].append({"role": role, "content": content, "ts": ts})
        count += 1

    thread_obj["updated_at"] = now_iso()
    save_thread_file(thread_obj, path)

    print(f"Imported {count} messages into thread '{thread_obj['name']}'")
    print(f"Thread file: {path}")

if __name__ == "__main__":
    main()
