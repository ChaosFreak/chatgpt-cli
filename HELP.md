# chatgpt_cli.py — Command-Line Help

A fast CLI for OpenAI Chat with persistent "Conversation Threads".

---

## Basics

Threads are stored as JSON under:
```
~/.chatgpt_cli/threads/<thread-id>.json
```

You can back these up, diff them, or export to Markdown.

Environment variables:
- `OPENAI_API_KEY` (required)
- `CHATGPT_CLI_DEFAULT_MODEL` (default: gpt-5)
- `CHATGPT_CLI_TEMPERATURE` (default: 0.2)
- `CHATGPT_CLI_TOP_P` (default: 1.0)
- `CHATGPT_CLI_STREAM` ("1" = try streaming, default: 1)

The script auto-detects unsupported parameters (temperature, top_p, max_tokens, stream) per model and retries without them.

---

## Commands

### 1. Thread Management

- **Create a thread**
  ```bash
  ./chatgpt_cli.py new --name "My Thread" --system "You are concise." --model gpt-5
  ```

- **List threads**
  ```bash
  ./chatgpt_cli.py list
  ```

- **Show a transcript in terminal**
  ```bash
  ./chatgpt_cli.py show --thread "My Thread"
  ```

- **Rename a thread**
  ```bash
  ./chatgpt_cli.py rename --thread "My Thread" --name "My Thread v2"
  ```

- **Delete a thread**
  ```bash
  ./chatgpt_cli.py delete --thread "My Thread v2"
  ```

---

### 2. Chatting

- **Interactive REPL**
  ```bash
  ./chatgpt_cli.py chat --thread "My Thread" --model gpt-5
  ```

  Inside the REPL:
  - `/show` — print the transcript so far
  - `/setmodel NAME` — switch models mid-thread (e.g., gpt-5, gpt-4o)
  - `/quit` or `/exit` — leave REPL

- **One-off send (no REPL)**
  ```bash
  ./chatgpt_cli.py send --thread "My Thread" -m "Summarize this pipeline."
  ```

  Options:
  - `--model gpt-5`
  - `--max_tokens 800`
  - `--no-stream` — force non-streaming output

---

### 3. Importing Text

You can import text into a thread as if it came from:
- `user` — a user prompt
- `system` — a system instruction
- `assistant` — a model reply

Examples:
```bash
# Import user content
./chatgpt_cli.py import --thread "My Thread" --file notes.txt --role user

# Import a system instruction
./chatgpt_cli.py import --thread "My Thread" --file primer.md --role system

# Import assistant content
./chatgpt_cli.py import --thread "My Thread" --file answer.txt --role assistant
```

If the thread doesn’t exist yet:
```bash
./chatgpt_cli.py import --thread "Imported Chat" --file convo.txt --role user --create --model gpt-5
```

#### Importing a whole conversation
Recreate transcripts by alternating imports:
```bash
./chatgpt_cli.py new --name "Migrated Chat" --model gpt-5
./chatgpt_cli.py import --thread "Migrated Chat" --file system.txt --role system
./chatgpt_cli.py import --thread "Migrated Chat" --file u1.txt --role user
./chatgpt_cli.py import --thread "Migrated Chat" --file a1.txt --role assistant
./chatgpt_cli.py import --thread "Migrated Chat" --file u2.txt --role user
./chatgpt_cli.py import --thread "Migrated Chat" --file a2.txt --role assistant
```

---

### 4. Exporting Threads

- **JSON (raw, machine-readable)**
  ```bash
  ./chatgpt_cli.py export --thread "My Thread" --out my_thread.json
  ```

- **Markdown (pretty transcript)**
  ```bash
  ./chatgpt_cli.py export --thread "My Thread" --out my_thread.md
  ```

---

### 5. Searching Threads

Search thread names and contents:
```bash
./chatgpt_cli.py search --query "Kalshi"
```

---

## Tips

- Use `--create` when importing or chatting to auto-create a thread if it doesn’t exist.
- Set defaults once:
  ```bash
  export CHATGPT_CLI_DEFAULT_MODEL=gpt-5
  ```
- `/setmodel` inside REPL is handy if you want to switch between gpt-5, gpt-4o, gpt-4.1, etc.
- If the model doesn’t support streaming (like gpt-5 for some orgs), the CLI auto-falls back to non-streaming and prints the complete reply.
- Threads are just JSON files. You can `git commit` them for version control or hand-edit if needed.
