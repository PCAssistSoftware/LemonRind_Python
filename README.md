# Lemon Rind

### "All the zest, none of the cloud."

A local AI assistant for [Lemonade Server](https://lemonade-server.ai/), written in Python. Everything runs on your own machine, including the model: your chats, memories and documents
never leave it. Use it in your browser, or in a terminal.

| Light | Dark |
|---|---|
| ![The chat screen in light mode](screenshots/overview-light.png) | ![The chat screen in dark mode](screenshots/overview-dark.png) |

It is a chat with an assistant that can *do* things. It streams its answers and shows its thinking, calls tools (search the web, read and edit files, draw pictures, use other programs
through MCP), remembers lasting facts about you, answers from your own documents, and runs saved jobs on a schedule. Each of those is a switchable module, and every one of them asks
before it does anything that changes something.

Lemon Rind also comes in [WPF, Avalonia and Blazor editions](https://github.com/PCAssistSoftware/LemonRind) (C# and VB.NET) with the same feature set. This is the Python edition, and it started as a learning project: the code is commented to explain the Python ideas as they appear.

## What you need

- A running **[Lemonade Server](https://lemonade-server.ai/)** (on this computer or on your network) with at least one chat model downloaded. A model that supports tool calling, such as
  `Qwen3-8B-GGUF`, gives the best results
- **Python 3.12 or newer** (if you install with uv, it fetches Python for you)
- **[uv](https://docs.astral.sh/uv/)** or **[pipx](https://pipx.pypa.io/)** for the easiest install (Option 1 below), or **git** (or just a ZIP download) to run from a copy of the source (Option 2)
- Optional, for the modules that use them:
  - an **embedding model** in Lemonade (for example `Qwen3-Embedding-0.6B-GGUF`) for Memory and Knowledge bases
  - a web search service: your own [SearXNG](https://docs.searxng.org/), or an API key for Jina, Tavily or Firecrawl
  - **Node.js**, for MCP servers that start with `npx`

## Get it running

### Option 1: install it as a command (easiest)

[uv](https://docs.astral.sh/uv/) or [pipx](https://pipx.pypa.io/) puts the app in its own private environment, so it cannot clash with anything else on the computer. uv also fetches Python 3.12 or newer if you don't have it.

```bash
uv tool install git+https://github.com/PCAssistSoftware/LemonRind_Python
lemonrind-web                    # or: lemonrind  for the terminal chat
```

Or with pipx (the same result, but it needs Python 3.12 or newer to be installed already):

```bash
pipx install git+https://github.com/PCAssistSoftware/LemonRind_Python
lemonrind-web                    # or: lemonrind  for the terminal chat
```

If the command is not found afterwards, run `uv tool update-shell` (or `pipx ensurepath`) and open a new terminal.

Behind an antivirus program that scans secure connections, add `--system-certs` to the uv command.

#### Update it

```bash
uv tool upgrade lemonrind        # if you installed with uv
pipx upgrade lemonrind           # if you installed with pipx
```

#### Remove it

```bash
uv tool uninstall lemonrind      # if you installed with uv
pipx uninstall lemonrind         # if you installed with pipx
```

Your chats and settings are kept in the data folder (see below), so neither of these touches them.

### Option 2: run it from a copy of the source

```bash
git clone https://github.com/PCAssistSoftware/LemonRind_Python.git
cd LemonRind_Python
```

No git? Use the green **Code** button on GitHub, choose **Download ZIP**, unzip it, and open a terminal in that folder.

#### Linux and macOS

```bash
bash lemonrind.sh      # creates its own environment on the first run, then starts the web app
```

`bash lemonrind.sh chat` starts the terminal chat instead, and anything else on the line (for example `--port 8090 --no-browser`) goes to the web app.

#### Windows (PowerShell)

```powershell
py -3 -m venv .venv                  # a private environment for this project
.venv\Scripts\Activate.ps1
pip install -e .
lemonrind-web                        # opens your browser; or: lemonrind  for the terminal chat
```

### Open it

However you started it, the web app opens in your browser at **http://127.0.0.1:8080**. Press Ctrl+C in the terminal to stop it.

**Is port 8080 already used by another program?** Start it on a different port with `--port`, and open that one instead:

```bash
lemonrind-web --port 8090          # from a source copy: bash lemonrind.sh --port 8090
```

If you forget, it stops straight away with a message that the port is in use and suggests another.

### Connect it to Lemonade

Point it at your Lemonade Server in **Settings > Lemonade**, or on the command line:

```bash
lemonrind-web --base-url http://my-server:13305/v1/
```

The default is `http://localhost:13305/v1/`.

## Using it from another device on your network

By default the web app listens on this computer only, so no other device can reach it. To use it from a phone, a tablet or another computer on your network:

**1. Set a password first.** Without one, anything that can reach the address could read your chats and make the assistant run its tools.

```bash
export LEMONRIND_PASSWORD='something long'          # Linux and macOS
```

```powershell
$env:LEMONRIND_PASSWORD = "something long"          # Windows PowerShell
```

**2. Start it listening on the network.** Add `--port 8090` if 8080 is taken, and `--no-browser` on a computer with no screen:

```bash
lemonrind-web --host 0.0.0.0
```

**3. Open it from the other device** using this computer's name or IP address and the port, for example `http://ubuntu:8080` or `http://192.168.1.20:8080`. An IP address is the safer choice for a phone, which often cannot look up computer names. To find this computer's address: `hostname -I` on Linux, `ipconfig` on Windows.

**4. If it still cannot be reached, check the firewall** on the computer running Lemon Rind. On Ubuntu: `sudo ufw allow 8080/tcp` (use your own port). On Windows, allow the app when the firewall asks the first time you start it.

Every page then asks for the password once. It travels over plain HTTP, so use it on a network you trust, and never make the port reachable from the internet. Without a password, listening on the network prints a warning.

## Command-line options

These work with `lemonrind-web` (the web app) and `lemonrind` (the terminal chat). From a source copy on Linux or macOS, put them after `bash lemonrind.sh`.

| Option | For | Meaning |
|---|---|---|
| `--port PORT` | web | the port to listen on (default `8080`; choose another if something else uses it) |
| `--host ADDRESS` | web | the address to listen on (default `127.0.0.1`, this computer only; `0.0.0.0` shares it on your network) |
| `--password` | web | require a password (better: the `LEMONRIND_PASSWORD` environment variable) |
| `--no-browser` | web | do not open a browser tab on start-up |
| `--base-url URL` | both | the Lemonade API address |
| `--data-dir DIR` | both | where the data folder is |
| `--model`, `--new`, `--verbose` | terminal | the model, a fresh chat, debug logging |

## Features

### Chatting

- Streaming replies with a collapsible **thinking** panel for models that reason (it can open by default if you prefer), Markdown with tables and code, a Copy button, and a Stop button
- A **model picker** grouped by kind (Chat, Image, Embedding, Other), read live from Lemonade, with an info button for the model's details and the real launch arguments, and a stopwatch while a
  model loads. Pick an image model and what you type is drawn as a picture
- Statistics for each reply (tokens in and out, speed, time to first token) and for the chat, a **context ring** beside the Send button (hover for the numbers) whose breakdown uses Lemonade's own tokenizer, and
  progress shown while a long prompt is read, and a **usage screen** (tokens and speed over time, per model and per tool, from every request the app has saved)
- **Attachments**: a picture for a vision model, or a PDF, Word, Excel, text or Markdown file to talk about
- Each tool call is shown with its real outcome, and the actual error if it failed
- The model always knows the real date and time
- Viewers for exactly what the model is given: the **system prompt**, the **turn context** (what would be added if you pressed Send now) and the **tools sent**. A **log viewer** shows the app's
  log and Lemonade's own server log, with level and text filters, pause and copy
- Light and dark themes, side panels you can resize by dragging, and a layout that works on a phone

### Modules

Every capability beyond the chat is a module you switch on or off in **Settings > Modules**. A module that is off offers the model no tools at all.

| Module | What it does |
|---|---|
| **Utilities** | the current time and a calculator |
| **Web search** | searches the web through SearXNG, Jina, Tavily or Firecrawl: pick one in Settings |
| **Web reader** | reads one page (the direct reader refuses private network addresses and checks every redirect), or crawls a site when a Firecrawl key is set |
| **File system** | reads and writes files in a workspace folder and any extra folders you allow; writes ask first and show what will change |
| **Coder** | finds, searches, outlines, reads and edits the files of a project folder; edits must match exactly once, are syntax-checked (Python, C#, JSON, TOML, XML; Visual Basic gets a warning) and show a diff before you approve them |
| **Memory** | learns lasting facts about you from your messages and recalls the relevant ones in later chats; reviewable, editable and pinnable in Settings |
| **Knowledge bases** | answers from your own documents: files, folders, web pages and notes, searched by meaning and attached per chat. Each one is a single `.kb` file you can download and add on another computer without re-reading the documents; if the other computer uses a different embedding model, one button re-embeds it |
| **MCP servers** | uses tools from other programs through the Model Context Protocol; paste a server's JSON to add it, and each new server asks permission before each tool use |
| **Images** | draws pictures with an image model on Lemonade, asking first; Save and Copy buttons sit under each picture |
| **Scheduler** | runs a saved prompt on a schedule (Daily, Weekly, Monthly or a full cron expression), saving each result as a tagged chat |
| **Auto-backup** | zips the whole data folder to another folder every few days, keeping the newest five (off by default) |

### Keeping prompts small

A deliberate design priority, because a local model's context is precious:

- The system prompt holds only stable content, so Lemonade's cache can reuse work from turn to turn
- Per-message extras (the time, recalled memories, matching passages from a knowledge base) go into the outgoing message only and are never stored in the chat
- When a chat nears the model's limit, the oldest messages are **summarised** into a running paragraph (the saved chat keeps everything)
- Memory is capped and filtered by similarity, so old chats do not bloat every new prompt
- Only the newest picture in a chat is sent to the model again
- A long tool result (a big web page, say) is handed over in pieces, and the model reads on with a built-in `read_more` tool, so nothing is lost and no single result fills the context

### Persona

**Settings > Persona** sets who the assistant is: its name and personality, a standing description of you, and its tone, length and emoji use. All of it is optional and costs nothing when blank.

### Organising your chats

Folders and tags, rename and move, full-text search across titles, messages, folders and tags, automatic tags that say how a chat was used (`image`, `coder`, `mcp`, `scheduled`), and **export to
Markdown** with the thinking, tool use and pictures kept.

### Safe by design

- One portable **data folder** holds the database, settings, pictures, workspace and logs: copy it anywhere and it travels with you
- File tools work inside folders you name, and a path is resolved before it is checked, so `..` and links cannot escape them
- Anything that writes, draws or uses an MCP server asks first; scheduled jobs refuse tools that need permission unless you allow them for that job
- Text fetched from the web is cleaned before the model sees it: invisible and look-alike characters are stripped so a page cannot hide instructions in plain sight
- An optional password protects the web app, with a lockout after repeated wrong guesses

### The terminal chat

`lemonrind` is the same assistant in a terminal, with the same settings and the same saved chats. `/help` lists every command: `/chats`, `/open`, `/search`, `/rename`, `/folder`, `/tag`, `/export`,
`/models`, `/model`, `/persona`, `/prompt`, `/attach`, `/modules`, `/memories`, `/kb`, `/mcp`, `/jobs` and more.

## A closer look

| | |
|---|---|
| ![A reply with its thinking and a tool call opened up](screenshots/chat-tool-and-thinking.png) | ![Search results with the matching words highlighted](screenshots/search.png) |
| Thinking and tool calls, opened up | Search that highlights what matched |
| ![The usage screen: tokens and speed over time, per model and per tool](screenshots/view-usage.png) | ![Asking permission before a tool runs](screenshots/approval-dialog.png) |
| Usage: tokens and speed over time, per model and per tool | Every tool that changes something asks first |
| ![A drawn picture in a chat, with Save and Copy buttons](screenshots/drawing.png) | ![The scheduled job form with Daily, Weekly and Monthly pickers](screenshots/job-form.png) |
| Drawing from an image model | A scheduled job, with a schedule picker and the next run times |

Every screen, including all fourteen Settings sections, the phone layout and the login page, is in the [gallery](screenshots/README.md).

## Settings and your data

Almost everything is in the **Settings** dialog (the gear in the header), laid out in sections: Lemonade, Assistant, Persona, Interface, Storage, Web search, File system access, Image generation,
Memory, Auto-backup, Modules, Knowledge bases, MCP servers and Scheduler. They are saved in `settings.json` in the data folder, which you can also edit by hand.

The data folder is `data/` in the project folder when you run from a checkout. An installed copy (Option 1) has no project folder, so it uses a fixed place for your user instead: `%LOCALAPPDATA%\LemonRind` on Windows, `~/Library/Application Support/LemonRind` on macOS and `~/.local/share/lemonrind` on Linux. To use another place, pass `--data-dir`, set `LEMONRIND_DATA_DIR`, or choose it in Settings > Storage. The app's own log is `logs/lemonrind.log` inside the data folder. The other start-up options are listed under Command-line options above.

## What it is built with

Python, with [NiceGUI](https://nicegui.io/) for the web interface, the official `openai` SDK to talk to Lemonade, the official `mcp` SDK for MCP, `pydantic` for settings and data, SQLite for storage,
`APScheduler` for schedules and `Rich` for the terminal. There is no AI agent framework: the tool loop is about 150 lines you can read in `src/lemonrind/chats/conversation.py`. The full list
of libraries is in `pyproject.toml`.

## Status and known limits

- The code has an automated test suite (it needs no server) and is checked with `ruff`, `mypy`, `bandit` and `pip-audit`; `scripts/audit.ps1` runs every check in one go on Windows
- All four search services (SearXNG, Jina, Tavily and Firecrawl) have been tried against the real services
- The launcher script `lemonrind.sh` has been run on Linux and under Git Bash on Windows; macOS is not yet tested
- The database is the Python edition's own design: its data folder is not interchangeable with the .NET editions'

See [`CHANGELOG.md`](CHANGELOG.md) for what changed in each version.

## Develop it

Using VS Code? The folder includes a `.vscode` setup (run and debug, the Testing panel, Ruff and mypy).

```bash
pip install -e ".[dev]"       # the app plus the developer tools (newest versions that fit pyproject.toml)
uv sync --extra dev           # or: exactly the versions in uv.lock (pip install uv first)
pytest                        # run the tests
pytest --cov=lemonrind        # and see which lines they never run
ruff check . && ruff format . # lint and format
mypy src tests                # check the type hints
bandit -q -c pyproject.toml -r src   # security scan of the code
python scripts/audit_dependencies.py # known vulnerabilities in the installed libraries
powershell -ExecutionPolicy Bypass -File scripts\audit.ps1   # Windows: every check above in one go, results in audit-results/ and a zip
uv lock --upgrade             # refresh uv.lock on purpose, then run the tests again
```

`uv.lock` records the exact version of every library (about 100, including the ones the app's own libraries need) that the tests passed with, so an install later or on another machine gets the same set.
It is optional: `pip install -e .` still works from `pyproject.toml` alone. (If uv reports a certificate error behind antivirus HTTPS scanning, add `--system-certs`.)

## AI-assisted development

Not vibe coded, but vibe assisted. AI helped with ideas, inspiration, and pointers, but code was reviewed, understood, tested, and validated by a human.

## Licence

MIT: see [`LICENSE`](LICENSE). The libraries it uses are all under permissive licences too.
