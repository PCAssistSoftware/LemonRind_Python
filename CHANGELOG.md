# Changelog

What changed in each version of the Python edition, newest first. Each entry is dated, and a change is added here when it is pushed.

## 0.2.8 - 09-10-2026

### Fixed

- **A removed default model no longer causes a red banner that keeps coming back.** If the default chat model saved in Settings was later removed from Lemonade, the page showed "Model '...' is not downloaded on this Lemonade" again at every health check, and scheduled jobs without a model of their own failed. Now the app carries on with the model Lemonade has loaded (or the smallest tool-capable chat model), says once that your default is gone, and leaves your saved setting alone. A model you name yourself (`--model`, or a job's own model) is still an error if it is missing.

## 0.2.7 - 09-10-2026

### Added

- **A long tool result can now be read to the end.** A big web page (or any long result) used to be cut off after 12,000 characters and the rest was lost, so the model went round in circles asking for "the rest" or rebuilding a page from other pages. Now the cut-off note tells the model how to continue, and a built-in `read_more` tool gives the next piece, and the next, until the end. It works for every tool: web pages from any search service, Playwright, MCP servers and the file tools.
- **A setting for the size of a piece:** Settings > Modules > "Longest piece of a tool result (characters)". Raise it if your model has a large context window.

### Changed

- **The default piece is 40,000 characters** (it was 12,000), about 10,000 tokens. If you saved your settings before, they still hold the old 12,000: change the box in Settings > Modules to use the new default.
- **The page reader no longer cuts a page at 8,000 characters** of its own: it gives the whole page to the same piece-by-piece rule.
- Pieces end at the end of a line (or at least a word), never in the middle of one.

## 0.2.6 - 09-10-2026

### Added

- **The model picker now knows about running scheduled jobs.** While a job runs, a note beside the picker says which job and which model ("Job running: Weekly report summary (Qwen3-8B-GGUF)"). If you pick a different model while a job is running, you are asked first, because loading another model can push the job's model out of Lemonade's memory and stop the job. Picking the job's own model needs no question, and nothing is blocked.
- **Watching a running job shows its model.** While you watch a job's chat live, the picker shows the job's model and cannot be changed. When the job ends it goes back to your own model.

## 0.2.5 - 09-10-2026

### Fixed

- **A scheduled job whose model has been removed from Lemonade can be edited again.** Opening it failed with "Invalid value" because the saved model was no longer in the list of models. The form now keeps the saved model in its list, marked "not listed by Lemonade now", so you can open the job, choose another model or leave it, and save.

## 0.2.4 - 08-10-2026

### Fixed

- **The job form no longer lets a long prompt run behind its buttons.** With a very long prompt, the text showed through underneath the Cancel and Save buttons, and part of it appeared below them. The form now has a fixed title, a scrolling middle and buttons that stay clear of the text, like the other windows.

## 0.2.3 - 08-10-2026

### Added

- **The time limit for a scheduled run can be changed in Settings > Scheduler.** It was fixed at 30 minutes. The new box ("Time limit for one run", in minutes, from 1 up to a day) takes effect from the next run and is saved at once.

## 0.2.2 - 07-10-2026

### Changed

- **A busy port is now explained.** Starting the web app on a port that another program is using used to print a long system error after a start-up line that looked like success. It now stops at once with a short message that the port is in use and suggests another, for example `lemonrind-web --port 8090`.

## 0.2.1 - 06-10-2026

### Fixed

- **Settings > Storage now adds up.** "What is using the space" gave the database's main file and one of its two side files, left out the knowledge bases, attached pictures and settings, and counted in units that did not match the computer's file manager. The database now includes both side files, knowledge bases and attached pictures have their own rows, anything else is listed as "Other", and the rows add up to the total. Sizes use the same units as the file manager: 1 KB is 1,024 bytes on Windows and 1,000 bytes on Linux and macOS.

## 0.2.0 - 06-10-2026

### Added

- **Install as a command** with `uv tool install` or `pipx install` straight from GitHub, so `lemonrind-web` and `lemonrind` work from any folder (see the README).

### Changed

- **An installed copy keeps its data in a fixed per-user folder** (`%LOCALAPPDATA%\LemonRind` on Windows, `~/Library/Application Support/LemonRind` on macOS, `~/.local/share/lemonrind` on Linux), where it used to use a `data` folder in whatever folder it was started from. Running from a source checkout is unchanged: the data is still in the project's `data` folder.

## 0.1.0 - 06-10-2026

The first version of the Python edition.

### Added

- **Chat** with any model on a Lemonade Server, in the browser (NiceGUI) or in a terminal. Replies stream in, with the model's thinking in a panel of its own.
- **Saved chats** in folders, with tags, rename, delete and Markdown export. Search covers titles, messages, folders and tags, and highlights what matched.
- **Tools the assistant can use**, each a module you can switch on or off: web search (SearXNG, Jina, Tavily or Firecrawl), a web page reader, file system access, code editing with syntax checks, picture drawing, and tools from MCP servers.
- **Asks before it changes anything.** Writing files, drawing pictures, running MCP tools and scheduling jobs all show exactly what they would do and wait for a yes.
- **Memory:** lasting facts about you, kept across chats and editable in Settings.
- **Knowledge bases:** your own documents (PDF, Word, Excel, text, Markdown, web pages), one portable file each, attached to a chat so its answers can draw on them.
- **Scheduled jobs** that run on a daily, weekly or monthly schedule (or a cron line) and save their results as chats.
- **Attachments:** a picture for a model that can see, or a document to talk about.
- **Long chats stay usable:** older messages are summarised automatically as a chat fills the model's memory, and a ring beside the message box shows how full it is.
- **Persona:** who the assistant is, who it is talking to, and how it writes.
- **Usage screen:** tokens and speed over time, per model and per tool, including scheduled jobs.
- **Settings** for everything above, an optional automatic backup of the data folder, and a movable data folder.
- **Light and dark themes**, a layout that works on a phone, and an optional password for when the app is shared on a network.
- **Launchers** for Windows, Linux and macOS (`lemonrind.sh`), a VS Code setup, and a lock file (`uv.lock`) with the exact library versions the tests passed with.

### Security

- Web pages are fetched only from public addresses, with every redirect checked again.
- Files are reachable only inside the folders you allow.
- Text from the web has invisible and look-alike characters removed before the model reads it.
- The code is scanned with bandit and the libraries are checked against known vulnerabilities with pip-audit (`scripts/audit.ps1` runs every check in one go).
