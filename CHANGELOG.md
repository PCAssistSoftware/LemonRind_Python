# Changelog

What changed in each version of the Python edition, newest first. Each entry is dated, and a change is added here when it is pushed.

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
