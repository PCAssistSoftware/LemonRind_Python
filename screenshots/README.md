# Gallery

A tour of Lemon Rind in pictures. Everything shown is made-up demo data: the chats, memories, knowledge base and jobs are invented, and the model is whatever the author's Lemonade Server had loaded.

The main screen is shown in both themes. Everything else is shown in light mode, because the dark theme only changes colours.

## The main screen

| Light | Dark |
|---|---|
| ![The chat screen in light mode](overview-light.png) | ![The chat screen in dark mode](overview-dark.png) |

Left: your chats, in folders, with tags and a search box. Middle: the conversation. Right: statistics for the last reply and the whole chat, the tools used, and the view screens. Along the bottom: the knowledge base attached to this chat, the message box, and the context ring.

## Chatting

| | |
|---|---|
| ![An empty new chat](new-chat.png) | ![A reply with a code block and a tool panel](chat-code.png) |
| A new chat. It is saved when the first reply arrives. | A reply with a code block, and the tool it used |
| ![A reply with its thinking and a tool call opened up](chat-tool-and-thinking.png) | ![A picture attached to a message](attachment-picture.png) |
| The model's thinking and a tool call, opened: the arguments it sent and the result it got | A picture attached to a message, for a model that can see |
| ![A PDF attached to a message](attachment-file.png) | ![A picture drawn by an image model](drawing.png) |
| A document attached to a message: its text is read and sent along | A picture drawn by an image model, with Save and Copy buttons |

## Chats, folders and search

| | |
|---|---|
| ![Search results with the matching words highlighted](search.png) | ![The menu on a chat](chat-menu.png) |
| Searching: the matching words are highlighted in the title, and a piece of the best matching message is shown | The menu on every chat: rename, move to a folder, tags, export as Markdown, delete |

## Models and what is sent

| | |
|---|---|
| ![The model picker, grouped by kind](model-picker.png) | ![Details of the chosen model](model-details.png) |
| The model picker, grouped into chat, image and embedding models | The details of a model: size, context window, backend, and what it is doing now |
| ![The context usage breakdown](context-breakdown.png) | ![The Modules tab of the right-hand panel](stats-modules-tab.png) |
| The context ring, opened: how full the model's memory is and what is using it | The Modules tab: what is switched on, at a glance |

## The view screens

Read-only screens at the bottom of the right-hand panel.

| | |
|---|---|
| ![The system prompt](view-system-prompt.png) | ![A preview of the turn context](view-turn-context.png) |
| View system prompt: what the model is told before every request | View turn context: what would be added to the message you are typing |
| ![The tools the model is offered](view-tools-sent.png) | ![The application log](view-logs.png) |
| View tools sent: every tool, with its description and a token count | View logs: this app's log (and Lemonade's own, on the other tab) |

![The usage screen](view-usage.png)

**View usage**: tokens and speed over time, per model and per tool, from every request the app has saved, scheduled jobs included.

## Settings

Fourteen sections, each saved with the Save button, except the lists (memories, knowledge bases, MCP servers, jobs), which act at once.

| | |
|---|---|
| ![Settings: Lemonade](settings-lemonade.png) | ![Settings: Assistant](settings-assistant.png) |
| Lemonade: the address, an optional key, the default models | Assistant: the base instructions |
| ![Settings: Persona](settings-persona.png) | ![Settings: Interface](settings-interface.png) |
| Persona: who the assistant is and how it writes | Interface: theme and the thinking panel |
| ![Settings: Storage](settings-storage.png) | ![Settings: Web search](settings-web-search.png) |
| Storage: where the data is kept and what uses the space | Web search: one of four services |
| ![Settings: File system access](settings-file-system.png) | ![Settings: Image generation](settings-image-generation.png) |
| File system access: the workspace and extra folders | Image generation: the model and the picture defaults |
| ![Settings: Memory](settings-memory.png) | ![Settings: Auto-backup](settings-auto-backup.png) |
| Memory: what the assistant remembers about you, and compaction | Auto-backup: a zip of the data folder at intervals |
| ![Settings: Modules](settings-modules.png) | ![Settings: Knowledge bases](settings-knowledge-bases.png) |
| Modules: switch whole abilities on and off | Knowledge bases: your own documents, one portable file each |
| ![Settings: MCP servers](settings-mcp-servers.png) | ![Settings: Scheduler](settings-scheduler.png) |
| MCP servers: tools from other programs | Scheduler: saved jobs and their last runs |

![The scheduled job form](job-form.png)

The job form: a Daily, Weekly or Monthly picker (or a cron line), and a preview of the next run times.

## Questions the app asks

| | |
|---|---|
| ![Asking permission to run a tool](approval-dialog.png) | ![Asking how big a picture to draw](draw-size-dialog.png) |
| Before a tool that changes something runs, the app shows exactly what it would do and waits for a yes | Before drawing, the app confirms the size, because drawing is slow and heavy |

## Sharing it, and small screens

| | | |
|---|---|---|
| ![The login page](login.png) | ![The chat on a phone](mobile-chat.png) | ![The chat list on a phone](mobile-chat-list.png) |
| The optional password page, for when the app is shared on a network | On a phone the side panels become drawers, and a wide table scrolls inside its reply | The chat list drawer |

## The terminal chat

The terminal chat (`lemonrind`, or `bash lemonrind.sh chat`) shares the same chats and settings. It is text, so it has no pictures here; its commands are listed in the [README](../README.md#the-terminal-chat).
