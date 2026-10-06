"""Page-wide CSS and JavaScript, added once per page.

NiceGUI is built on Vue and the Quasar component library, with Tailwind for utility classes
(``"text-caption w-full"`` and so on). Most styling is done with those classes right where a widget is
created. What does not fit a class goes here.

Light and dark colours are CSS *variables* (``--lr-...``) with a different value under ``body.body--dark``,
the class Quasar puts on the page in dark mode. Components use the variable, so they follow the theme
without any Python code knowing which theme is active.
"""

CSS = """
:root {
    --lr-assistant-bg: #eceef2;
    --lr-assistant-fg: #1b1f27;
    --lr-user-bg: #2f6bff;
    --lr-user-fg: #ffffff;
    --lr-muted: #5b6472;
    --lr-pane: #f3f4f6;
    --lr-bar-bg: #ffffff;
    --lr-bar-fg: #1b1f27;
}
body.body--dark {
    --lr-assistant-bg: #232a38;
    --lr-assistant-fg: #e6e9ef;
    --lr-user-bg: #3b6fe0;
    --lr-muted: #9aa3b2;
    --lr-pane: #11151c;
    --lr-bar-bg: #161b24;
    --lr-bar-fg: #e6e9ef;
}

/* chat bubbles: rounded, no speech-bubble tail, colours from the variables above */
.q-message-text { border-radius: 14px; }
.q-message-text:last-child:before { display: none; }
.q-message-text--received { background: var(--lr-assistant-bg) !important; color: var(--lr-assistant-fg) !important; }
.q-message-text--sent { background: var(--lr-user-bg) !important; color: var(--lr-user-fg) !important; }
.q-message-text-content { overflow-wrap: anywhere; }
.q-message-text--sent .q-message-text-content, .q-message-text--sent .q-message-text-content * { color: var(--lr-user-fg) !important; }
.q-message-text--sent .q-message-stamp { opacity: 0.8; }
.q-message-text--received .q-message-text-content { color: var(--lr-assistant-fg); }

/* markdown inside a bubble: long lines and code blocks wrap instead of scrolling sideways */
.lr-markdown { overflow-wrap: anywhere; color: var(--lr-assistant-fg); }
.lr-markdown p { margin: 0 0 0.5em 0; }
.lr-markdown p:last-child { margin-bottom: 0; }
.lr-markdown pre {
    white-space: pre-wrap;
    word-break: break-word;
    background: rgba(127, 127, 127, 0.15);
    padding: 0.5em 0.7em;
    border-radius: 8px;
    font-size: 0.85em;
}
.lr-markdown code { font-size: 0.9em; }
.lr-markdown img { max-width: min(100%, 30rem); height: auto; border-radius: 8px; margin: 0.3em 0; }
.lr-markdown table { border-collapse: collapse; }
.lr-markdown th, .lr-markdown td { border: 1px solid rgba(127, 127, 127, 0.4); padding: 4px 8px; }
/* On a narrow screen (a phone) a wide table or a long tool label must not push a reply wider than the screen. Flex
   children may not shrink below their content unless told min-width: 0, so each box in the chain is told, and a table
   that is still too wide scrolls inside its own box instead of stretching the page. */
.q-message, .q-message-container, .q-message-container > div, .q-message-text, .q-message-text-content {
    min-width: 0; max-width: 100%;
}
.lr-markdown { min-width: 0; max-width: 100%; overflow-x: auto; }
.lr-markdown table { display: block; max-width: 100%; overflow-x: auto; }
.lr-tool, .lr-tool .q-expansion-item__container, .lr-tool .q-item { min-width: 0; max-width: 100%; }
.lr-tool .q-item { overflow: hidden; }
.lr-tool .q-item > .row { flex: 1 1 0; min-width: 0; max-width: 100%; }
.lr-tool .q-item .row > .ellipsis { min-width: 0; flex: 0 1 auto; }

.lr-thinking-text { white-space: pre-wrap; color: var(--lr-muted); font-size: 0.85em; }
.lr-muted { color: var(--lr-muted); }
.lr-mark { background: rgba(255, 193, 7, 0.35); color: inherit; border-radius: 2px; padding: 0 1px; }
/* A row of dialog buttons that stays at the bottom edge of a scrolling card. The background is inherited from
   the card so the form does not show through; the top border marks where the scrolling content ends. */
.lr-sticky-footer { position: sticky; bottom: 0; background: inherit; padding-top: 8px; z-index: 1;
                    border-top: 1px solid rgba(127, 127, 127, 0.3); }
.lr-one-line { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.lr-snippet { font-size: 0.75rem; color: var(--lr-muted); line-height: 1.3; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }

/* small tag badges under a chat's title in the list */
.lr-tag { font-size: 0.7rem; padding: 1px 6px; border-radius: 10px; color: var(--lr-muted) !important; }

/* group headings inside the model picker, with a divider above every group but the first */
.lr-model-group {
    font-size: 0.7rem; letter-spacing: 0.06em; text-transform: uppercase; font-weight: 600;
    color: var(--lr-muted); padding: 10px 16px 4px; margin-top: 4px;
    border-top: 1px solid rgba(127, 127, 127, 0.3); cursor: default;
}
.q-virtual-scroll__content > .lr-model-group:first-child { border-top: none; margin-top: 0; }

/* warning-coloured text (progress lines, notes): a dark amber on light backgrounds and a light one on dark, so it
   stays readable (Quasar's own warning yellow is pale on a grey bubble) */
.lr-warn { color: #9a4f00; }
body.body--dark .lr-warn { color: #f5b942; }

/* the "Unfiled" heading in the chat list: as strong as a folder's name */
.lr-unfiled { color: var(--lr-bar-fg) !important; opacity: 0.9; padding-left: 16px; margin-top: 10px; }

/* once the "Lemon Rind has stopped" cover is up, NiceGUI's own small "Connection lost" notice is redundant */
body.lr-stopped .nicegui-error-popup { display: none !important; }

/* the percentage inside the context ring beside the Send button */
.lr-wrap { overflow-wrap: anywhere; word-break: break-word; }
.lr-ring-text { font-size: 0.62rem; font-weight: 500; line-height: 1; color: var(--lr-muted); }

/* two boxes with a line between them: the Stats / Modules switch (the open one is filled) */
.lr-segments {
    border: 1px solid rgba(127, 127, 127, 0.5); border-radius: 8px; overflow: hidden; gap: 0 !important;
    position: sticky; top: 0; z-index: 3; background: var(--lr-pane);
    box-shadow: 0 -24px 0 0 var(--lr-pane);  /* hides list text peeking through the panel's padding above the switch */
}
.lr-segment { flex: 1 1 0; border-radius: 0 !important; font-weight: 600; }
.lr-segment + .lr-segment { border-left: 1px solid rgba(127, 127, 127, 0.5); }

/* Lemonade's log as a table: Time, Level, Source, Message */
.lr-log { font-family: ui-monospace, Consolas, "Cascadia Mono", monospace; font-size: 0.78rem; line-height: 1.35; }
.lr-log-head, .lr-log-row { display: grid; grid-template-columns: 6.4rem 3.6rem 8.5rem minmax(0, 1fr); column-gap: 0.75rem; padding: 2px 8px; align-items: baseline; }
.lr-log-head {
    position: sticky; top: 0; z-index: 1; background: var(--lr-bar-bg); color: var(--lr-muted);
    font-family: Roboto, sans-serif; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
    border-bottom: 1px solid rgba(127, 127, 127, 0.4);
}
.lr-log-row:nth-child(even) { background: rgba(127, 127, 127, 0.07); }
.lr-log-row:hover { background: rgba(47, 107, 255, 0.12); }
.lr-log-row-warn, .lr-log-row-warning { background: rgba(184, 134, 11, 0.14) !important; }
.lr-log-row-error, .lr-log-row-fatal { background: rgba(211, 47, 47, 0.14) !important; }
.lr-log-time { color: var(--lr-muted); white-space: nowrap; }
.lr-log-lvl { font-weight: 700; color: #2F6BFF; }
.lr-log-lvl-warn, .lr-log-lvl-warning { color: #B8860B; }
.lr-log-lvl-error { color: #D32F2F; }
.lr-log-lvl-fatal { color: #ffffff; background: #D32F2F; border-radius: 3px; text-align: center; }
.lr-log-tag { color: var(--lr-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lr-log-msg { overflow-wrap: anywhere; white-space: pre-wrap; }
.lr-log-day { padding: 5px 8px 2px; font-family: Roboto, sans-serif; font-size: 0.72rem; font-weight: 600; color: var(--lr-muted); border-top: 1px dashed rgba(127, 127, 127, 0.4); }

/* the navigation down the left of the Settings dialog */
.lr-settings-nav { border-right: 1px solid rgba(127, 127, 127, 0.25); min-width: 3.2rem; }
@media (min-width: 600px) { .lr-settings-nav { width: 14rem; } }
.lr-settings-nav .q-item { border-radius: 8px; margin: 1px 0; }
/* tool call panels: arguments and results in a small monospace box that scrolls if long */
.lr-tool { border: 1px solid rgba(127, 127, 127, 0.3); border-radius: 8px; }
.lr-tool-text { white-space: pre-wrap; font-family: monospace; font-size: 0.8em; overflow-wrap: anywhere;
    max-height: 14rem; overflow: auto; display: block; }
.lr-pane { background: var(--lr-pane); }

/* header and message box follow the theme (Quasar paints them in the primary colour by default) */
.q-header, .q-footer { background: var(--lr-bar-bg) !important; color: var(--lr-bar-fg) !important; }
.q-drawer { background: var(--lr-pane) !important; }
.q-footer { border-top: 1px solid rgba(127, 127, 127, 0.25); }

/* keep the message box and the page content at a readable width */
.lr-column { width: 100%; max-width: 860px; margin-left: auto; margin-right: auto; }

/* the strip on a side panel's inner edge that you drag to resize it */
.lr-resizer { position: absolute; top: 0; bottom: 0; width: 8px; cursor: col-resize; z-index: 5; touch-action: none; }
.lr-resizer-left { right: -4px; }
.lr-resizer-right { left: -4px; }
.lr-resizer:hover, body.lr-resizing .lr-resizer { background: rgba(47, 107, 255, 0.35); }
body.lr-resizing { cursor: col-resize; user-select: none; }

/* shown when the server has been gone for a few seconds (the app was stopped, or the computer slept) */
.lr-offline {
    position: fixed; inset: 0; z-index: 9000; display: none; align-items: center; justify-content: center;
    background: rgba(0, 0, 0, 0.55); color: #fff; text-align: center; padding: 24px;
}
.lr-offline.lr-offline-on { display: flex; }
.lr-offline-card { background: var(--lr-bar-bg); color: var(--lr-bar-fg); border-radius: 12px; padding: 28px 36px; max-width: 420px; }
.lr-offline-card b { display: block; font-size: 1.25rem; margin-bottom: 8px; }
"""

# Follow new content only while the reader is already near the bottom; scrolling up to read must not be
# overridden by a reply that is still streaming.
JS = """
<script>
window.lrFollow = true;
window.addEventListener('scroll', function () {
    const d = document.documentElement;
    window.lrFollow = (d.scrollHeight - window.innerHeight - window.scrollY) < 120;
}, { passive: true });
// Drag handles on the inner edges of the side panels. Each move is reported to Python (event "lr_resize"), which sets
// the panel's width; the handles are re-added if the page rebuilds a panel, and are inactive while a panel floats
// over the page (a narrow window).
window.lrAddResizers = function () {
    [['left', '.q-drawer--left'], ['right', '.q-drawer--right']].forEach(function (pair) {
        const side = pair[0];
        const drawer = document.querySelector(pair[1]);
        if (!drawer || drawer.querySelector(':scope > .lr-resizer')) return;
        const bar = document.createElement('div');
        bar.className = 'lr-resizer lr-resizer-' + side;
        bar.addEventListener('pointerdown', function (down) {
            if (drawer.classList.contains('q-drawer--on-top')) return;
            down.preventDefault();
            bar.setPointerCapture(down.pointerId);
            document.body.classList.add('lr-resizing');
            const widthAt = function (ev) {
                return Math.round(side === 'left' ? ev.clientX : document.documentElement.clientWidth - ev.clientX);
            };
            const move = function (ev) { emitEvent('lr_resize', { side: side, width: widthAt(ev), done: false }); };
            const up = function (ev) {
                bar.removeEventListener('pointermove', move);
                bar.removeEventListener('pointerup', up);
                document.body.classList.remove('lr-resizing');
                emitEvent('lr_resize', { side: side, width: widthAt(ev), done: true });
            };
            bar.addEventListener('pointermove', move);
            bar.addEventListener('pointerup', up);
        });
        drawer.appendChild(bar);
    });
};
window.addEventListener('load', function () {
    window.lrAddResizers();
    setInterval(window.lrAddResizers, 1500);
});
window.lrScrollDown = function (force) {
    if (force || window.lrFollow) {
        window.scrollTo({ top: document.documentElement.scrollHeight });
    }
};

// NiceGUI shows a small "Connection lost" notice when the server goes away. A tab cannot close itself, so after the
// server has been gone for a few seconds this puts a clear, full-page notice over the (now dead) page instead.
(function () {
    const title = document.title;
    let timer = null;
    function overlay() {
        let box = document.getElementById('lr-offline');
        if (!box) {
            box = document.createElement('div');
            box.id = 'lr-offline';
            box.className = 'lr-offline';
            box.innerHTML = '<div class="lr-offline-card"><b>Lemon Rind has stopped</b>' +
                'The app is no longer running, so this page cannot do anything. You can close this tab. ' +
                'If it should still be running, start it again and the page will reconnect on its own.</div>';
            document.body.appendChild(box);
        }
        return box;
    }
    function check() {
        const popup = document.getElementById('popup');
        const lost = popup && popup.getAttribute('aria-hidden') === 'false';
        if (lost && !timer) {
            timer = setTimeout(function () {
                overlay().classList.add('lr-offline-on');
                document.body.classList.add('lr-stopped');
                document.title = '(stopped) ' + title;
            }, 4000);
        } else if (!lost && timer) {
            clearTimeout(timer);
            timer = null;
            const box = document.getElementById('lr-offline');
            if (box) box.classList.remove('lr-offline-on');
            document.body.classList.remove('lr-stopped');
            document.title = title;
        }
    }
    window.addEventListener('load', function () {
        const popup = document.getElementById('popup');
        if (popup) new MutationObserver(check).observe(popup, { attributes: true, attributeFilter: ['aria-hidden'] });
    });
})();
</script>
"""
