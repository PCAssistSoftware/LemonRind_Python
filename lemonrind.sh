#!/usr/bin/env bash
# One-step launcher for Linux and macOS (it also works in Git Bash on Windows).
#
#   bash lemonrind.sh                   start the web app          (then open http://127.0.0.1:8080)
#   bash lemonrind.sh --port 8090       anything else is passed to the web app: see  bash lemonrind.sh --help
#   bash lemonrind.sh chat              the terminal chat instead of the web app
#   bash lemonrind.sh install           only create the environment and install, then stop
#
# Starting it with "bash" needs nothing else. (To start it as ./lemonrind.sh instead, the file needs the executable
# permission: run  chmod +x lemonrind.sh  once; copying a file from Windows loses that permission.)
#
# The first run creates a private Python environment in .venv next to this file and installs the app into it.
# After that it starts straight away. If pyproject.toml changes (a new version of the app), it reinstalls itself.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

die() { echo "lemonrind.sh: $*" >&2; exit 1; }

# A virtual environment keeps its programs in bin/ on Linux and macOS, and in Scripts/ on Windows.
bin_of() { if [ -d "$1/Scripts" ]; then echo "$1/Scripts"; else echo "$1/bin"; fi; }

# Does the environment in this folder actually run here? (One copied from another system does not.)
works() { "$(bin_of "$1")/python" -c 'import sys' >/dev/null 2>&1; }

find_python() {
    local candidate
    for candidate in python3.14 python3.13 python3.12 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 &&
            "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

# An environment cannot be shared between operating systems: a .venv copied over from Windows has Scripts/ and
# Windows programs in it. So if .venv is there but does not run, use a separate one named for this system.
venv="$here/.venv"
if [ -d "$venv" ] && ! works "$venv"; then
    venv="$here/.venv-$(uname -s | tr '[:upper:]' '[:lower:]')"
    echo "The environment in .venv was made on another system, so a separate one is used here: ${venv##*/}"
fi
stamp="$venv/.lemonrind-installed"

if [ ! -d "$venv" ]; then
    python="$(find_python)" || die "Python 3.12 or newer is needed. On Debian or Ubuntu:  sudo apt install python3 python3-venv"
    echo "Creating the Python environment in ${venv##*/} (first run only) ..."
    "$python" -m venv "$venv" ||
        die "could not create the environment. On Debian or Ubuntu the venv module is separate:  sudo apt install python3-venv"
fi
bin="$(bin_of "$venv")"

# Install (or reinstall) when the stamp is missing or older than pyproject.toml.
if [ ! -f "$stamp" ] || [ pyproject.toml -nt "$stamp" ]; then
    echo "Installing Lemon Rind and its libraries (this takes a minute the first time) ..."
    "$bin/python" -m pip install --quiet --disable-pip-version-check -e . || die "the install failed (see the messages above)"
    touch "$stamp"
fi

case "${1:-}" in
    install)
        echo "Installed. Start it with:  ./lemonrind.sh"
        ;;
    chat)
        shift
        exec "$bin/lemonrind" "$@"
        ;;
    web)
        shift
        exec "$bin/lemonrind-web" "$@"
        ;;
    *)
        exec "$bin/lemonrind-web" "$@"
        ;;
esac
