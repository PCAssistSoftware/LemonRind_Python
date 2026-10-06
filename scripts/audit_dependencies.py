"""Check the installed libraries against the public vulnerability databases (pip-audit).

Run it with the project's Python:  python scripts/audit_dependencies.py

Why a wrapper rather than plain ``pip-audit``: it has to reach the internet, and behind an antivirus program that
inspects secure connections the certificates in Python's own bundle are not the ones in use, so the lookup fails
with a certificate error. ``truststore`` makes Python use the operating system's certificates instead, which is what
a browser does. Any extra command-line arguments are passed on to pip-audit.
"""

from __future__ import annotations

import sys

import truststore


def main() -> None:
    truststore.inject_into_ssl()  # must happen before pip-audit opens a connection
    from pip_audit._cli import audit  # imported late so the line above takes effect first

    # --local: the packages installed in this environment; --skip-editable: not this project itself (it is not on
    # the public index, so there is nothing to look up).
    sys.argv = ["pip-audit", "--local", "--skip-editable", *sys.argv[1:]]
    audit()


if __name__ == "__main__":
    main()
