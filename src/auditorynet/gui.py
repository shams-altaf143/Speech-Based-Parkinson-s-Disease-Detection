"""Launcher for the Streamlit GUI (console script: auditorynet-gui)."""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from streamlit.web import cli as stcli

    app = Path(__file__).with_name("gui_app.py")
    sys.argv = ["streamlit", "run", str(app),
                "--server.fileWatcherType", "none",     # avoids torch/streamlit watcher errors
                "--browser.gatherUsageStats", "false"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
