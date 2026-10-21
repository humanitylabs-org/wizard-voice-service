"""Initialize app-owned config once; never import host config or personal rules."""
import os
from pathlib import Path
import shutil


def initialize():
    data = Path(os.environ.get("WIZARD_VOICE_DATA_DIR", "/data"))
    data.mkdir(parents=True, exist_ok=True)
    config = Path(os.environ.get("WIZARD_VOICE_CONFIG", str(data / "config.yaml")))
    if not config.exists():
        # Exclusive create keeps concurrent setup from replacing owner configuration.
        with config.open("x", encoding="utf-8") as target:
            target.write(Path("/app/packaging/default-config.yaml").read_text())
        config.chmod(0o600)
    os.environ["WIZARD_VOICE_CONFIG"] = str(config)


if __name__ == "__main__":
    initialize()
    os.execvp("python", ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8803", "--workers", "1", "--no-access-log"])
