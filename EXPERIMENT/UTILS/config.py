import yaml
from pathlib import Path


def load_yaml(
    path: str,
):
    folder = Path(path)
    files = list(folder.rglob("*.yaml"))

    cfg = dict()

    for file in files:
        kwargs = dict(
            file=file,
            mode="r",
            encoding="utf-8",
        )
        with open(**kwargs) as f:
            cfg[file.stem] = yaml.safe_load(f)
    
    return cfg