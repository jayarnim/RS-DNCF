import yaml
from pathlib import Path
from .parser.parser import parser
from .parser.schema import schema


def config_builder(path):
    path = Path(path)

    kwargs = dict(
        file=path,
        mode="r",
        encoding="utf-8",
    )
    with open(**kwargs) as f:
        cfg = yaml.safe_load(f)

    return parser(cfg)


def schema_builder(*args):
    return schema(*args)