from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent),
    keep_trailing_newline=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render(name: str, **kwargs) -> str:
    return _env.get_template(name).render(**kwargs)
