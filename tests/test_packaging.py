import tomllib
from pathlib import Path


def test_parquet_engine_is_a_core_dependency():
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]
    assert any(item.split("==", 1)[0].lower() == "pyarrow" for item in dependencies)
