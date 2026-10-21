"""Layout and dependency rules that import-linter cannot check."""

import ast
import inspect
import re
import sys
import tomllib
from importlib.metadata import distributions
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {
    "openknowledge_contracts": ROOT / "packages" / "contracts",
    "openknowledge_core": ROOT / "packages" / "core",
    "openknowledge_adapters": ROOT / "packages" / "adapters",
    "openknowledge_pipelines": ROOT / "pipelines",
}
FEATURE_FILES = {"models.py", "commands.py", "service.py", "ports.py", "errors.py"}
# No project, core feature, adapter or extra may exist outside these sets.
ALLOWED_PROJECTS = {"packages/contracts", "packages/core", "packages/adapters", "pipelines"}
ALLOWED_FEATURES = {"ingestion", "search"}
ALLOWED_ADAPTERS = {"semantic_scholar", "gcs", "clickhouse", "batch", "opensearch"}
ALLOWED_EXTRAS = {"semantic-scholar", "gcs", "clickhouse", "batch", "opensearch"}


def source(package: str) -> Path:
    return PROJECTS[package] / "src" / package


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def name_of(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return name_of(node.func)
    return ""


def subpackages(root: Path) -> set[str]:
    return {path.name for path in root.iterdir() if (path / "__init__.py").is_file()}


def test_every_port_method_is_a_coroutine() -> None:
    ports = sorted(source("openknowledge_core").rglob("ports.py"))
    assert ports
    for path in ports:
        for node in ast.walk(parse(path)):
            if isinstance(node, ast.ClassDef) and "Protocol" in map(name_of, node.bases):
                methods = [item for item in node.body if isinstance(item, ast.FunctionDef)]
                assert not methods, f"{path.name}: {node.name}.{methods[0].name} is not async def"


def test_core_features_have_the_standard_files_and_nothing_is_called_utils() -> None:
    for feature in subpackages(source("openknowledge_core")):
        files = {path.name for path in (source("openknowledge_core") / feature).glob("*.py")}
        assert FEATURE_FILES <= files, f"{feature} lacks {FEATURE_FILES - files}"
    for package in PROJECTS:
        assert not [path for path in source(package).rglob("*") if path.stem == "utils"]


def test_no_cached_or_module_level_clients() -> None:
    for package in PROJECTS:
        for path in sorted(source(package).rglob("*.py")):
            module = parse(path)
            for node in ast.walk(module):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    cached = {"cache", "lru_cache"} & set(map(name_of, node.decorator_list))
                    assert not cached, f"{path.name}: {node.name} is cached"
            for statement in module.body:
                value = statement.value if isinstance(statement, ast.Assign) else None
                if isinstance(value, ast.Call):
                    made = name_of(value.func)
                    client = made.endswith(("Client", "Storage", "Session"))
                    assert not client, f"{path.name} builds a {made} at module level"


def distribution(spec: str) -> str:
    match = re.match(r"[A-Za-z0-9._-]+", spec)
    assert match is not None
    return re.sub(r"[-_.]+", "-", match.group()).lower()


def declared(package: str, extra: str | None) -> set[str]:
    project = tomllib.loads((PROJECTS[package] / "pyproject.toml").read_text())["project"]
    specs: list[str] = [*project.get("dependencies", [])]
    if extra is not None:
        specs += project["optional-dependencies"][extra]
    return {distribution(spec) for spec in specs} | {distribution(package)}


def imported(paths: list[Path]) -> set[str]:
    """Full dotted names, with each ``from`` import's name appended, since it may be a submodule."""
    names: set[str] = set()
    for path in paths:
        for node in ast.walk(parse(path)):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return {name for name in names if name.split(".")[0] not in sys.stdlib_module_names}


def installed_modules() -> dict[str, set[str]]:
    """Each module in the installed file lists, leaving out namespaces such as gcloud.aio."""
    modules: dict[str, set[str]] = {}
    for found in distributions():
        owner = distribution(found.name)
        for file in found.files or []:
            name = inspect.getmodulename(file.name)
            if name is not None:
                parts = file.parent.parts if name == "__init__" else (*file.parent.parts, name)
                modules.setdefault(".".join(parts), set()).add(owner)
    return modules


def providers(name: str, modules: dict[str, set[str]]) -> set[str]:
    """The longest installed prefix decides, so gcloud.aio.auth needs gcloud-aio-auth declared."""
    parts = name.split(".")
    for depth in range(len(parts), 0, -1):
        if found := modules.get(".".join(parts[:depth])):
            return found
    return set()


@pytest.mark.parametrize(
    ("package", "subpackage", "extra"),
    [
        ("openknowledge_contracts", None, None),
        ("openknowledge_core", None, None),
        ("openknowledge_adapters", None, None),
        ("openknowledge_adapters", "semantic_scholar", "semantic-scholar"),
        ("openknowledge_adapters", "gcs", "gcs"),
        ("openknowledge_adapters", "clickhouse", "clickhouse"),
        ("openknowledge_adapters", "batch", "batch"),
        ("openknowledge_adapters", "opensearch", "opensearch"),
        ("openknowledge_pipelines", None, None),
    ],
)
def test_every_import_is_a_declared_dependency(
    package: str, subpackage: str | None, extra: str | None
) -> None:
    """Each adapter subpackage may import only its own extra, and the package root none."""
    root = source(package)
    if subpackage is not None:
        paths = sorted((root / subpackage).rglob("*.py"))
    elif package == "openknowledge_adapters":
        paths = [root / "__init__.py"]
    else:
        paths = sorted(root.rglob("*.py"))
    allowed = declared(package, extra)
    modules = installed_modules()
    for name in imported(paths):
        provided = providers(name, modules)
        if not provided and name.startswith("openknowledge_"):
            provided = {distribution(name.split(".")[0])}
        assert provided & allowed, f"{root.name}/{subpackage or ''} imports undeclared {name}"


def test_nothing_exists_outside_the_allowed_sets() -> None:
    projects = {
        path.parent.relative_to(ROOT).as_posix()
        for path in [*ROOT.glob("*/pyproject.toml"), *ROOT.glob("packages/*/pyproject.toml")]
    }
    assert projects == ALLOWED_PROJECTS
    assert subpackages(source("openknowledge_core")) <= ALLOWED_FEATURES
    assert subpackages(source("openknowledge_adapters")) <= ALLOWED_ADAPTERS
    adapters = tomllib.loads((PROJECTS["openknowledge_adapters"] / "pyproject.toml").read_text())
    assert set(adapters["project"]["optional-dependencies"]) <= ALLOWED_EXTRAS
