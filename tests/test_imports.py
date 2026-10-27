import subprocess
import sys
from pathlib import Path

from sale_forecasting.core.config import CONFIG_PATH, PROJECT_ROOT, model_export_path


def test_all_modules_import_outside_project_directory(tmp_path):
    """Importing modules must not require the repository as working directory."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib, pkgutil, sale_forecasting; "
            "modules = pkgutil.walk_packages("
            "sale_forecasting.__path__, sale_forecasting.__name__ + '.'); "
            "[importlib.import_module(module.name) for module in modules]",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_configuration_paths_resolve_from_project_root():
    assert PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert CONFIG_PATH == PROJECT_ROOT / "config" / "config.yml"
    assert CONFIG_PATH.is_file()
    assert model_export_path.is_absolute()
