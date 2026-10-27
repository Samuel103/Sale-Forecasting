from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yml"

with CONFIG_PATH.open(encoding="utf-8") as config_file:
    settings = yaml.safe_load(config_file)

model_settings = settings["model"]
group_cols = settings["preprocessing"]["group_cols"]
categorical_cols = settings["preprocessing"]["categorical_cols"]
feature_cols = settings["preprocessing"]["feature_cols"]
inference_days = settings["preprocessing"]["inference_days"]
general = settings["general"]
if inference_days is not None and (type(inference_days) is not int or inference_days <= 0):
    raise ValueError("preprocessing.inference_days must be a positive integer or null.")
model_export_path = Path(settings["training"]["model_export_path"]).expanduser()
if not model_export_path.is_absolute():
    model_export_path = PROJECT_ROOT / model_export_path
training_data_path = Path(settings["training"]["data_path"]).expanduser()
if not training_data_path.is_absolute():
    training_data_path = PROJECT_ROOT / training_data_path
