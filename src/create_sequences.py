from pathlib import Path
import yaml

with open("configs/dataset.yml", "r") as f:
    cfg = yaml.safe_load(f)

train_sequences = len(
    list(Path(cfg["paths"]["train"]["annotations"]).glob("*.csv"))
)

test_sequences = len(
    list(Path(cfg["paths"]["test"]["annotations"]).glob("*.csv"))
)

print("Train sequences:", train_sequences)
print("Test sequences:", test_sequences)
