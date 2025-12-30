from braindecodetest.experiments import run_eeg_experiment

overrides = {
    "seed": 42,
    "n_splits": 5,
    "batch_size": 32,
    "epochs": 1,  # 設定した epochs
    "patience": 5,
    "lr": 0.001,
    "cache_root": "/home/kawamura/cache",
    "subject_ids": [1],  # 設定した subject_ids
    "AUG_PROB": 0.5,
    "SNR_SEGMENTS": [2, 3],
    "FILTER_ORDER": 4,
}

result = run_eeg_experiment(
    overrides=overrides,
    dry_run=True,  # dry_run モード
    python_executable=None,
)

print("計画されたコマンド:", result["planned_cmd"])
print("config.json のパス:", result["config_path"])