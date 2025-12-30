【システム環境】
- Server: Cygnus (Ubuntu, GPU 24GB)
- Stack: LangGraph + Ollama (DeepSeek-R1:14b)
- Env: agent_env により bci2020 をサブプロセスで呼び出す構成

【実装済み】
- LangGraph の SystemMessage に `tools.eeg_experiment` の仕様を注入
- `tools/eeg_experiment.run_eeg_experiment` で `train_cv.py` を安全に呼び出す dry_run/実行パスとログ抽出を提供
- README に dry-run 手順と返り値構造、セキュリティ制約を記載

【次のアクション】
1. `tools.eeg_experiment` で `dry_run=False` と `subject_ids` を最小化し、`overall_mean_acc` を拾って実行状況を確認する
2. LangGraph の会話で `dry_run` → `run` の順が守られていることをレビューする
3. 実行結果を status として `tools/runs` 以下で記録・確認する
