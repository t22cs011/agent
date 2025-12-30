あなたはBCI研究のパートナーです。続きのタスクを行います。

【システム環境】
- Server: Cygnus (Ubuntu, GPU 24GB)
- Stack: LangGraph + Ollama (DeepSeek-R1:14b)
- Env: `agent_env`（Controller）が `bci2020`（Worker）をサブプロセスで呼び出す構成

【現在の状況 (STATUS.md)】
【システム環境】
- Server: Cygnus (Ubuntu, GPU 24GB)
- Stack: LangGraph + Ollama (DeepSeek-R1:14b)
- Env: agent_env により bci2020 をサブプロセスで呼び出す構成

【実装済み】
- LangGraph の SystemMessage に `tools.eeg_experiment` の仕様を注入
- `tools/eeg_experiment.run_eeg_experiment` で dry_run/実行/ログ尾/`overall_mean_acc` 抽出を一貫提供
- README とドキュメントに dry-run/返り値/ホワイトリスト制約を記載

【次のアクション】
1. `tools.eeg_experiment` で `dry_run=False`＋ `subject_ids` を絞って `overall_mean_acc` を確認
2. LangGraph 対話で dry_run → 本番実行の順が守られているかレビュー
3. 実行結果を `runs/eeg` に保存し、状態報告に使う

【実装済みの主要コード】
- `main.py`: `initial_state` の先頭に SystemMessage を差し込み、Copilot に dry_run → 本番実行のフローを守るよう促しています。
- `tools/eeg_experiment.py`: `SUPPORTED_KEYS` で override を制限し、`dry_run`/実行/ログストリーム/`overall_mean_acc` を返す具現化。

# EEG解析自律エージェント向けツールガイド

## 1. Project Overview (研究背景)

- **目的:** EEG Conformer を用いた言語デコーディング（EEG-based Language Decoding）のために、仮説立案〜実験実行〜結果分析を自律的に回せる研究サイクルを構築する。
- **目標:** 2026年修士課程進学に向け、試行錯誤を自律的に行う「階層型マルチエージェントシステム」を整備し、毎週のトレーニング実験と評価を継続的に完遂できるようにする。
- **ユーザー:** Shungook（Future BCI Product Manager）として、エージェントを「意思決定と監督」に集中させながら、重めの学習処理は安全に Worker に委譲する。

## 2. System Architecture (システム構成)

本プロジェクトは Controller（意思決定）と Worker（実験実行）を明確に分離した二層構造です。

- **Hardware:** Server `Cygnus`（Ubuntu、Titan RTX 24GB VRAM）
- **Controller (Agent):**
  - **Stack:** LangGraph + Ollama（`deepseek-r1:14b`）
  - **Environment:** `agent_env`（軽量な意思決定ロジック）
  - **Role:** 実験計画の立案、コード生成、ステータス報告
- **Worker (Experiment):**
  - **Stack:** Braindecode / PyTorch
  - **Environment:** `bci2020`（既存研究環境）
  - **Role:** [`tools.eeg_experiment.run_eeg_experiment`](tools/eeg_experiment.py) を介して `train_cv.py` を起動し、学習・評価を実行
  - **Interface:** JSON 設定ファイルを受け取り、ログから抽出された Metrics（例: Accuracy）を返す

Controller は `tools.eeg_experiment` とのやり取りを通じて、実験の dry-run → 実行 → メトリクス確認というサイクルを実装します。Worker 側はホワイトリスト化されたパラメータと隔離された Python 環境（`TARGET_PYTHON`）でのみ動作します。

## Keeping GPU services alive via tmux
GPU が使えない原因は [`main.py`](main.py) の `preflight_checks` で `nvidia-smi` が見つからない・GPU が検出されない事にあります。Cygnus 上の GPU ドライバや Ollama サービスが落ちているとこのチェックが失敗するため、以下のように `tmux` で常駐させておきます。

1. `tmux new -s cygnus-gpu` で新しいセッションを作り、`sudo systemctl start nvidia-persistenced`（または `sudo modprobe nvidia` が必要ならその後）→ `nvidia-smi` を実行して GPU を確認。ログを閉じずに `Ctrl-b d` でデタッチしておきます。
2. Ollama サービスも `tmux new -s ollama` で立ち上げ、`OLLAMA_MODELS=/data/kawamura/.ollama ollama serve` を実行し続けることで `main.py` の `service_available` チェックを常に満たします。
3. `nvidia-smi` の出力にデバイスが表示されるまで 1→2 を繰り返し、`main.py` の `use_gpu` が `True` になるのを確認してください（`preflight_checks` のログに `Greaph Ready` などが出ます）。

このように `tmux` で GPU ドライバ系サービスと Ollama サーバを常駐させ、定期的に `nvidia-smi` を叩いて GPU 状態を確認することで、今後 GPU が使えない状態を防げます。
## `run_eeg_experiment` の使い方
```python
from typing import Any, Dict, Optional
from pathlib import Path

def run_eeg_experiment(
    overrides: Dict[str, Any],
    *,
    run_label: Optional[str] = None,
    dry_run: bool = False,
    python_executable: Optional[str] = None,
    script_path: Path = DEFAULT_SCRIPT_PATH,
    timeout: Optional[float] = None,
    workspace_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    ...
```
- `overrides`: `SUPPORTED_KEYS`（seed, n_splits, batch_size, epochs, patience, lr, cache_root, subject_ids、AUG_PROB など補助パラメータ）以外を受け付けません。ホワイトリスト違反は `ValueError` を返します。
- `run_label`: `runs/eeg/<timestamp>_<label>` 名の末尾文字列。
- `dry_run`: `True` のときは config を書いて `planned_cmd` を返すのみで実行しません。
- `python_executable`: 特定の Python バイナリを指定できます（デフォルトは `TARGET_PYTHON`）。

### Dry-run の例
```bash
python - <<'PY'
from tools import eeg_experiment

plan = eeg_experiment.run_eeg_experiment(
    {
        "epochs": 1,
        "batch_size": 64,
        "subject_ids": [1],
    },
    run_label="smoke",
    dry_run=True,
)
print(plan["planned_cmd"])
print(plan["config_path"])
PY
```
この段階で config JSON、`run_dir`、`planned_cmd` を確認し、`dry_run=False` でのみ `train_cv.py` を起動してください。

### 戻り値の構造
- `run_dir`: 実行アーティファクトの保存先（例: `runs/eeg/20251229T154318_smoke_test`）
- `config_path`: 書き出された JSON
- `planned_cmd`: 実行されるコマンド
- `log_path`: `train.log`（stdout/stderr を蓄積）
- `stdout_tail`, `stderr_tail`: 末尾ログ
- `exit_code`: 実行コード（dry_run は `None`）
- `metrics`: 正規表現で抽出した `overall_mean_acc` など
- `error`: 実行中の例外発生時に文字列で記録

この辞書を公式ステータスとして扱い、次のアクションや報告ではそのまま引用してください。

## セキュリティ／堅牢性
- `SUPPORTED_KEYS` 以外の `overrides` キーは拒否されるため、LLM の任意コマンド挿入を防ぎます。
- `train_cv.py` 実行は常に `/data/kawamura/miniforge3/envs/bci2020/bin/python`（`TARGET_PYTHON`）から行い、信頼済み環境で制御します。

【今回の依頼】
STATUS.md にある「次のアクション」を実行したいので、サポートしてください。