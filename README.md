承知いたしました。ご指定のディレクトリ構造の明確化を反映した `README.md` の更新版と、その新しい仕様に基づいてCopilotに作業を指示するためのプロンプトを作成しました。

### 1. 更新版 README.md

ご提示いただいた内容を統合し、構造を整理しました。「2. System Architecture」の中にハードウェア構成などの説明を残しつつ、「2.1. Directory Structure」を追加しています。

```markdown
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

### 2.1. Directory Structure & Data Locations (重要)
本プロジェクトは、AgentのコードとBCI研究の本番データ/コードが別ディレクトリに分かれています。Docker環境から本番データにアクセスするには、適切なボリュームマウントが必要です。

- **Agent Root (`/home/kawamura/agent`)**
  - **内容:** Agentのソースコード、`docker-compose.yml`、`STATUS.md`
  - **Dockerマウント:** コンテナ内の `/app` にマウントされています。
  - **役割:** 実験の制御、ログの収集。

- **BCI Project Root (`/home/kawamura/bci_project/braindecodetest`)**
  - **内容:** BCI 2020用の本番ソースコード、および大規模データセット（`data/`）
  - **Dockerマウント:** **デフォルトではマウントされていません。**
  - **役割:** 学習データの提供。本番実験を行う際は、このディレクトリをコンテナ（例: `/mnt/bci_source`）にマウントし、そこからデータを読み込む必要があります。

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

* `overrides`: `SUPPORTED_KEYS`（seed, n_splits, batch_size, epochs, patience, lr, cache_root, subject_ids、AUG_PROB など補助パラメータ）以外を受け付けません。ホワイトリスト違反は `ValueError` を返します。
* `run_label`: `runs/eeg/<timestamp>_<label>` 名の末尾文字列。
* `dry_run`: `True` のときは config を書いて `planned_cmd` を返すのみで実行しません。
* `python_executable`: 特定の Python バイナリを指定できます（デフォルトは `TARGET_PYTHON`）。

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

* `run_dir`: 実行アーティファクトの保存先（例: `runs/eeg/20251229T154318_smoke_test`）
* `config_path`: 書き出された JSON
* `planned_cmd`: 実行されるコマンド
* `log_path`: `train.log`（stdout/stderr を蓄積）
* `stdout_tail`, `stderr_tail`: 末尾ログ
* `exit_code`: 実行コード（dry_run は `None`）
* `metrics`: 正規表現で抽出した `overall_mean_acc` など
* `error`: 実行中の例外発生時に文字列で記録

この辞書を公式ステータスとして扱い、次のアクションや報告ではそのまま引用してください。

## セキュリティ／堅牢性

* `SUPPORTED_KEYS` 以外の `overrides` キーは拒否されるため、LLM の任意コマンド挿入を防ぎます。
* `train_cv.py` 実行は常に `/data/kawamura/miniforge3/envs/bci2020/bin/python`（`TARGET_PYTHON`）から行い、信頼済み環境で制御します。

```

---

### 2. Copilot向けプロンプト

READMEの更新内容を踏まえ、Copilotに「ボリュームマウントの設定」と「本番データへのパス切り替え」を指示するプロンプトです。

```markdown
# タスク: 本番データ環境の統合と検証

README.md が更新され、AgentディレクトリとBCIプロジェクトディレクトリの関係性（Section 2.1）が明確化されました。
この仕様に基づき、ダミーデータ運用から本番データ運用へ移行するため、以下の手順を実行してください。

## 目的
ホスト側のBCIプロジェクトルート（`/home/kawamura/bci_project/braindecodetest`）をコンテナにマウントし、`train_cv.py` から実データを参照可能にする。

## 実行手順

### Step 1: docker-compose.yml の修正
`docker-compose.yml` を編集し、以下のボリュームマウントを追加してください。
- **Host Path:** `/home/kawamura/bci_project/braindecodetest`
- **Container Path:** `/mnt/bci_source`
- **Options:** `read_only: true` (または `:ro`)

編集後、`docker compose up -d` を実行してコンテナに変更を適用してください。

### Step 2: 学習スクリプトの修正 (`bci_code/train_cv.py`)
`bci_code/train_cv.py` を修正し、マウントされたパスからデータを読み込むロジックを実装してください。

1. **データルートの優先順位:**
   - 優先: `/mnt/bci_source/data/preprocessed_mixed_256hz_v2` (READMEのBCI Project Rootに基づく)
   - フォールバック: 現在のSynthetic Data生成ロジック（データが見つからない場合のみ警告を出して使用）
2. **ログ出力:**
   - どちらのデータソースを使用しているか、INFOレベルでログに出力してください（例: `INFO | Loading Real Data from /mnt/bci_source/...`）。

### Step 3: 動作検証
`tools.eeg_experiment` を使用して動作確認を行ってください。

1. **Dry Run:** `dry_run=True` で実行し、エラーが出ないことを確認。
2. **Smoke Test:** `dry_run=False`, `epochs=1` で実行し、ログに「Real Data」の読み込み成功メッセージが出ることを確認してください。

**注意:** 作業中は `README.md` の "2.1. Directory Structure & Data Locations" を常に参照してください。

```