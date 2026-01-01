# EEG解析自律エージェント向けツールガイド

## 1. Project Overview (研究背景)
- **目的:** EEG Conformer を用いた言語デコーディング（EEG-based Language Decoding）のために、仮説立案〜実験実行〜結果分析を自律的に回せる研究サイクルを構築する。
- **目標:** 2026年修士課程進学に向け、試行錯誤を自律的に行う「階層型マルチエージェントシステム」を整備し、毎週のトレーニング実験と評価を継続的に完遂できるようにする。
- **ユーザー:** Shungook（Future BCI Product Manager）として、エージェントを「意思決定と監督」に集中させながら、重めの学習処理は安全に Worker に委譲する。

## 2. System Architecture (システム構成)
本プロジェクトは Controller（意思決定）と Worker（実験実行）を明確に分離した構成です。

- **Hardware:** Server `Sirius` (Ubuntu, RTX 3090 24GB)
- **Infrastructure:** Docker Compose (Worker + MLflow)
- **Controller (Agent):**
  - **Stack:** LangGraph + Ollama (`deepseek-r1:14b`)
  - **Role:** 実験計画、コード生成、ステータス管理 (`STATUS.md`)
- **Worker (Experiment):**
  - **Container:** `bci` (PyTorch 2.x, Braindecode, CUDA 12.1)
  - **Role:** [`tools.eeg_experiment`](tools/eeg_experiment.py) 経由で `docker exec` され、学習を実行。
- **Logger (Experiment Tracking):**
  - **Service:** `mlflow_server` (Local Host)
  - **Role:** 実験パラメータ、メトリクス (Accuracy, Loss)、モデルの記録・可視化。

### 2.1. Directory Structure & Data Locations (重要)
Agentコードと研究用データセットは別ディレクトリにあり、Dockerのボリュームマウントで接続します。

- **Agent Root (`/home/kawamura/agent`)**
  - **Docker Mount:** `/app`
  - **内容:** Agentソース, `docker-compose.yml`, `STATUS.md`
- **BCI Project Root (`/home/kawamura/bci_project/braindecodetest`)**
  - **Docker Mount:** `/mnt/bci_source` (Read-Only)
  - **内容:** 本番用ソースコード (`bci_code/`), データセット (`data/`)
  - **注意:** 学習スクリプトは `/mnt/bci_source/data/...` からデータを読み込む必要があります。

## 3. Workflow & Usage

### 3.1. エージェントによる実験実行
Agent は `tools/eeg_experiment.py` を使用して Docker コンテナ内の学習スクリプトを呼び出します。

```python
# Agent内部での呼び出しイメージ
run_eeg_experiment(
    overrides={"epochs": 10, "batch_size": 32},
    dry_run=False
)
# -> 内部で `docker exec bci python bci_code/train_cv.py ...` が実行される
```

## 4. ここまでの作業履歴と今後のタスク

### 実施済みの主な作業
- Docker コンテナ化: `Dockerfile` と `docker-compose.yml` を用いて `bci` サービスを構築・起動しました。
- `main.py` の互換性修正: Python の型表記や `WORKSPACE_DIR` を環境変数で制御するよう修正しました。
- Ollama モデルの取り扱い: ホストで pull した Ollama モデルをコンテナに `/mnt/ollama` としてマウントし、非 root ユーザー (UID=10151) が参照できるようにしました。
- ランタイム依存の修正: 実行中のコンテナに `langchain-ollama`, `langchain-core`, `langgraph`, `ollama` 等を順次インストールして依存エラーを解消しました。
- ネットワーク調整: ホストの Ollama サービス（127.0.0.1:11434）へコンテナから接続できるよう `network_mode: host` を一時的に適用し、動作確認を行いました。
- 恒久化: 依存をインストールしたコンテナを `agent-bci-llm:latest` としてコミットし、`docker-compose.yml` をこのイメージを参照するよう更新しました。

### 現在の状態（要点）
- `main.py` はフルエージェントモードで起動し、ホストの Ollama サービス上のモデル `deepseek-r1:14b` を利用可能です。
- ただし現在は `docker-compose.yml` に `network_mode: host` を使っているため、環境依存性とセキュリティ上の注意が必要です。
- 大きな科学系パッケージをビルド時に pip/conda で一括導入するとビルド中にメモリ不足やセグフォルトが発生することがあり、今回は実行中コンテナへ手動でインストールしてイメージへコミットする対応を取りました。

### 今後の推奨タスク（優先順位付き）
1. （推奨）CI/ビルドマシンで `INSTALL_SCIENCE_PACKAGES=true` / `INSTALL_LLM_PACKAGES=true` を指定して `docker build` を行い、依存を Docker イメージへ焼き込む（ビルド用に十分なメモリが必要）。
   - コマンド例:
   ```bash
   docker build --build-arg INSTALL_SCIENCE_PACKAGES=true --build-arg INSTALL_LLM_PACKAGES=true -t agent-bci-llm:latest .
   ```
2. （運用）`agent-bci-llm:latest` をプライベートレジストリへ push して、他ホストから再現可能にする。
3. Ollama サービスの運用自動化: systemd ユニットまたは専用コンテナで Ollama を立ち上げ、`network_mode: host` に依存しない構成を作る。
4. エージェントの End-to-End テスト: `tools/eeg_experiment` を介した学習実行フローを実データで検証し、`mlflow_server` 連携を確認する。
5. `docker-compose.yml` のドキュメント化: 現在の `network_mode: host` 選択理由と、元に戻す手順を README に追記する。

必要であれば上記タスクを順に実行して差し上げます。変更履歴や追加の補足が必要なら教えてください。
