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
