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

## コンテナ運用の最短手順

- **開発者向け（sudo なし）**
  - 起動: `docker compose up -d --build`
  - 停止: `docker compose down`
  - ログ閲覧: `docker compose logs -f bci`
- **運用担当向け（sudo/systemd 利用）**
  - リポジトリ同梱の `systemd/agent-bci.service` を `/etc/systemd/system/` に配置し、`sudo systemctl daemon-reload` を実行
  - 起動・自動起動有効化: `sudo systemctl enable --now agent-bci`
  - ログ閲覧: `sudo journalctl -u agent-bci -f`

## CI/CD & Heavy Image Build

- `mamba` と重いサイエンス系パッケージはビルドメモリを多く消費するため、可能なら Self-Hosted Runner で `.github/workflows/ci-build-image.yml` を実行
- Self-Hosted が無い場合でも、ビルドマシンは 16GB 以上の RAM を推奨
- 本番用イメージでは `INSTALL_SCIENCE_PACKAGES=true` を build-arg で指定し、必要な依存をイメージに焼き込む

## Debugging Handbook

- 状態確認: `docker compose ps`
- ログ確認: `docker compose logs -f bci`（systemd 運用時は `journalctl` でも可）
- コンテナへ入る: `docker exec -it <container_id> /bin/bash`
- Python 環境確認（コンテナ内）: `python --version` と `pip list | grep mne` で主要依存が揃っているか確認

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

## Quick Start — 確実に起動するための手順

以下はこのリポジトリをホスト上で確実に起動するための短い手順です。環境に応じた調整（ユーザ名、パス、コンテナ名など）をしてください。

1. ホストで Ollama を起動し、目的のモデル（例: `deepseek-r1:14b`）を pull しておく:

```bash
# Ollama サービスが systemd 管理されている場合
sudo systemctl start ollama
ollama pull deepseek-r1:14b
ollama list
```

2. （推奨）依存を含むイメージをビルドする（ビルドマシンに十分なメモリが必要）:

```bash
# 重いサイエンス依存や LLM クライアントをイメージへ焼き込みたい場合
docker build \
  --build-arg INSTALL_SCIENCE_PACKAGES=true \
  --build-arg INSTALL_LLM_PACKAGES=true \
  -t agent-bci-llm:latest .
```

3. 既存イメージを使う場合は `docker compose up` で `bci` サービスを起動する（`network_mode: host` を使用している場合、ホストの Ollama に直接接続できます）:

```bash
docker compose up -d --force-recreate bci
docker compose ps
```

4. コンテナ内で `main.py` を実行してエージェントが `System Ready!` になることを確認する:

```bash
container=$(docker compose ps -q bci)
docker exec -it "$container" /bin/bash -c "python3 /app/main.py"
# プロンプトが表示され、Model: deepseek-r1:14b が表示されれば OK
```

5. もしコンテナ側で言語モデルクライアント等が欠けている場合（`No module named 'langgraph'` 等）、コンテナ内で一時的にインストールし、動作確認後イメージをコミットすると再現性が高まります:

```bash
docker exec -it "$container" /bin/bash
pip install langgraph langchain-ollama langchain-core
# 動作確認後、コンテナをコミット
docker commit "$container" agent-bci-llm:latest
```

## トラブルシュート（よくある問題と対処）

- Ollama の `500` エラー / ランナープロセス終了:
  - まずホスト側のログを確認してください:
    ```bash
    sudo journalctl -u ollama -n 500 --no-pager
    sudo tail -n 500 ~/.ollama/serve.log || true
    sudo dmesg | grep -i -E 'oom|killed' || true
    ```
  - GPU メモリ不足なら、動作中の他プロセスを停止するかモデルの quantize レベルを見直してください。

- `No module named 'langgraph'` など Python モジュールエラー:
  - ホストで `TARGET_PYTHON` に指定した環境（例: `bci2020`）を作成し依存をインストールするか、コンテナ内で `pip install` してイメージをコミットしてください。

- ホストの Ollama にコンテナから接続できない:
  - 一時的に `network_mode: host` を使うか、`extra_hosts` / ポートフォワーディングを設定してください（`127.0.0.1` バインドの制限を回避するため）。

## 確認コマンド（起動後のチェックリスト）

```bash
# Ollama が応答するか
curl -sS http://127.0.0.1:11434/health
ollama list

# コンテナ側で agent が起動しているか
docker compose ps
docker logs $(docker compose ps -q bci) --tail 200

# エージェントから LLM へ問い合わせテスト
docker exec -it $(docker compose ps -q bci) /bin/bash -c "python3 /app/main.py" # -> USER> で質問
```

## 権限と systemd の設置について（重要）

### 概要
`systemd` ユニット (`systemd/agent-bci.service`) を `/etc/systemd/system` に配置して有効化するには管理者権限が必要です。リポジトリ内のユニットファイルは `/home/kawamura/agent/systemd/agent-bci.service` にあります。

### 対処方法（選択肢）

- 1) 管理者が直接実行する（推奨）

  管理者は次を実行してください:
  ```bash
  sudo cp /home/kawamura/agent/systemd/agent-bci.service /etc/systemd/system/agent-bci.service
  sudo systemctl daemon-reload
  sudo systemctl enable --now agent-bci.service
  sudo systemctl status agent-bci.service --no-pager
  ```

- 2) 管理者に依頼する（管理者に上のコマンド列を渡す）

- 3) `testuser` に sudo 権限があり、そのアカウントを使える場合

  ```bash
  su - testuser
  sudo cp /home/kawamura/agent/systemd/agent-bci.service /etc/systemd/system/agent-bci.service
  sudo systemctl daemon-reload
  sudo systemctl enable --now agent-bci.service
  ```

- 4) `kawamura` を sudoers に追加してもらう（管理者が実行）

  ```bash
  sudo usermod -aG sudo kawamura
  # その後、kawamura でログインし直して上の管理者手順を実行
  ```

- 5) root/sudo が使えない場合の代替（手動で常時起動させる）

  systemd を使わずに Docker Compose のまま永続化する方法:
  ```bash
  # コンテナを起動
  docker compose up -d bci
  # 自動再起動ポリシーを設定
  docker update --restart=unless-stopped $(docker compose ps -q bci)
  ```

### 推奨フロー
1. 管理者にユニット設置を依頼する（最も確実）。
2. すぐに使いたい場合は、5) の方法で Docker Compose のまま永続化する（`systemd` なし）。

必要なら、こちらで管理者用のコマンド列を整形して渡すテンプレートを作成します。

## 今回行った変更と現在の構成（要約）

- 変更点
  - `main.py` を改良して、LLM の出力から精度を抽出・保存し、目標精度に達するまで自己修正フローを回せるようにしました（`TARGET_ACC` で閾値設定可）。
  - `Dockerfile` を修正し、重いサイエンスパッケージは `mamba` を用いた分割インストール／ビルド引数で制御できるようにしました。`ARG INSTALL_SCIENCE_PACKAGES`／`ARG INSTALL_EEG_PACKAGES` によりビルド時インストールを切替可能です。
  - `docker-compose.yml` を更新して、ローカルビルドを有効化し、`restart: unless-stopped` を追加しました（サービスは永続起動可能）。
  - コンテナ内で手動インストール→イメージ化するためのスクリプト `scripts/interactive_install_and_commit.sh` を追加しました。
  - systemd ユニット `systemd/agent-bci.service` を追加（管理者が `/etc/systemd/system` に配置して有効化することを想定）。
  - `README.md` に sudo/systemd の注意と手順を追加しました。

- 現在の稼働構成
  - Compose サービス名: `bci`（イメージ: `agent-bci-llm:latest`）
  - コンテナ動作: `docker compose up -d bci`（イメージがなければローカルでビルドされます）
  - 自動再起動: `unless-stopped` を設定済み（`docker update --restart=unless-stopped <container>`）
  - systemd: 管理者が `systemd/agent-bci.service` を `/etc/systemd/system` に配置すれば OS 起動時に自動でコンテナ起動可能

## コンテナが down した場合の対応手順

1. まず状態を確認:
```bash
docker compose ps
docker logs $(docker compose ps -q bci) --tail 200
```

2. コンテナが停止している／Exit 状態の場合は再起動:
```bash
docker compose up -d bci
```

3. 再起動ポリシーが機能していない場合（手動で設定する）:
```bash
docker update --restart=unless-stopped $(docker compose ps -q bci)
```

4. systemd 管理下にある場合は systemd の状態確認／再起動:
```bash
sudo systemctl status agent-bci.service --no-pager
sudo systemctl restart agent-bci.service
sudo journalctl -u agent-bci.service -n 200 --no-pager
```

5. イメージを最新に差し替えたい場合（コミット済みイメージを反映）:
```bash
docker compose pull || true
docker compose up -d --force-recreate bci
```

6. ログや原因が分からない場合は、最近のログ（`docker logs` / `journalctl`）をまず確認し、必要なら `docker exec -it <container> bash` で直接デバッグしてください。



上の手順を `README` に追記しました。必要なら自動化用の起動スクリプト（`scripts/up.sh` など）を作成してお渡しします。

## CI: Build image on a build server (GitHub Actions)

To reliably bake the heavy science and LLM dependencies into an image, use the provided GitHub Actions workflow to build on a larger build host and push to a registry.

- Workflow path: `.github/workflows/ci-build-image.yml`
- Inputs (when triggering `workflow_dispatch`): `image_name`, `image_tag`, `push`, `use_self_hosted`, `install_science`, `install_llm`.

Required secrets:
- For GHCR (recommended): `GITHUB_TOKEN` is used automatically by Actions.
- For Docker Hub: set `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` in repository Secrets.

Trigger example (from Actions UI):
- `image_name`: `ghcr.io/<your-org>/agent-bci-llm`
- `image_tag`: `latest`
- `push`: `true`
- `use_self_hosted`: `true` (if you have a large self-hosted runner)

Notes:
- GitHub-hosted runners have limited RAM; for large builds set `use_self_hosted=true` and run on a machine with sufficient RAM (>=32GB recommended).
- The workflow uses `docker/build-push-action` with build-args `INSTALL_SCIENCE_PACKAGES` and `INSTALL_LLM_PACKAGES` to enable heavy installs.


**Session Handoff**

- **Last agent output (from attaching to container and running `main.py`):**

```
[agent] Tail last 200 lines of bci logs (press Ctrl-C to stop):
[agent] Attaching and running /app/main.py inside container 39c670de74f3c8d50b18399b2416e78a915f57ec2146a0e7cc532adb2accc373
🚀 Initializing Agent with deepseek-r1:14b...
🐍 Target Python: /data/kawamura/miniforge3/envs/bci2020/bin/python
🚀 System Ready! (Model: deepseek-r1:14b)
Type 'exit' or 'quit' to end.

USER> Understand this workspace and summarise what's inside.
```

- **Implemented (what the current workspace / image can do):**
  - **Agent runtime:** Container `bci` can run `python3 /app/main.py` and reach `System Ready! (Model: deepseek-r1:14b)`.
  - **Ollama integration:** Host Ollama is reachable from the container (we used `network_mode: host`); `ollama run deepseek-r1:14b 'Hello'` returns a valid text response.
  - **Docs & automation:** `README.md` updated with Quick Start and Troubleshooting; `scripts/up.sh` was added to automate build/pull/up/attach flows.
  - **Manual fixes:** Missing Python/LangGraph dependencies were identified and can be installed inside the running container; a committed image `agent-bci-llm:latest` exists in the workflow.

- **Not implemented / Known issues (what still needs attention):**
  - **`scripts/up.sh` permissions:** Created but the non-root `testuser` cannot mark it executable in-place (permission denied). Fix requires owner/permission change by a user with appropriate rights.
  - **Trusted Python path missing:** The runtime shows `TARGET_PYTHON`/agent expects `/data/kawamura/miniforge3/envs/bci2020/bin/python` which does not exist in the runtime environment; this causes executor tasks (e.g., `run_eeg_experiment`) to fail with `No such file or directory` errors.
  - **Incomplete image bake:** Large science and LLM packages are not fully baked into a reproducible image (they were installed manually in a container and committed). This is fragile for CI/other hosts.
  - **Missing serve.log:** `/home/kawamura/.ollama/serve.log` was not present during log collection; this may limit post-mortem diagnostics.
  - **Operational dependency on host networking:** Current setup relies on `network_mode: host` to access a host-bound Ollama; this reduces portability and increases security considerations.

- **Action items for next Copilot session / handoff checklist:**
  - **Fix `scripts/up.sh` execution permissions:** Either change ownership/permissions or provide a wrapper invocation (e.g., `bash scripts/up.sh`) and document sudo requirements.
  - **Provide or configure `bci2020` Python environment:** Create the conda env on host (or adjust `TARGET_PYTHON`) and add steps to `README` to create/install required packages, or bake them into the Docker image.
  - **Bake final image on CI:** Run `docker build --build-arg INSTALL_SCIENCE_PACKAGES=true --build-arg INSTALL_LLM_PACKAGES=true -t agent-bci-llm:latest .` on a machine with adequate RAM and publish to a private registry.
  - **Add `scripts/down.sh`:** Provide a clean stop/cleanup script to bring services down and optionally remove dangling images/containers.
  - **Automate Ollama startup or document:** Provide a reproducible way to start Ollama (systemd unit or containerized Ollama) so local `127.0.0.1` binding doesn't force host-network mode.
  - **Collect serve.log location / rotate logs:** Ensure Ollama is configured to write `serve.log` to a known path and that log rotate/retention is documented.

- **Quick commands to hand off to the next session**

```bash
# Run the helper (no exec bit required if you use bash explicitly)
bash scripts/up.sh --pull-model --attach

# If you need to set executable bit (run as a user with write perms to the repo):
sudo chown $(whoami):$(id -gn) scripts/up.sh || true
chmod +x scripts/up.sh

# Create the missing conda env example (if TARGET_PYTHON should point here):
conda create -n bci2020 python=3.11 -y
conda activate bci2020
pip install -r requirements_docker.txt

# Bake final image on a build host with enough RAM:
docker build --build-arg INSTALL_SCIENCE_PACKAGES=true --build-arg INSTALL_LLM_PACKAGES=true -t agent-bci-llm:latest .
```

## Recent Recovery Note

- On this workstation we encountered local `.git` permission issues that prevented committing directly into the checked-out repository. To recover, a writable HTTPS clone was created at `/tmp/agent-clone` and all fixes and PR work were performed there.
- A Pull Request (#8) was created from `prepcompexp-clone` and successfully merged into `main` (merge commit: `d15e2cfe...`).
- A CI test run was triggered and completed successfully (example run id: `20637811531`).
- Local uncommitted/staged edits from the original worktree were backed up as patches:
  - `/tmp/staged-changes.patch` (staged changes snapshot)
  - `/tmp/unstaged-changes.patch` (unstaged changes snapshot)
- If you want to re-apply any of the backed-up edits into this repository, run:

```bash
# from repository root
git apply /tmp/staged-changes.patch
# or
git apply /tmp/unstaged-changes.patch
```

This note documents the recovery steps taken during the recent session so future developers understand why a temporary clone was used.

## 5. Recent PR & Communication
- **PR branch:** prepcompexp-clone
- **Action:** PR created to merge runtime fixes, Ollama HTTP detection fallback, `TARGET_PYTHON` env override, and README updates (Japanese quick start & debugging).
- **Suggested next steps:** review the PR, run CI build (`ci-build-image.yml`) on a machine with sufficient RAM, and merge when checks pass.

### Gemini Notification Prompt
以下のプロンプトを Gemini（LLM）へ投げて、今回の変更内容とデプロイ準備の要点を通知してください。

Japanese prompt (copy-and-paste):
"あなたはこのリポジトリの運用担当です。以下の要点を理解して、CIでのイメージビルド実行のためのチェックリストを作成してください：\n- ブランチ: prepcompexp-clone\n- 変更: `main.py` に `TARGET_PYTHON` の環境変数対応と Ollama HTTP フォールバック追加、`docker-compose.yml` に `TARGET_PYTHON=python` を追加、`README.md` に日本語のクイックスタート/デバッグ追記\n- 現在の制約: 一部ファイルの所有権がホストで変更不可（sudo 必要）、CI ビルドは大容量メモリを推奨\n- 期待する出力: (1) CI ビルドで必要な事前チェックリスト、(2) マージ後のデプロイ手順（安全な順序で）、(3) ビルド失敗時に収集すべきログと診断コマンド\n出力は日本語で、箇条書きか短い段落で簡潔に提供してください。"

English prompt (short):
"You are the ops engineer for this repository. Summarize the changes on branch `prepcompexp-clone` and produce: (1) a pre-merge CI checklist, (2) post-merge deploy steps, and (3) clear log/diagnostic commands to run if a build fails. Keep it concise and actionable."


---

This section is intended for the next Copilot session or developer picking up the task; it captures the last observed runtime output and the precise set of implemented vs outstanding items to continue work without re-running discovery steps.
