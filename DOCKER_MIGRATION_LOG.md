# Docker Environment Migration Log
**Date:** 2025-12-30
**Server:** Sirius (Ubuntu / RTX 3090)
**Author:** Shungook (with AI Assistant)

## 1. プロジェクト概要
EEG Conformerを用いた言語デコーディング研究（BCI Competition 2020 Track 3）の実験環境を、従来のConda環境から再現性の高いDocker環境へ移行した。

## 2. 達成されたアーキテクチャ
- **Controller (Host):** Agent (LangGraph + Ollama)
  - 役割: 実験計画の立案、Dockerコンテナへのコマンド発行
- **Worker (Container):** `bci` (PyTorch + Braindecode)
  - 役割: 計算リソース（GPU）を使用した実際の学習処理
  - ベースイメージ: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`

## 3. 実施作業まとめ

### A. 環境定義 (Docker化)
1. **`requirements_docker.txt` の作成**
   - 既存の `environment_bci2020.yml` から、学習に必要な主要ライブラリ（braindecode, moabb, mne, scikit-learn等）のみを厳選して抽出。
   - バージョン不整合を防ぐため、主要ライブラリのバージョンを固定。
2. **`Dockerfile` の構築**
   - NVIDIA PyTorch公式イメージを採用し、CUDA 12.4環境を構築。
   - 日本時間（Asia/Tokyo）へのタイムゾーン設定を追加。
   - `pip` インストール時のセグメンテーション違反（Sirius固有の問題）を回避するため、インストール手順を分割。
3. **`docker-compose.yml` の整備**
   - サービス名: `bci`
   - GPUリソース（NVIDIA Driver）のパススルー設定。
   - ホストのカレントディレクトリをコンテナ内の `/app` にマウント。
   - PyTorchのDataLoader用に `shm_size: 8gb` を確保。

### B. Agentツール改修 (`tools/eeg_experiment.py`)
1. **実行コマンドの変更**
   - `subprocess` で直接 `python` を呼ぶ方式から、`docker exec` を経由する方式へ変更。
2. **パス問題の解決**
   - ホスト側の「絶対パス」を、コンテナ内（`/app` 起点）の「相対パス」に自動変換するロジックを実装。
   - 古いハードコードされたパスを除去。
3. **権限問題 (NFS) の解決**
   - コンテナ内でのファイル生成が `root` 所有になるのを防ぐため、`docker exec` に `-u $(id -u):$(id -g)` オプションを動的に付与するよう修正。

### C. サーバー環境設定 (Sirius)
1. **NVIDIA Container Toolkit の導入**
   - DockerコンテナからGPUを認識させるため、ドライバとツールキットをインストールし、Dockerデーモンを再起動。
2. **権限設定**
   - ユーザーを `docker` グループに追加（`sudo` なしでの実行用）。

## 4. 検証結果 (Smoke Test)
- **コマンド:** Agentから1エポック分の学習ジョブを投入。
- **結果:**
  - ✅ Dockerコンテナの自動起動と接続
  - ✅ コンテナ内でのCUDA (RTX 3090) 認識
  - ✅ Agentによる実験結果（Accuracy）の受け取り
- **特記事項:**
  - コンテナ直下へのファイル書き込み権限エラーが出たが、本番データの出力先（`runs/`）への書き込みはユーザーマッピングにより保証されている。

## 5. 次のステップ
- ダミーの `bci_code/train_cv.py` を、実際のBraindecodeを用いた本番用学習コードに置き換える。
- Agentによる自律的なパラメータ探索実験を開始する。