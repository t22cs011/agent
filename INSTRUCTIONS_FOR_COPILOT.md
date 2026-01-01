🤖 Instructions for Next Development Session

このドキュメントは、次の開発セッション（Copilot/Agent）へ向けた引き継ぎ資料です。
現在の実装状況、発生している問題、および次に実装すべき機能（Step 1）をまとめています。

🎯 Main Goal: "Step 1 - 評価ロジックの強化"

現在のエージェントは「エラーが出ずに終了すれば成功」と判断しますが、これを**「目標精度（Accuracy）が出るまで試行錯誤する」**ように改良してください。

具体的な実装タスク

main.py の executor_node を改良する:

実行したスクリプト（train_cv.py 等）の標準出力（stdout）から、精度（例: Accuracy: 0.85）を正規表現で抽出する処理を追加する。

抽出したメトリクスを state に保存する。

should_continue のロジックを変更する:

単なる Exit Code 0 ではなく、「Accuracy が目標値（例: 0.8）を超えたか？」を判定基準にする。

目標未達の場合は、LLMに「現在の精度はXでした。目標Yに届くようにハイパーパラメータ（lr, epoch等）を修正してください」というフィードバックを返し、ループを継続させる。

DeepSeek-R1 対策（"No code block found" 回避）:

coder_node の SystemPrompt を強化し、「必ず python ...  ブロックを出力すること」「思考（thinking）だけでなく実装コードを含めること」を強く指示する。

必要であれば正規表現を緩和し、コードブロックの抽出精度を上げる。

🛠️ Environment Troubleshooting Guide (Required Fixes)

現在、TARGET_PYTHON のパスが見つからないエラーが頻発しています。
エージェントを動かす前に、以下の手順でホスト側の Conda 環境を確実に整備してください。

1. Verify Conda Environment Existence

List all Conda environments to check if bci2020 exists.

conda env list


If the environment is not listed, create it with the desired Python version.

conda create --name bci2020 python=3.8 -y


2. Activate the Conda Environment

Ensure you are working within the bci2020 environment before proceeding.

conda activate bci2020


3. Check Python Installation in the Environment

After activating, verify that Python is installed correctly by checking its version.

python --version


Ensure the path to Python within the environment is correct.

which python


Target Output: /data/kawamura/miniforge3/envs/bci2020/bin/python (Adjust based on your actual path)

4. Set TARGET_PYTHON Environment Variable

Confirm that TARGET_PYTHON in main.py or .env points to the correct Python executable found in step 3.

export TARGET_PYTHON=$(which python)
echo $TARGET_PYTHON


5. Check File Permissions

Ensure the user has read/execute permissions for the Python executable.

ls -l $TARGET_PYTHON
chmod 755 $TARGET_PYTHON


6. Install Required Packages

The environment needs the dependencies to run the experiments.

pip install torch numpy braindecode  # Add other requirements from requirements_docker.txt


📂 Current Implementation Status

(Refer to IMPLEMENTATION_STATUS.md for details)

Current Level: 2 (Self-correction on syntax errors)

Next Level: 3 (Metric-based self-correction)

Please proceed with Step 1 implementation based on the above.