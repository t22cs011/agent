import os
import subprocess
import re
import operator
import sys
import time
from typing import TypedDict, Annotated, List

# --- ライブラリのインポート修正 ---
HAS_FRAMEWORK = True
IMPORT_ERROR_MSG = ''
try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
except Exception as e:
    HAS_FRAMEWORK = False
    IMPORT_ERROR_MSG = str(e)
    # We'll provide a fallback interactive loop later so the script still accepts input.

EEG_TOOL_IMPORT_ERROR: str | None = None
try:
    from tools import eeg_experiment
except Exception as exc:
    eeg_experiment = None  # type: ignore[assignment]
    EEG_TOOL_IMPORT_ERROR = str(exc)

# ==========================================
# ⚙️ 設定セクション
# ==========================================

# 1. 実行部隊 (bci2020) のPythonパス
TARGET_PYTHON = "/data/kawamura/miniforge3/envs/bci2020/bin/python"

# 2. 作業ディレクトリ (Workspace)
WORKSPACE_DIR = "/home/kawamura/agent"

# 3. 使用するLLMモデル
# 思考過程を見るため、DeepSeek-R1系を使用
LLM_MODEL = "deepseek-r1:14b"

# ==========================================

# ディレクトリ初期化
if not os.path.exists(WORKSPACE_DIR):
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    print(f"📁 Created workspace at: {WORKSPACE_DIR}")

print(f"🚀 Initializing Agent with {LLM_MODEL}...")
print(f"🐍 Target Python: {TARGET_PYTHON}")

EEG_TOOL_SPEC = """ツール run_eeg_experiment(overrides: dict, run_label: str | None = None, dry_run: bool = False, python_executable: str | None = None) を使って /home/kawamura/bci_project/braindecodetest/experiments/train_cv.py を呼び出します。
- overrides にはホワイトリスト (seed, n_splits, batch_size, epochs, patience, lr, cache_root, subject_ids など) を使ってパラメータを指定してください。
- 設定や想定コマンドに確信を持つまで、**必ず dry_run=True で実行し、config.json の場所と planned_cmd を確認**してから dry_run=False での実行に移ってください。
- このツールは安全のため trusted python 実行環境 (`TARGET_PYTHON`) を使って train_cv.py をサブプロセスで起動します。
- 返り値は JSON 形式（辞書）で、`run_dir`, `config_path`, `planned_cmd`, `log_path`, `stdout_tail`, `stderr_tail`, `exit_code`, `metrics`, `error` などのキーを含みます。これを公式ステータスとして扱い、次のアクションや報告にも JSON 内容を引用してください。
- `metrics` にはログから抽出された `overall_mean_acc` が含まれます。"""


# --- 事前チェック: GPU と Ollama モデルの存在確認 ---
def run_cmd(cmd: str, env=None, timeout: int = 30):
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=timeout)
        return proc.returncode, proc.stdout + proc.stderr
    except Exception as e:
        return 1, str(e)


def tcp_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    import socket
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False


def start_local_ollama(models_dir: str, logdir: str) -> subprocess.Popen:
    """Start `ollama serve` as a background process using models_dir. Returns Popen or raises."""
    env = os.environ.copy()
    env['OLLAMA_MODELS'] = models_dir
    log_path = os.path.join(logdir, 'ollama_serve.log')
    out = open(log_path, 'a')
    # Start serve in background
    # Ensure models_dir is traversable/writable by this user before starting.
    try:
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
        # quick accessibility check: try listing
        os.listdir(models_dir)
    except PermissionError as e:
        raise PermissionError(f"permission denied for models_dir={models_dir}: {e}")

    cmd = ['ollama', 'serve']
    proc = subprocess.Popen(cmd, stdout=out, stderr=out, env=env)
    # wait for port 11434 to open
    for _ in range(30):
        if tcp_port_open('127.0.0.1', 11434, timeout=1.0):
            details = f'Started local ollama serve (pid={proc.pid}), log: {log_path}'
            return proc
        time.sleep(1.0)
    # if not open, leave proc running and raise
    raise RuntimeError(f'ollama serve did not open port in time, see {log_path}')


def pull_model_to(models_dir: str, model: str) -> bool:
    env = os.environ.copy()
    env['OLLAMA_MODELS'] = models_dir
    code, out = run_cmd(f'OLLAMA_MODELS={models_dir} ollama pull {model}', env=env, timeout=3600)
    # detect permission errors so caller can fallback
    lower = out.lower() if out else ''
    if 'permission denied' in lower or 'ensure path elements are traversable' in lower:
        raise PermissionError(f'Permission denied when pulling model to {models_dir}: {out}')
    return code == 0


def preflight_checks():
    details = {}

    # nvidia-smi の確認（警告のみ）
    code, out = run_cmd('which nvidia-smi')
    details['nvidia_smi_present'] = (code == 0)
    if code == 0:
        code2, out2 = run_cmd('nvidia-smi')
        details['nvidia_smi_output'] = out2
        if 'No devices were found' in out2 or 'Unable to determine the device handle' in out2:
            use_gpu = False
        else:
            use_gpu = True
    else:
        details['nvidia_smi_output'] = ''
        use_gpu = False

    # Ollama の model list をいくつかの方法で確認する（サービス経由、候補ディレクトリ）
    model_present = False
    details['checks'] = {}

    # 1) サーバが返す model list（デフォルト）
    code_svc, out_svc = run_cmd('ollama list')
    details['checks']['ollama_list_default'] = out_svc
    service_available = (code_svc == 0)
    if service_available and LLM_MODEL in out_svc:
        model_present = True

    # 2) OLLAMA_MODELS 環境変数または候補パスで確認
    candidates = []
    if os.environ.get('OLLAMA_MODELS'):
        candidates.append(os.environ.get('OLLAMA_MODELS'))
    # add known custom path and home
    candidates.extend(['/data/kawamura/.ollama', os.path.expanduser('~/.ollama')])

    details['permission_issues'] = []

    for c in candidates:
        if not c:
            continue
        cmd = f'OLLAMA_MODELS={c} ollama list'
        code_c, out_c = run_cmd(cmd)
        details['checks'][f'ollama_list_{c}'] = out_c
        lower_c = (out_c or '').lower()
        if 'permission denied' in lower_c or 'ensure path elements' in lower_c:
            details['permission_issues'].append({'path': c, 'output': out_c})
        if LLM_MODEL in out_c:
            model_present = True
            # prefer that models_dir
            details['model_dir_found'] = c
            break

    # check default server output for permission errors as well
    if code_svc != 0:
        lower_s = (out_svc or '').lower()
        if 'permission denied' in lower_s or 'ensure path elements' in lower_s:
            details['permission_issues'].append({'path': 'server_default', 'output': out_svc})

    return use_gpu, model_present, service_available, details


# 実行前チェック
use_gpu, model_present, service_available, diag = preflight_checks()
if not use_gpu:
    print('\n⚠️ GPU未検出または応答なしです（警告）。続行しますが処理は遅くなります。以下を確認してください:')
    print('- `nvidia-smi` の出力を確認してください')
    print('- カーネルログに `nvidia-modeset` のエラーがないか確認してください')
    print('\nnvidia-smi 出力サマリ:')
    print(diag.get('nvidia_smi_output', '(no output)'))
if not service_available:
    print('\n⚠️ Ollama サービスに接続できません（Connection refused 等）。フォールバックモードで起動します。')
    print('サービスが稼働しているか、ポートや systemd の状態を確認してください。')

if not model_present:
    print(f"\n⚠️ Ollama にモデル '{LLM_MODEL}' が見つかりません（警告）。本スクリプトはフォールバックモードで起動します。\nモデルを pull するには次を実行してください:\n  OLLAMA_MODELS={os.environ.get('OLLAMA_MODELS', os.path.expanduser('~/.ollama'))} ollama pull {LLM_MODEL}")
    # continue in fallback mode; model_present remains False

# 権限に関する診断を出力（/data 以下が root 所有で書き込み不可など）
if diag.get('permission_issues'):
    print('\n⚠️ Permission issues detected when probing Ollama model paths:')
    for p in diag['permission_issues']:
        path = p.get('path')
        print(f" - Path: {path} -> possible permission denied. See Ollama logs for details.")
    print('\n推奨: システムで /data/kawamura/.ollama を使う場合、管理者に次を実行してもらってください:')
    print('  sudo chown -R kawamura:kawamura /data/kawamura/.ollama && sudo chmod -R 755 /data/kawamura/.ollama')

# LLM の初期化はサービスとモデルが利用可能なときのみ行う
FULL_AGENT = False
LOCAL_OLLAMA_PROC = None
models_dir_to_use = None

try:
    if HAS_FRAMEWORK:
        # choose models dir if found
        models_dir_to_use = None
        if 'model_dir_found' in diag:
            models_dir_to_use = diag['model_dir_found']

        home_models = os.path.expanduser('~/.ollama')

        # if model not present anywhere, attempt to pull into home dir
        if not model_present:
            print('モデルが見つからないため、ホームディレクトリに pull を試みます...')
            os.makedirs(home_models, exist_ok=True)
            try:
                pulled = pull_model_to(home_models, LLM_MODEL)
                if pulled:
                    print(f'Pull 成功: {LLM_MODEL} -> {home_models}')
                    models_dir_to_use = home_models
                    model_present = True
                else:
                    print('モデルの pull に失敗しました。手動で pull してください。')
            except PermissionError as e:
                print('⚠️ ホームディレクトリへの pull 中に権限エラーが発生しました:', e)
                print('モデルの pull に失敗しました。手動で pull してください。')

        # If service not available or service doesn't see the model, start local ollama serve
        if model_present and not service_available:
            # prefer model_dir_found, then home_models
            if not models_dir_to_use:
                models_dir_to_use = os.environ.get('OLLAMA_MODELS', home_models)
            print(f"サービス未接続のため、ローカルで ollama serve を OLLAMA_MODELS={models_dir_to_use} で起動します...")
            try:
                LOCAL_OLLAMA_PROC = start_local_ollama(models_dir_to_use, WORKSPACE_DIR)
                print(f'ローカル ollama serve 起動: pid={LOCAL_OLLAMA_PROC.pid}')
                service_available = True
            except Exception as e:
                # If permission error on the chosen models_dir, try to fall back to home_models
                if isinstance(e, PermissionError) or ('permission denied' in str(e).lower()):
                    if models_dir_to_use != home_models:
                        print('権限エラーを検出したため、ホームの ~/.ollama にフォールバックして再試行します...')
                        try:
                            os.makedirs(home_models, exist_ok=True)
                            LOCAL_OLLAMA_PROC = start_local_ollama(home_models, WORKSPACE_DIR)
                            print(f'ローカル ollama serve 起動: pid={LOCAL_OLLAMA_PROC.pid} (home_models)')
                            models_dir_to_use = home_models
                            service_available = True
                        except Exception as e2:
                            print('ホームディレクトリへのフォールバックも失敗しました:', e2)
                            print('注: systemd の設定で /data/kawamura/.ollama を使う場合は、管理者により次を実行して下さい:')
                            print('  sudo chown -R kawamura:kawamura /data/kawamura/.ollama && sudo chmod -R 755 /data/kawamura/.ollama')
                    else:
                        print('ローカル ollama serve の起動に失敗しました:', e)

        # Finally, if service is available and model_present, initialize ChatOllama
        if service_available and model_present:
            try:
                # ensure client uses same OLLAMA_MODELS when interacting
                env = os.environ.copy()
                if models_dir_to_use:
                    env['OLLAMA_MODELS'] = models_dir_to_use
                from langchain_ollama import ChatOllama
                from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
                llm = ChatOllama(
                    model=LLM_MODEL,
                    temperature=0,
                    callbacks=[StreamingStdOutCallbackHandler()]
                )
                FULL_AGENT = True
            except Exception as e:
                print('\n⚠️ LLM クライアントライブラリの初期化に失敗しました:')
                print(e)
                FULL_AGENT = False
    else:
        FULL_AGENT = False
except Exception as e:
    print('事前初期化で例外:', e)
    FULL_AGENT = False

# --- State定義 ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    code_filename: str
    iterations: int

# --- Node: Coder ---
def coder_node(state: AgentState):
    messages = state["messages"]
    print(f"\n--- 🤖 Coder ({LLM_MODEL}) is thinking & coding ---\n")
    
    # LLMに指示を投げる（ストリーミングで表示される）
    response = llm.invoke(messages)
    
    # ストリーミング後は改行を入れて見やすくする
    print("\n\n--------------------------------------------------")
    
    return {
        "messages": [response],
        "iterations": state["iterations"] + 1
    }

# --- Node: Executor ---
def executor_node(state: AgentState):
    last_message = state["messages"][-1].content
    filename = state.get("code_filename", "script.py")
    filepath = os.path.join(WORKSPACE_DIR, filename)

    print(f"\n⚙️ Executor: Extracting code for {filename}...")

    # Markdownのコードブロックを抽出
    # DeepSeekは <think> タグなどを出すため、確実に python ブロックだけを狙う
    code_match = re.search(r"```python(.*?)```", last_message, re.DOTALL)
    
    if not code_match:
        print("⚠️ No code block found.")
        return {"messages": [HumanMessage(content="Error: No python code block found. Please wrap code in ```python ... ```.")]}
    
    code = code_match.group(1).strip()

    # --- 🛡️ 安全装置 ---
    FORBIDDEN_KEYWORDS = [
        "pip install", "pip uninstall", "conda install",
        "rm -rf", "shutil.rmtree", "os.remove",
        "sudo", "chmod", "/etc/", "/var/"
    ]
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in code:
            msg = f"Security Error: Forbidden keyword '{keyword}' found."
            print(f"⚠️ Blocked: {keyword}")
            return {"messages": [HumanMessage(content=msg)]}

    # ファイル保存
    with open(filepath, "w") as f:
        f.write(code)
    
    print(f"⚙️ Executor: Running on bci2020 environment...")
    
    # 実行
    try:
        result = subprocess.run(
            [TARGET_PYTHON, filepath],
            cwd=WORKSPACE_DIR,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output = result.stdout + result.stderr
        
        if result.returncode == 0:
            print("✅ Execution Success")
            # 出力が長すぎる場合は省略して表示
            disp_output = output[:500] + "..." if len(output) > 500 else output
            print(f"Output:\n{disp_output}")
            return {"messages": [HumanMessage(content=f"Execution Success! Output:\n{output}")]}
        else:
            print("❌ Execution Failed")
            print(f"Error:\n{output}")
            return {"messages": [HumanMessage(content=f"Execution Failed:\n{output}\nPlease fix the code.")]}
            
    except Exception as e:
        print(f"❌ System Error: {e}")
        return {"messages": [HumanMessage(content=f"System Error: {str(e)}")]}

# --- 終了判定 ---
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    
    if isinstance(last_message, HumanMessage) and "Execution Success" in last_message.content:
        return END
    
    if state["iterations"] > 5:
        print("🛑 Max iterations reached.")
        return END
    
    return "coder"

if FULL_AGENT:
    # --- グラフ構築 ---
    workflow = StateGraph(AgentState)
    workflow.add_node("coder", coder_node)
    workflow.add_node("executor", executor_node)
    workflow.set_entry_point("coder")
    workflow.add_edge("coder", "executor")
    workflow.add_conditional_edges("executor", should_continue, {END: END, "coder": "coder"})
    app = workflow.compile()

    # --- 対話ループ ---
    if __name__ == "__main__":
        print(f"🚀 System Ready! (Model: {LLM_MODEL})")
        print("Type 'exit' or 'quit' to end.\n")

        while True:
            try:
                user_input = input("USER> ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                if not user_input.strip():
                    continue

                print("\n--- 🤖 Agent Working ---")
                
                tool_note = ""
                if EEG_TOOL_IMPORT_ERROR:
                    tool_note = f"\n※ tools.eeg_experiment の読み込みで次のエラーが発生しました: {EEG_TOOL_IMPORT_ERROR}"

                initial_state = {
                    "messages": [
                        SystemMessage(content=EEG_TOOL_SPEC + tool_note),
                        HumanMessage(content=user_input)
                    ],
                    "code_filename": "generated_script.py",
                    "iterations": 0
                }
                
                # stream() を回すことでグラフを実行
                for event in app.stream(initial_state):
                    pass 
                
                print("--- ✅ Task Completed ---\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
else:
    # フォールバック: 必要なフレームワークがない場合でも入力を受け付ける簡易 REPL を提供
    if __name__ == "__main__":
        print("⚠️ ランタイム依存ライブラリが見つかりません:")
        print(IMPORT_ERROR_MSG)
        print("\n簡易モードで起動します。実際に LLM を動かすには以下を確認してください:")
        print(" - conda activate bci2020 (または環境) で langchain_ollama 等をインストール")
        print(" - または pip install langchain-ollama langchain-core など\n")

        prompts_file = os.path.join(WORKSPACE_DIR, "user_prompts.txt")
        print(f"ユーザー入力は {prompts_file} に追記されます。'exit' で終了。\n")

        while True:
            try:
                user_input = input("USER> ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                if not user_input.strip():
                    continue
                with open(prompts_file, "a") as f:
                    f.write(user_input.strip() + "\n\n")
                print("入力を保存しました。依存関係を整えた後に本スクリプトを再実行してください.")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break