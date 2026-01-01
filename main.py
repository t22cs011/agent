import os
import subprocess
import re
import operator
import sys
import time
from typing import TypedDict, Annotated, List, Optional

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

# Minimal fallbacks to avoid NameError when framework imports are unavailable
if not HAS_FRAMEWORK:
    class BaseMessage:
        def __init__(self, content: Optional[str] = None):
            self.content = content or ""

    class HumanMessage(BaseMessage):
        pass

    class SystemMessage(BaseMessage):
        pass

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
TARGET_PYTHON = os.environ.get('TARGET_PYTHON', '/data/kawamura/miniforge3/envs/bci2020/bin/python')

# 1.1 目標精度 (環境変数で上書き可)
TARGET_ACC = float(os.environ.get('TARGET_ACC', '0.8'))
# 2. 作業ディレクトリ (Workspace)
WORKSPACE_DIR = os.environ.get('WORKSPACE_DIR', "/home/kawamura/agent")

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

    # 1) サーバが返す model list（デフォルト） -- if CLI isn't available, try HTTP port check instead
    code_svc, out_svc = run_cmd('ollama list')
    details['checks']['ollama_list_default'] = out_svc
    service_available = (code_svc == 0)
    if not service_available:
        # try simple TCP probe to common host targets (host.docker.internal resolves on Linux with host-gateway)
        try_hosts = [os.environ.get('OLLAMA_HOST', '').replace('http://', '').split(':')[0] if os.environ.get('OLLAMA_HOST') else 'host.docker.internal', '127.0.0.1']
        for h in try_hosts:
            try:
                if tcp_port_open(h, 11434, timeout=0.5):
                    service_available = True
                    details['checks']['tcp_probe'] = f'port 11434 open on {h}'
                    break
            except Exception:
                pass
    if service_available and out_svc and LLM_MODEL in out_svc:
        model_present = True

    # If the CLI isn't available inside the container but the Ollama HTTP service is reachable,
    # try querying the HTTP API for available models (more robust inside minimal containers).
    if service_available and not model_present:
        try:
            import urllib.request, urllib.error
            host = os.environ.get('OLLAMA_HOST', '127.0.0.1:11434')
            if host.startswith('http://') or host.startswith('https://'):
                url_base = host
            else:
                url_base = f"http://{host}"
            tags_url = f"{url_base}/api/tags"
            req = urllib.request.Request(tags_url, headers={"User-Agent": "agent-check/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode('utf-8', errors='ignore')
                if LLM_MODEL in body or LLM_MODEL.split(':')[0] in body:
                    model_present = True
                    details['checks']['ollama_http_tags'] = body[:2000]
        except Exception:
            # ignore HTTP probe failures; fall back to filesystem checks
            pass

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
        # prefer filesystem check of mounted models dir (no need for CLI)
        models_dir = os.path.join(c, 'models')
        details['checks'][f'ollama_models_dir_{c}'] = ''
        try:
            if os.path.isdir(models_dir):
                details['checks'][f'ollama_models_dir_{c}'] = ','.join(os.listdir(models_dir)[:20])
                # check for model name prefix (before ':' tag)
                model_prefix = LLM_MODEL.split(':')[0]
                for entry in os.listdir(models_dir):
                    if model_prefix in entry:
                        model_present = True
                        details['model_dir_found'] = c
                        break
                if model_present:
                    break
        except PermissionError as e:
            details['permission_issues'].append({'path': c, 'output': str(e)})

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
                # Try to use langchain chat models as a replacement if available
                try:
                    from langchain.chat_models import ChatOpenAI
                    from langchain.schema import HumanMessage as LCHumanMessage, SystemMessage as LCSystemMessage

                    class LangChainAdapter:
                        def __init__(self, model):
                            self.model = model

                        def invoke(self, messages):
                            lc_msgs = []
                            for m in messages:
                                content = getattr(m, 'content', '') or ''
                                if isinstance(m, SystemMessage):
                                    lc_msgs.append(LCSystemMessage(content=content))
                                else:
                                    lc_msgs.append(LCHumanMessage(content=content))
                            try:
                                if hasattr(self.model, 'predict_messages'):
                                    resp = self.model.predict_messages(lc_msgs)
                                    return HumanMessage(content=getattr(resp, 'content', str(resp)))
                                if hasattr(self.model, '__call__'):
                                    out = self.model.__call__(lc_msgs)
                                    if isinstance(out, str):
                                        return HumanMessage(content=out)
                                    return HumanMessage(content=getattr(out, 'content', str(out)))
                                if hasattr(self.model, 'generate'):
                                    gen = self.model.generate([lc_msgs])
                                    text = ''
                                    if getattr(gen, 'generations', None):
                                        try:
                                            text = gen.generations[0][0].text
                                        except Exception:
                                            text = str(gen)
                                    else:
                                        text = str(gen)
                                    return HumanMessage(content=text)
                            except Exception as ex:
                                return HumanMessage(content=f"LangChain model error: {ex}")

                    lc_model = ChatOpenAI(temperature=0)
                    llm = LangChainAdapter(lc_model)
                    print('✅ Using LangChain Chat model as LLM backend.')
                    FULL_AGENT = True
                except Exception as ex2:
                    print('\n❌ LangChain chat models are not available:')
                    print(ex2)
                    print('\nPlease install LangChain and a supported chat provider (e.g. OpenAI) in the Python environment:')
                    print('  pip install langchain openai')
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
    metrics: Optional[dict]

# --- Node: Coder ---
def coder_node(state: AgentState):
    messages = state["messages"]
    print(f"\n--- 🤖 Coder ({LLM_MODEL}) is thinking & coding ---\n")
    # 強化プロンプト: 必ず python のコードブロックを出力させる
    strong_system = SystemMessage(content=(
        "IMPORTANT: Always output a runnable Python script inside a single ```python ... ``` code block. "
        "Do not output only thoughts. Provide the complete implementation and ensure the code can be executed with the target Python."
    ))

    # LLMに指示を投げる（ストリーミングで表示される）
    messages_with_instruction = [strong_system] + messages
    response = llm.invoke(messages_with_instruction)
    
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
    # DeepSeekは <think> タグなどを出すため、まずは ```python ``` を優先、それがなければ任意のコードブロックを許容する
    code_match = re.search(r"```python(.*?)```", last_message, re.DOTALL | re.IGNORECASE)

    if not code_match:
        # try any code block
        code_match = re.search(r"```(.*?)```", last_message, re.DOTALL)
        if code_match:
            print("⚠️ Found code block without language; assuming python.")

    if not code_match:
        # try simple <python>...</python> tags
        code_match = re.search(r"<python>(.*?)</python>", last_message, re.DOTALL | re.IGNORECASE)

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
        # ログから精度を抽出する (柔軟にマッチ)
        # NOTE: LLM 側の出力は `Accuracy: 0.85` などこの正規表現で拾える形式を必ず守らせること。フォーマットが崩れると精度を記録できない。
        acc_match = re.search(r"(?:Accuracy|overall_mean_acc|overall mean acc)[\s:=]*([0-9]*\.?[0-9]+)", output, re.IGNORECASE)
        accuracy = None
        if acc_match:
            try:
                accuracy = float(acc_match.group(1))
            except Exception:
                accuracy = None
        # state に metrics を保存
        state_metrics = state.get('metrics', {}) or {}
        if accuracy is not None:
            state_metrics['accuracy'] = accuracy
        state['metrics'] = state_metrics
        
        if result.returncode == 0:
            print("✅ Execution Success")
            # 出力が長すぎる場合は省略して表示
            disp_output = output[:500] + "..." if len(output) > 500 else output
            print(f"Output:\n{disp_output}")
            if accuracy is not None:
                return {"messages": [HumanMessage(content=f"Execution Success! Output:\n{output}\nDetected accuracy: {accuracy}")], "metrics": state.get('metrics'), "iterations": state["iterations"] + 1}
            else:
                return {"messages": [HumanMessage(content=f"Execution Success! Output:\n{output}")], "metrics": state.get('metrics'), "iterations": state["iterations"] + 1}
        else:
            print("❌ Execution Failed")
            print(f"Error:\n{output}")
            # 失敗時も metrics が取れていれば LLM にフィードバックする
            if state.get('metrics') and state['metrics'].get('accuracy') is not None:
                acc_val = state['metrics']['accuracy']
                feedback = (
                    f"Current accuracy was {acc_val:.4f}, which is below target {TARGET_ACC}. "
                    "Please modify hyperparameters (lr, epochs, batch_size) to improve accuracy and provide a new runnable script."
                )
                return {"messages": [HumanMessage(content=f"Execution Failed:\n{output}\n{feedback}")], "metrics": state.get('metrics'), "iterations": state["iterations"] + 1}
            return {"messages": [HumanMessage(content=f"Execution Failed:\n{output}\nPlease fix the code.")], "iterations": state["iterations"] + 1}
            
    except Exception as e:
        print(f"❌ System Error: {e}")
        return {"messages": [HumanMessage(content=f"System Error: {str(e)}")], "iterations": state["iterations"] + 1}

# --- 終了判定 ---
def should_continue(state: AgentState):
    # 終了判定を精度ベースに変更
    metrics = state.get('metrics') or {}
    acc = metrics.get('accuracy')
    if acc is not None:
        if acc >= TARGET_ACC:
            print(f"✅ Target accuracy reached: {acc} >= {TARGET_ACC}")
            return END
        else:
            print(f"🔄 Accuracy {acc} < {TARGET_ACC}, continuing to coder.")
            return "coder"

    # フォールバック: 実行成功メッセージを基に終了判定
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