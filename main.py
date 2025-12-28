import os
import subprocess
import re
import operator
from typing import TypedDict, Annotated, List
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, BaseMessage

# ==========================================
# ⚙️ 設定セクション (ここを環境に合わせて変更)
# ==========================================

# 1. 実行部隊 (bci2020) のPythonパス
#    ここに指定したPython環境を使ってコードが実行されます
TARGET_PYTHON = "/data/kawamura/miniforge3/envs/bci2020/bin/python"

# 2. 作業ディレクトリ (Workspace)
#    画像の指示通り、I/Oが重くなる処理はNFS(/home)ではなく
#    ローカルSSD(/data)で行うように設定します
WORKSPACE_DIR = "/home/kawamura/agent"

# 3. 使用するLLMモデル
# LLM_MODEL = "qwen2.5-coder:7b"
LLM_MODEL = "qwen2.5-coder:32b"  # ← ここを32bに変更

# ==========================================

# ディレクトリが存在することを確認
if not os.path.exists(WORKSPACE_DIR):
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    print(f"📁 Created workspace at: {WORKSPACE_DIR}")

print(f"🚀 Initializing Agent with {LLM_MODEL}...")
print(f"🐍 Target Python: {TARGET_PYTHON}")

# LLMの初期化
llm = ChatOllama(model=LLM_MODEL, temperature=0)

# --- State定義 (エージェント間の記憶) ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    code_filename: str
    iterations: int

# --- Node: Coder (コードを書く・修正する) ---
def coder_node(state: AgentState):
    messages = state["messages"]
    print("\n🤖 Coder: Thinking...")
    
    # LLMに指示を投げる
    response = llm.invoke(messages)
    
    return {
        "messages": [response],
        "iterations": state["iterations"] + 1
    }

# --- Node: Executor (実行する + 安全装置) ---
def executor_node(state: AgentState):
    last_message = state["messages"][-1].content
    filename = state.get("code_filename", "script.py")
    filepath = os.path.join(WORKSPACE_DIR, filename)

    # Markdownのコードブロックを抽出
    code_match = re.search(r"```python(.*?)```", last_message, re.DOTALL)
    
    if not code_match:
        return {"messages": [HumanMessage(content="Error: No python code block found. Please wrap code in ```python ... ```.")]}
    
    code = code_match.group(1).strip()

    # --- 🛡️ 安全装置 (Safety Guardrails) ---
    FORBIDDEN_KEYWORDS = [
        "pip install", "pip uninstall", "conda install", # 環境破壊防止
        "rm -rf", "shutil.rmtree", "os.remove", # 削除防止
        "sudo", "chmod", "/etc/", "/var/" # システム領域保護
    ]
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in code:
            msg = f"Security Error: Forbidden keyword '{keyword}' found. Do not use system modification commands."
            print(f"⚠️ Blocked: {keyword}")
            return {"messages": [HumanMessage(content=msg)]}
    # --------------------------------------

    # ファイル保存 (ローカルSSD /data に保存される)
    with open(filepath, "w") as f:
        f.write(code)
    
    print(f"⚙️ Executor: Running {filename} on bci2020 environment...")
    
    # 実行
    try:
        # ここで TARGET_PYTHON (bci2020) を使って実行する
        result = subprocess.run(
            [TARGET_PYTHON, filepath],
            cwd=WORKSPACE_DIR, # カレントディレクトリも /data にする
            capture_output=True,
            text=True,
            timeout=60 # 60秒でタイムアウト
        )
        
        output = result.stdout + result.stderr
        
        if result.returncode == 0:
            print("✅ Execution Success")
            return {"messages": [HumanMessage(content=f"Execution Success! Output:\n{output}")]}
        else:
            print("❌ Execution Failed")
            return {"messages": [HumanMessage(content=f"Execution Failed:\n{output}\nPlease fix the code.")]}
            
    except Exception as e:
        return {"messages": [HumanMessage(content=f"System Error: {str(e)}")]}

# --- 終了判定 ---
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    
    if isinstance(last_message, HumanMessage) and "Execution Success" in last_message.content:
        return END
    
    if state["iterations"] > 5: # 最大5回試行
        print("🛑 Max iterations reached.")
        return END
    
    return "coder"

# --- グラフ構築 ---
workflow = StateGraph(AgentState)
workflow.add_node("coder", coder_node)
workflow.add_node("executor", executor_node)
workflow.set_entry_point("coder")
workflow.add_edge("coder", "executor")
workflow.add_conditional_edges("executor", should_continue, {END: END, "coder": "coder"})
app = workflow.compile()

# --- 修正版: 対話ループ ---
if __name__ == "__main__":
    print(f"🚀 System Ready! (Model: {LLM_MODEL})")
    print("Type 'exit' or 'quit' to end.\n")

    # 会話履歴を保持するリスト（コンテキストを維持したい場合）
    # 今回はシンプルに毎回新しいタスクとして扱いますが、
    # 必要ならここを工夫して履歴を継承させます。

    while True:
        try:
            user_input = input("USER> ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue

            print("\n--- 🤖 Agent Working ---")
            
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "code_filename": "generated_script.py", # 上書きしていく
                "iterations": 0
            }
            
            for _ in app.stream(initial_state):
                pass # 経過はprint済み
            
            print("--- ✅ Task Completed ---\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break