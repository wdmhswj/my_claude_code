"""
s01_agent_loop.py - Agent 循环

图示:
    +-------------+     +-----+
    | User Prompt | --> | LLM | -- stop_reason == "tool_use" --> False --> Stop
    +-------------+     +-----+                 |
                            ^                   V
                            |                  True
                            |                   |
                            |                    V
                            | tool_result +-----------------+
                            +-------------| Tool Execution  |
                                          +-----------------+


"""                                             

import os
import subprocess

try:
    import readline
    # macOS 的 libedit 在处理中文输入时有退格问题，这四行修复它
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None) # 关闭官方认证方式

# 客户端
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.getenv("MODEL_ID")

SYSTEM = f"You are a code agent at {os.getcwd()}. Use bash to solve tasks. Act, do not explain."

# 工具定义: 暂时只有 bash
TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
]

# 工具执行函数
def run_bash(command: str) -> str:
    dangerous = [
        "rm -rf /",
        "sudo",
        "shutdown",
        "reboot",
        "> /dev/"
    ]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked!!!"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(), capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"
    
# 核心 pattern: 一个 while 循环调用工具执行直到模型发送停止命令
def agent_loop(messages: list):
    while True:
        # 客户端通过API发起请求并接收响应
        resp = client.messages.create(model=MODEL, system=SYSTEM, messages=messages, tools=TOOLS, max_tokens=8000)

        # 向messages添加模型回复
        messages.append({"role": "assistant", "content": resp.content})

        # 检查是否停止
        if resp.stop_reason != "tool_use":
            return
        
        # 执行工具
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                print(f"${block.input['command']}")
                output = run_bash(block.input["command"])
                print(output[:200])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

        # 向messages中添加工具调用结果
        messages.append({"role": "user", "content": results})

# Entry point
if __name__ == "__main__":
    print("s01: Agent loop")
    print("输入问题, 回车发送. 输入 q 推出. \n")

    history = []
    while True:
        try:
            query = input(">>")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)

        # 打印模型最后的文本输出
        resp_content = history[-1]["content"]
        if isinstance(resp_content, list):
            for block in resp_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()