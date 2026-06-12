"""
s06: Subagent - 下发 messages[] 消息列表为空的 sub-agents 实现 context 隔离 

  Parent Agent                           Subagent
  +------------------+                  +------------------+
  | messages=[...]   |                  | messages=[task]  | <-- fresh
  |                  |   dispatch       |                  |
  | tool: task       | ---------------> | own while loop   |
  |   prompt="..."   |                  |   bash/read/...  |
  |                  |   summary only   |   (max 30 turns) |
  | result = "..."   | <--------------- | return last text |
  +------------------+                  +------------------+
        ^                                      |
        |       intermediate results DISCARDED  |
        +--------------------------------------+

  Subagent tools: bash, read, write, edit, glob (NO task — no recursion)


  

Changes from s05:
  + task 工具 + spwan_subagent() with fresh messages[]
  + 安全限制: max 30 turns per subagent 
  + extract_text() 辅助函数
  Subagent 不能递归的派发 sub-subagents (通过将 task tool 排除在 sub_tools 外实现)

Run: python s06_subagent/code.py
"""                                             

import ast, os, json, subprocess
from pathlib import Path

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


WORKDIR = Path.cwd() # 工作目录
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL")) # 客户端
MODEL = os.getenv("MODEL_ID") # 模型ID
CURRENT_TODOS: list[dict] = [] # todos

# s05 change: SYSTEM prompt 中添加 planning
# SYSTEM = f"You are a code agent at {WORKDIR}. Use tools to solve tasks. Act, do not explain."
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Before starting ang multi-step task, use todo_write to plan your steps. "
    "Update status as you go."
)

# s06: subagent 拥有它自己的 system prompt - 没有 task 工具, 没有无限递归循环
SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."

)

# 工具执行函数
def run_bash(command: str) -> str:
    # dangerous = [
    #     "rm -rf /",
    #     "sudo",
    #     "shutdown",
    #     "reboot",
    #     "> /dev/"
    # ]
    # if any(d in command for d in dangerous):
    #     return "Error: Dangerous command blocked!!!"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(), capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except Exception as e:
        return f"Error: {e}"



# =======================================================
# s02: 新增 安全路径判断 + 4个新工具
# =======================================================

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escape workspace: {p}")
    return path

def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

def run_glob(pattern: str) -> str:
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"

# =======================================================
# s05: todo_write tool - 只plan, 不执行
# =======================================================

def _normalize_todos(todos):
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError:
            try:
                todos = ast.literal_eval(todos)
            except (SyntaxError, ValueError):
                return None, "Error: todos must be a list or JSON array string"
    if not isinstance(todos, list):
        return None, "Error: todos must be a list"
    for i, t in enumerate(todos):
        if not isinstance(t, dict):
            return None, f"Error: todos[{i}] must be an object"
        if "content" not in t or "status" not in t:
            return None, f"Error: todos[{i}] missing 'content' or 'status'"
        if t["status"] not in ("pending", "in_progress", "completed"):
            return None, f"Error: todos[{i}] has invalid status '{t['status']}'"
    return todos, None

def run_todo_write(todos: list) -> str:
    global CURRENT_TODOS
    todos, error = _normalize_todos(todos)
    if error:
        return error
    CURRENT_TODOS = todos
    lines = ["\n## Current Tasks"]
    for t in CURRENT_TODOS:
        icon = {"pending": " ", "in_progress": "▸", "completed": "✓"}[t["status"]]
        lines.append(f"  [{icon}] {t['content']}")
    print("\n".join(lines))
    return f"Updated {len(CURRENT_TODOS)} tasks"


# =======================================================
# s02: 工具定义扩展到 5 个
# =======================================================

TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in a file once.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    # s05: new tool
    {
        "name": "todo_write", 
        "description": "Create and manage a task list for your current coding session.",
        "input_schema": {
            "type": "object", 
            "properties": {
                "todos": {
                    "type": "array", 
                    "items": {
                        "type": "object", 
                        "properties": {
                            "content": {"type": "string"}, 
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}
                        }, 
                                                                        "required": ["content", "status"]
                                                                    }}}, 
            "required": ["todos"]
        }
    },
]

# =======================================================
# s02: 增加 工具分发映射
# =======================================================

TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
    "todo_write": run_todo_write,
}

# =======================================================
# s06: 增加 Subagent - fresh messsages[], summary only
# =======================================================

SUB_TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in a file once.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
]
# NO "task" tool — prevent recursive spawning
# No "todo_write" tool - 子任务暂时不需要计划列表

SUB_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}

def extract_text(content) -> str:
    """从 message content blocks 中提取文本"""
    if not isinstance(content, list):
        return str(content)
    return "\n".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")

def spawn_subagent(description: str) -> str:
    """Spawn a subagent with fresh messages[], return summary only."""
    print(f"\n[Subagent spawned]")
    messages = [{"role": "user", "content": description}]

    for _ in range(30):
        resp = client.messages.create(model=MODEL, system=SUB_SYSTEM, messages=messages, tools=SUB_TOOLS, max_tokens=8000)

        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            break
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                blocked = trigger_hooks("PreToolUse", block)
                if blocked:
                    messages.append({"type": "tool_result", "tool_use_id": block.id, "content": str(blocked)})
                    continue
                handler = SUB_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown: {block.name}"
                trigger_hooks("PostToolUse", block, output)
                print(f"    [sub] {block.name}: {str(output)[:100]}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        
        messages.append({"role": "user", "content": results})

    # 提取subagent运行的最后结果
    result = extract_text(messages[-1]["content"])
    if not result:
        for message in reversed(messages):
            if message["role"] == "assitant":
                result = extract_text(message["content"])
                if result:
                    break
        if not result:
            result = "Subagent stopped after 30 turns without final answer."
    print(f"[Subagent done]")
    return result

TOOLS.append({
    "name": "task",
    "description": "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
    "input_schema": {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]},
})
TOOL_HANDLERS["task"] = spawn_subagent


# =======================================================
# s04: 增加 钩子回调系统 (s03 中的权限管理通过回调实现)
# =======================================================

HOOKS = {
    "UserPromptSubmit": [],
    "PreToolUse": [],
    "PostToolUse": [],
    "Stop": [],
}

# 注册钩子
def register_hook(event: str, callback):
    HOOKS[event].append(callback)

# 触发钩子
def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:  # teaching shortcut: block this tool call
            return result
    return None


# =======================================================
# s03: 3-Gate Permission Pipeline
# =======================================================

# Gate 1: 硬编码拒绝列表
DENY_LIST = [
    "rm -rf /",
    "sudo",
    "shutdown",
    "reboot",
    "mkfs",
    "dd if=",
    "> /dev/sda",
]
DESTRUCTIVE = [
    "rm ",
    "> /etc/",
    "chmod 777",
]

# def check_deny_list(command: str) -> str | None:
#     for pattern in DENY_LIST:
#         if pattern in command:
#             return f"Blocked: '{pattern}' is on the deny list!!!"
#     return None

# # Gate 2: 规则匹配
# PERMISSION_RULES = [
#     {
#         "tools": ["write_file", "edit_file"],
#         "check": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
#         "message": "Writing outside workspace"
#     },
#     {
#         "tools": ["bash"],
#         "check": lambda args: any(kw in args.get("command", "") for kw in ["rm ", "> /etc/", "chmod 777"]),
#         "message": "Potentially destructive command"
#     }
# ]

# def check_rules(tool_name: str, args: dict) -> str | None:
#     for rule in PERMISSION_RULES:
#         if tool_name in rule["tools"] and rule["check"](args):
#             return rule["message"]
#     return None

# # Gate 3: 用户审批
# def ask_user(tool_name: str, args: dict, reason: str) -> str:
#     print(f"\n⚠  {reason}")
#     print(f"   Tool: {tool_name}({args})")
#     choice = input("   Allow? [y/N] ").strip().lower()
#     return "allow" if choice in ("y", "yes") else "deny"

# # Pipeline
# def check_permission(block) -> bool:
#     if block.name == "bash":
#         reason = check_deny_list(block.input.get("command", ""))
#         if reason:
#             print(f"\n⛔ {reason}")
#             return False
#     reason = check_rules(block.name, block.input)
#     if reason:
#         decision = ask_user(block.name, block.input, reason)
#         if decision == "deny":
#             return False
#     return True

def permission_hook(block):
    """PreToolUse: s03 check_permission() logic moved here."""
    if block.name == "bash":
        for pattern in DENY_LIST:
            if pattern in block.input.get("command", ""):
                print(f"\n⛔ Blocked: '{pattern}'")
                return "Permission denied by deny list"
        for kw in DESTRUCTIVE:
            if kw in block.input.get("command", ""):
                print(f"\n⚠  Potentially destructive command")
                print(f"   Tool: {block.name}({block.input})")
                choice = input("   Allow? [y/N] ").strip().lower()
                if choice not in ("y", "yes"):
                    return "Permission denied by user"
    if block.name in ("write_file", "edit_file"):
        path = block.input.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print(f"\n⚠  Writing outside workspace")
            print(f"   Tool: {block.name}({block.input})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    return None

def log_hook(block):
    """PreToolUse: log every tool call."""
    args_preview = str(list(block.input.values())[:2])[:60]
    print(f"[HOOK] {block.name}({args_preview})")
    return None

def large_output_hook(block, output):
    """PostToolUse: warn on large output."""
    if len(str(output)) > 100000:
        print(f"[HOOK] ⚠ Large output from {block.name}: {len(str(output))} chars")
    return None

# UserPromptSubmit hook: log user input before it reaches the LLM
def context_inject_hook(query: str):
    print(f"[HOOK] UserPromptSubmit: working in {WORKDIR}")
    return None

# Stop hook: print summary when loop is about to exit
def summary_hook(messages: list):
    tool_count = sum(1 for m in messages
                     for b in (m.get("content") if isinstance(m.get("content"), list) else [])
                     if isinstance(b, dict) and b.get("type") == "tool_result")
    print(f"[HOOK] Stop: session used {tool_count} tool calls")
    return None

register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


# =======================================================
# s01: output = run_bash(block.input["command"])
# s02: output = TOOL_HANDLERS[block.name](**block.input)
# =======================================================


rounds_since_todo = 0

# 核心 pattern: 一个 while 循环调用工具执行直到模型发送停止命令
def agent_loop(messages: list):
    global rounds_since_todo
    while True:
        # s05: nag reminder
        if rounds_since_todo >= 3 and messages:
            messages.append({"role": "user", "content": "<reminder>Update your todos.</reminder>"})
            rounds_since_todo = 0

        # 客户端通过API发起请求并接收响应
        resp = client.messages.create(model=MODEL, system=SYSTEM, messages=messages, tools=TOOLS, max_tokens=8000)

        # 向messages添加模型回复
        messages.append({"role": "assistant", "content": resp.content})

        # 检查是否停止
        if resp.stop_reason != "tool_use":
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return
        

        # 执行工具
        rounds_since_todo += 1
        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue

            # print(f"${block.name}")

            # s04 change: 使用 hook 代替硬编码的权限检查
            # if not check_permission(block):
            #     results.append({
            #         "type": "tool_result",
            #         "tool_use_id": block.id,
            #         "content": "Permission denied."
            #     })
            #     continue
            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(blocked)
                })
                continue

            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            # print(output[:200])
            trigger_hooks("PostToolUse", block, output)
            
            # s05: 当todo_write被调用时重置nag counter
            if block.name == "todo_write":
                rounds_since_todo = 0

            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })

        # 向messages中添加工具调用结果
        messages.append({"role": "user", "content": results})

# Entry point
if __name__ == "__main__":
    print("s06: TODO WRITE")
    print("输入问题, 回车发送. 输入 q 推出. \n")

    history = []
    while True:
        try:
            query = input(">>")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        trigger_hooks("UserPromptSubmit", query)
        history.append({"role": "user", "content": query})
        agent_loop(history)

        # 打印模型最后的文本输出
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(f"\n👌👌👌\n👌回答:\n{block.text}")
        print()