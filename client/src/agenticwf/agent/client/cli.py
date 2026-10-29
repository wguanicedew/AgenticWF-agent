"""agenticwf agent — the agent from the command line.

    agenticwf agent chat "hello"                     a new chat; prints the reply (session id on stderr)
    agenticwf agent chat "and then?" --session SID   continue a chat
    agenticwf agent sessions                         your chats
    agenticwf agent task "summarise run 42" --wait   a background task; --wait prints its result
    agenticwf agent tasks [TID]                      your tasks, or one of them
    agenticwf agent approve CALL --session SID       approve a tool call that waits (--reject, --task TID)
    agenticwf agent answer TID "yes, run 42"         answer a task's question
    agenticwf agent cancel TID
    agenticwf agent call GET /models                 any agent API call (the path under /api/v1)

Behind Caddy it signs in as `agenticwf token` does (`agenticwf login`, AGENTICWF_TOKEN or
AGENTICWF_CLIENT_SECRET; chats need a person, not a machine client) and finds the agent at
AGENTICWF_API_URL. The personal agent has its own token instead:

    export AGENTICWF_AGENT_URL=http://127.0.0.1:8765
    export AGENTICWF_AGENT_TOKEN_FILE=~/.agenticwf-agent/token     # or AGENTICWF_AGENT_TOKEN

`--json` prints the API's data as it is (for jq).
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx

from agenticwf.agent.client import FINISHED, AgentClient, AgentClientError


def _options(args) -> dict:
    return {"task_type": args.task_type, "mode": args.mode, "model": args.model}


def _chat(agent: AgentClient, args):
    sid = args.session
    if not sid:
        sid = agent.create_session(args.title or " ".join(args.text.split())[:60])["id"]
        print(f"session: {sid}", file=sys.stderr)
    return agent.send(sid, args.text, **_options(args))


def _task(agent: AgentClient, args):
    submitted = agent.submit_task(args.prompt, **_options(args))
    if not args.wait:
        return submitted
    print(f"task: {submitted['task_id']}", file=sys.stderr)
    return agent.wait_task(submitted["task_id"], timeout=args.timeout)


def _approve(agent: AgentClient, args):
    return agent.decide(args.call, approve=not args.reject, reason=args.reason or "", remember=args.remember,
                        session=args.session, task=args.task)


def _call(agent: AgentClient, args):
    return agent.request(args.method.upper(), args.path, json=json.loads(args.data) if args.data else None)


COMMANDS = {
    "chat": _chat,
    "sessions": lambda agent, args: agent.sessions(),
    "task": _task,
    "tasks": lambda agent, args: agent.task(args.tid) if args.tid else agent.tasks(),
    "approve": _approve,
    "answer": lambda agent, args: agent.answer_task(args.tid, args.text),
    "cancel": lambda agent, args: agent.cancel_task(args.tid),
    "call": _call,
}


def _show_waiting(waiting: list[dict], where: str) -> None:
    for task in waiting:
        for call in task.get("approvals", []):
            print(f"approval needed: {call['name']} {json.dumps(call.get('arguments'), ensure_ascii=False)}"
                  f"{' (' + call['reason'] + ')' if call.get('reason') else ''}\n"
                  f"  agenticwf agent approve {call['call']} {where}   (or --reject)")
        for question in task.get("questions", []):
            print(f"question: {question['question']}")


def _show_task(task: dict) -> None:
    if "task_id" in task:   # just submitted
        print(f"{task['task_id']}  {task['status']}")
        return
    if task["status"] in FINISHED and task.get("result"):
        print(task["result"])
    else:
        print(f"{task['id']}  {task['status']}")
    if task.get("error"):
        print(f"error: {task['error']}", file=sys.stderr)
    if task["status"] == "waiting":
        print(f"  waiting for an approval or an answer: agenticwf agent call GET /tasks/{task['id']}/turns")


def _show(command: str, data) -> None:
    if command == "chat":
        if data["status"] == "succeeded":
            print(data.get("reply", ""))
        elif data["status"] == "waiting":
            _show_waiting(data.get("waiting", []), f"--session {data['session']}")
        else:
            print(f"still working ({data['status']}); the reply will be in the session: "
                  f"agenticwf agent call GET /sessions/{data['session']}")
    elif command == "sessions":
        for item in data:
            print(f"{item['id']}  {item.get('created_at', '')}  {item.get('messages', 0):>3} messages  {item.get('title', '')}")
    elif command in ("task", "tasks") and isinstance(data, dict):
        _show_task(data)
    elif command == "tasks":
        for item in data:
            print(f"{item['id']}  {item['status']:<9}  {item.get('created_at', '')}  {' '.join(item['prompt'].split())[:60]}")
    elif data is not None:
        print(json.dumps(data, indent=2, ensure_ascii=False))


def run(args) -> int:
    try:
        with AgentClient.from_env(url=args.url, **args.auth) as agent:
            data = COMMANDS[args.agent_command](agent, args)
    except AgentClientError as exc:
        print(f"agenticwf agent: {exc}", file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        print(f"agenticwf agent: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        _show(args.agent_command, data)
    if args.agent_command == "task" and args.wait and data["status"] in ("failed", "cancelled"):
        return 1
    return 0


def register(sub) -> None:
    """The `agenticwf.commands` entry point: adds `agent` to the `agenticwf` command."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", help="the agent's base URL, where it serves /api/v1 (AGENTICWF_AGENT_URL)")
    common.add_argument("--json", action="store_true", help="print the API's data as JSON")
    run_options = argparse.ArgumentParser(add_help=False)
    run_options.add_argument("--task-type", help="a task type name (default: the session's or the agent's)")
    run_options.add_argument("--mode", choices=["default", "plan", "auto"])
    run_options.add_argument("--model", help="a model id from `agenticwf agent call GET /models`")

    parser = sub.add_parser("agent", help="the agent: chats and background tasks", description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.set_defaults(run=run)
    commands = parser.add_subparsers(dest="agent_command", required=True)
    chat = commands.add_parser("chat", parents=[common, run_options], help="send a chat message, print the reply")
    chat.add_argument("text")
    chat.add_argument("--session", help="continue this chat (default: a new one)")
    chat.add_argument("--title", help="the new chat's title (default: the message)")
    commands.add_parser("sessions", parents=[common], help="your chats")
    task = commands.add_parser("task", parents=[common, run_options], help="submit a background task")
    task.add_argument("prompt")
    task.add_argument("--wait", action="store_true", help="wait until it finishes, or waits for you")
    task.add_argument("--timeout", type=float, help="with --wait: give up waiting after this many seconds")
    tasks = commands.add_parser("tasks", parents=[common], help="your tasks, or one of them")
    tasks.add_argument("tid", nargs="?")
    approve = commands.add_parser("approve", parents=[common], help="approve (or reject) a tool call that waits")
    approve.add_argument("call", help="the call id the chat or task shows")
    where = approve.add_mutually_exclusive_group(required=True)
    where.add_argument("--session", help="in this chat")
    where.add_argument("--task", help="in this background task")
    approve.add_argument("--reject", action="store_true")
    approve.add_argument("--reason")
    approve.add_argument("--remember", action="store_true", help="approve this tool for the rest of the task")
    answer = commands.add_parser("answer", parents=[common], help="answer a task's question")
    answer.add_argument("tid")
    answer.add_argument("text")
    cancel = commands.add_parser("cancel", parents=[common], help="cancel a background task")
    cancel.add_argument("tid")
    call = commands.add_parser("call", parents=[common], help="any agent API call")
    call.add_argument("method")
    call.add_argument("path", help="under /api/v1, e.g. /models")
    call.add_argument("--data", help="JSON request body")
