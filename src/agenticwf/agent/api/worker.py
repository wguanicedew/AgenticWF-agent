"""
A worker process: runs jobs beside the API process, sharing them through the database
(docs/deployment.md, Worker processes).

    agenticwf-agent-worker [--workers N]

It needs `[harness] shared = true` in agent.cfg, for the API process and every worker process
alike. It serves no requests and holds no sessions: it claims jobs (intake, task turns, replies,
consolidation, cron reports) from agent_jobs, fairly between owners across every process, and
sends the events of what it does to the API process through agent_events, which delivers them to
the sessions and live update streams there. The API process schedules crons and resumes
unfinished work after a restart; a worker that stops leaves its jobs to the others once their
leases expire.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import signal

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime import telemetry
from agenticwf.agent.runtime.bootstrap import RuntimeConfig, build_thread_manager
from agenticwf.agent.runtime.harness.relay import EventRelay

LOG = get_logger(__name__)


async def run(config: RuntimeConfig, stop: asyncio.Event) -> None:
    from agenticwf.agent.api.app import database_engine   # the same engine as the API process

    telemetry.setup("agenticwf-agent-worker")
    manager = build_thread_manager(config, database_engine(config))
    relay = EventRelay()
    manager.harness.remote = relay.write
    if manager.crons is not None:   # a cron's events (a run finished) go to the API process too
        manager.crons.notify = lambda cron, event: relay.write(f"cron:{cron.id}", "", event)
    manager.workers.start()
    LOG.info("worker process: %d workers on the shared job queue", manager.workers.size)
    await stop.wait()
    await manager.workers.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=None, help="worker loops (default: [harness] workers)")
    args = parser.parse_args()
    config = RuntimeConfig.from_core_config()
    if not config.shared:
        raise SystemExit("a worker process needs [harness] shared = true in agent.cfg (and the API process too)")
    if args.workers:
        config = dataclasses.replace(config, workers=args.workers)

    async def serve() -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for name in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(name, stop.set)
        await run(config, stop)

    asyncio.run(serve())


if __name__ == "__main__":
    main()
