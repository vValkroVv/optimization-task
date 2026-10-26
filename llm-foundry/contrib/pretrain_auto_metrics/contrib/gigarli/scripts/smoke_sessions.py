from __future__ import annotations

import argparse
import asyncio
import json

from gigarli import (
    CloseSessionRequest,
    CreateSessionRequest,
    RequestLimits,
    RLIClient,
    SessionEnv,
    StepAction,
    StepActionArgs,
    StepRequest,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test for GigaRLI sessions.")
    parser.add_argument(
        "--base-url",
        default="https://rli.sberdevices.ru",
        help="Base URL for the GigaRLI API.",
    )
    parser.add_argument(
        "--env-name",
        default="terminal-bench-modernize-scientific-stack",
        help="Environment name used during session creation.",
    )
    parser.add_argument(
        "--env-version",
        default="1.0.0",
        help="Environment version used during session creation.",
    )
    parser.add_argument(
        "--task-id",
        default="modernize-scientific-stack",
        help="Task identifier used during session creation.",
    )
    parser.add_argument(
        "--ttl-ms",
        type=int,
        default=3_600_000,
        help="Session TTL in milliseconds.",
    )
    parser.add_argument(
        "--deadline-ms",
        type=int,
        default=60_000,
        help="Deadline for the probe step in milliseconds.",
    )
    parser.add_argument(
        "--command",
        default="set -x; pwd; whoami; echo HOME=$HOME",
        help="Command executed via POST /v1/sessions/step.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification.",
    )
    parser.add_argument(
        "--keep-session",
        action="store_true",
        help="Do not close the session at the end of the smoke test.",
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    client = RLIClient(
        base_url=args.base_url,
        verify=not args.insecure,
        timeout=120.0,
    )

    created = await client.create_session(
        CreateSessionRequest(
            env=SessionEnv(
                name=args.env_name,
                version=args.env_version,
                task_id=args.task_id,
            ),
            ttl_ms=args.ttl_ms,
            use_pip_proxy=True,
        )
    )
    print("create_session:")
    print(
        json.dumps(
            created.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )
    assert created.session_id, "create_session: missing session_id"
    assert created.backend_ref, "create_session: missing backend_ref"
    assert created.backend_type, "create_session: missing backend_type"
    assert created.terminal_url, "create_session: missing terminal_url"
    assert created.workspace_path is not None, "create_session: missing workspace_path"

    session = await client.get_session(created.session_id)
    print("\nget_session:")
    print(
        json.dumps(
            session.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )
    assert session.session_id == created.session_id, "get_session: session_id mismatch"
    assert session.status == "RUNNING", f"get_session: unexpected status {session.status!r}"
    assert session.env_name == args.env_name, "get_session: env_name mismatch"
    assert session.env_version == args.env_version, "get_session: env_version mismatch"
    assert session.task_id == args.task_id, "get_session: task_id mismatch"

    step = await client.step(
        StepRequest(
            session_id=created.session_id,
            mode="single",
            actions=[
                StepAction(
                    type="env_action",
                    name="bash",
                    args=StepActionArgs(cmd=args.command),
                )
            ],
            limits=RequestLimits(deadline_ms=args.deadline_ms),
        )
    )
    print("\nstep:")
    print(
        json.dumps(
            step.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )
    assert step.terminated is False, "step: session terminated unexpectedly"
    assert step.events, "step: no events returned"
    assert step.events[0].exit_code == 0, "step: non-zero exit code"
    assert step.applied_limits is not None, "step: missing applied_limits"
    assert (
        step.applied_limits.deadline_ms == args.deadline_ms
    ), "step: deadline_ms mismatch"
    assert "+ pwd" in step.observation.output, "step: expected command trace missing"
    assert "HOME=/root" in step.observation.output, "step: expected HOME output missing"

    if args.keep_session:
        print(f"\nsession kept open: {created.session_id}")
        return

    await client.close_session(CloseSessionRequest(session_id=created.session_id))
    print(f"\nclose_session: ok ({created.session_id})")

    closed = await client.get_session(created.session_id)
    print("\npost_close_get_session:")
    print(
        json.dumps(
            closed.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )
    assert closed.session_id == created.session_id, "post-close get_session: session_id mismatch"
    assert closed.status == "CLOSED", f"post-close get_session: unexpected status {closed.status!r}"
    assert closed.closed_at is not None, "post-close get_session: missing closed_at"


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
