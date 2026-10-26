import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ci import review_mr

pytestmark = pytest.mark.unit


@pytest.fixture
def review_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[SimpleNamespace, Callable[[], None]]:
    state = SimpleNamespace(
        head="b" * 40,
        findings=[],
        published=[],
        fail_once=False,
        review=review_mr.review,
    )
    for name, value in {
        "OPENAI_BASE_URL": "https://model.invalid/v1",
        "PAM_REVIEW_MODEL": "reviewer",
        "OPENAI_API_KEY": "fake-model-token",
        "HTTPS_PROXY": "http://proxy.invalid:8080",
        "PAM_REVIEW_GITLAB_TOKEN": "fake-gitlab-token",
        "CI_API_V4_URL": "https://gitlab.invalid/api/v4",
        "CI_MERGE_REQUEST_PROJECT_ID": "1",
        "CI_MERGE_REQUEST_IID": "1",
        "CI_JOB_URL": "https://gitlab.invalid/jobs/1",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("CI_MERGE_REQUEST_SOURCE_BRANCH_SHA", raising=False)
    monkeypatch.delenv("CI_MERGE_REQUEST_DIFF_BASE_SHA", raising=False)
    monkeypatch.setattr(review_mr, "ROOT", tmp_path)
    monkeypatch.setattr(review_mr, "git", lambda *args: "")
    monkeypatch.setattr(review_mr, "review", lambda *args: iter(state.findings))

    def get(
        url: str, *, params: dict | None = None, **kwargs: object
    ) -> SimpleNamespace:
        headers = {}
        data: object
        if url.endswith("/diffs"):
            data = [
                {
                    "old_path": "sample.py",
                    "new_path": "sample.py",
                    "diff": "@@ -0,0 +1,2 @@\n+first = 1\n+second = 2",
                }
            ]
        elif url.endswith("/discussions"):
            assert params is not None
            if params["page"] == 1:
                data = [{"notes": [{"body": "An unrelated discussion."}]}]
                headers = {"X-Next-Page": "2"}
            else:
                data = [{"notes": [{"body": body}]} for body in state.published]
        else:
            data = {
                "sha": state.head,
                "diff_refs": {
                    "base_sha": "a" * 40,
                    "start_sha": "a" * 40,
                    "head_sha": state.head,
                },
                "web_url": "https://gitlab.invalid/mr/1",
                "title": "Review retry",
                "description": "Preserve already published findings.",
            }
        return SimpleNamespace(status_code=200, headers=headers, json=lambda: data)

    def post(url: str, *, json: dict, **kwargs: object) -> SimpleNamespace:
        if state.fail_once and len(state.published) == 1:
            state.fail_once = False
            return SimpleNamespace(status_code=503)
        state.published.append(json["body"])
        note = {"id": len(state.published)}
        data = {"notes": [note]} if url.endswith("/discussions") else note
        return SimpleNamespace(status_code=201, json=lambda: data)

    gitlab = Mock(headers={}, get=get, post=post)
    monkeypatch.setattr(review_mr.requests, "Session", lambda: gitlab)

    def run() -> None:
        monkeypatch.setenv("CI_COMMIT_SHA", state.head)
        review_mr.main()

    return state, run


def test_retry_skips_published_findings_but_keeps_changed_content_and_commits(
    review_run: tuple[SimpleNamespace, Callable[[], None]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, run = review_run
    state.findings = [
        {
            "severity": "Worth fixing",
            "title": title,
            "path": "sample.py",
            "side": "new",
            "line": line,
            "comment": title,
        }
        for line, title in ((1, "First finding"), (2, "Second finding"))
    ]
    state.fail_once = True

    with pytest.raises(RuntimeError, match="comment failed.*503"):
        run()
    assert len(state.published) == 1
    first_body = state.published[0]

    monkeypatch.setenv("CI_JOB_URL", "https://gitlab.invalid/jobs/2")
    run()
    assert len(state.published) == 2
    assert state.published[0] == first_body
    assert "Second finding" in state.published[1]

    state.findings[0]["comment"] = "Updated first finding."
    run()
    assert len(state.published) == 3
    assert "Updated first finding." in state.published[2]

    state.head = "c" * 40
    run()
    assert len(state.published) == 5


def test_retry_does_not_duplicate_no_findings_note(
    review_run: tuple[SimpleNamespace, Callable[[], None]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, run = review_run
    run()
    monkeypatch.setenv("CI_JOB_URL", "https://gitlab.invalid/jobs/2")
    run()
    assert len(state.published) == 1
    assert state.published[0].startswith("No findings.")


def test_model_failure_preserves_first_comment_and_retry_publishes_remaining(
    review_run: tuple[SimpleNamespace, Callable[[], None]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, run = review_run
    monkeypatch.setattr(review_mr, "review", state.review)
    monkeypatch.setattr(
        review_mr, "run_tool", lambda *args: {"content": "1: first = 1"}
    )
    read = {
        "finish_reason": "tool_calls",
        "message": {
            "content": None,
            "tool_calls": [
                {
                    "id": "read",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"sample.py"}',
                    },
                }
            ],
        },
    }
    first: dict = {
        "finish_reason": "stop",
        "message": {
            "content": json.dumps(
                {
                    "finding": {
                        "severity": "Worth fixing",
                        "title": "First finding",
                        "path": "sample.py",
                        "side": "new",
                        "line": 1,
                        "comment": "First finding.",
                    }
                }
            ),
        },
    }
    second = {
        "finish_reason": "stop",
        "message": {
            "content": json.dumps(
                {
                    "finding": {
                        "severity": "Worth fixing",
                        "title": "Second finding",
                        "path": "sample.py",
                        "side": "new",
                        "line": 2,
                        "comment": "Second finding.",
                    }
                }
            ),
        },
    }
    responses = iter([read, first, review_mr.requests.Timeout()])

    def model_post(*args: object, **kwargs: object) -> SimpleNamespace:
        choice = next(responses)
        if isinstance(choice, Exception):
            assert len(state.published) == 1
            raise choice
        return SimpleNamespace(status_code=200, json=lambda: {"choices": [choice]})

    monkeypatch.setattr(review_mr.requests, "post", model_post)
    with pytest.raises(RuntimeError, match="Review API request failed.*Timeout"):
        run()
    report = (tmp_path / "review/review.md").read_text()
    assert report.startswith("Review incomplete.")
    assert "First finding" in report
    assert len(state.published) == 1
    assert (tmp_path / "review/response.json").read_text() == first["message"][
        "content"
    ]

    responses = iter(
        [
            read,
            first,
            second,
            {"finish_reason": "stop", "message": {"content": '{"done": true}'}},
        ]
    )
    run()
    assert len(state.published) == 2
    assert "Second finding" in state.published[1]
    assert (tmp_path / "review/review.md").read_text().startswith("Review complete.")
