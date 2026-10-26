import pytest
from pydantic import ValidationError

from node_runner.src.trace.events.model import TraceEvent

pytestmark = pytest.mark.unit


class TestTraceEventJobKind:
    def test_job_event_carries_run_metadata(self):
        ev = TraceEvent(
            kind="job",
            event="start",
            trace_id="job-a",
            name="job",
            run_name="run-a",
            checkpoint_path="/ckpt/a",
            model_class="sft",
            model_size="7b",
            model_username="u",
            global_nnodes=4,
        )
        assert ev.event_id == "job"
        dumped = ev.model_dump(mode="json", exclude_none=True)
        assert dumped["run_name"] == "run-a"
        assert dumped["global_nnodes"] == 4
        assert "phase" not in dumped

    def test_job_event_does_not_require_phase(self):
        # Unlike "phase"/"shooter", "job" is not an event with a phase.
        TraceEvent(kind="job", event="start", trace_id="job-a", name="job")


class TestTraceEventValidation:
    def test_secondary_model_is_rejected_outside_a_shooter(self):
        with pytest.raises(ValidationError, match="secondary_model"):
            TraceEvent(
                kind="phase",
                event="start",
                trace_id="job-a",
                name="parallel",
                phase="parallel",
                secondary_model="judge",
            )

    def test_an_end_event_without_a_status_is_rejected(self):
        with pytest.raises(ValidationError, match="requires status"):
            TraceEvent(kind="barrier", event="end", trace_id="job-a", name="X")

    def test_a_phase_event_id_is_derived_from_the_phase(self):
        ev = TraceEvent(
            kind="phase",
            event="start",
            trace_id="job-a",
            name="scoring",
            phase="scoring",
        )

        assert ev.event_id == "phase:scoring"
