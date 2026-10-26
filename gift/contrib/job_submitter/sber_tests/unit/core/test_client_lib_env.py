import importlib

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def client_lib():
    # Imported lazily: the module reads the environment at import time.
    return importlib.import_module("job_submitter.c_lib.client_lib")


def test_namespace_and_workspace_do_not_need_nb_prefix(monkeypatch, client_lib):
    monkeypatch.delenv("NB_PREFIX")

    assert client_lib._namespace() == "tests"
    assert client_lib._workspace_id() == "ws-tests"


def test_namespace_and_workspace_fall_back_to_nb_prefix(monkeypatch, client_lib):
    monkeypatch.delenv("NAMESPACE")
    monkeypatch.delenv("WORKSPACE_ID")
    monkeypatch.setenv("NB_PREFIX", "/notebook/ns-1/ws-1")

    assert client_lib._namespace() == "ns-1"
    assert client_lib._workspace_id() == "ws-1"


def test_notebook_namespace_is_taken_from_nb_prefix(monkeypatch, client_lib):
    monkeypatch.setenv("NAMESPACE", "notebook")
    monkeypatch.setenv("NB_PREFIX", "/notebook/ns-1/ws-1")

    assert client_lib._namespace() == "ws-1"
