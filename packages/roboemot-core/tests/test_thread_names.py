"""Threads carry their Python name as their OS name (SPEC 13, diagnostics)."""

from __future__ import annotations

import inspect
import threading

import pytest
from roboemot import thread_names
from roboemot.runtime import runtime


@pytest.fixture
def written(monkeypatch):
    """Linux-like seams: every name a thread writes is recorded with the thread that wrote it."""
    names: list[tuple[str, str]] = []
    monkeypatch.setattr(thread_names, "is_supported", lambda: True)
    monkeypatch.setattr(
        thread_names, "write_comm", lambda n: names.append((n, threading.current_thread().name))
    )
    assert thread_names.install()
    yield names
    thread_names.uninstall()


def run(thread: threading.Thread) -> None:
    thread.start()
    thread.join()


def test_a_started_thread_writes_its_name_from_inside_itself_before_its_work(written):
    seen: list[int] = []
    run(threading.Thread(target=lambda: seen.append(len(written)), name="roboemot-light"))
    assert written == [("roboemot-light", "roboemot-light")]
    assert seen == [1]  # the name was there when the work started


def test_the_name_is_cut_to_what_the_kernel_keeps(written):
    run(threading.Thread(target=lambda: None, name="roboemot-listener"))
    assert written[0][0] == "roboemot-listen"
    thread_names.name_current("è" * 10)  # 20 bytes: cut at a whole character
    assert written[-1][0] == "è" * 7


def test_a_subclass_with_its_own_run_is_named_and_keeps_no_wrapper(written):
    class Worker(threading.Thread):
        ran = False

        def run(self) -> None:
            self.ran = True

    worker = Worker(name="motion-loop")
    run(worker)
    assert worker.ran and written == [("motion-loop", "motion-loop")]
    assert "run" not in worker.__dict__


def test_a_name_that_cannot_be_written_does_not_stop_the_thread(monkeypatch, written):
    def refuse(name: str) -> None:
        raise PermissionError(name)

    monkeypatch.setattr(thread_names, "write_comm", refuse)
    done: list[bool] = []
    run(threading.Thread(target=lambda: done.append(True), name="speech-output"))
    assert done == [True]


def test_it_changes_nothing_where_it_does_not_apply_and_installs_once(monkeypatch):
    start = threading.Thread.start
    monkeypatch.setattr(thread_names, "is_supported", lambda: False)
    assert not thread_names.install()
    assert threading.Thread.start is start
    monkeypatch.setattr(thread_names, "is_supported", lambda: True)
    assert thread_names.install() and thread_names.install()
    thread_names.uninstall()
    assert threading.Thread.start is start


def test_python_with_native_names_and_other_systems_are_left_alone(monkeypatch):
    monkeypatch.setattr(thread_names.sys, "platform", "linux")
    monkeypatch.setattr(thread_names.sys, "version_info", (3, 12, 0))
    assert thread_names.supported()
    monkeypatch.setattr(thread_names.sys, "version_info", (3, 14, 0))
    assert not thread_names.supported()
    monkeypatch.setattr(thread_names.sys, "version_info", (3, 12, 0))
    monkeypatch.setattr(thread_names.sys, "platform", "darwin")
    assert not thread_names.supported()


def test_the_runtime_installs_the_names_before_it_builds_any_thread():
    source = inspect.getsource(runtime.Runtime.run)
    assert 0 < source.index("thread_names.install()") < source.index("assemble(")
