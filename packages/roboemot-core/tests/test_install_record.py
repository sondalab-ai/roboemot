"""SPEC 4A, Packaging: the installer's change record and uninstall plan (robot-agnostic)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from roboemot.install.build import distribution_names, min_python, third_party_requirements
from roboemot.install.pyenv import RemotePython, report_changes
from roboemot.install.record import (
    RECORD_VERSION,
    InstallRecord,
    Replaced,
    canonical,
    plan_uninstall,
    unknown_keys,
)
from roboemot.install.ui import WizardError

ROOT = Path(__file__).resolve().parents[3]
OWN = ["roboemot-core", "roboemot-reachymini"]


def rec(**kw) -> InstallRecord:
    return InstallRecord("roboemot", "bot", OWN, **kw)


def test_canonical_names():
    assert canonical("Foo_Bar.baz") == "foo-bar-baz"


def test_merge_records_added_and_replaced_but_not_own():
    r = rec()
    before = {"numpy": "2.0.0", "pip": "26.0"}
    after = {"numpy": "2.1.0", "pip": "26.0", "scipy": "1.18.1", "roboemot-core": "0.1"}
    r.pending_before = before
    r.merge(before, after)
    assert r.added == ["scipy"]
    assert r.replaced == {"numpy": Replaced("2.0.0", "2.1.0")}
    assert r.pending_before is None


def test_reinstall_merge_keeps_the_original_version():
    r = rec(added=["scipy"], replaced={"numpy": Replaced("2.0.0", "2.1.0")})
    r.merge({"numpy": "2.1.0", "scipy": "1.18.1"}, {"numpy": "2.2.0", "scipy": "1.18.1", "x": "1"})
    assert r.replaced["numpy"] == Replaced("2.0.0", "2.2.0")
    assert r.added == ["scipy", "x"]


def test_record_round_trips_through_json():
    r = rec(added=["scipy"], replaced={"numpy": Replaced("2.0.0", "2.1.0")}, pending_before={})
    assert InstallRecord.from_json(json.loads(json.dumps(r.to_json()))) == r


def test_plan_removes_own_and_added_restores_replaced():
    r = rec(added=["scipy"], replaced={"numpy": Replaced("2.0.0", "2.1.0")})
    installed = {
        "roboemot-core": "0.1",
        "roboemot-reachymini": "0.1",
        "scipy": "1",
        "numpy": "2.1.0",
    }
    requires = {"roboemot-core": ["numpy", "scipy"], "roboemot-reachymini": ["roboemot-core"]}
    plan = plan_uninstall(r, installed, requires)
    assert plan.remove == ["roboemot-core", "roboemot-reachymini", "scipy"]
    assert plan.restore == {"numpy": "2.0.0"}
    assert plan.kept == {} and plan.drifted == {}


def test_plan_keeps_what_another_app_now_needs_and_its_dependencies():
    r = rec(added=["scipy", "fortran-rt"])
    installed = {"roboemot-core": "0.1", "scipy": "1", "fortran-rt": "1", "other-app": "3"}
    requires = {"scipy": ["fortran-rt"], "other-app": ["scipy"], "roboemot-core": ["scipy"]}
    plan = plan_uninstall(r, installed, requires)
    assert plan.remove == ["roboemot-core"]
    assert plan.kept == {"scipy": ["other-app"], "fortran-rt": ["scipy"]}


def test_plan_leaves_versions_someone_else_changed():
    r = rec(replaced={"numpy": Replaced("2.0.0", "2.1.0"), "httpx": Replaced("0.27", "0.28")})
    plan = plan_uninstall(r, {"numpy": "2.3.0", "httpx": "0.27"}, {})
    assert plan.restore == {} and plan.drifted == {"numpy": "2.3.0"}
    assert plan.empty


def test_plan_without_record_removes_only_own_installed_packages():
    plan = plan_uninstall(rec(), {"roboemot-core": "0.1", "scipy": "1"}, {})
    assert plan.remove == ["roboemot-core"]


def test_constraints_pin_everything_but_own():
    pins = RemotePython.constraints({"numpy": "2.0", "roboemot-core": "0.1"}, {"roboemot-core"})
    assert pins == "numpy==2.0\n"


def test_report_changes_splits_added_and_changed():
    report = {
        "install": [
            {"metadata": {"name": "SciPy", "version": "1.18.1"}},
            {"metadata": {"name": "numpy", "version": "2.1.0"}},
            {"metadata": {"name": "httpx", "version": "0.28"}},
        ]
    }
    added, changed = report_changes(report, {"numpy": "2.0.0", "httpx": "0.28"})
    assert added == {"scipy": "1.18.1"} and changed == {"numpy": "2.0.0 -> 2.1.0"}


def test_build_metadata_from_the_checkout():
    dirs = [ROOT / "packages" / p for p in OWN]
    assert distribution_names(dirs) == OWN
    assert min_python(dirs) == (3, 11)
    reqs = third_party_requirements(dirs)
    assert not any(r.startswith("roboemot") for r in reqs)
    assert any(r.startswith("scipy") for r in reqs) and any(r.startswith("psutil") for r in reqs)


def test_extras_add_their_requirements_to_the_dependency_check():
    dirs = [ROOT / "packages" / p for p in OWN]
    base = set(third_party_requirements(dirs))
    voice = set(third_party_requirements(dirs, ("voice",))) - base
    assert any(r.startswith("vosk") for r in voice)
    assert not any(r.startswith("opencv") for r in voice)


# ---------------------------------------------------------------- the flow with a non-Reachy robot


class BareRobot:
    """A robot with no daemon: just SSH, a venv and pip (emulated)."""

    target = "dev@bare"

    def __init__(self) -> None:
        self.env = {"pip": "26.0", "numpy": "2.0.0"}
        self.files: dict[str, str] = {}

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def copy(self, files, dest) -> None:
        pass

    def stream(self, command, on_line, check: bool = True) -> str:
        out = self.run(command, check=check)
        for line in out.splitlines():
            on_line(line)
        return out

    def run(self, command: str, check: bool = True) -> str:
        import shlex as sh

        if " -m pip " in command:
            args = command.split(" -m pip ", 1)[1].split(" 2>&1", 1)[0]
            args = args.replace("--progress-bar off ", "")
            targets = [t for t in sh.split(args) if not t.startswith("-")][1:]
            if args.startswith("list"):
                rows = [{"name": n, "version": v} for n, v in self.env.items()]
                return json.dumps(rows) + "\n__rc=0"
            if args.startswith("uninstall"):
                for t in targets:
                    self.env.pop(t, None)
            elif "--dry-run" in args:
                return json.dumps({"install": []}) + "\n__rc=0"
            elif "--force-reinstall" not in args:
                self.env.update({"bare-app": "1.0", "tomli": "2.0"})
            return "\n__rc=0"
        plain = " ".join(sh.split(command)) if command.startswith("/opt") else command
        if command.startswith("test -e"):
            path = sh.split(command)[2]
            there = any(f == path or f.startswith(path + "/") for f in self.files)
            return "yes" if there else "no"
        if command.startswith("realpath "):
            return sh.split(command)[1]
        if command.startswith("test -"):
            return "yes" if command.startswith("test -x") else "no"
        if "platform.machine" in plain:
            return "x86_64\n3.12"
        if "distributions()" in plain:
            return json.dumps({n: [] for n in self.env})
        if "find_spec" in plain:
            return json.dumps(
                {"bare_app": ["/opt/venv/site/bare_app"]} if "bare-app" in self.env else {}
            )
        if "purelib" in plain:
            return "/opt/venv/site"
        if command.startswith("df "):
            return "/dev/sda 1 1 99999999 1% /"
        if command.startswith("cat "):
            return self.files.get(sh.split(command)[1], "")
        if command == 'printf %s "$HOME"':
            return "/home/dev"
        if command == 'printf %s "${ROBOEMOT_DATA_DIR:-}"':
            return ""
        if command.startswith("umask 077 && mkdir -p "):  # the app config
            t = sh.split(command)
            self.files[t[-1]] = t[t.index("printf") + 2]
            return ""
        if command.startswith("rm -rf "):
            path = sh.split(command)[2]
            self.files = {f: v for f, v in self.files.items() if not f.startswith(path + "/")}
            return ""
        if command.startswith("printf %s"):
            t = sh.split(command)
            self.files[t[8]] = t[2]
            return ""
        if command.startswith("mktemp"):
            return "/tmp/roboemot-install.a1"
        if command.startswith("rm -f "):
            self.files.pop(sh.split(command)[2], None)
        return ""


class BarePlatform:
    manifest_id = "bare-v1"
    display_name = "Bare robot"
    app_name = "bare"
    default_host = "bare.local"
    default_user = "dev"
    arch = "x86_64"
    python = "/opt/venv/bin/python"
    env_dir = "/opt/venv"
    base_python = None
    record_path = "/opt/.bare-install.json"
    space_paths = ("/tmp",)
    distributions = ("bare-app",)
    modules = ("bare_app",)
    extras = ()
    clearable_caches = ()

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp

    def connect(self, ui, host): ...
    def plan_env(self, remote):
        return ""

    def create_env(self, ui, remote, plan): ...
    def env_users(self, py):
        return []

    def env_ready(self, py):
        return True

    def probe(self, ui, py): ...
    def verify(self, ui, py): ...
    def running_app(self):
        return None

    def stop_app(self): ...
    def app_installed(self):
        return True

    def app_url(self, host):
        return None

    def start_blocker(self):
        return None

    def start(self, ui): ...
    def autostart(self):
        return None

    def set_autostart(self, name): ...
    def cleanup(self, ui, py, dry_run): ...
    def verify_removed(self, ui): ...

    def stage(self, root, dest):
        pkg = dest / "bare-app"
        pkg.mkdir(parents=True)
        (pkg / "pyproject.toml").write_text(
            '[project]\nname = "bare-app"\nrequires-python = ">=3.10"\ndependencies = ["tomli"]\n'
        )
        return [pkg]


def test_install_and_uninstall_work_without_any_reachy_code(tmp_path, monkeypatch):
    import roboemot.install as inst

    robot = BareRobot()
    monkeypatch.setattr(inst, "load_platform", lambda rid: BarePlatform(tmp_path))
    monkeypatch.setattr(inst, "Remote", lambda user, host, d: robot)
    monkeypatch.setattr(inst, "build_wheels", lambda dirs, out: [out / "bare_app-1.0.whl"])
    opts = dict(host="bare", user="dev", robot="bare-v1", dev=False, start=False, startup=False)
    ui = inst.Ui(interactive=False, assume_yes=True)
    before = dict(robot.env)
    inst.run_install(inst.Options(**opts), ui, tmp_path)
    assert json.loads(robot.files[BarePlatform.record_path])["added"] == ["tomli"]
    inst.run_uninstall(inst.Options(**opts, uninstall=True), ui, tmp_path)
    assert robot.env == before and robot.files == {}


def test_a_record_from_before_env_created_still_loads():
    old = rec().to_json()
    del old["env_created"], old["env_baseline"]
    loaded = InstallRecord.from_json(old)
    assert loaded.env_created is None and loaded.env_baseline == []


def test_env_created_round_trips():
    r = rec(env_created="/venvs/apps_venv")
    assert InstallRecord.from_json(json.loads(json.dumps(r.to_json()))).env_created == (
        "/venvs/apps_venv"
    )


def test_a_record_from_a_newer_installer_is_refused_with_a_clear_error():
    """R4 S1/T4: a record written by a newer installer is a WizardError, not a TypeError."""
    newer = {**rec().to_json(), "version": RECORD_VERSION + 1, "brand_new": 1}
    with pytest.raises(WizardError, match="newer than this installer"):
        InstallRecord.from_json(newer)


def test_unknown_keys_of_the_same_version_are_dropped_and_named():
    data = {**rec(added=["scipy"]).to_json(), "brand_new": [1]}
    loaded = InstallRecord.from_json(data)
    assert loaded.added == ["scipy"] and unknown_keys(data) == ["brand_new"]


def test_an_old_record_is_written_back_at_the_current_version():
    old = {"app": "a", "platform": "p", "packages": ["x"], "version": 1}
    assert InstallRecord.from_json(old).to_json()["version"] == RECORD_VERSION == 2


@pytest.mark.parametrize("version", ["2", None, True, 1.5])
def test_an_unreadable_record_version_is_named_as_such(version):
    with pytest.raises(WizardError, match="unreadable version"):
        InstallRecord.from_json({**rec().to_json(), "version": version})
