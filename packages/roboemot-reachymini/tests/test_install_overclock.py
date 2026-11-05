"""SPEC 4A, Installer: the opt-in CPU overclock (roboemot.install.overclock) with the Reachy Mini
Wireless platform (roboemot_reachymini.overclock), against the fake robot and daemon of
test_install_wizard extended with the boot config, vcgencmd, sudo and a reboot. Never the real
robot."""

# ruff: noqa: F811  (the wizard's fixtures are imported, then taken as arguments)

from __future__ import annotations

import json
import shlex

import pytest
import roboemot.install as inst
from roboemot.install import Options, Ui, WizardError
from roboemot.install import overclock as oc_mod
from roboemot.install.record import InstallRecord
from roboemot_reachymini import overclock as reachy_oc
from roboemot_reachymini.install import ReachyMiniWireless
from sondalab_palette.term import MissingFlag
from test_install_wizard import (  # noqa: F401  (fixtures)
    RECORD,
    ROOT,
    FakeRobot,
    answers,
    daemon,
    flat,
    no_settle,
    options,
    pc_zone,
    plain_terminal,
    quiet_ui,
    robot,
)

CONFIG = reachy_oc.CONFIG
TRIAL = reachy_oc.TRIAL
ORIGINAL = (
    "# For more options and information see\n"
    "# http://rptl.io/configtxt\n"
    "dtparam=audio=on\n"
    "camera_auto_detect=1\n"
    "\n"
    "[cm4]\n"
    "otg_mode=1\n"
    "\n"
    "[all]\n"
    "arm_boost=1\n"
    "dtoverlay=vc4-kms-v3d\n"
)
CPU_KHZ = {False: "1500000", True: "1800000"}
ARM = {False: "arm_freq=1500", True: "arm_freq=1800"}


class BootRobot(FakeRobot):
    """The fake robot with a boot partition, firmware answers, sudo and a reboot."""

    def __init__(self) -> None:
        super().__init__()
        self.boot = {CONFIG: ORIGINAL}  # /boot/firmware files
        self.staged: dict[str, str] = {}
        self.model = "Raspberry Pi Compute Module 4 Rev 1.1"
        self.throttled = 0
        self.temp_milli = 52_000
        self.firmware = "Mar 12 2025 10:00:00\nCopyright (c) 2012 Broadcom\nversion abc (release)"
        self.bootloader = "2025/02/12 10:51:52\nversion 1 (release)\ntimestamp 1739357512"
        self.passwordless = True
        self.sudo_fails = False
        self.roots: list[tuple[str, bool]] = []
        self.reboots: list[str] = []
        self.boot_id = "boot-1"
        self.tryboot = 0  # the device tree's tryboot flag of the current boot
        self.overclocked = False  # whether the current boot runs the block
        self.never_back = False
        self.down_polls = 2  # reachable() answers False this many times after a reboot
        self.after_boot: dict[str, object] = {}  # attributes a trial boot sets (a failure)
        self._down = 0

    def run(self, command: str, check: bool = True, stdin: str | None = None) -> str:
        words = shlex.split(command) if "'" in command or '"' in command else command.split()
        if command.startswith("test -e /boot/firmware/") and "printf x" in command:
            path = words[2]
            return f"x{self.boot[path]}x" if path in self.boot else ""
        if command.startswith("sudo -n true"):
            self.commands.append(command)
            return "yes" if self.passwordless else "no"
        if command.startswith("umask 077 && cat > "):
            self.staged[words[-1]] = stdin or ""
            return ""
        if command == oc_mod.BOOT_ID:
            return self.boot_id
        if command.startswith("tr -d "):
            return self.model
        if command.startswith("vcgencmd get_throttled"):
            return f"throttled={hex(self.throttled)}"
        if command.startswith(f"cat {reachy_oc.THERMAL}"):
            return str(self.temp_milli)
        if command.startswith("vcgencmd version"):
            return self.firmware
        if command.startswith("vcgencmd bootloader_version"):
            return self.bootloader
        if command.startswith("vcgencmd get_config arm_freq"):
            return ARM[self.overclocked]
        if command.startswith(f"cat {reachy_oc.CPU_MAX_FREQ}"):
            return CPU_KHZ[self.overclocked]
        if command.startswith("od -An -tu1 "):
            return f"0 0 0 {self.tryboot}"
        if command.startswith("rm -rf /tmp/roboemot-install"):
            folder = words[2]
            self.staged = {p: t for p, t in self.staged.items() if not p.startswith(folder + "/")}
        return super().run(command, check=check, stdin=stdin)

    def run_root(self, script: str, tty: bool) -> int:
        self.roots.append((script, tty))
        if self.sudo_fails:
            return oc_mod.SUDO_DENIED_RC
        lines = script.splitlines()
        if lines[0].startswith("reboot"):
            return self._reboot(lines[0])
        assert lines[0] == "set -e" and lines[-1] == "sync", script
        for line in lines[1:-1]:
            w = shlex.split(line)
            if w[:2] == ["test", "!"]:
                if w[3] in self.boot:
                    return oc_mod.CHANGED_RC
            elif w[:2] == ["cmp", "-s"]:
                if self.boot.get(w[3]) != self.staged[w[2]]:
                    return oc_mod.CHANGED_RC
            elif w[0] == "rm":
                self.boot.pop(w[2], None)
            elif w[0] == "cp":
                assert w[2] == w[6] == w[7] + oc_mod.NEW_SUFFIX, line
                assert w[3:6] == ["&&", "mv", "-f"], line
                self.boot[w[7]] = self.staged[w[1]]
            else:
                raise AssertionError(f"unexpected root line: {line}")
        return 0

    def _reboot(self, line: str) -> int:
        self.reboots.append(line)
        self.boot_id = f"boot-{len(self.reboots) + 1}"
        self._down = self.down_polls
        trial = line == reachy_oc.TRIAL_REBOOT and TRIAL in self.boot
        self.tryboot = int(trial)
        booted = self.boot[TRIAL] if trial else self.boot[CONFIG]
        self.overclocked = reachy_oc.BEGIN in booted
        for name, value in self.after_boot.items():
            setattr(self, name, value)
        return 255  # ssh loses the connection

    def reachable(self) -> bool:
        if self.never_back:
            return False
        self._down -= 1
        return self._down < 0

    def disconnect(self) -> None:
        pass

    @property
    def record(self) -> dict:
        return json.loads(self.files[RECORD])


@pytest.fixture()
def boot_robot(monkeypatch):
    return BootRobot()


@pytest.fixture()
def offline_boot(monkeypatch, boot_robot, daemon):
    monkeypatch.setattr(inst, "load_platform", lambda rid: ReachyMiniWireless(port=daemon.port))
    monkeypatch.setattr(inst, "Remote", lambda user, host, control_dir: boot_robot)
    monkeypatch.setattr(
        inst, "build_wheels", lambda dirs, out: [out / f"{d.name}-0.1.whl" for d in dirs]
    )
    clock = iter(range(0, 10**6, 7))
    monkeypatch.setattr(oc_mod.time, "monotonic", lambda: float(next(clock)))
    monkeypatch.setattr(oc_mod.time, "sleep", lambda s: None)
    return boot_robot


@pytest.fixture()
def robot(boot_robot):  # noqa: F811  (the wizard's fixtures use the boot robot)
    return boot_robot


def install(ui: Ui | None = None, **kw) -> None:
    opts = options(**{"start": False, "startup": False, "overclock": True, **kw})
    inst.run_install(opts, ui or quiet_ui(yes=True), ROOT)


def uninstall(**kw) -> None:
    inst.run_uninstall(options(uninstall=True, **kw), quiet_ui(yes=True), ROOT)


def overclocked() -> str:
    return ORIGINAL + reachy_oc.BLOCK


# ---------------------------------------------------------------- the block


def test_the_block_sets_only_1800_mhz_for_a_cm4_and_resets_the_filter():
    lines = reachy_oc.BLOCK.splitlines()
    assert lines == [reachy_oc.BEGIN, "[all]", "[cm4]", "arm_freq=1800", "[all]", reachy_oc.END]
    assert "force_turbo" not in reachy_oc.BLOCK and "over_voltage" not in reachy_oc.BLOCK


def test_with_and_without_the_block_keep_every_other_byte():
    for text in (ORIGINAL, ORIGINAL.rstrip("\n"), ""):
        new, newline = oc_mod.with_block(text, reachy_oc.BLOCK)
        assert new.startswith(text) and newline == (bool(text) and not text.endswith("\n"))
        rec = inst.OverclockRecord(CONFIG, reachy_oc.BLOCK, text, newline)
        assert oc_mod.without_block(new, rec) == text
        edited = new.replace("arm_freq=1800", "arm_freq=2000")
        assert oc_mod.without_block(edited, rec) is None


# ---------------------------------------------------------------- support checks


@pytest.mark.parametrize(
    ("attr", "value", "check"),
    [
        ("model", "Raspberry Pi 4 Model B Rev 1.4", "Board"),
        ("throttled", 0x50000, "Throttling"),  # under-voltage and throttling since boot
        ("throttled", 0x1, "Throttling"),  # under-voltage now
        ("throttled", 0x80000, "Throttling"),  # soft temperature limit since boot
        ("temp_milli", 70_000, "Temperature"),
        ("firmware", "Oct  1 2020 10:00:00\nCopyright", "Firmware"),
        ("firmware", "", "Firmware"),
    ],
)
def test_a_failed_support_check_leaves_the_robot_unchanged_and_names_it(
    offline_boot, daemon, capsys, attr, value, check
):
    setattr(offline_boot, attr, value)
    install()
    out = flat(capsys.readouterr().out)
    assert f"! {check}" in out and "No overclock" in out and "the robot is unchanged" in out
    assert offline_boot.boot == {CONFIG: ORIGINAL} and offline_boot.roots == []
    assert offline_boot.record["overclock"] is None


@pytest.mark.parametrize(
    ("boot", "why"),
    [
        ({CONFIG: ORIGINAL + "arm_freq=2000\n"}, "already sets arm_freq (line 12)"),
        ({CONFIG: ORIGINAL + "over_voltage_delta=50000\n"}, "already sets over_voltage_delta"),
        ({CONFIG: ORIGINAL + "force_turbo=1\n"}, "already sets force_turbo"),
        ({CONFIG: ORIGINAL + reachy_oc.BLOCK}, "already has a RoboEmot block"),
        ({CONFIG: ORIGINAL, TRIAL: "foreign\n"}, "tryboot.txt exists and is not the installer's"),
        ({}, "config.txt not found"),
    ],
)
def test_a_boot_config_that_cannot_take_the_block_is_refused(
    offline_boot, daemon, capsys, boot, why
):
    offline_boot.boot = dict(boot)
    install()
    out = flat(capsys.readouterr().out)
    assert why in out and "No overclock" in out
    assert offline_boot.boot == boot and offline_boot.roots == []


def test_arm_boost_in_the_stock_config_is_no_conflict():
    assert reachy_oc.conflicts(ORIGINAL) == []
    assert reachy_oc.conflicts("#arm_freq=2000\nx=1 # arm_freq=2\n") == []


def test_build_dates_are_read_from_vcgencmd():
    assert reachy_oc.firmware_date("Mar  7 2023 10:52:00\nCopyright") == reachy_oc.date(2023, 3, 7)
    assert reachy_oc.bootloader_date("2020/09/03 12:11:43\nversion x\ntimestamp 1599135103") == (
        reachy_oc.date(2020, 9, 3)
    )
    assert reachy_oc.bootloader_date("2021/04/29 14:55:08") == reachy_oc.date(2021, 4, 29)
    assert reachy_oc.bootloader_date("") is None


# ---------------------------------------------------------------- dry run


def test_the_dry_run_prints_the_checks_and_the_diff_and_writes_nothing(
    offline_boot, daemon, capsys
):
    install(dry_run=True)
    out = capsys.readouterr().out
    for check in ("Board", "Throttling", "Temperature", "Firmware", "Bootloader", "Boot config"):
        assert f"✓ {check}" in out
    assert "+arm_freq=1800" in out and f"--- {CONFIG}" in out
    assert "Write the trial config" in flat(out)
    assert offline_boot.boot == {CONFIG: ORIGINAL} and offline_boot.roots == []
    assert offline_boot.staged == {} and offline_boot.reboots == []
    assert RECORD not in offline_boot.files


# ---------------------------------------------------------------- tryboot path


def test_the_trial_boot_is_checked_then_promoted_with_every_other_line_unchanged(
    offline_boot, daemon, capsys
):
    install(overclock_reboot=True)
    out = flat(capsys.readouterr().out)
    assert offline_boot.reboots == [reachy_oc.TRIAL_REBOOT]
    assert offline_boot.boot == {CONFIG: overclocked()}  # the trial config is gone
    assert offline_boot.boot[CONFIG].startswith(ORIGINAL)
    assert "Promoted the overclock" in out and "no second reboot" in out
    rec = offline_boot.record["overclock"]
    assert rec["state"] == "on" and rec["backup"] == ORIGINAL and rec["trial"] is None
    assert rec["block"] == reachy_oc.BLOCK
    assert all(not tty for _, tty in offline_boot.roots)  # passwordless sudo: no prompt


def test_the_trial_config_is_a_full_copy_of_the_boot_config_plus_the_block(offline_boot, daemon):
    install(overclock_reboot=False)
    assert offline_boot.boot == {CONFIG: ORIGINAL, TRIAL: overclocked()}
    assert offline_boot.record["overclock"]["state"] == "trial"
    assert offline_boot.reboots == []


def test_the_reboot_question_shows_the_warnings_and_defaults_to_no(
    offline_boot, daemon, monkeypatch, capsys
):
    monkeypatch.setattr(Ui, "ask_secret", lambda self, question: None)
    ui = Ui(interactive=True, assume_yes=True, read=answers(*["\n"] * 10))
    install(ui=ui, secrets=[])
    captured = capsys.readouterr()
    assert "Reboot the robot now into the trial config?" in captured.err
    assert "rpiboot" in flat(captured.out) and "the head may drop" in flat(captured.out)
    assert offline_boot.reboots == [] and TRIAL in offline_boot.boot
    assert "--overclock --overclock-reboot" in flat(captured.out)


def test_yes_without_the_reboot_flag_refuses_the_reboot(offline_boot, daemon, capsys):
    install()  # -y, no terminal, no --overclock-reboot
    out = flat(capsys.readouterr().out)
    assert "Did not reboot" in out and "--overclock-reboot" in out
    assert offline_boot.reboots == [] and offline_boot.boot[CONFIG] == ORIGINAL


def test_a_robot_that_does_not_come_back_is_left_for_a_power_cycle_and_cleaned_next_run(
    offline_boot, daemon, capsys
):
    offline_boot.never_back = True
    install(overclock_reboot=True)
    out = flat(capsys.readouterr().out)
    assert "Robot not back" in out and "Power-cycle the robot" in out
    assert "boots the unchanged /boot/firmware/config.txt" in out
    assert offline_boot.record["overclock"]["state"] == "trying"
    assert offline_boot.boot == {CONFIG: ORIGINAL, TRIAL: overclocked()}
    # the owner power-cycles: the flag is gone, the robot runs the unchanged config.txt
    offline_boot.never_back, offline_boot.tryboot, offline_boot.overclocked = False, 0, False
    install(overclock_reboot=True)
    out = flat(capsys.readouterr().out)
    assert "Trial boot not finished" in out and "Removed trial config" in out
    assert offline_boot.boot == {CONFIG: ORIGINAL}
    assert offline_boot.record["overclock"] is None
    assert offline_boot.reboots == [reachy_oc.TRIAL_REBOOT]  # no retry in that run


@pytest.mark.parametrize(
    ("after_boot", "failed"),
    [
        ({"overclocked": False}, "arm_freq"),
        ({"throttled": 0x50005}, "Throttling"),
        ({"temp_milli": 81_000}, "Temperature"),
        ({"tryboot": 0, "overclocked": False}, "Booted the trial config"),
    ],
)
def test_a_failed_check_after_the_trial_boot_removes_the_trial_and_does_not_promote(
    offline_boot, daemon, capsys, after_boot, failed
):
    offline_boot.after_boot = after_boot
    install(overclock_reboot=True)
    out = flat(capsys.readouterr().out)
    assert f"Overclock not promoted {failed}" in out
    assert offline_boot.boot == {CONFIG: ORIGINAL}
    assert offline_boot.record["overclock"] is None


def test_a_boot_config_edited_during_the_trial_is_not_promoted(offline_boot, daemon, capsys):
    install(overclock_reboot=False)
    offline_boot.boot[CONFIG] = ORIGINAL + "dtparam=i2c_arm=on\n"
    offline_boot.tryboot, offline_boot.overclocked = 1, True  # booted by hand
    install()
    assert "Overclock not promoted" in flat(capsys.readouterr().out)
    assert offline_boot.boot == {CONFIG: ORIGINAL + "dtparam=i2c_arm=on\n"}


def test_a_trial_booted_by_hand_is_promoted_on_the_next_run(offline_boot, daemon, capsys):
    install(overclock_reboot=False)
    offline_boot._reboot(reachy_oc.TRIAL_REBOOT)  # the owner: sudo reboot '0 tryboot'
    install()
    assert "Promoted the overclock" in flat(capsys.readouterr().out)
    assert offline_boot.boot == {CONFIG: overclocked()}


def test_a_failed_sudo_leaves_the_robot_unchanged(offline_boot, daemon):
    offline_boot.sudo_fails = True
    with pytest.raises(WizardError, match="Root command failed"):
        install(overclock_reboot=True)
    assert offline_boot.boot == {CONFIG: ORIGINAL} and offline_boot.reboots == []
    assert offline_boot.record["overclock"] is None


def test_a_refused_trial_reboot_keeps_the_trial_unbooted(offline_boot, daemon, monkeypatch):
    real = offline_boot.run_root

    def refuse_reboot(script: str, tty: bool) -> int:
        return oc_mod.SUDO_DENIED_RC if script.startswith("reboot") else real(script, tty)

    monkeypatch.setattr(offline_boot, "run_root", refuse_reboot)
    with pytest.raises(WizardError, match="sudo refused the reboot"):
        install(overclock_reboot=True)
    assert offline_boot.record["overclock"]["state"] == "trial" and offline_boot.reboots == []
    assert offline_boot.boot[CONFIG] == ORIGINAL


def test_without_a_terminal_a_sudo_that_asks_stops_before_anything_changes(offline_boot, daemon):
    offline_boot.passwordless = False
    env = dict(offline_boot.env)
    with pytest.raises(WizardError, match="The overclock needs root"):
        install()
    assert offline_boot.env == env and RECORD not in offline_boot.files
    assert offline_boot.boot == {CONFIG: ORIGINAL} and daemon.mutations == []


def test_on_a_terminal_sudo_asks_over_the_terminal(offline_boot, daemon, monkeypatch):
    offline_boot.passwordless = False
    monkeypatch.setattr(Ui, "ask_secret", lambda self, question: None)
    install(
        ui=Ui(interactive=True, assume_yes=True, read=answers()), secrets=[], overclock_reboot=True
    )
    assert offline_boot.roots and all(tty for _, tty in offline_boot.roots)


# ---------------------------------------------------------------- fallback path


def test_an_old_bootloader_writes_the_boot_config_directly_and_never_reboots(
    offline_boot, daemon, capsys
):
    offline_boot.bootloader = "2020/09/03 12:11:43\nversion x (release)\ntimestamp 1599135103"
    install(overclock_reboot=True)
    out = flat(capsys.readouterr().out)
    assert "no tryboot" in out and "Wrote the overclock block" in out
    assert offline_boot.boot == {CONFIG: overclocked()} and offline_boot.reboots == []
    assert "sudo reboot" in out  # the reboot hint
    assert offline_boot.record["overclock"]["state"] == "on"


# ---------------------------------------------------------------- revert


def promoted(robot_: BootRobot) -> None:
    install(overclock_reboot=True)
    assert robot_.boot == {CONFIG: overclocked()}


def test_a_reinstall_without_the_option_removes_exactly_the_block(offline_boot, daemon, capsys):
    promoted(offline_boot)
    offline_boot.boot[CONFIG] += "dtparam=i2c_arm=on\n"  # the owner's later edit stays
    install(overclock=False)
    out = flat(capsys.readouterr().out)
    assert offline_boot.boot == {CONFIG: ORIGINAL + "dtparam=i2c_arm=on\n"}
    assert "Removed overclock block" in out and "Did not reboot" in out
    assert offline_boot.record["overclock"] is None


def test_a_boot_config_without_a_final_newline_comes_back_byte_identical(offline_boot, daemon):
    offline_boot.boot[CONFIG] = ORIGINAL.rstrip("\n")
    install(overclock_reboot=True)
    assert offline_boot.boot[CONFIG] == ORIGINAL.rstrip("\n") + "\n" + reachy_oc.BLOCK
    install(overclock=False)
    assert offline_boot.boot == {CONFIG: ORIGINAL.rstrip("\n")}


def test_the_revert_reboot_has_its_own_question(offline_boot, daemon, capsys):
    promoted(offline_boot)
    install(overclock=False, overclock_reboot=True)
    assert offline_boot.reboots == [reachy_oc.TRIAL_REBOOT, reachy_oc.REBOOT]
    assert not offline_boot.overclocked


def test_uninstall_removes_the_block(offline_boot, daemon, capsys):
    promoted(offline_boot)
    uninstall()
    assert offline_boot.boot == {CONFIG: ORIGINAL} and RECORD not in offline_boot.files
    assert "Removed overclock block" in flat(capsys.readouterr().out)
    assert offline_boot.reboots == [reachy_oc.TRIAL_REBOOT]  # no reboot without the flag


def test_uninstall_removes_a_trial_config(offline_boot, daemon):
    install(overclock_reboot=False)
    uninstall()
    assert offline_boot.boot == {CONFIG: ORIGINAL} and RECORD not in offline_boot.files


def test_uninstall_dry_run_shows_the_diff_and_writes_nothing(offline_boot, daemon, capsys):
    promoted(offline_boot)
    roots = list(offline_boot.roots)
    uninstall(dry_run=True)
    out = capsys.readouterr().out
    assert "-arm_freq=1800" in out and "Remove overclock block" in flat(out)
    assert offline_boot.boot == {CONFIG: overclocked()} and offline_boot.roots == roots


def test_an_edited_block_is_refused_with_the_difference(offline_boot, daemon, capsys):
    promoted(offline_boot)
    edited = overclocked().replace("arm_freq=1800", "arm_freq=2000")
    offline_boot.boot[CONFIG] = edited
    install(overclock=False)
    out = capsys.readouterr().out
    assert "Left the overclock alone" in flat(out)
    assert "-arm_freq=1800" in out and "+arm_freq=2000" in out
    assert offline_boot.boot == {CONFIG: edited}
    assert offline_boot.record["overclock"]["state"] == "on"
    with pytest.raises(WizardError, match="Uninstall incomplete"):
        uninstall()
    assert offline_boot.boot == {CONFIG: edited} and RECORD in offline_boot.files


def test_a_block_removed_by_hand_is_forgotten(offline_boot, daemon, capsys):
    promoted(offline_boot)
    offline_boot.boot[CONFIG] = ORIGINAL
    install(overclock=False)
    assert "No overclock block left" in flat(capsys.readouterr().out)
    assert offline_boot.record["overclock"] is None and offline_boot.boot == {CONFIG: ORIGINAL}


def test_a_reinstall_with_the_option_keeps_the_block(offline_boot, daemon, capsys):
    promoted(offline_boot)
    install(overclock_reboot=True)
    assert "Overclock on" in flat(capsys.readouterr().out)
    assert offline_boot.boot == {CONFIG: overclocked()}
    assert offline_boot.reboots == [reachy_oc.TRIAL_REBOOT]


# ---------------------------------------------------------------- opt-in


def test_yes_on_a_terminal_does_not_turn_it_on(offline_boot, daemon, monkeypatch, capsys):
    monkeypatch.setattr(Ui, "ask_secret", lambda self, question: None)
    ui = Ui(interactive=True, assume_yes=True, read=answers())
    install(ui=ui, overclock=None, secrets=[])
    captured = capsys.readouterr()
    assert "Overclock the CPU" not in captured.err
    assert "Overclock off -y does not turn it on, --overclock does" in flat(captured.out)
    assert offline_boot.boot == {CONFIG: ORIGINAL} and offline_boot.roots == []


def test_the_overclock_question_defaults_to_no(offline_boot, daemon, monkeypatch, capsys):
    monkeypatch.setattr(Ui, "ask_secret", lambda self, question: None)
    ui = Ui(interactive=True, assume_yes=False, read=answers(*["\n"] * 20))
    opts = options(overclock=None, start=False, startup=False, piper=False, models=False)
    inst.run_install(Options(**{**vars(opts), "yes": False, "secrets": []}), ui, ROOT)
    assert "Overclock the CPU (1.8 GHz, from 1.5)?" in capsys.readouterr().err
    assert offline_boot.boot == {CONFIG: ORIGINAL} and offline_boot.roots == []


@pytest.mark.parametrize("yes", [False, True])
def test_a_run_without_a_terminal_needs_an_overclock_flag(offline_boot, daemon, yes):
    opts = options(overclock=None, start=False, startup=False, yes=yes, piper=False, models=False)
    with pytest.raises(MissingFlag, match="--overclock or --no-overclock"):
        inst.run_install(opts, quiet_ui(yes=yes), ROOT)
    assert offline_boot.boot == {CONFIG: ORIGINAL}


def test_a_platform_without_an_overclock_refuses_the_flag_and_is_a_no_op_without(
    offline_boot, daemon, monkeypatch, capsys
):
    bare = ReachyMiniWireless(port=daemon.port)
    bare.overclock = None
    monkeypatch.setattr(inst, "load_platform", lambda rid: bare)
    with pytest.raises(WizardError, match="No overclock for Reachy Mini Wireless"):
        install()
    inst.run_install(options(start=False, startup=False, overclock=None), quiet_ui(yes=True), ROOT)
    assert "No overclock the boot config is not touched" in flat(capsys.readouterr().out)
    assert offline_boot.boot == {CONFIG: ORIGINAL} and offline_boot.roots == []


def test_the_rerun_line_repeats_the_overclock_flags():
    line = inst._rerun(Options(host="h", overclock=True, overclock_reboot=False))
    assert "--overclock" in line and "--no-overclock-reboot" in line
    assert inst.parse_args(["--overclock", "--overclock-reboot"]).overclock_reboot is True
    assert inst.parse_args([]).overclock is None


def test_the_record_keeps_the_backup_and_round_trips(offline_boot, daemon):
    install(overclock_reboot=False)
    data = offline_boot.record
    assert data["version"] == 4 and data["overclock"]["backup"] == ORIGINAL
    assert InstallRecord.from_json(data).to_json() == data


# ---------------------------------------------------------------- diagnostics


def test_the_debug_report_says_whether_the_block_is_there(tmp_path):
    from roboemot.diagnostics import Target
    from roboemot_reachymini.diagnostics import overclock_state

    boot = tmp_path / "boot" / "firmware"
    assert overclock_state(Target(), tmp_path)["available"] is False
    boot.mkdir(parents=True)
    (boot / "config.txt").write_text(overclocked())
    assert overclock_state(Target(), tmp_path) == {
        "available": True,
        "block_present": True,
        "trial_pending": False,
    }
    assert overclock_state(Target("192.168.1.20"), tmp_path)["available"] is False
