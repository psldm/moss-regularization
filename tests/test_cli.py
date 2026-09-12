"""Tests for the moss-reg command-line interface."""

import json

import pytest

from moss_reg.cli import main


def test_validate_passes_and_writes_json(tmp_path, capsys):
    out = tmp_path / "validate.json"
    rc = main(["validate", "--json", str(out)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "0 failed" in text and "XFAIL" in text
    data = json.loads(out.read_text())
    assert data["failed"] is False
    statuses = {r["name"]: r["status"] for r in data["results"]}
    assert all(s in ("PASS", "XFAIL") for s in statuses.values())
    assert sum(s == "XFAIL" for s in statuses.values()) == 1


def test_validate_strict_fails_on_known_issue():
    assert main(["validate", "--strict"]) == 1


def _report(tmp_path):
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["schema"] == "moss-reg-report/1"
    return {r["command"]: r for r in data["runs"]}


def test_compare_particles_writes_figure_and_report(tmp_path):
    rc = main(["compare", "--type", "particles", "--output", str(tmp_path),
               "--n", "48", "--t-max", "1.5", "--compactness", "0.1"])
    assert rc == 0
    assert (tmp_path / "02_shell_crossing_arrest.png").is_file()
    run = _report(tmp_path)["compare:particles"]
    assert set(run) >= {"git", "environment", "params", "metrics", "checks", "figures", "timestamp_utc"}
    assert "commit" in run["git"] and "dirty" in run["git"]
    assert run["params"]["c"] == pytest.approx(10 ** 0.5, rel=1e-12)
    assert run["metrics"]["regime"].startswith("physical")
    assert all(run["checks"].values())


def test_compare_particles_flags_inconclusive_horizon(tmp_path):
    # classical run does not cross within t_max: checks fail, exit code 1
    rc = main(["compare", "--type", "particles", "--output", str(tmp_path),
               "--n", "48", "--t-max", "0.8"])
    assert rc == 1
    assert _report(tmp_path)["compare:particles"]["checks"]["undamped_crossing_detected"] is False


def test_compare_c_overrides_compactness(tmp_path):
    rc = main(["compare", "--type", "particles", "--output", str(tmp_path),
               "--n", "48", "--t-max", "1.5", "--c", "0.02"])
    assert rc == 0
    run = _report(tmp_path)["compare:particles"]
    assert run["params"]["c"] == 0.02
    assert run["params"]["compactness"] == pytest.approx(2500.0)
    assert "superluminal" in run["metrics"]["regime"]


def test_compare_fluid_writes_figure_and_report(tmp_path):
    rc = main(["compare", "--type", "fluid", "--output", str(tmp_path),
               "--n", "32", "--t-max", "0.5", "--lambda", "0.5", "--re", "200"])
    assert rc == 0
    assert (tmp_path / "03_cfd_stability.png").is_file()
    run = _report(tmp_path)["compare:fluid"]
    assert run["checks"]["taylor_green_exact_decay"] is True
    assert run["checks"]["energy_identity"] is True
    assert run["metrics"]["dt_limits_initial"]["damping"] > 0


def test_reports_merge_by_command(tmp_path):
    main(["compare", "--type", "fluid", "--output", str(tmp_path), "--n", "32", "--t-max", "0.2"])
    main(["compare", "--type", "particles", "--output", str(tmp_path), "--n", "32", "--t-max", "0.5"])
    main(["compare", "--type", "fluid", "--output", str(tmp_path), "--n", "32", "--t-max", "0.3"])
    runs = _report(tmp_path)
    assert set(runs) == {"compare:fluid", "compare:particles"}
    assert runs["compare:fluid"]["params"]["t_max"] == 0.3


def test_sweep_quick_writes_figure_and_report(tmp_path):
    rc = main(["sweep", "--quick", "--output", str(tmp_path), "--t-max", "1.5"])
    assert rc == 0
    assert (tmp_path / "02_shell_crossing_sweep.png").is_file()
    run = _report(tmp_path)["sweep:particles"]
    assert run["params"]["quick"] is True
    assert len(run["metrics"]["runs"]) == len(run["params"]["c_values"]) * len(run["params"]["n_values"])
    assert all(run["checks"].values())


def test_legacy_benchmark_commands_still_work(tmp_path):
    assert main(["benchmark", "--type", "decay", "--output", str(tmp_path)]) == 0
    assert (tmp_path / "01_velocity_decay.png").is_file()
    assert main(["benchmark", "--type", "particles", "--output", str(tmp_path),
                 "--n", "32", "--t-max", "0.5", "--c", "0.005"]) == 0
    assert (tmp_path / "02_shell_crossing_arrest.png").is_file()


def test_unknown_benchmark_type_rejected():
    with pytest.raises(SystemExit):
        main(["benchmark", "--type", "nonsense"])


def test_no_command_rejected():
    with pytest.raises(SystemExit):
        main([])
