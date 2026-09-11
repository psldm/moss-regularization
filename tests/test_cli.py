"""Smoke tests for the moss-reg command-line interface."""

import pytest

from moss_reg.cli import main


def test_decay_benchmark_writes_figure(tmp_path):
    rc = main(["benchmark", "--type", "decay", "--output", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "01_velocity_decay.png").is_file()


def test_fluid_benchmark_writes_figure(tmp_path):
    rc = main(
        [
            "benchmark",
            "--type",
            "fluid",
            "--output",
            str(tmp_path),
            "--n",
            "32",
            "--t-max",
            "1.0",
            "--lambda",
            "0.5",
            "--re",
            "200",
        ]
    )
    assert rc == 0
    assert (tmp_path / "03_cfd_stability.png").is_file()


def test_particles_benchmark_writes_figure(tmp_path):
    rc = main(
        [
            "benchmark",
            "--type",
            "particles",
            "--output",
            str(tmp_path),
            "--n",
            "64",
            "--t-max",
            "1.0",
            "--c",
            "0.005",
        ]
    )
    assert rc == 0
    assert (tmp_path / "02_shell_crossing_arrest.png").is_file()


def test_unknown_benchmark_type_rejected():
    with pytest.raises(SystemExit):
        main(["benchmark", "--type", "nonsense"])


def test_no_command_rejected():
    with pytest.raises(SystemExit):
        main([])
