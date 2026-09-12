"""The LaTeX supplement is generated from report.json only."""

import json

from moss_reg import validate as V
from moss_reg.cli import main
from moss_reg.report import make_entry, write_report
from moss_reg.supplement import esc, generate_supplement, num


def test_escaping_and_number_formatting():
    assert esc("a_b%c&d") == r"a\_b\%c\&d"
    assert num(None) == "---"
    assert num("inf") == r"$\infty$"
    assert num(True) == "PASS" and num(False) == "FAIL"
    assert num(3) == "3"
    assert num(0.5) == "0.5"
    assert num(2.5e-7) == r"$2.500\times10^{-7}$"


def test_supplement_from_real_report(tmp_path):
    # a small but real report: validate + one fluid comparison
    main(["compare", "--type", "fluid", "--output", str(tmp_path), "--n", "32", "--t-max", "0.3"])
    results = V.run_checks()
    entry = make_entry(
        "validate", {}, {"failed": V.has_failures(results), "results": [r.__dict__ for r in results]},
        [], checks={r.name: r.status for r in results}, argv=["run-all"],
    )
    write_report(tmp_path, entry)
    out = tmp_path / "paper" / "supplement.tex"
    rc = main(["supplement", "--report", str(tmp_path / "report.json"), "--output", str(out),
               "--copy-figures", "--subtitle", "test run"])
    assert rc == 0
    tex = out.read_text()
    assert tex.startswith("\\documentclass") and tex.rstrip().endswith("\\end{document}")
    assert "\\section{Provenance}" in tex
    assert "Self-validation" in tex and "XFAIL" in tex
    assert "ODE coupling" in tex                      # detail rows present
    assert "Taylor--Green vortex" in tex
    assert "\\includegraphics[width=\\textwidth]{figures/03_cfd_stability.png}" in tex
    assert (out.parent / "figures" / "03_cfd_stability.png").is_file()
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["runs"][0]["params"]["t_max"] == 0.3
    assert "0.3" in tex
