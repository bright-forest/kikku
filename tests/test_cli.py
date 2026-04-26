"""Tests for v2 parse_cli, TestSpec, RunSpec, merge order (kikku-runspec-v2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from kikku.run.cli import parse_cli
from kikku.run.types import RunSpec, SimSpec

MODES = ["compare", "sweep", "simulate", "mpi", "gpu", "plots"]


@pytest.fixture
def base_dir(tmp_path) -> Path:
    """calibration + settings; keys disjoint except we add overlap variant elsewhere."""
    calib = {
        "calibration": {
            "beta": 0.96,
            "r": 0.02,
        }
    }
    settings = {
        "settings": {
            "grid_size": 3000,
            "T": 20,
        }
    }
    (tmp_path / "calibration.yaml").write_text(yaml.dump(calib))
    (tmp_path / "settings.yaml").write_text(yaml.dump(settings))
    return tmp_path


def _ps(monkeypatch, base: Path, argv: list, **kw):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prologue"] + argv,
    )
    d = {
        "name": "m",
        "base_spec": str(base),
        "modes": MODES,
        "output": str(base / "out"),
    }
    d.update(kw)
    return parse_cli(**d)


def test_single_test_default(monkeypatch, base_dir):
    run = _ps(monkeypatch, base_dir, [])
    assert isinstance(run, RunSpec)
    assert run.mode == "single"
    assert len(run.test_set) == 1
    t = run.test_set[0]
    assert t.params == {}
    assert t.methods is None
    assert t.settings == {}
    assert t.label == ""
    assert run.sim is None


def test_compare_two_rows_and_labels(monkeypatch, base_dir):
    run = _ps(
        monkeypatch,
        base_dir,
        ["--compare", "beta=0.92,0.99"],
    )
    assert run.mode == "compare"
    assert len(run.test_set) == 2
    assert {run.test_set[0].label, run.test_set[1].label} == {repr(0.92), repr(0.99)}
    bvals = {run.test_set[0].params["beta"], run.test_set[1].params["beta"]}
    assert bvals == {0.92, 0.99}


def test_sweep_cartesian_product(monkeypatch, base_dir):
    pl = json.dumps([{"beta": 0.9}, {"beta": 0.91}])
    ml = json.dumps(
        [
            {
                "stage.sub.target.scheme1": "FUES",
            },
            {
                "stage.sub.target.scheme1": "MSS",
            },
        ]
    )
    run = _ps(
        monkeypatch,
        base_dir,
        [
            "--sweep",
            f"--params-range={pl}",
            f"--methods-range={ml}",
        ],
    )
    assert run.mode == "sweep"
    assert len(run.test_set) == 4
    sigs = {
        (t.params.get("beta"), tuple(t.methods.items()) if t.methods else None)
        for t in run.test_set
    }
    assert len(sigs) == 4
    t0 = run.test_set[0]
    assert ("stage", "sub", "target", "scheme1") in (t0.methods or {})


def test_params_settings_overlap_raises(monkeypatch, tmp_path):
    cal = {"calibration": {"dup": 1, "a": 1}}
    st = {"settings": {"dup": 2, "b": 2}}
    (tmp_path / "calibration.yaml").write_text(yaml.dump(cal))
    (tmp_path / "settings.yaml").write_text(yaml.dump(st))
    with pytest.raises(ValueError, match="params_keys"):
        _ps(monkeypatch, tmp_path, [])


def test_unknown_param_key(monkeypatch, base_dir):
    with pytest.raises(ValueError, match="Unknown params key"):
        _ps(
            monkeypatch,
            base_dir,
            ["--params-override", "notakey=1"],
        )


def test_methods_override_too_few_dotted_parts(monkeypatch, base_dir):
    with pytest.raises(ValueError, match="three dot-separated"):
        _ps(
            monkeypatch,
            base_dir,
            ["--methods-override", "no=FUES"],
        )


def test_merge_precedence_spec_then_override(monkeypatch, base_dir):
    run = _ps(
        monkeypatch,
        base_dir,
        [
            "--params-spec",
            '{"beta":0.95}',
            "--params-override",
            "beta=0.97",
        ],
    )
    assert run.test_set[0].params["beta"] == 0.97


def test_merge_precedence_override_then_spec(monkeypatch, base_dir):
    run = _ps(
        monkeypatch,
        base_dir,
        [
            "--params-override",
            "beta=0.97",
            "--params-spec",
            '{"beta":0.95}',
        ],
    )
    assert run.test_set[0].params["beta"] == 0.95


def test_dict_literal_matches_atfile(tmp_path, monkeypatch, base_dir):
    f = tmp_path / "p.yaml"
    f.write_text("beta: 0.88\nr: 0.03\n")
    a = _ps(
        monkeypatch,
        base_dir,
        [f"--params-spec=@{f}"],
    )
    b = _ps(
        monkeypatch,
        base_dir,
        ['--params-spec', '{"beta":0.88,"r":0.03}'],
    )
    assert a.test_set[0] == b.test_set[0]


def test_compare_and_sweep_mutually_exclusive(monkeypatch, base_dir):
    with pytest.raises(ValueError, match="compare and --sweep"):
        _ps(
            monkeypatch,
            base_dir,
            [
                "--compare",
                "beta=0.9,0.91",
                "--sweep",
            ],
        )


def test_simulate_creates_simspec(monkeypatch, base_dir):
    r = _ps(monkeypatch, base_dir, ["--simulate", "--n-sim", "5000", "--seed", "3"])
    assert isinstance(r.sim, SimSpec)
    assert r.sim.n_sim == 5000
    assert r.sim.seed == 3

    r2 = _ps(monkeypatch, base_dir, [])
    assert r2.sim is None


def test_params_range_atfile_roundtrip(tmp_path, monkeypatch, base_dir):
    f = tmp_path / "pr.yaml"
    f.write_text(
        yaml.dump(
            [
                {"beta": 0.9},
                {"beta": 0.91},
            ]
        )
    )
    inline = json.dumps([{"beta": 0.9}, {"beta": 0.91}])
    a = _ps(
        monkeypatch,
        base_dir,
        [f"--params-range=@{f}"],
    )
    b = _ps(
        monkeypatch,
        base_dir,
        [f"--params-range={inline}"],
    )
    assert a.test_set == b.test_set
    assert a.mode == "sweep" and b.mode == "sweep"
    assert len(a.test_set) == 2


def test_params_spec_unknown_key_in_literal_dict(monkeypatch, base_dir):
    with pytest.raises(ValueError, match="Unknown params key"):
        _ps(
            monkeypatch,
            base_dir,
            ['--params-spec', '{"notakey":1}'],
        )


def test_settings_override_misroutes_param_key_to_settings(monkeypatch, base_dir):
    with pytest.raises(ValueError, match="params-override|params-spec"):
        _ps(
            monkeypatch,
            base_dir,
            ["--settings-override", "beta=0.96"],
        )


def test_compare_on_methods_path_tuple_keys(monkeypatch, base_dir):
    run = _ps(
        monkeypatch,
        base_dir,
        [
            "--compare",
            "adjuster_cons.cntn_to_dcsn_mover.bellman_backward=EGM,NEGM",
        ],
    )
    assert run.mode == "compare"
    assert len(run.test_set) == 2
    path = (
        "adjuster_cons",
        "cntn_to_dcsn_mover",
        "bellman_backward",
    )
    for t in run.test_set:
        assert t.methods is not None
        assert path in t.methods
        assert t.methods[path] in ("EGM", "NEGM")


def test_params_override_folds_into_params_range_rows(monkeypatch, base_dir):
    run = _ps(
        monkeypatch,
        base_dir,
        [
            "--params-override",
            "r=0.03",
            "--params-range",
            json.dumps([{"beta": 0.9}, {"beta": 0.91}]),
        ],
    )
    assert run.mode == "sweep"
    for t in run.test_set:
        assert t.params["r"] == 0.03
    assert {t.params["beta"] for t in run.test_set} == {0.9, 0.91}


def test_compare_plus_params_range_rejected(monkeypatch, base_dir):
    with pytest.raises(ValueError, match="--compare cannot be combined"):
        _ps(
            monkeypatch,
            base_dir,
            [
                "--compare",
                "beta=0.9,0.91",
                "--params-range",
                json.dumps([{"r": 0.03}]),
            ],
        )
