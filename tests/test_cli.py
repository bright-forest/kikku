"""Tests for v3 parse_cli, TestSpec, RunSpec, merge order (kikku-runspec-v3)."""

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
    """v3: no calibration/settings reads; any directory is valid base_spec path."""
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
    assert t.slots == {}
    assert t.label == ""
    assert run.sim is None
    assert not hasattr(run, "params_keys")


def test_compare_two_rows_and_labels(monkeypatch, base_dir):
    run = _ps(
        monkeypatch,
        base_dir,
        ["--compare", "$draw.beta=0.92,0.99"],
    )
    assert run.mode == "compare"
    assert len(run.test_set) == 2
    # v4 §5.2: labels are str(value), unquoted in tables.
    assert {run.test_set[0].label, run.test_set[1].label} == {str(0.92), str(0.99)}
    bvals = {
        run.test_set[0].slots["draw"]["beta"],
        run.test_set[1].slots["draw"]["beta"],
    }
    assert bvals == {0.92, 0.99}


def test_sweep_cartesian_product(monkeypatch, base_dir):
    pl = json.dumps(
        [
            {"draw": {"beta": 0.9}},
            {"draw": {"beta": 0.91}},
        ]
    )
    ml = json.dumps(
        [
            {"ms": {"tag": "FUES"}},
            {"ms": {"tag": "MSS"}},
        ]
    )
    run = _ps(
        monkeypatch,
        base_dir,
        [
            "--sweep",
            f"--slot-range={pl}",
            f"--slot-range={ml}",
        ],
    )
    assert run.mode == "sweep"
    assert len(run.test_set) == 4
    keys = {
        (t.slots.get("draw", {}).get("beta"), t.slots.get("ms", {}).get("tag"))
        for t in run.test_set
    }
    assert len(keys) == 4
    t0 = run.test_set[0]
    assert t0.slots.get("ms", {}).get("tag") in ("FUES", "MSS")


def test_merge_precedence_spec_then_override(monkeypatch, base_dir):
    run = _ps(
        monkeypatch,
        base_dir,
        [
            "--slot-spec",
            '{"draw":{"beta":0.95}}',
            "--slot-override",
            "$draw.beta=0.97",
        ],
    )
    assert run.test_set[0].slots["draw"]["beta"] == 0.97


def test_merge_precedence_override_then_spec(monkeypatch, base_dir):
    run = _ps(
        monkeypatch,
        base_dir,
        [
            "--slot-override",
            "$draw.beta=0.97",
            "--slot-spec",
            '{"draw":{"beta":0.95}}',
        ],
    )
    assert run.test_set[0].slots["draw"]["beta"] == 0.95


def test_dict_literal_matches_atfile(tmp_path, monkeypatch, base_dir):
    f = tmp_path / "p.yaml"
    f.write_text(yaml.dump({"draw": {"beta": 0.88, "r": 0.03}}))
    a = _ps(
        monkeypatch,
        base_dir,
        [f"--slot-spec=@{f}"],
    )
    b = _ps(
        monkeypatch,
        base_dir,
        ['--slot-spec', '{"draw":{"beta":0.88,"r":0.03}}'],
    )
    assert a.test_set[0] == b.test_set[0]


def test_compare_and_sweep_mutually_exclusive(monkeypatch, base_dir):
    with pytest.raises(ValueError, match="compare and --sweep"):
        _ps(
            monkeypatch,
            base_dir,
            [
                "--compare",
                "$draw.a=0.9,0.91",
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


def test_slot_range_atfile_roundtrip(tmp_path, monkeypatch, base_dir):
    f = tmp_path / "pr.yaml"
    f.write_text(
        yaml.dump(
            [
                {"draw": {"beta": 0.9}},
                {"draw": {"beta": 0.91}},
            ]
        )
    )
    inline = json.dumps([{"draw": {"beta": 0.9}}, {"draw": {"beta": 0.91}}])
    a = _ps(
        monkeypatch,
        base_dir,
        [f"--slot-range=@{f}"],
    )
    b = _ps(
        monkeypatch,
        base_dir,
        [f"--slot-range={inline}"],
    )
    assert a.test_set == b.test_set
    assert a.mode == "sweep" and b.mode == "sweep"
    assert len(a.test_set) == 2


def test_compare_plus_slot_range_rejected(monkeypatch, base_dir):
    with pytest.raises(ValueError, match="--compare cannot be combined"):
        _ps(
            monkeypatch,
            base_dir,
            [
                "--compare",
                "$draw.beta=0.9,0.91",
                "--slot-range",
                json.dumps([{"draw": {"r": 0.03}}]),
            ],
        )


def test_slot_override_folds_into_slot_range_rows(monkeypatch, base_dir):
    run = _ps(
        monkeypatch,
        base_dir,
        [
            "--slot-override",
            "$draw.r=0.03",
            "--slot-range",
            json.dumps([{"draw": {"beta": 0.9}}, {"draw": {"beta": 0.91}}]),
        ],
    )
    assert run.mode == "sweep"
    for t in run.test_set:
        assert t.slots["draw"]["r"] == 0.03
    assert {t.slots["draw"]["beta"] for t in run.test_set} == {0.9, 0.91}


def test_invalid_slot_override_missing_dollar(monkeypatch, base_dir):
    with pytest.raises(ValueError, match="Invalid --slot-override"):
        _ps(
            monkeypatch,
            base_dir,
            ["--slot-override", "draw.beta=1"],
        )


def test_slot_range_for_nested_methods_block(monkeypatch, base_dir):
    """v3: use --slot-range (not --compare) for deep method_switch shapes."""
    r1 = json.dumps(
        [
            {
                "method_switch": {
                    "methods": [
                        {
                            "on": "st",
                            "schemes": [{"scheme": "upper_envelope", "method": "FUES"}],
                        }
                    ]
                }
            },
            {
                "method_switch": {
                    "methods": [
                        {
                            "on": "st",
                            "schemes": [{"scheme": "upper_envelope", "method": "NEGM"}],
                        }
                    ]
                }
            },
        ]
    )
    run = _ps(
        monkeypatch,
        base_dir,
        [f"--slot-range={r1}"],
    )
    assert run.mode == "sweep"
    assert len(run.test_set) == 2
    m0 = run.test_set[0].slots["method_switch"]["methods"][0]["schemes"][0]["method"]
    m1 = run.test_set[1].slots["method_switch"]["methods"][0]["schemes"][0]["method"]
    assert {m0, m1} == {"FUES", "NEGM"}


def test_recursive_at_resolution(tmp_path, monkeypatch, base_dir):
    inner = tmp_path / "inner.yaml"
    inner.write_text("beta: 0.5\n")
    outer = tmp_path / "outer.yaml"
    outer.write_text(f'draw: "@{inner}"\n')
    run = _ps(
        monkeypatch,
        base_dir,
        [f"--slot-spec=@{outer}"],
    )
    assert run.test_set[0].slots["draw"] == {"beta": 0.5}


def test_brief_import_check():
    from kikku.run import parse_cli, sweep, TestSpec, RunSpec

    assert callable(parse_cli) and callable(sweep)
    assert TestSpec and RunSpec


# ---------------------------------------------------------------------------
# v4: deep-path overrides (§3)
# ---------------------------------------------------------------------------


def test_slot_override_one_level(monkeypatch, base_dir):
    """v4 backward-compat: single-level path is unchanged from v3."""
    run = _ps(monkeypatch, base_dir, ["--slot-override", "$draw.beta=0.97"])
    assert run.test_set[0].slots == {"draw": {"beta": 0.97}}


def test_slot_override_two_levels(monkeypatch, base_dir):
    """`$draw.calibration.beta=v` → {draw: {calibration: {beta: v}}}."""
    run = _ps(
        monkeypatch,
        base_dir,
        ["--slot-override", "$draw.calibration.beta=0.96"],
    )
    assert run.test_set[0].slots == {"draw": {"calibration": {"beta": 0.96}}}


def test_slot_override_three_plus_levels(monkeypatch, base_dir):
    """Unbounded depth (§3.4)."""
    run = _ps(
        monkeypatch,
        base_dir,
        ["--slot-override", "$draw.calibration.shocks.sigma=0.02"],
    )
    assert run.test_set[0].slots == {
        "draw": {"calibration": {"shocks": {"sigma": 0.02}}}
    }


def test_slot_override_type_collision(monkeypatch, base_dir):
    """Argv-order, last-writer-wins (§3.3): scalar then deep path."""
    run = _ps(
        monkeypatch,
        base_dir,
        [
            "--slot-override", "$draw.b=42",
            "--slot-override", "$draw.b.c=99",
        ],
    )
    assert run.test_set[0].slots == {"draw": {"b": {"c": 99}}}


def test_slot_override_multiple_levels_share_prefix(monkeypatch, base_dir):
    """Two deep overrides under the same prefix merge (no overwrite)."""
    run = _ps(
        monkeypatch,
        base_dir,
        [
            "--slot-override", "$draw.calibration.beta=0.96",
            "--slot-override", "$draw.calibration.tau=0.05",
            "--slot-override", "$draw.settings.n_a=500",
        ],
    )
    assert run.test_set[0].slots == {
        "draw": {
            "calibration": {"beta": 0.96, "tau": 0.05},
            "settings": {"n_a": 500},
        }
    }


# ---------------------------------------------------------------------------
# v4: axis-form ranges (§4)
# ---------------------------------------------------------------------------


def test_slot_range_axis_form_scalars(monkeypatch, base_dir):
    """`$path=[1,2,3]` → 3 rows, each setting the path."""
    run = _ps(
        monkeypatch,
        base_dir,
        ["--sweep", "--slot-range", "$draw.calibration.tau=[0.05, 0.10, 0.15]"],
    )
    assert run.mode == "sweep"
    assert len(run.test_set) == 3
    taus = [t.slots["draw"]["calibration"]["tau"] for t in run.test_set]
    assert taus == [0.05, 0.10, 0.15]


def test_slot_range_axis_form_dicts(monkeypatch, base_dir):
    """Axis values can be dicts (e.g. method-block payloads)."""
    payload = (
        '$method_switch=['
        '{"methods":[{"on":"m","schemes":[{"scheme":"upper_envelope","method":"FUES"}]}]},'
        '{"methods":[{"on":"m","schemes":[{"scheme":"upper_envelope","method":"NEGM"}]}]}'
        "]"
    )
    run = _ps(
        monkeypatch,
        base_dir,
        ["--sweep", f"--slot-range={payload}"],
    )
    assert len(run.test_set) == 2
    methods = {
        t.slots["method_switch"]["methods"][0]["schemes"][0]["method"]
        for t in run.test_set
    }
    assert methods == {"FUES", "NEGM"}


def test_slot_range_two_axes_cartesian(monkeypatch, base_dir):
    """Two `--slot-range` flags → Cartesian (2 × 3 = 6 rows)."""
    run = _ps(
        monkeypatch,
        base_dir,
        [
            "--sweep",
            "--slot-range", "$draw.calibration.tau=[0.05, 0.10]",
            "--slot-range", "$draw.settings.n_a=[100, 200, 300]",
        ],
    )
    assert len(run.test_set) == 6
    pairs = {
        (t.slots["draw"]["calibration"]["tau"], t.slots["draw"]["settings"]["n_a"])
        for t in run.test_set
    }
    assert pairs == {
        (0.05, 100), (0.05, 200), (0.05, 300),
        (0.10, 100), (0.10, 200), (0.10, 300),
    }


def test_slot_range_axis_plus_bundle_list(monkeypatch, base_dir):
    """Axis form × bundle-list form → Cartesian (2 × 2 = 4)."""
    bundle = json.dumps([{"draw": {"r": 0.02}}, {"draw": {"r": 0.04}}])
    run = _ps(
        monkeypatch,
        base_dir,
        [
            "--sweep",
            "--slot-range", "$draw.beta=[0.96, 0.99]",
            "--slot-range", bundle,
        ],
    )
    assert len(run.test_set) == 4
    pairs = {
        (t.slots["draw"]["beta"], t.slots["draw"]["r"])
        for t in run.test_set
    }
    assert pairs == {(0.96, 0.02), (0.96, 0.04), (0.99, 0.02), (0.99, 0.04)}


def test_slot_range_axis_form_atfile(tmp_path, monkeypatch, base_dir):
    """Spec §4.5: `$path=@file.yaml` reads the file as a YAML list."""
    f = tmp_path / "betas.yaml"
    f.write_text("- 0.92\n- 0.94\n- 0.96\n")
    run = _ps(
        monkeypatch,
        base_dir,
        ["--sweep", f"--slot-range=$draw.calibration.beta=@{f}"],
    )
    betas = [t.slots["draw"]["calibration"]["beta"] for t in run.test_set]
    assert betas == [0.92, 0.94, 0.96]


# ---------------------------------------------------------------------------
# v4: label derivation (§5.2, §7.4)
# ---------------------------------------------------------------------------


def test_compare_str_label_unquoted(monkeypatch, base_dir):
    """`--compare` row labels are str(value), unquoted in tables (§5.2)."""
    run = _ps(
        monkeypatch,
        base_dir,
        ["--compare", "$method_switch.tag=FUES,NEGM"],
    )
    labels = {t.label for t in run.test_set}
    # str() of a YAML-loaded scalar; "FUES" is not Python repr "'FUES'".
    assert labels == {"FUES", "NEGM"}
    assert all(not lbl.startswith("'") for lbl in labels)


def test_compare_deep_path_two_levels(monkeypatch, base_dir):
    """v4 `--compare` allows deep paths just like `--slot-override` (§5.1)."""
    run = _ps(
        monkeypatch,
        base_dir,
        ["--compare", "$draw.calibration.beta=0.92,0.96"],
    )
    assert run.mode == "compare"
    assert len(run.test_set) == 2
    betas = {
        t.slots["draw"]["calibration"]["beta"] for t in run.test_set
    }
    assert betas == {0.92, 0.96}


def test_axis_diff_label_for_sweep(monkeypatch, base_dir):
    """Non-compare sweep label is `slot.path=value,…` per §7.4."""
    run = _ps(
        monkeypatch,
        base_dir,
        [
            "--sweep",
            "--slot-range", "$draw.calibration.tau=[0.05, 0.10]",
            "--slot-range", "$draw.settings.n_a=[100, 200]",
        ],
    )
    labels = {t.label for t in run.test_set}
    # Expected: "draw.calibration.tau=0.05,draw.settings.n_a=100", etc.
    assert "draw.calibration.tau=0.05,draw.settings.n_a=100" in labels
    assert "draw.calibration.tau=0.1,draw.settings.n_a=200" in labels
    assert len(labels) == 4


# ---------------------------------------------------------------------------
# v4: mode classification (§5.3)
# ---------------------------------------------------------------------------


def test_sweep_without_slot_range_rejected(monkeypatch, base_dir):
    """Bare --sweep is now an error (§5.3 refines v3)."""
    with pytest.raises(ValueError, match="--sweep requires"):
        _ps(monkeypatch, base_dir, ["--sweep"])


def test_slot_range_without_sweep_implicit_mode(monkeypatch, base_dir):
    """`--slot-range` without `--sweep` → mode='sweep' implicitly (§5.3)."""
    run = _ps(
        monkeypatch,
        base_dir,
        ["--slot-range", "$draw.beta=[0.92, 0.96]"],
    )
    assert run.mode == "sweep"
    assert len(run.test_set) == 2
