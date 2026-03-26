"""Tests for kikku.run.cli — parse_run and RunSpec."""

import pytest
import yaml
from pathlib import Path

from kikku.run.cli import parse_run, RunSpec


@pytest.fixture
def syntax_dir(tmp_path):
    """Minimal syntax directory with calibration.yaml and settings.yaml."""
    calib = {
        'calibration': {
            'beta': 0.96,
            'r': 0.02,
            'delta': 1.0,
        }
    }
    settings = {
        'settings': {
            'grid_size': 3000,
            'T': 20,
            'plot_age': 5,
        }
    }
    (tmp_path / 'calibration.yaml').write_text(yaml.dump(calib))
    (tmp_path / 'settings.yaml').write_text(yaml.dump(settings))
    return tmp_path


def _run(monkeypatch, syntax_dir, argv, **kwargs):
    """Helper: mock sys.argv and call parse_run."""
    import sys
    monkeypatch.setattr(sys, 'argv', ['test'] + argv)
    defaults = dict(
        name='test_model',
        syntax=str(syntax_dir),
        methods=['FUES', 'NEGM'],
        modes=['compare', 'sweep', 'simulate'],
        output=str(syntax_dir / 'results'),
    )
    defaults.update(kwargs)
    return parse_run(**defaults)


def test_parse_run_defaults(monkeypatch, syntax_dir):
    run = _run(monkeypatch, syntax_dir, [])
    assert isinstance(run, RunSpec)
    assert run.name == 'test_model'
    assert run.method is None
    assert run.mode == 'single'
    assert run.compare_methods is None
    assert run.verbose is False
    assert run.simulate is False
    assert run.sweep_grids is None
    assert run.calib['beta'] == 0.96
    assert run.settings['grid_size'] == 3000
    assert run.config == {}
    assert run.mpi is False
    assert run.gpu is False


def test_parse_run_method_selection(monkeypatch, syntax_dir):
    run = _run(monkeypatch, syntax_dir, ['--method', 'NEGM'])
    assert run.method == 'NEGM'


def test_parse_run_method_overrides(monkeypatch, syntax_dir):
    run = _run(monkeypatch, syntax_dir, [
        '--method-override', 'adjuster_cons.cntn_to_dcsn_mover.upper_envelope=NEGM',
        '--method-override', 'keeper_cons.upper_envelope=RFC',
    ])
    assert run.method_overrides == {
        ('adjuster_cons', 'cntn_to_dcsn_mover', 'upper_envelope'): 'NEGM',
        ('keeper_cons', 'cntn_to_dcsn_mover', 'upper_envelope'): 'RFC',
    }


def test_parse_run_method_overrides_empty_by_default(monkeypatch, syntax_dir):
    run = _run(monkeypatch, syntax_dir, [])
    assert run.method_overrides == {}


def test_parse_run_compare_mode(monkeypatch, syntax_dir):
    run = _run(monkeypatch, syntax_dir, ['--compare', 'FUES', 'NEGM'])
    assert run.mode == 'compare'
    assert run.compare_methods == ('FUES', 'NEGM')


def test_parse_run_compare_invalid_method(monkeypatch, syntax_dir):
    with pytest.raises(SystemExit):
        _run(monkeypatch, syntax_dir, ['--compare', 'FUES', 'INVALID'])


def test_mutually_exclusive_modes(monkeypatch, syntax_dir):
    with pytest.raises(SystemExit):
        _run(monkeypatch, syntax_dir,
             ['--compare', 'FUES', 'NEGM', '--sweep'])


def test_tier_enforcement_calib_in_settings(monkeypatch, syntax_dir):
    """Known calib key 'beta' via --setting-override should error."""
    with pytest.raises(SystemExit, match='beta'):
        _run(monkeypatch, syntax_dir,
             ['--setting-override', 'beta=0.99'])


def test_tier_enforcement_settings_in_calib(monkeypatch, syntax_dir):
    """Known settings key 'grid_size' via --calib-override should error."""
    with pytest.raises(SystemExit, match='grid_size'):
        _run(monkeypatch, syntax_dir,
             ['--calib-override', 'grid_size=5000'])


def test_override_precedence(monkeypatch, syntax_dir, tmp_path):
    """CLI beats override-file beats base YAML."""
    override_file = tmp_path / 'overrides.yaml'
    override_file.write_text(yaml.dump({'beta': 0.90, 'r': 0.05}))

    run = _run(monkeypatch, syntax_dir, [
        '--override-file', str(override_file),
        '--calib-override', 'beta=0.85',
    ])
    assert run.calib['beta'] == 0.85
    assert run.calib['r'] == 0.05


def test_output_dir_creation(monkeypatch, syntax_dir):
    run = _run(monkeypatch, syntax_dir, [])
    assert Path(run.output_dir).exists()


def test_modes_control_help(monkeypatch, syntax_dir):
    """modes=['compare'] only — --sweep should not be recognized."""
    with pytest.raises(SystemExit):
        _run(monkeypatch, syntax_dir, ['--sweep'],
             modes=['compare'])


def test_extra_args(monkeypatch, syntax_dir):
    run = _run(monkeypatch, syntax_dir,
               ['--use-taxes'],
               extra_args={'--use-taxes': {'action': 'store_true'}})
    assert run.extra['use_taxes'] is True


def test_sweep_grids_parsing(monkeypatch, syntax_dir):
    run = _run(monkeypatch, syntax_dir,
               ['--sweep', '--sweep-grids', '500,1000,2000'])
    assert run.mode == 'sweep'
    assert run.sweep_grids == [500, 1000, 2000]


def test_runspec_frozen(monkeypatch, syntax_dir):
    run = _run(monkeypatch, syntax_dir, [])
    with pytest.raises(AttributeError):
        run.method = 'NEGM'


def test_config_override_warning(monkeypatch, syntax_dir):
    """--config-override with a known calib key should warn."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        run = _run(monkeypatch, syntax_dir,
                    ['--config-override', 'beta=0.99'])
        calib_warnings = [x for x in w if 'beta' in str(x.message)]
        assert len(calib_warnings) >= 1


def test_setting_override_merges(monkeypatch, syntax_dir):
    run = _run(monkeypatch, syntax_dir,
               ['--setting-override', 'grid_size=5000'])
    assert run.settings['grid_size'] == 5000
    assert run.settings['T'] == 20
