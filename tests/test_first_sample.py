"""CopyPasteConfig.first_sample and the run_survey_banks.py options added for the RQ2 samples
(docs/notes/rq2_samples_provenance.md)."""

from __future__ import annotations

import json
import sys

import pytest
import yaml

import adr.harness.copy_paste as cp_mod
import adr.harness.doc_upload as du_mod
from adr.harness.copy_paste import CopyPasteConfig, run_copy_paste
from scripts import run_survey_banks as rsb
from tests.test_harness import _setup, make_client


# ---------------------------------------------------------------- harness: first_sample
def _responses(tmp_path, run_id):
    path = tmp_path / "runs" / run_id / "responses.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _samples(rows):
    return sorted({int(r["session_id"].rsplit(":", 1)[1]) for r in rows})


def test_default_first_sample_is_zero(tmp_path):
    client, _ = make_client(tmp_path)
    assessment, items, variants, mechs = _setup()
    cfg = CopyPasteConfig(
        run_id="a", models=["fake-a"], n_samples=3, temperature=0.7, results_dir=str(tmp_path)
    )
    assert cfg.first_sample == 0
    run_copy_paste(client, assessment, items, variants, mechs, cfg)
    assert _samples(_responses(tmp_path, "a")) == [0, 1, 2]


def test_first_sample_adds_new_samples_without_repeating_an_index(tmp_path):
    client, fake = make_client(tmp_path)
    assessment, items, variants, mechs = _setup()
    base = CopyPasteConfig(
        run_id="base", models=["fake-a"], n_samples=2, temperature=0.7, results_dir=str(tmp_path)
    )
    run_copy_paste(client, assessment, items, variants, mechs, base)
    calls = fake.calls

    more = CopyPasteConfig(
        run_id="more",
        models=["fake-a"],
        n_samples=3,
        first_sample=2,
        temperature=0.7,
        results_dir=str(tmp_path),
    )
    rows = run_copy_paste(client, assessment, items, variants, mechs, more)
    got = _responses(tmp_path, "more")
    assert _samples(got) == [2, 3, 4]
    assert {r.session_id for r in rows} == {f"more:fake-a:{s}" for s in (2, 3, 4)}
    # new indices -> new cache keys -> real calls, never a replay of samples 0-1
    old_keys = {r["cache_key"] for r in _responses(tmp_path, "base")}
    assert not old_keys & {r["cache_key"] for r in got}
    assert not any(r["cached"] for r in got)
    assert fake.calls > calls
    conf = yaml.safe_load((tmp_path / "runs" / "more" / "config.yaml").read_text())
    assert (conf["first_sample"], conf["n_samples"]) == (2, 3)


def test_first_sample_reuses_the_cache_for_the_same_index(tmp_path):
    """Sample i under another run id is the same cached call (the run id is not in the key)."""
    client, fake = make_client(tmp_path)
    assessment, items, variants, mechs = _setup()
    a = CopyPasteConfig(
        run_id="x",
        models=["fake-a"],
        n_samples=2,
        first_sample=3,
        temperature=0.7,
        results_dir=str(tmp_path),
    )
    run_copy_paste(client, assessment, items, variants, mechs, a)
    calls = fake.calls
    b = CopyPasteConfig(
        run_id="y",
        models=["fake-a"],
        n_samples=1,
        first_sample=4,
        temperature=0.7,
        results_dir=str(tmp_path),
    )
    run_copy_paste(client, assessment, items, variants, mechs, b)
    assert fake.calls == calls
    assert all(r["cached"] for r in _responses(tmp_path, "y"))


# ---------------------------------------------------------------- survey driver options
@pytest.fixture
def driver(tmp_path, monkeypatch):
    """cmd_run with the data loading and the harness replaced: returns the configs it built."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rsb, "load", lambda bank: (None, {}, {}))
    monkeypatch.setattr(rsb, "mechanisms_for", lambda bank: {})
    monkeypatch.setattr(rsb, "client_for", lambda fake: None)
    built = []

    def fake_run(client, assessment, items, variants, mechs, cfg):
        built.append(cfg)
        run_dir = tmp_path / cfg.results_dir / "runs" / cfg.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump({"stopped_for_budget": False}))
        (run_dir / "errors.jsonl").write_text("")
        return []

    monkeypatch.setattr(cp_mod, "run_copy_paste", fake_run)
    monkeypatch.setattr(du_mod, "run_doc_upload", fake_run)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["run_survey_banks.py", "run", *argv])
        rsb.main()
        return built[-1]

    return run


def test_driver_defaults_are_unchanged(driver):
    cfg = driver("--bank", "survey-gss", "--mode", "cp")
    assert (cfg.run_id, cfg.n_samples, cfg.first_sample, cfg.temperature) == (
        f"v1-cp-L0-survey-gss-s{rsb.CP_SAMPLES}",
        rsb.CP_SAMPLES,
        0,
        rsb.CP_TEMP,
    )
    cfg = driver("--bank", "survey-ipip", "--mode", "du")
    assert (cfg.run_id, cfg.n_samples, cfg.temperature, cfg.questions_per_doc) == (
        "v1-du-L0-survey-ipip-q20",
        rsb.DU_SAMPLES,
        rsb.DU_TEMP,
        20,
    )


def test_driver_cp_overrides(driver):
    cfg = driver(
        "--bank",
        "survey-ipip",
        "--mode",
        "cp",
        "--samples",
        "15",
        "--first-sample",
        "5",
        "--run-id",
        "v1-cp-L0-survey-ipip-s5-19",
    )
    assert (cfg.run_id, cfg.n_samples, cfg.first_sample, cfg.temperature) == (
        "v1-cp-L0-survey-ipip-s5-19",
        15,
        5,
        rsb.CP_TEMP,
    )
    assert cfg.models == [rsb.MODEL] == ["gpt-mini"]


def test_driver_du_overrides(driver):
    cfg = driver(
        "--bank",
        "assignment-asurre-train",
        "--mode",
        "du",
        "--samples",
        "20",
        "--temperature",
        "0.7",
        "--run-id",
        "v1-du-L0-assignment-asurre-train-t07-s20",
    )
    assert (cfg.run_id, cfg.n_samples, cfg.temperature, cfg.max_tokens) == (
        "v1-du-L0-assignment-asurre-train-t07-s20",
        20,
        0.7,
        6000,
    )


def test_driver_first_sample_is_copy_paste_only(driver):
    with pytest.raises(SystemExit, match="copy-paste only"):
        driver("--bank", "survey-gss", "--mode", "du", "--first-sample", "5")


# ---------------------------------------------------------------- numeric controls (#45 review)
@pytest.mark.parametrize("bad", [{"n_samples": 0}, {"n_samples": -2}, {"first_sample": -1}])
def test_copy_paste_rejects_bad_sample_controls_before_any_call(tmp_path, bad):
    client, fake = make_client(tmp_path)
    assessment, items, variants, mechs = _setup()
    cfg = CopyPasteConfig(run_id="bad", models=["fake-a"], temperature=0.7,
                          results_dir=str(tmp_path), **bad)
    with pytest.raises(ValueError, match="n_samples must be at least 1|first_sample must be 0"):
        run_copy_paste(client, assessment, items, variants, mechs, cfg)
    assert fake.calls == 0
    assert not (tmp_path / "runs" / "bad").exists()  # no empty or partial run is written


@pytest.mark.parametrize("n", [0, -1])
def test_doc_upload_rejects_n_samples_below_one(tmp_path, n):
    from adr.harness.doc_upload import DocUploadConfig, run_doc_upload

    client, fake = make_client(tmp_path)
    assessment, items, variants, mechs = _setup()
    cfg = DocUploadConfig(run_id="bad", models=["fake-a"], n_samples=n, results_dir=str(tmp_path))
    with pytest.raises(ValueError, match="n_samples must be at least 1"):
        run_doc_upload(client, assessment, items, variants, mechs, cfg)
    assert fake.calls == 0


@pytest.mark.parametrize("argv", [
    ("--mode", "cp", "--samples", "0"),          # was silently replaced by the default
    ("--mode", "du", "--samples", "0"),
    ("--mode", "cp", "--samples", "-3"),
    ("--mode", "cp", "--first-sample", "-1"),
])
def test_driver_rejects_bad_sample_options(driver, argv, capsys):
    with pytest.raises(SystemExit) as stop:
        driver("--bank", "survey-gss", *argv)
    assert stop.value.code == 2 and "must be at least" in capsys.readouterr().err


def test_driver_samples_one_is_one_not_the_default(driver):
    cfg = driver("--bank", "survey-gss", "--mode", "cp", "--samples", "1")
    assert cfg.n_samples == 1 != rsb.CP_SAMPLES


# ---------------------------------------------------------------- rescore --runs
class _Stop(Exception):
    pass


@pytest.fixture
def rescore(monkeypatch, tmp_path):
    monkeypatch.setattr(rsb, "style_mechanism", lambda path: None)
    monkeypatch.setattr(rsb, "sha256", lambda path: "x")
    loaded = []

    def load(bank):
        loaded.append(bank)
        raise _Stop

    monkeypatch.setattr(rsb, "load", load)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["run_survey_banks.py", "rescore", *argv])
        rsb.main()

    return run, loaded


def test_rescore_runs_touches_only_the_listed_banks(rescore):
    run, loaded = rescore
    with pytest.raises(_Stop):
        run("--runs", "survey-gss=v1-cp-L0-survey-gss-s5-19")
    assert loaded == ["survey-gss"]


def test_rescore_default_is_the_six_standard_runs(rescore):
    run, loaded = rescore
    with pytest.raises(_Stop):
        run()
    assert loaded == [next(iter(rsb.BANKS))]


@pytest.mark.parametrize("spec", ["nobank=v1-x", "survey-gss=", "survey-gss"])
def test_rescore_runs_rejects_bad_specs(rescore, spec):
    run, loaded = rescore
    with pytest.raises(SystemExit, match="BANK=RUN_ID"):
        run("--runs", spec)
    assert loaded == []
