"""The cloud commands on the command line (#113)."""

from aerial_search.cloud import launcher

# --- the command line -------------------------------------------------------------


def test_sagemaker_fetch_lists_what_it_fetched(monkeypatch, capsys, tmp_path) -> None:
    from aerial_search import cli

    monkeypatch.setattr(cli, "_aws_session", lambda: "session")
    monkeypatch.setattr(launcher, "fetch", lambda job, session, out: [tmp_path / "r"])
    cli.main(["sagemaker-fetch", "job-1"])
    assert capsys.readouterr().out.strip() == str(tmp_path / "r")
