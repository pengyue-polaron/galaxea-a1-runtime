from galaxea_a1_runtime.cli import main


def test_cli_safety_report_json(capsys):
    rc = main(["safety-report", "--json"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "safe_command_path" in captured.out
