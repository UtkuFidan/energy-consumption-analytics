import pandas as pd
import pytest

from energy import cli


def test_markdown_table_keeps_integers():
    df = pd.DataFrame({"clients": [169], "share": [51.368]}, index=pd.Index([0], name="archetype"))
    md = cli._md(df)
    assert "| 0 | 169 | 51.368 |" in md


def test_cli_requires_a_command(capsys):
    with pytest.raises(SystemExit):
        cli.main([])
