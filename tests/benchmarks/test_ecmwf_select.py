import json

from benchmarks._data import select_enfo_messages


def _line(param, number, offset, length):
    return json.dumps({"param": param, "number": number, "_offset": offset, "_length": length})


def test_selects_matching_param_and_member_sorted_by_offset():
    lines = [
        _line("2t", "2", 200, 10),
        _line("2t", "1", 100, 10),
        _line("tp", "1", 50, 10),       # wrong param
        _line("2t", "3", 300, 10),      # member not requested
    ]
    selected = select_enfo_messages(lines, params=("2t",), members=(1, 2))
    assert [e["_offset"] for e in selected] == [100, 200]
    assert {e["number"] for e in selected} == {"1", "2"}


def test_empty_selection_returns_empty_list():
    lines = [_line("tp", "1", 0, 10)]
    assert select_enfo_messages(lines, params=("2t",), members=(1,)) == []
