from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tse"))
from bu_decoder import extract_president

FIX = Path(__file__).parent / "fixtures" / "bu_2026_r2_budapest_s1696.bu"


def test_budapest_section_1696_round2():
    r = extract_president(FIX.read_bytes(), 6258)
    assert r is not None
    assert r.votes == {13: 115, 22: 60}
    assert (r.blank, r.null) == (3, 1)
    assert r.comparecimento == sum(r.votes.values()) + r.blank + r.null
