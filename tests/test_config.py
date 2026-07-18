from app.config import parse_zone


def test_parse_zone():
    assert parse_zone("0,0;1,0;1,1") == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))


def test_parse_zone_rejects_out_of_range():
    try:
        parse_zone("0,0;2,0;1,1")
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range zone accepted")

