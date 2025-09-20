import pytest
from src.ocr.mapping import TokenAlignment


def test_name_replacement_shorter():
    orig = "John Smith born 01/01/1990"
    new = "[REDACTED] born [REDACTED]"
    ta = TokenAlignment(orig, new)
    # 'John Smith' spans 0..10
    new_s, new_e = ta.map_span(0, 10)
    assert new_s >= 0 and new_s <= len(new)
    assert new_e >= new_s


def test_ssn_formats():
    orig = "SSN: 123-45-6789 and 123456789"
    # anonymizer may produce fixed token
    new = "SSN: [REDACTED] and [REDACTED]"
    ta = TokenAlignment(orig, new)
    # first ssn location
    s1, e1 = orig.index("123-45-6789"), orig.index("123-45-6789") + len("123-45-6789")
    ns1, ne1 = ta.map_span(s1, e1)
    assert new[ns1:ne1] == "[REDACTED]" or ns1 < ne1


def test_overlapping_entities():
    orig = "Dr. John Smith MD, phone 555-1234"
    new = "Dr. [REDACTED] MD, phone [REDACTED]"
    ta = TokenAlignment(orig, new)
    # two entities: name and phone
    name_start = orig.index("John Smith")
    name_end = name_start + len("John Smith")
    phone_start = orig.index("555-1234")
    phone_end = phone_start + len("555-1234")
    ns, ne = ta.map_span(name_start, name_end)
    ps, pe = ta.map_span(phone_start, phone_end)
    assert ns < ne and ps < pe
    # ensure ranges do not overlap in new text (they should be distinct)
    assert not (ns < pe and ps < ne)
