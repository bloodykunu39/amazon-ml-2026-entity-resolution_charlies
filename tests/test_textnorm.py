import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textnorm import Normalizer  # noqa: E402
from features_house import house_rel  # noqa: E402

N = Normalizer()


def core(s):
    return set(N.name_views(s)["n_core"].split())


def variants(s):
    v = N.name_views(s)
    return {v["n_core"], v["n_alt"]}


def test_ocr_and_legal():
    v = N.name_views("Cabrera 5ecure Sciences LP")
    assert set(v["n_core"].split()) == {"cabrera", "secure", "sciences"}
    assert v["n_legal"] == "lp"


def test_alias_right_side():
    assert "asset building committee" in variants("Quoavi Co doing business as Asset Building Committee")
    assert "cabrera secure sciences" in variants("Jaxaria Formerly Cabrera Secure Sciences")
    assert "keystone emera partners" in variants("Fayekor a/k/a Keystone Emera Partners")
    assert "pune city globe" in variants("Wexsol Co t/a Pune City Globe Ltd")
    assert "institut zulu" in variants("Yumaciraavi DBA Institut du Zulu")


def test_domain_pipe():
    v = N.name_views("Cabrera  Secure Sciences | www.cabreras.com")
    assert set(v["n_core"].split()) == {"cabrera", "secure", "sciences"}
    assert v["n_dom"] == "cabreras"
    assert N.name_views("wilfordhancock.com")["n_dom"] == "wilfordhancock"
    assert N.name_views("Newlifezioncom")["n_dom"] == "newlifezion"
    assert N.name_views("@bennettacademy")["n_dom"] == "bennettacademy"


def test_honorific():
    assert core("Mr Leisure-Dmfigns") == {"leisure", "dmfigns"}
    assert core("Shri Leisure  Delgins") == {"leisure", "delgins"}


def test_dotted_legal():
    v = N.name_views("REAL JAY MANAGEMENT L.L.P.")
    assert v["n_legal"] == "llp" and set(v["n_core"].split()) == {"real", "jay", "management"}


def test_french_societe():
    assert core("Sté Dupont Transports SARL") == core("Société Dupont Transports") == {"dupont", "transports"}


def test_translit():
    assert {"city", "power"} <= core("सिटी पावर प्रा. लि.")
    assert N.name_views("सिटी पावर प्रा. लि.")["n_legal"] in ("ltd pvt", "ltd", "pvt")


def A(s, c="US"):
    return N.addr_views(s, c)


def test_us_addr_reorder_saint():
    a, b = A("00709 Hackberry Saint, Tilden, Texas"), A("TX, 709 Hackberry Street, Tilden")
    for x in (a, b):
        assert x["a_house"] == "709" and x["a_street"] == "hackberry" and x["a_stype"] == "st"
        assert x["a_city"] == "tilden" and x["a_state"] == "tx"


def test_us_cdp_glued():
    a = A("315. 80TH ST, CHICAGOCDP, IL")
    assert a["a_house"] == "315" and a["a_city"] == "chicago" and a["a_state"] == "il"
    b = A("165 BARREN RIVER DR, ERLANGER KY, KY")
    assert b["a_city"] == "erlanger" and b["a_state"] == "ky"


def test_india_state():
    a = A("14, 11TH CROSS, SWIMMING POOL EXTENSION MALLESWARAM, BANGALORE, ಕರ್ನಾಟಕ", "India")
    assert a["a_state"] == N.state_of("ka", "India") == N.state_of("karnataka", "India") != ""
    b = A("Panvel, MH, Shop No 8-50, Raigarh(mh)", "India")
    assert b["a_state"] == N.state_of("maharashtra", "India") != ""


def test_france_addr():
    a = A("N°2 R. BASSELART, LILLE, Hauts-de-France", "France")
    assert a["a_house"] == "2" and a["a_stype"] == "rue" and a["a_street"] == "basselart" and a["a_city"] == "lille"
    b = A("56 BIS R. DE LA VILLE AUX ROSES", "France")
    assert b["a_house"] == "56" and b["a_hsuf"] == "bis" and b["a_stype"] == "rue"
    assert A("x, Bordeaux, Nouvelle-Aquitaine", "France")["a_state"] == A("x, Bordeaux, Gironde", "France")["a_state"] != ""


@pytest.mark.parametrize("a,b,kind", [
    ("1344", "344", "suffix"), ("709", "00709", "equal"), ("4514D", "4514", "equal_base"),
    ("5001-5003", "5001", "range"),
])
def test_house_relations(a, b, kind):
    x, y = A(f"{a} Main St, Tilden, TX"), A(f"{b} Main St, Tilden, TX")
    rel = house_rel(x["a_house"], x["a_hsuf"], x["a_hrange"], y["a_house"], y["a_hsuf"], y["a_hrange"])
    assert rel[kind] == 1, (x, y, rel)
