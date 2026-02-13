# egx_watchlist.py
# EGX – Yahoo Finance tickers (EGS************.CA)
# Scanner uses EGS codes, UI shows short symbols

from __future__ import annotations
from typing import Dict

# -----------------------------
# Yahoo-SAFE EGX intraday tickers
# -----------------------------
EGX_SAFE_INTRADAY = [
    "EGS60121C018.CA",  # CIB
    "EGS48031C016.CA",  # ETEL
    "EGS3G0Z1C014.CA",  # SWDY
    "EGS69101C011.CA",  # HRHO
    "EGS691S1C011.CA",  # TMGH
    "EGS655L1C012.CA",  # PHDC
    "EGS65571C019.CA",  # MNHD
    "EGS37091C013.CA",  # EAST
    "EGS95001C011.CA",  # ORAS
    "EGS30901C010.CA",  # JUFO
    "EGS380P1C010.CA",  # AMOC
    "EGS65851C015.CA",  # OCDI
    "EGS30451C016.CA",  # HELI
    "EGS70271C019.CA",  # CCAP
    "EGS38191C010.CA",  # ABUK
    "EGS673Y1C015.CA",  # EMFD
    "EGS74191C018.CA",  # SIDP
    "EGS540S1C014.CA",  # GOUR
    "EGS729J1C018.CA",  # CLHO
    "EGS738I1C018.CA",  # CNFN
    "EGS70321C012.CA",  # ORHD
    "EGS696S1C016.CA",  # OFH
    "EGS60041C018.CA",  # CIEB
    "EGS745L1C014.CA",  # FWRY
    "EGS60231C015.CA",  # SCBN
    "EGS30201C015.CA",  # DSUG
    "EGS600M1C017.CA",  # UBEE
]

# -----------------------------
# UI display names
# -----------------------------
EGX_DISPLAY: Dict[str, str] = {
    "EGS60121C018.CA": "CIB",
    "EGS48031C016.CA": "ETEL",
    "EGS3G0Z1C014.CA": "SWDY",
    "EGS69101C011.CA": "HRHO",
    "EGS691S1C011.CA": "TMGH",
    "EGS655L1C012.CA": "PHDC",
    "EGS65571C019.CA": "MNHD",
    "EGS37091C013.CA": "EAST",
    "EGS95001C011.CA": "ORAS",
    "EGS30901C010.CA": "JUFO",
    "EGS380P1C010.CA": "AMOC",
    "EGS65851C015.CA": "OCDI",
    "EGS30451C016.CA": "HELI",
    "EGS70271C019.CA": "CCAP",
    "EGS38191C010.CA": "ABUK",
    "EGS673Y1C015.CA": "EMFD",
    "EGS74191C018.CA": "SIDP",
    "EGS540S1C014.CA": "GOUR",
    "EGS729J1C018.CA": "CLHO",
    "EGS738I1C018.CA": "CNFN",
    "EGS70321C012.CA": "ORHD",
    "EGS696S1C016.CA": "OFH",
    "EGS60041C018.CA": "CIEB",
    "EGS745L1C014.CA": "FWRY",
    "EGS60231C015.CA": "SCBN",
    "EGS30201C015.CA": "DSUG",
    "EGS600M1C017.CA": "UBEE",
}

def display_name(ticker: str) -> str:
    return EGX_DISPLAY.get(ticker, ticker)
