"""Party presentation metadata — colours and Swedish display names.

Single source of truth for how parties are *shown* (the webapp imports these via
the JSON export). Simulation/bloc config lives in `simulate.py`; this module is
purely presentation.

Colours are the canonical Swedish party hexes, with two legibility adjustments
for charting on a white canvas (see `trefyranio-party-colors` memory):
  * V darkened from #DA291C → #AF0000 (canonical V red is ~identical to S's),
  * SD deepened from #DDDD00 → #C9A227 (canonical bright yellow is ~invisible).
The Swedish blue+yellow stays in the LOGO only; these are the data-viz colours.
"""

from __future__ import annotations

PARTY_HEX = {
    "S": "#E8112D",    # Socialdemokraterna — red
    "M": "#52BDEC",    # Moderaterna — sky blue
    "SD": "#C9A227",   # Sverigedemokraterna — gold (canonical #DDDD00, deepened)
    "C": "#009933",    # Centerpartiet — green
    "V": "#AF0000",    # Vänsterpartiet — dark red (canonical #DA291C, darkened vs S)
    "KD": "#000077",   # Kristdemokraterna — navy
    "MP": "#83CF39",   # Miljöpartiet — lime
    "L": "#006AB3",    # Liberalerna — medium blue
    "Övr": "#9AA0A6",  # Övriga — neutral grey
}

# Data-collection method per pollster (for the poll-detail modal). From the
# pollster research; "Sifo" is the Verian/Kantar lineage label used by the data.
POLLSTER_METHOD = {
    "Novus": "Telefon, SMS och post",
    "Sifo": "Webbpanel",
    "Demoskop": "Webbundersökning (Iniziopanelen)",
    "Ipsos": "Telefon och webbpanel",
    "Indikator": "Postal enkät",
    "Sentio": "Självrekryterad webbpanel",
    "SCB": "Slumpmässigt urval (telefon/webb)",
    "Skop": "Webbpanel",
    "Inizio": "Webbpanel",
    "YouGov": "Självrekryterad webbpanel",
}

PARTY_NAME_SV = {
    "S": "Socialdemokraterna",
    "M": "Moderaterna",
    "SD": "Sverigedemokraterna",
    "C": "Centerpartiet",
    "V": "Vänsterpartiet",
    "KD": "Kristdemokraterna",
    "MP": "Miljöpartiet",
    "L": "Liberalerna",
    "Övr": "Övriga",
}
