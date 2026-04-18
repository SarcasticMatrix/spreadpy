import json
from pathlib import Path


FUTURES_UNIVERSE_PATH = Path(__file__).parent / "futures_universe.json"


def load_futures_universe() -> dict[str, dict[str, str]]:
    """Load the futures universe from the bundled JSON (keyed by category)."""
    with open(FUTURES_UNIVERSE_PATH) as f:
        return json.load(f)
    

def get_all_tickers():

    universe = load_futures_universe()

    tickers = [
        x
        for xs in [list(universe[key].keys()) for key in universe.keys()]
        for x in xs
    ]
    return tickers



if __name__ == "__main__":
    import warnings
    import yfinance as yf
    warnings.filterwarnings("ignore")

    TICKERS = [
        "CL=F", "BZ=F", "NG=F",                          # energy
        "OJ=F", "GC=F", "SI=F", "HG=F",                  # metals + OJ
        "ZC=F", "ZO=F", "KE=F", "ZR=F", "ZS=F",          # grains
        "GF=F", "HE=F", "LE=F",                          # livestock
        "CC=F", "KC=F", "CT=F", "LBS=F", "SB=F",         # softs
    ]
    PERIOD   = "730d"
    INTERVAL = "1h"

    print(get_all_tickers())