import pandas as pd 
from dotenv import load_dotenv
import os
import time
import json
import requests
load_dotenv()

API_KEY = os.environ.get("GRAPH_API_KEY") 

# token ETH addresses
USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"  
USDT_ADDRESS = "0xdac17f958d2ee523a2206206994597c13d831ec7"  
DAI_ADDRESS = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
WETH_ADDRESS = '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'
USDE_ADDRESS = "0x4c9EDD5852cd905f086C759E8383e09bff1E68B3"


def run_query(query: str, endpoint: str, variables: dict | None = None):
    payload = {"query": query, "variables": variables or {}}
    r = requests.post(endpoint, json=payload)
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(data["errors"])
    return data["data"]


def gql(query: str, endpoint: str, variables: dict | None = None) -> dict:
    while True:
        r = requests.post(endpoint, json={"query": query, "variables": variables or {}})
        if r.status_code == 429:                           # hit the rate limit → wait & retry
            time.sleep(int(r.headers.get("Retry-After", "2")))
            continue
        r.raise_for_status()
        data = r.json()
        if "errors" in data:
            raise RuntimeError(json.dumps(data["errors"], indent=2))
        return data["data"]

def to_row(snap: dict) -> dict:
        row = {
            "timestamp"      : int(snap["timestamp"]),
            "block"          : int(snap["blockNumber"]),
            "TVL_USD"        : float(snap["totalValueLockedUSD"]),
            "supplied_USD"   : float(snap["totalDepositBalanceUSD"]),
            "borrowed_USD"   : float(snap["totalBorrowBalanceUSD"]),
            "liquidations"   : float(snap['hourlyLiquidateUSD']),
            "token_balance"  : float(snap["inputTokenBalance"]),
        }

        # initialise APR columns with NaN
        for side in ("lender", "borrower"):
            for typ in ("variable", "stable"):
                row[f"{side}_{typ}_apr"] = None

        SECONDS_PER_YEAR = 60 * 60 * 24 * 365
        for r in snap["rates"]:
            col = f"{r['side'].lower()}_{r['type'].lower()}_apr"
            row[col] = float(r["rate"])
            # row[col] = float(r["rate"]) / 1e27 * SECONDS_PER_YEAR * 100  # % APR
        return row
    
def fetch_daily_snapshots(query, mkt_id: str, endpoint: str, batch: int = 1000, start_ts = 0) -> list[dict]:
      out, skip = [], 0
      while True:
          chunk = gql(query, endpoint, {"mkt": mkt_id, "first": batch, "skip": skip, 'ts': start_ts})\
                      ["marketHourlySnapshots"]
          if not chunk:
              break
          out.extend(chunk)
          skip += len(chunk)
      return out

def get_data_coin_aave(name, start_ts, endpoint):
    coins = {
        "usdt": USDT_ADDRESS,
        "usdc": USDC_ADDRESS,
        "dai": DAI_ADDRESS,
        "eth": WETH_ADDRESS,
        "usde" : USDE_ADDRESS
    }

    coin_address = coins[name]

    MKT_QUERY = """
    query($token: String!) {
        markets(where: {inputToken: $token}) {
        id
        name
        inputToken { symbol }
        }
    }
    """
    markets = gql(MKT_QUERY, endpoint = endpoint, variables = {"token": coin_address})["markets"]
    if not markets:
        raise ValueError(f"{name.upper()} market not found in this subgraph.")
    MARKET_ID = markets[0]["id"]
    print("Found market:", markets[0]["name"], "→", MARKET_ID)


    SNAP_QUERY = """
    query($mkt: String!, $first: Int!, $skip: Int!, $ts: Int!) {
        marketHourlySnapshots(
        where: {market: $mkt, timestamp_gt: $ts}
        orderBy: timestamp
        orderDirection: asc
        first: $first
        skip: $skip
        ) {
        id
        timestamp
        blockNumber
        totalValueLockedUSD
        totalDepositBalanceUSD
        totalBorrowBalanceUSD
        hourlyLiquidateUSD
        inputTokenBalance
        rates {
            side          # LENDER / BORROWER
            type          # VARIABLE / STABLE
            rate          # per-second ray (1e27)
        }
        }
    }
    """

    raw_snaps = fetch_daily_snapshots(SNAP_QUERY, MARKET_ID, endpoint, batch = 1000, start_ts= start_ts)
    print("Fetched", len(raw_snaps), "hourly snapshots")

    
    df = pd.DataFrame(map(to_row, raw_snaps))
    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    df.index = df.index.ceil('h')
    df = df.groupby(level = 0).mean()
    full_idx = pd.date_range(df.index[0].ceil("H"),
                            df.index[-1].ceil("H"),
                            freq="1H")
    df = df.reindex(full_idx,method = 'ffill')
    return df 

  
if __name__ == "__main__": 
    for coin in ['usdt', 'usdc', 'dai', 'usde', 'eth']:

        id_v3_eth = "JCNWRypm7FYwV8fx5HhzZPSFaMxgkPuw4TnR3Gpi81zk" # v3 ETH subgraph
        url_v3_eth = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{id_v3_eth}"
        try:
            prev = pd.read_parquet(f'./data/AAVE/aave_v3_{coin}_eth.parquet')
        except FileNotFoundError :
            full = get_data_coin_aave(coin, start_ts = 0, endpoint = url_v3_eth)
            full.to_parquet(f'./data/AAVE/aave_v3_{coin}_eth.parquet')
            print('----- Collected Full Historical AAVE v3 ETH Data -----')
        else:
            try:
                new = get_data_coin_aave(coin, start_ts = int(prev.index[-1].timestamp()), endpoint = url_v3_eth)
            except Exception as e:
                print(e)
            else:
                full = pd.concat((prev,new), axis = 0)
                full.to_parquet(f'./data/AAVE/aave_v3_{coin}_eth.parquet')
                print('----- Updated Historical AAVE v3 ETH Data -----')
            
    

        id_v2_eth = "C2zniPn45RnLDGzVeGZCx2Sw3GXrbc9gL4ZfL8B8Em2j" # v2 ETH subgraph
        url_v2_eth = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{id_v2_eth}" 
        try:
            prev = pd.read_parquet(f'./data/AAVE/aave_v2_{coin}_eth.parquet')
        except FileNotFoundError :
            full = get_data_coin_aave(coin, start_ts = 0, endpoint = url_v2_eth)
            full.to_parquet(f'./data/AAVE/aave_v2_{coin}_eth.parquet')
            print('----- Collected Full Historical AAVE v2 ETH Data -----')
        else:
            try:
                new = get_data_coin_aave(coin, start_ts = int(prev.index[-1].timestamp()), endpoint = url_v2_eth)
            except Exception as e:
                print(e)
            else:
                full = pd.concat((prev,new), axis = 0)
                full.to_parquet(f'./data/AAVE/aave_v2_{coin}_eth.parquet')
                print('----- Updated Historical AAVE v2 ETH Data -----')

   