from __future__ import annotations

import time
import math
import requests
import pandas as pd
from tqdm import tqdm
import os 
from dotenv import load_dotenv
load_dotenv()


def gql(query: str, variables: dict, retries: int = 6, backoff: float = 0.8) -> dict:
    for i in range(retries):
        r = requests.post(SUBGRAPH_URL, json={"query": query, "variables": variables}, timeout=60)
        if r.status_code == 200:
            j = r.json()
            if "errors" in j:
                raise RuntimeError(j["errors"])
            return j["data"]
        time.sleep(backoff * (2**i))
    r.raise_for_status()


# -----------------------
# Math helpers
# -----------------------

def hour_floor(ts: int) -> int:
    return ts - (ts % 3600)

def sqrtPriceX96_to_price_token1_per_token0(sqrtPriceX96: int, dec0: int, dec1: int) -> float:
    """
    price(token1 per token0) = (sqrtP/2^96)^2 * 10^(dec0-dec1)
    """
    sp = sqrtPriceX96 / (2 ** 96)
    return (sp * sp) * (10 ** (dec0 - dec1))


# -----------------------
# Queries
# -----------------------

POOL_META_Q = """
query($id: ID!) {
  pool(id: $id) {
    id
    feeTier
    token0 { id symbol decimals }
    token1 { id symbol decimals }
  }
}
"""

SWAPS_Q = """
query($pool: String!, $start: Int!, $end: Int!, $first: Int!) {
  swaps(
    first: $first
    where: { pool: $pool, timestamp_gte: $start, timestamp_lt: $end }
    orderBy: timestamp
    orderDirection: asc
  ) {
    id
    timestamp
    amount0
    amount1
    amountUSD
  }
}
"""

POOL_HOURS_Q = """
query($pool: String!, $start: Int!, $end: Int!, $after: Int!, $first: Int!) {
  poolHourDatas(
    first: $first
    where: { pool: $pool, periodStartUnix_gte: $start, periodStartUnix_lt: $end, periodStartUnix_gt: $after }
    orderBy: periodStartUnix
    orderDirection: asc
  ) {
    id
    periodStartUnix
    sqrtPrice
    tick
    liquidity
    tvlUSD
    volumeUSD
    feesUSD
    open
    high
    low
    close
  }
}
"""

_POOL_META_CACHE: dict[str, dict] = {}

def fetch_pool_meta(pool_id: str) -> dict:
    pool_id = pool_id.lower()
    if pool_id in _POOL_META_CACHE:
        return _POOL_META_CACHE[pool_id]
    meta = gql(POOL_META_Q, {"id": pool_id})["pool"]
    _POOL_META_CACHE[pool_id] = meta
    return meta


def fetch_hourly_net_swaps_for_pool(
    pool_id: str,
    fee_tier: int,
    start_ts: int,
    end_ts: int,
    chunk_days: int = 14,
    page_size: int = 1000,
    polite_sleep: float = 0.05,
    fill_missing_hours: bool = True,
) -> pd.DataFrame:
    """
    Returns hourly net sums of swap deltas (pool perspective):
      net_amount0 = sum(amount0)
      net_amount1 = sum(amount1)
      net_amountUSD = sum(amountUSD)
      swap_count = number of swaps
    """
    pool_id = pool_id.lower()
    chunk_seconds = chunk_days * 86400

    acc = {}  # hour_ts -> [count, net0, net1, netUSD]
    chunk_start = start_ts

    pbar = tqdm(total=max(0, end_ts - start_ts), desc=f"net swaps {pool_id[:6]}..{pool_id[-4:]}")

    while chunk_start < end_ts:
        chunk_end = min(end_ts, chunk_start + chunk_seconds)

        cursor = chunk_start
        while True:
            swaps = gql(SWAPS_Q, {"pool": pool_id, "start": cursor, "end": chunk_end, "first": page_size})["swaps"]
            if not swaps:
                break

            last_ts_in_page = None
            for s in swaps:
                ts = int(s["timestamp"])
                last_ts_in_page = ts
                h = hour_floor(ts)

                a0 = float(s["amount0"])
                a1 = float(s["amount1"])
                aUSD = float(s["amountUSD"])

                if h not in acc:
                    acc[h] = [0, 0.0, 0.0, 0.0]
                acc[h][0] += 1
                acc[h][1] += a0
                acc[h][2] += a1
                acc[h][3] += aUSD

            # advance by 1 second to avoid repeating the last second
            cursor = int(last_ts_in_page) + 1

            if len(swaps) < page_size:
                break
            if polite_sleep:
                time.sleep(polite_sleep)

        pbar.update(chunk_end - chunk_start)
        chunk_start = chunk_end

    pbar.close()

    if not acc:
        df = pd.DataFrame(columns=["pool", "feeTier", "hour", "datetime", "swap_count",
                                   "net_amount0", "net_amount1", "net_amountUSD"])
        return df

    df = pd.DataFrame(
        [
            {
                "pool": pool_id,
                "feeTier": fee_tier,
                "hour": int(h),
                "datetime": pd.to_datetime(int(h), unit="s", utc=True),
                "swap_count": int(v[0]),
                "net_amount0": float(v[1]),
                "net_amount1": float(v[2]),
                "net_amountUSD": float(v[3]),
            }
            for h, v in acc.items()
        ]
    ).sort_values("hour").reset_index(drop=True)

    if not fill_missing_hours:
        return df

    # build continuous hourly grid and fill gaps with zeros
    full = pd.DataFrame({"hour": range(hour_floor(start_ts), hour_floor(end_ts) + 3600, 3600)})
    full["datetime"] = pd.to_datetime(full["hour"], unit="s", utc=True)
    df = full.merge(df, on=["hour", "datetime"], how="left")
    df["pool"] = df["pool"].fillna(pool_id)
    df["feeTier"] = df["feeTier"].fillna(fee_tier).astype(int)
    for c in ["swap_count", "net_amount0", "net_amount1", "net_amountUSD"]:
        df[c] = df[c].fillna(0)
    df["swap_count"] = df["swap_count"].astype(int)

    return df


def fetch_hourly_prices_for_pool(
    pool_id: str,
    fee_tier: int,
    start_ts: int,
    end_ts: int,
    chunk_days: int = 180,
    page_size: int = 1000,
    polite_sleep: float = 0.05,
) -> pd.DataFrame:
    """
    Pulls poolHourDatas and computes price from sqrtPrice.
    """
    pool_id = pool_id.lower()
    meta = fetch_pool_meta(pool_id)
    dec0, dec1 = int(meta["token0"]["decimals"]), int(meta["token1"]["decimals"])

    out = []
    chunk_seconds = chunk_days * 86400
    chunk_start = start_ts

    pbar = tqdm(total=max(0, end_ts - start_ts), desc=f"hourly px {pool_id[:6]}..{pool_id[-4:]}")

    while chunk_start < end_ts:
        chunk_end = min(end_ts, chunk_start + chunk_seconds)

        after = chunk_start - 1
        while True:
            batch = gql(
                POOL_HOURS_Q,
                {"pool": pool_id, "start": chunk_start, "end": chunk_end, "after": after, "first": page_size},
            )["poolHourDatas"]
            if not batch:
                break

            out.extend(batch)
            after = int(batch[-1]["periodStartUnix"])

            if len(batch) < page_size:
                break
            if polite_sleep:
                time.sleep(polite_sleep)

        pbar.update(chunk_end - chunk_start)
        chunk_start = chunk_end

    pbar.close()

    df = pd.DataFrame(out)
    if df.empty:
        return df

    df["hour"] = df["periodStartUnix"].astype(int)
    df = df.drop(columns=["periodStartUnix"])

    for c in ["sqrtPrice","tick","liquidity","tvlUSD","volumeUSD","feesUSD","open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["price1_per_0"] = df["sqrtPrice"].apply(
        lambda sp: sqrtPriceX96_to_price_token1_per_token0(int(sp), dec0, dec1) if pd.notna(sp) else math.nan
    )
    df["price0_per_1"] = 1.0 / df["price1_per_0"]

    df["pool"] = pool_id
    df["feeTier"] = fee_tier
    df["datetime"] = pd.to_datetime(df["hour"], unit="s", utc=True)

    return (
        df.sort_values(["pool", "feeTier", "hour"])
          .drop_duplicates(["pool", "feeTier", "hour"])
          .reset_index(drop=True)
    )




def collect_hourly_prices_and_net_swaps(
    pools,
    start_ts: int | None = None,
    end_ts: int | None = None,
    fill_missing_hours_for_swaps: bool = True,
) -> pd.DataFrame:
    if start_ts is None:
        start_ts = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp())
    if end_ts is None:
        end_ts = int(pd.Timestamp.now(tz="UTC").timestamp())

    frames = []
    for p in pools:
        pool_id = p["id"].lower()
        fee_tier = int(p["feeTier"])

        # 1) swaps aggregated to hour (optionally with zero-filled hours)
        swaps_h = fetch_hourly_net_swaps_for_pool(
            pool_id, fee_tier, start_ts, end_ts,
            fill_missing_hours=fill_missing_hours_for_swaps,
        )

        # 2) hourly price from poolHourDatas (only returns hours where subgraph has a row)
        px_h = fetch_hourly_prices_for_pool(pool_id, fee_tier, start_ts, end_ts)

        # merge
        if swaps_h.empty and px_h.empty:
            continue

        # If swaps are zero-filled, left join swaps grid to prices.
        # If swaps are not zero-filled, outer join keeps all hours from both.
        how = "left" if fill_missing_hours_for_swaps else "outer"

        merged = swaps_h.merge(
            px_h[[
                "pool", "feeTier", "hour",
                "price1_per_0", "price0_per_1",
                "tick", "liquidity", "tvlUSD", "volumeUSD", "feesUSD",
                "open", "high", "low", "close",
                "sqrtPrice",
            ]] if not px_h.empty else px_h,
            on=["pool", "feeTier", "hour"],
            how=how,
            suffixes=("", "_px"),
        )

        # ensure datetime exists
        if "datetime" not in merged.columns or merged["datetime"].isna().all():
            merged["datetime"] = pd.to_datetime(merged["hour"], unit="s", utc=True)

        frames.append(merged)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True).sort_values(["pool", "feeTier", "hour"]).reset_index(drop=True)
    return df


def trim_leading_zero_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove initial hours per (pool, feeTier) where there is no activity and no price row.
    Adjust the definition as needed.
    """
    gcols = ["pool", "feeTier"]

    inactive = (
        (df["swap_count"].fillna(0) == 0)
        & (df["net_amountUSD"].fillna(0) == 0)
        & (df["price1_per_0"].isna())
    )

    first_hour = (
        df.loc[~inactive]
          .groupby(gcols)["hour"]
          .min()
          .rename("first_hour")
          .reset_index()
    )

    return (
        df.merge(first_hour, on=gcols, how="inner")
          .query("hour >= first_hour")
          .drop(columns=["first_hour"])
          .sort_values(gcols + ["hour"])
          .reset_index(drop=True)
    )


if __name__ == "__main__":

    API_KEY = os.environ.get("GRAPH_API_KEY") 
    SUBGRAPH_URL = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
    
    ### USDC/USDT v3 pools
    POOLS_USDC_USDT = [
        {"feeTier": 100,   "id": "0x3416cf6c708da44db2624d63ea0aaef7113527c6"},
        {"feeTier": 500,   "id": "0x7858e59e0c01ea06df3af3d20ac7b0003275d4bf"},
        {"feeTier": 10000, "id": "0xbb256c2f1b677e27118b0345fd2b3894d2e6d487"},
        {"feeTier": 3000,  "id": "0xee4cf3b78a74affa38c6a926282bcd8b5952818d"},
    ]
    POOLS_DAI_USDC = [
    {"feeTier": 100,   "id": "0x5777d92f208679db4b9778590fa3cab3ac9e2168"},  
    {"feeTier": 500,   "id": "0x6c6bc977e13df9b0de53b251522280bb72383700"},  
    {"feeTier": 3000,  "id": "0xa63b490aa077f541c9d64bfc1cc0db2a752157b5"},   
    {"feeTier": 10000, "id": "0x6958686b6348c3d6d5f2dca3106a5c09c156873a"}, 
    ]


    for pair, pool in zip(['USDC_USDT','DAI_USDC'], [POOLS_USDC_USDT,POOLS_DAI_USDC]):
        start_ts = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp())
        end_ts = int(pd.Timestamp.now(tz="UTC").timestamp())

        hourly = collect_hourly_prices_and_net_swaps(
            pools=pool,
            start_ts=start_ts,
            end_ts=end_ts,
            fill_missing_hours_for_swaps=True,  # gives you a continuous hourly grid per pool
        )


        hourly = trim_leading_zero_hours(hourly)

        if "price1_per_0" in hourly.columns:
            hourly["depeg_bps"] = (hourly["price1_per_0"] - 1.0) * 1e4

        hourly.to_parquet(f"./data/{pair}_hourly_metrics.parquet", index=False)