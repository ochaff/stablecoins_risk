from __future__ import annotations

import os, math, time, requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get("GRAPH_API_KEY") 
# ---------------- user-provided ----------------
SUBGRAPH_ID = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV" 
ENDPOINT = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}"


POOL_ID = '0x3416cf6c708da44db2624d63ea0aaef7113527c6'
PEG_TICK = 0
N_BUCKETS = 50                       # +/- N buckets around peg (bucket = tickSpacing)
TOKEN0_USD = 1.0                     # USDC
TOKEN1_USD = 1.0                     # USDT

# strongly recommended: bound tick range for queries (must cover your +/- window)
MIN_TICK = PEG_TICK - 500
MAX_TICK = PEG_TICK + 400
# BLOCKS = pd.read_parquet('hourly_blocks.parquet')
_BLOCKS_DF = None


# =======================
# GraphQL helpers
# =======================
def gql_post(endpoint: str, query: str, variables: dict, timeout: int = 60, retries: int = 5) -> dict:
    last_err = None
    for _ in range(retries):
        try:
            r = requests.post(endpoint, json={"query": query, "variables": variables}, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            if "errors" in j:
                raise RuntimeError(j["errors"])
            return j["data"]
        except Exception as e:
            last_err = e
            time.sleep(0.3)
    raise RuntimeError(f"GraphQL request failed: {last_err}")

POOL_STATE_QUERY = """
query($pool: String!, $block: Int!) {
  pool(id: $pool, block: { number: $block }) {
    id
    tick
    liquidity
    sqrtPrice
    token0 { symbol decimals }
    token1 { symbol decimals }
  }
}
"""

TICKS_QUERY = """
query(
  $poolAddr: Bytes!,
  $block: Int!,
  $first: Int!,
  $minTick: BigInt!,
  $maxTick: BigInt!,
  $tickIdxGt: BigInt!
) {
  ticks(
    first: $first,
    orderBy: tickIdx,
    orderDirection: asc,
    block: { number: $block },
    where: {
      poolAddress: $poolAddr,
      tickIdx_gte: $minTick,
      tickIdx_lte: $maxTick,
      tickIdx_gt: $tickIdxGt
    }
  ) {
    tickIdx
    liquidityNet
    liquidityGross
  }
}
"""

def fetch_pool_state_at_block(pool_id: str, block: int) -> dict:
    data = gql_post(ENDPOINT, POOL_STATE_QUERY, {"pool": pool_id, "block": int(block)})
    if data["pool"] is None:
        raise RuntimeError(f"Pool not found at block={block}")
    p = data["pool"]
    # normalize types
    p["tick"] = int(p["tick"])
    p["liquidity"] = int(p["liquidity"])
    p["sqrtPrice"] = int(p["sqrtPrice"])
    p["tickSpacing"] = 1
    p["token0"]["decimals"] = int(p["token0"]["decimals"])
    p["token1"]["decimals"] = int(p["token1"]["decimals"])
    return p

def fetch_ticks_at_block(pool_addr: str, block: int, min_tick: int, max_tick: int, first: int = 1000) -> pd.DataFrame:
    rows = []
    tick_gt = str(min_tick - 1)
    while True:
        data = gql_post(
            ENDPOINT,
            TICKS_QUERY,
            {
                "poolAddr": pool_addr,
                "block": int(block),
                "first": int(first),
                "minTick": str(min_tick),
                "maxTick": str(max_tick),
                "tickIdxGt": str(tick_gt),
            },
        )
        batch = data["ticks"]
        if not batch:
            break
        rows.extend(batch)
        tick_gt = batch[-1]["tickIdx"]
        if len(batch) < first:
            break

    if not rows:
        return pd.DataFrame(columns=["tickIdx", "liquidityNet", "liquidityGross"])

    df = pd.DataFrame(rows)
    df["tickIdx"] = df["tickIdx"].astype("int64")
    # keep as python ints to avoid float rounding in reconstruction
    df["liquidityNet"] = df["liquidityNet"].apply(int)
    df["liquidityGross"] = df["liquidityGross"].apply(int)
    return df.sort_values("tickIdx").reset_index(drop=True)


# =======================
# v3 math + reconstruction
# =======================
def sqrtP_from_tick_np(ticks: np.ndarray) -> np.ndarray:
    return np.power(1.0001, ticks.astype(np.float64) / 2.0)

def price_from_tick_np(ticks: np.ndarray) -> np.ndarray:
    return np.power(1.0001, ticks.astype(np.float64))

def build_active_liquidity_grid(
    grid_ticks: np.ndarray,
    initialized_ticks_df: pd.DataFrame,
    ref_tick: int,
    ref_liq: int,
) -> np.ndarray:
    """
    Reconstruct active liquidity L at each grid tick boundary using:
      prefix(t) = sum liquidityNet at initialized ticks <= t
      L(t) = ref_liq + prefix(t) - prefix(ref_tick_floor)
    where ref_tick_floor is the largest grid tick <= ref_tick.
    """
    liq_net = np.zeros(len(grid_ticks), dtype=object)  # python ints

    if not initialized_ticks_df.empty:
        m = {int(t): int(n) for t, n in zip(initialized_ticks_df["tickIdx"], initialized_ticks_df["liquidityNet"])}
        for i, t in enumerate(grid_ticks):
            v = m.get(int(t))
            if v is not None:
                liq_net[i] = v

    prefix = np.cumsum(liq_net)  # dtype=object

    # anchor at last grid tick <= ref_tick
    i_ref = int(np.searchsorted(grid_ticks, ref_tick, side="right") - 1)
    if i_ref < 0:
        raise ValueError("ref_tick below grid range; widen MIN_TICK/MAX_TICK.")
    ref_prefix = prefix[i_ref]

    L = np.array([ref_liq + (p - ref_prefix) for p in prefix], dtype=object)
    # convert to float for bar math/plotting
    return L.astype(np.float64)

def bars_from_active_liquidity(
    tick_lowers: np.ndarray,
    L: np.ndarray,
    sqrtP_current: float,
    step: int,
    dec0: int,
    dec1: int,
    token0_usd: float,
    token1_usd: float,
) -> pd.DataFrame:
    """
    Convert active liquidity L (for interval [t, t+step]) into per-bucket token0/token1 amounts
    and USD values, like the Uniswap UI liquidity chart.
    """
    tick_lower = tick_lowers
    tick_upper = tick_lowers + step

    sa = sqrtP_from_tick_np(tick_lower)
    sb = sqrtP_from_tick_np(tick_upper)
    sp = float(sqrtP_current)

    amt0_raw = np.zeros_like(L, dtype=np.float64)
    amt1_raw = np.zeros_like(L, dtype=np.float64)

    left = sp <= sa
    right = sp >= sb
    mid = (~left) & (~right)

    amt0_raw[left]  = L[left]  * (1.0 / sa[left] - 1.0 / sb[left])
    amt1_raw[right] = L[right] * (sb[right] - sa[right])
    amt0_raw[mid]   = L[mid]   * (1.0 / sp - 1.0 / sb[mid])
    amt1_raw[mid]   = L[mid]   * (sp - sa[mid])

    amt0 = amt0_raw / (10 ** dec0)
    amt1 = amt1_raw / (10 ** dec1)

    usd0 = amt0 * token0_usd
    usd1 = amt1 * token1_usd

    price_mid = price_from_tick_np(tick_lower + step / 2.0)

    return pd.DataFrame({
        "tickLower": tick_lower.astype(int),
        "tickUpper": tick_upper.astype(int),
        "priceMid": price_mid.astype(float),
        "usd_token0": usd0.astype(float),
        "usd_token1": usd1.astype(float),
        "usd_total": (usd0 + usd1).astype(float),
        "active_liquidity_L": L.astype(float),
    })



def _load_blocks_parquet(path: str = "hourly_blocks.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)

    if "hour_unix" in df.columns:
        ts_col = "hour_unix"
    elif "query_ts" in df.columns:
        ts_col = "query_ts"
    else:
        raise ValueError("Parquet must contain 'hour_unix' or 'query_ts' column")

    if "block_number" not in df.columns:
        raise ValueError("Parquet must contain 'block_number' column")

    df = df[[ts_col, "block_number"]].dropna().copy()
    df[ts_col] = df[ts_col].astype("int64")
    df["block_number"] = df["block_number"].astype("int64")
    df = df.sort_values(ts_col).reset_index(drop=True)
    df = df.rename(columns={ts_col: "ts"})
    return df


def block_by_timestamp(ts: int | datetime, parquet_path: str = "hourly_blocks.parquet") -> int:
    global _BLOCKS_DF
    if _BLOCKS_DF is None:
        _BLOCKS_DF = _load_blocks_parquet(parquet_path)
    if isinstance(ts, datetime):
        dt = ts.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        ts = int(dt.timestamp())
    else:
        ts = int(ts) - (int(ts) % 3600)
    pos = _BLOCKS_DF["ts"].searchsorted(ts)
    if pos >= len(_BLOCKS_DF) or _BLOCKS_DF.at[pos, "ts"] != ts:
        raise KeyError(f"Hour timestamp {ts} not found in {parquet_path}")

    return int(_BLOCKS_DF.at[pos, "block_number"])

# =======================
# Hourly liquidity curve collector
# =======================
def collect_hourly_liquidity_curves(
    start_dt_utc: datetime,
    end_dt_utc: datetime,
    pool_id: str,
    peg_tick: int,
    n_buckets: int,
    min_tick: int,
    max_tick: int,
    peg_alignment: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - states_df: one row per hour (block, tick, liquidity, tickSpacing, decimals)
      - bars_df:   per-hour per-bucket liquidity curve in USD
    """
    # hourly grid
    start = start_dt_utc.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    end = end_dt_utc.replace(tzinfo=timezone.utc)
    hours = []
    cur = start
    while cur < end:
        hours.append(cur)
        cur += timedelta(hours=1)

    state_rows = []
    all_bars = []

    for h in hours:
        ts = int(h.timestamp())
        block = block_by_timestamp(ts)

        pool = fetch_pool_state_at_block(pool_id, block)
        step = pool["tickSpacing"]
        dec0 = pool["token0"]["decimals"]
        dec1 = pool["token1"]["decimals"]

        # align min/max to step grid
        min_aligned = (min_tick // step) * step
        max_aligned = (max_tick // step) * step

        # fetch initialized ticks in range at this block
        ticks_df = fetch_ticks_at_block(pool_id, block, min_aligned, max_aligned)

        # grid of tick boundaries (bucket boundaries)
        grid_ticks = np.arange(min_aligned, max_aligned + step, step, dtype=np.int64)

        # reconstruct active L at each boundary
        L_grid = build_active_liquidity_grid(
            grid_ticks=grid_ticks,
            initialized_ticks_df=ticks_df,
            ref_tick=pool["tick"],
            ref_liq=pool["liquidity"],
        )

        # choose bucket centers around peg_tick
        if peg_alignment :
            peg_aligned = (peg_tick // step) * step
            bar_lowers = np.arange(peg_aligned - n_buckets * step, peg_aligned + n_buckets * step + step, step, dtype=np.int64)
        else:
            active_aligned = (pool["tick"] // step) * step
            bar_lowers = np.arange(
                active_aligned - n_buckets * step,
                active_aligned + n_buckets * step + step,
                step,
                dtype=np.int64
            )
        # map L_grid onto bar_lowers (L at boundary)
        # grid_ticks and bar_lowers are aligned; use index mapping
        idx0 = np.searchsorted(grid_ticks, bar_lowers)
        ok = (idx0 >= 0) & (idx0 < len(grid_ticks))
        L_bar = np.zeros(len(bar_lowers), dtype=np.float64)
        L_bar[ok] = L_grid[idx0[ok]]

        # current sqrtP from tick (consistent with your earlier approach)
        sqrtP_current = float(1.0001 ** (pool["tick"] / 2.0))

        bars = bars_from_active_liquidity(
            tick_lowers=bar_lowers,
            L=L_bar,
            sqrtP_current=sqrtP_current,
            step=step,
            dec0=dec0,
            dec1=dec1,
            token0_usd=TOKEN0_USD,
            token1_usd=TOKEN1_USD,
        )

        bars.insert(0, "hour", h)
        bars.insert(1, "timestamp", ts)
        bars.insert(2, "block", block)
        bars.insert(3, "poolTick", pool["tick"])

        all_bars.append(bars)

        state_rows.append({
            "hour": h,
            "timestamp": ts,
            "block": block,
            "poolTick": pool["tick"],
            "poolLiquidity": pool["liquidity"],
            "tickSpacing": step,
            "token0": pool["token0"]["symbol"],
            "token1": pool["token1"]["symbol"],
            "dec0": dec0,
            "dec1": dec1,
        })

    states_df = pd.DataFrame(state_rows)
    bars_df = pd.concat(all_bars, ignore_index=True) if all_bars else pd.DataFrame()
    return states_df, bars_df


def _dt_tag(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def collect_hourly_liquidity_curves_batched(
    *,
    start_dt_utc: datetime,
    end_dt_utc: datetime,
    pool_id,
    peg_tick: int,
    n_buckets: int,
    min_tick: int,
    max_tick: int,
    out_dir: str = "./data/liquidity_batches",
    batch_hours: int = 1000,
    # retry behavior for “reference tick too low” / tick-range issues:
    retry_min_tick_step: int = 10_000,  # how much to lower min_tick each retry
    max_retries: int = 5,
    skip_if_exists: bool = True,
    peg_alignment: bool = True,
):
    """
    Splits [start_dt_utc, end_dt_utc) into batch_hours chunks.
    Writes one parquet per batch for states_df and bars_df.

    On failure, retries ONLY that failed interval with progressively lower min_tick.
    """
    assert start_dt_utc.tzinfo is not None and end_dt_utc.tzinfo is not None
    assert start_dt_utc < end_dt_utc

    out = Path(out_dir)
    (out / "states").mkdir(parents=True, exist_ok=True)
    (out / "bars").mkdir(parents=True, exist_ok=True)

    chunk_start = start_dt_utc

    while chunk_start < end_dt_utc:
        chunk_end = min(chunk_start + timedelta(hours=batch_hours), end_dt_utc)
        if not peg_alignment:
            tag = f"{_dt_tag(chunk_start)}__{_dt_tag(chunk_end)}"
            states_path = out / "states" / f"states_pricecentered_{tag}.parquet"
            bars_path = out / "bars" / f"bars_pricecentered_{tag}.parquet"
        else:
            tag = f"{_dt_tag(chunk_start)}__{_dt_tag(chunk_end)}"
            states_path = out / "states" / f"states_{tag}.parquet"
            bars_path = out / "bars" / f"bars_{tag}.parquet"

        if skip_if_exists and states_path.exists() and bars_path.exists():
            chunk_start = chunk_end
            continue

        # Try normal min_tick first, then lower only for this failing chunk
        last_err = None
        for attempt in range(max_retries + 1):
            attempt_min_tick = min_tick - attempt * retry_min_tick_step

            try:
                states_df, bars_df = collect_hourly_liquidity_curves(
                    start_dt_utc=chunk_start,
                    end_dt_utc=chunk_end,
                    pool_id=pool_id,
                    peg_tick=peg_tick,
                    n_buckets=n_buckets,
                    min_tick=attempt_min_tick,
                    max_tick=max_tick,
                    peg_alignment=peg_alignment,
                )

                # Write batch outputs
                states_df.to_parquet(states_path, index=False)
                bars_df.to_parquet(bars_path, index=False)

                last_err = None
                break

            except Exception as e:  # replace with your specific error if possible
                last_err = e

                # If you can detect the “reference tick too low” condition, do it here.
                # Example (customize): only retry when error message indicates tick-range issue.
                msg = str(e).lower()
                tick_related = ("tick" in msg) or ("range" in msg) or ("reference" in msg)

                if (attempt >= max_retries) or (not tick_related):
                    # Either out of retries, or not the tick issue -> fail fast
                    raise

                # else: retry with lower min_tick for THIS chunk only

        if last_err is not None:
            raise last_err

        chunk_start = chunk_end


if __name__ == "__main__":
    start_dt = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    end_dt = datetime(2026, 1, 19, 8, 0, 0, tzinfo=timezone.utc)

    collect_hourly_liquidity_curves_batched(
        start_dt_utc=start_dt,
        end_dt_utc=end_dt,
        pool_id=POOL_ID,
        peg_tick=PEG_TICK,
        n_buckets=N_BUCKETS,
        min_tick=MIN_TICK,
        max_tick=MAX_TICK,
        out_dir="./data/liquidity_batches",
        batch_hours=1000,
        retry_min_tick_step=1_000, 
        max_retries=5,
        peg_alignment = False,
    )