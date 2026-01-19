from __future__ import annotations

import os, math, time, requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv 
load_dotenv()


# --- add near your other endpoints / env vars ---
GRAPH_API_KEY = os.environ["GRAPH_API_KEY"]  # or whatever your gateway key env var is

BLOCKS_SUBGRAPH_ID = "236pc6mnPH2EdGJxR5wunibyGsagq1JsSx5e2hx5tdoE"  # blocklytics/ethereum-blocks
BLOCKS_ENDPOINT = f"https://gateway.thegraph.com/api/{GRAPH_API_KEY}/subgraphs/id/{BLOCKS_SUBGRAPH_ID}"


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


def next_hour(dt: pd.Timestamp) -> pd.Timestamp:
    """Ceil to the next hour (exclusive)."""
    dt = pd.Timestamp(dt)
    if dt.tz is None:
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")
    dt_floor = dt.floor("h")
    return dt_floor

def hourly_timestamps_from_df_to_now(df: pd.DataFrame, ts_col: str = "timestamp") -> list[int]:
    """
    Uses the last value in df[ts_col] as the starting point (exclusive),
    and returns a list of unix timestamps (hourly) up to current UTC hour (inclusive).
    Assumes df[ts_col] is unix seconds.
    """
    if df.empty:
        raise ValueError("df is empty")

    last_ts = int(df[ts_col].max())
    last_dt = pd.to_datetime(last_ts, unit="s", utc=True)

    start_dt = next_hour(last_dt)  # start after the last timestamp in df
    end_dt = pd.Timestamp(datetime.now(timezone.utc)).floor("h")

    if start_dt > end_dt:
        return []

    hour_index = pd.date_range(start=start_dt, end=end_dt, freq="h", tz="UTC")
    return [int(x.timestamp()) for x in hour_index]


def fetch_blocks_after_timestamps(
    timestamps: list[int],
    batch_size: int = 48,
    sleep_s: float = 0.05,
) -> dict[int, int]:
    """
    For each unix timestamp ts, returns the first block with timestamp >= ts.
    Uses GraphQL aliases to batch many lookups per request.

    Returns: {ts: block_number}
    """
    out = []

    for b in range(0, len(timestamps), batch_size):
        batch = timestamps[b:b + batch_size]

        # Build one query with aliases h0..hN
        var_defs = []
        fields = []
        variables = {}

        for i, ts in enumerate(batch):
            vn = f"ts{i}"
            var_defs.append(f"${vn}: BigInt!")
            variables[vn] = int(ts)

            # Prefer timestamp_gte; if your schema only supports *_gt, switch to timestamp_gt.
            fields.append(f"""
              h{i}: blocks(
                first: 1
                orderBy: timestamp
                orderDirection: asc
                where: {{ timestamp_gte: ${vn} }}
              ) {{
                number
                timestamp
                id
                parentHash
              }}
            """)

        query = f"query ({', '.join(var_defs)}) {{\n{''.join(fields)}\n}}"
        data = gql_post(BLOCKS_ENDPOINT, query, variables)

        for i, ts in enumerate(batch):
            k = f"h{i}"
            arr = data.get(k, [])
            if not arr:
                raise RuntimeError(f"No block found for ts={ts} (alias {k})")
            out.append({'hour_utc': pd.to_datetime(datetime.fromtimestamp(ts,timezone.utc)),
            'hour_unix': ts,
            'block_number' : arr[0]['number'],
            'block_timestamp': arr[0]['timestamp'],
            'block_id': arr[0]['id'],
            'parent_hash': arr[0]['parentHash']
            })



    return pd.DataFrame(out)


if __name__ == '__main__':
    df = pd.read_parquet('hourly_blocks.parquet')
    new_hourly_ts = hourly_timestamps_from_df_to_now(df, ts_col="hour_unix")
    timestamps = new_hourly_ts[1:]
    out = fetch_blocks_after_timestamps(timestamps=timestamps)
    out.block_number = out.block_number.astype(int)
    out.block_timestamp = out.block_timestamp.astype(int)
    
    df = pd.concat((df, pd.DataFrame(out)), axis = 0)
    df.to_parquet('./hourly_blocks.parquet')
    print('------ Hourly blocks timestamps updated -----')
    