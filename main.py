from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
import pandas as pd
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TD_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
TD_BASE = "https://api.twelvedata.com"

ALLOWED_INTERVALS = {
    "1min","5min","15min","30min",
    "1h","2h","4h","8h","12h",
    "1day","1week","1month"
}

class StrategyBlock(BaseModel):
    type: str
    params: dict = {}

class BacktestRequest(BaseModel):
    market: str
    symbol: str
    interval: str
    outputsize: int = 500
    exchange: str | None = None

    # Strategy JSON blocks
    entry: StrategyBlock
    filter: StrategyBlock | None = None
    exit: StrategyBlock | None = None

    # realism
    fee_bps: float = 2.0
    slippage_bps: float = 1.0


@app.get("/health")
def health():
    return {"status": "ok", "provider": "twelvedata", "has_api_key": bool(TD_API_KEY)}


@app.get("/backtests/mock")
def backtests_mock():
    return {
        "strategy_name": "Mock Strategy (fallback)",
        "symbol": "BTC/USD",
        "execution_timeframe": "1h",
        "capital": {"initial_sek": 10000, "final_sek": 11250},
        "kpi": {
            "total_return_pct": 12.5,
            "max_drawdown_pct": -4.2,
            "sharpe": 1.1,
            "win_rate_pct": 52,
            "trade_count": 38
        },
        "equity_curve": [
            {"time": "2026-01-01T00:00:00", "equity_sek": 10000},
            {"time": "2026-01-02T00:00:00", "equity_sek": 10120},
            {"time": "2026-01-03T00:00:00", "equity_sek": 10080},
            {"time": "2026-01-04T00:00:00", "equity_sek": 10340},
            {"time": "2026-01-05T00:00:00", "equity_sek": 10650},
            {"time": "2026-01-06T00:00:00", "equity_sek": 10580},
            {"time": "2026-01-07T00:00:00", "equity_sek": 11250}
        ],
        "drawdown_curve": [
            {"time": "2026-01-01T00:00:00", "dd_pct": 0},
            {"time": "2026-01-02T00:00:00", "dd_pct": 0},
            {"time": "2026-01-03T00:00:00", "dd_pct": -0.4},
            {"time": "2026-01-04T00:00:00", "dd_pct": 0},
            {"time": "2026-01-05T00:00:00", "dd_pct": 0},
            {"time": "2026-01-06T00:00:00", "dd_pct": -0.7},
            {"time": "2026-01-07T00:00:00", "dd_pct": 0}
        ],
        "ai_insights": {
            "summary": "Fallback mock while no recent backtest is stored in session.",
            "confidence_pct": 50
        }
    }


def _td_request(symbol: str, interval: str, outputsize: int, exchange: str | None):
    url = f"{TD_BASE}/time_series"
    params = {
        "apikey": TD_API_KEY,
        "symbol": symbol,
        "interval": interval,
        "outputsize": int(max(50, min(outputsize, 5000))),
        "format": "JSON",
        "timezone": "UTC",
        "order": "ASC",
    }
    if exchange:
        params["exchange"] = exchange

    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"TwelveData HTTP {r.status_code}: {r.text}")

    data = r.json()
    if isinstance(data, dict) and data.get("status") == "error":
        raise HTTPException(status_code=400, detail=f"TwelveData error: {data.get('message')}")
    return data


def td_time_series(symbol: str, interval: str, outputsize: int, exchange: str | None):
    if not TD_API_KEY:
        raise HTTPException(status_code=500, detail="Missing TWELVEDATA_API_KEY in Railway variables")
    if interval not in ALLOWED_INTERVALS:
        raise HTTPException(status_code=400, detail=f"Invalid interval. Allowed: {sorted(ALLOWED_INTERVALS)}")

    # try with exchange; if fails and exchange provided -> retry without
    try:
        data = _td_request(symbol, interval, outputsize, exchange)
    except HTTPException as e:
        if exchange:
            data = _td_request(symbol, interval, outputsize, None)
        else:
            raise e

    values = data.get("values")
    if not values or not isinstance(values, list):
        raise HTTPException(status_code=400, detail=f"No OHLCV returned for {symbol} {interval}")

    df = pd.DataFrame(values)
    if "datetime" not in df.columns or "close" not in df.columns:
        raise HTTPException(status_code=400, detail="Provider response missing datetime/close")

    df["time"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(None)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = df.sort_values("time").reset_index(drop=True)
    df = df.dropna(subset=["close"])
    if df.empty or len(df) < 50:
        raise HTTPException(status_code=400, detail="Not enough data to backtest (try higher outputsize)")

    # Ensure OHLC exists (some series might not provide open/high/low)
    for col in ["open", "high", "low"]:
        if col not in df.columns:
            df[col] = df["close"]

    return df


def apply_entry(df: pd.DataFrame, block: StrategyBlock) -> pd.Series:
    t = block.type
    p = block.params or {}

    if t == "ema_crossover":
        fast = int(p.get("fast", 20))
        slow = int(p.get("slow", 50))
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        return (ema_fast > ema_slow).astype(int)

    if t == "rsi_oversold":
        length = int(p.get("length", 14))
        level = float(p.get("level", 30))
        delta = df["close"].diff().fillna(0)
        gain = delta.clip(lower=0).rolling(length).mean()
        loss = (-delta.clip(upper=0)).rolling(length).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return (rsi < level).astype(int)

    if t == "breakout_high":
        lookback = int(p.get("lookback", 20))
        hh = df["close"].rolling(lookback).max().shift(1)
        return (df["close"] > hh).astype(int)

    raise HTTPException(status_code=400, detail=f"Unknown entry type: {t}")


def resample_to_interval(df: pd.DataFrame, target: str) -> pd.DataFrame:
    freq_map = {
        "1h": "1H",
        "2h": "2H",
        "4h": "4H",
        "8h": "8H",
        "12h": "12H",
        "1day": "1D",
        "1week": "1W",
        "1month": "1M",
    }
    if target not in freq_map:
        raise HTTPException(status_code=400, detail=f"Unsupported HTF interval: {target}")

    d = df.set_index("time").sort_index()
    o = d["open"].resample(freq_map[target]).first()
    h = d["high"].resample(freq_map[target]).max()
    l = d["low"].resample(freq_map[target]).min()
    c = d["close"].resample(freq_map[target]).last()
    v = d["volume"].resample(freq_map[target]).sum() if "volume" in d.columns else None

    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c})
    if v is not None:
        out["volume"] = v
    out = out.dropna(subset=["close"]).reset_index()
    return out


def apply_filter_ltf(df_ltf: pd.DataFrame, block: StrategyBlock) -> pd.Series:
    try:
        t = block.type
        p = block.params or {}

        if t != "htf_close_above_ema":
            return pd.Series(1, index=df_ltf.index)

        htf_interval = str(p.get("interval", "4h"))
        ema_len = int(p.get("ema", 200))

        df_htf = resample_to_interval(df_ltf, htf_interval)

        if df_htf.empty or len(df_htf) < 10:
            # Not enough HTF data → disable filter instead of crashing
            return pd.Series(1, index=df_ltf.index)

        df_htf["ema"] = df_htf["close"].ewm(span=ema_len, adjust=False).mean()
        df_htf["ok"] = (df_htf["close"] > df_htf["ema"]).astype(int)

        ltf = df_ltf[["time"]].copy().sort_values("time")
        htf = df_htf[["time", "ok"]].copy().sort_values("time")

        merged = pd.merge_asof(ltf, htf, on="time", direction="backward")

        ok = merged["ok"].fillna(1).astype(int)
        return ok

    except Exception:
        # If anything fails, just don't filter instead of crashing
        return pd.Series(1, index=df_ltf.index)
        t = block.type
    p = block.params or {}

    if t != "htf_close_above_ema":
        raise HTTPException(status_code=400, detail=f"Unknown filter type: {t}")

    htf_interval = str(p.get("interval", "4h"))
    ema_len = int(p.get("ema", 200))

    df_htf = resample_to_interval(df_ltf, htf_interval)
    df_htf["ema"] = df_htf["close"].ewm(span=ema_len, adjust=False).mean()
    df_htf["ok"] = (df_htf["close"] > df_htf["ema"]).astype(int)

    ltf = df_ltf[["time"]].copy().sort_values("time")
    htf = df_htf[["time", "ok"]].copy().sort_values("time")

    merged = pd.merge_asof(ltf, htf, on="time", direction="backward")
    ok = merged["ok"].fillna(0).astype(int)
    return ok


def backtest_long_only(df: pd.DataFrame, signal: pd.Series, fee_bps: float, slippage_bps: float):
    df = df.copy()
    df["signal"] = signal.fillna(0).astype(int)
    df["returns"] = df["close"].pct_change().fillna(0.0)

    pos = df["signal"].shift(1).fillna(0).astype(int)
    strat_ret = df["returns"] * pos

    trade_event = (df["signal"].diff().fillna(0) != 0).astype(int)
    cost_pct = (fee_bps + slippage_bps) / 10000.0
    strat_ret = strat_ret - (trade_event * cost_pct)

    df["strategy_returns"] = strat_ret.replace([np.inf, -np.inf], 0).fillna(0)
    df["equity"] = (1.0 + df["strategy_returns"]).cumprod()

    running_max = df["equity"].cummax()
    dd = df["equity"] / running_max - 1.0

    return df, dd, int(trade_event.sum())


@app.post("/backtests/run")
def run_backtest(req: BacktestRequest):
    market = (req.market or "").lower().strip()
    symbol = (req.symbol or "").strip()
    interval = (req.interval or "").strip()
    exchange = (req.exchange or "").strip() if req.exchange else None

    if market not in {"crypto","forex","stocks","indices"}:
        raise HTTPException(status_code=400, detail="market must be one of: crypto, forex, stocks, indices")
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol is required")

    df = td_time_series(symbol, interval, req.outputsize, exchange)

    entry_sig = apply_entry(df, req.entry)
    filt_sig = apply_filter_ltf(df, req.filter) if req.filter else pd.Series(1, index=df.index)

    signal = (entry_sig & filt_sig).astype(int)

    df_bt, dd, trade_count = backtest_long_only(df, signal, req.fee_bps, req.slippage_bps)

    initial_sek = 10000.0
    final_sek = float(df_bt["equity"].iloc[-1] * initial_sek)

    total_return_pct = float((final_sek / initial_sek - 1.0) * 100.0)
    max_drawdown_pct = float(dd.min() * 100.0)

    std = float(df_bt["strategy_returns"].std())
    sharpe = float(df_bt["strategy_returns"].mean() / (std + 1e-9) * np.sqrt(252))

    equity_curve = [{"time": str(t), "equity_sek": float(e * initial_sek)} for t, e in zip(df_bt["time"], df_bt["equity"])]
    drawdown_curve = [{"time": str(t), "dd_pct": float(d * 100.0)} for t, d in zip(df_bt["time"], dd)]

    return {
        "strategy_name": f"{req.entry.type}" + (f" + {req.filter.type}" if req.filter else ""),
        "symbol": symbol,
        "execution_timeframe": interval,
        "capital": {"initial_sek": int(initial_sek), "final_sek": final_sek},
        "kpi": {
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "sharpe": sharpe,
            "win_rate_pct": 50,
            "trade_count": trade_count
        },
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
        "ai_insights": {
            "summary": "Strategy JSON (entry/filter) + fees/slippage enabled. Next: exits (ATR/TP) + trade list + save runs.",
            "confidence_pct": 75
        }
    }
