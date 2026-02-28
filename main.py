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

class BacktestRequest(BaseModel):
    market: str              # crypto | forex | stocks | indices
    symbol: str
    interval: str
    outputsize: int = 500
    exchange: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "provider": "twelvedata", "has_api_key": bool(TD_API_KEY)}


@app.get("/backtests/mock")
def backtests_mock():
    return {
        "strategy_name": "Mock Strategy",
        "symbol": "BTC/USD",
        "execution_timeframe": "1h",
        "capital": {"initial_sek": 10000, "final_sek": 12000},
        "kpi": {
            "total_return_pct": 20,
            "max_drawdown_pct": -5,
            "sharpe": 1.5,
            "win_rate_pct": 55,
            "trade_count": 50
        },
        "equity_curve": [
            {"time": "2026-01-01T00:00:00", "equity_sek": 10000},
            {"time": "2026-01-02T00:00:00", "equity_sek": 10150},
            {"time": "2026-01-03T00:00:00", "equity_sek": 10080},
            {"time": "2026-01-04T00:00:00", "equity_sek": 10320},
        ],
        "drawdown_curve": [
            {"time": "2026-01-01T00:00:00", "dd_pct": 0},
            {"time": "2026-01-02T00:00:00", "dd_pct": 0},
            {"time": "2026-01-03T00:00:00", "dd_pct": -0.69},
            {"time": "2026-01-04T00:00:00", "dd_pct": 0},
        ],
        "ai_insights": {"summary": "Mock data.", "confidence_pct": 60}
    }


def _td_request_time_series(symbol: str, interval: str, outputsize: int, exchange: str | None):
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

    # 1) Try with exchange (if provided)
    try:
        data = _td_request_time_series(symbol, interval, outputsize, exchange)
    except HTTPException as e:
        # 2) If exchange was provided, retry once WITHOUT exchange (fixes many ETF/index cases)
        if exchange:
            try:
                data = _td_request_time_series(symbol, interval, outputsize, None)
            except HTTPException:
                raise e
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
    if df.empty or len(df) < 20:
        raise HTTPException(status_code=400, detail="Not enough data to backtest (try higher outputsize)")

    return df


@app.post("/backtests/run")
def run_backtest(req: BacktestRequest):
    market = (req.market or "").lower().strip()
    symbol = (req.symbol or "").strip()
    interval = (req.interval or "").strip()
    exchange = (req.exchange or "").strip() if req.exchange else None

    if market not in {"crypto", "forex", "stocks", "indices"}:
        raise HTTPException(status_code=400, detail="market must be one of: crypto, forex, stocks, indices")
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol is required")

    df = td_time_series(symbol, interval, req.outputsize, exchange)

    # Strategy v1: EMA 20/50 crossover
    ema_fast, ema_slow = 20, 50
    df["ema_fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()
    df["signal"] = (df["ema_fast"] > df["ema_slow"]).astype(int)

    df["returns"] = df["close"].pct_change().fillna(0.0)
    df["strategy_returns"] = (df["returns"] * df["signal"].shift(1).fillna(0)).replace([np.inf, -np.inf], 0).fillna(0)
    df["equity"] = (1.0 + df["strategy_returns"]).cumprod()

    running_max = df["equity"].cummax()
    drawdown = df["equity"] / running_max - 1.0

    initial_sek = 10000.0
    final_sek = float(df["equity"].iloc[-1] * initial_sek)

    total_return_pct = float((final_sek / initial_sek - 1.0) * 100.0)
    max_drawdown_pct = float(drawdown.min() * 100.0)

    std = float(df["strategy_returns"].std())
    sharpe = float(df["strategy_returns"].mean() / (std + 1e-9) * np.sqrt(252))

    trade_count = int((df["signal"].diff().fillna(0) != 0).sum())

    equity_curve = [{"time": str(t), "equity_sek": float(e * initial_sek)} for t, e in zip(df["time"], df["equity"])]
    drawdown_curve = [{"time": str(t), "dd_pct": float(d * 100.0)} for t, d in zip(df["time"], drawdown)]

    return {
        "strategy_name": f"EMA {ema_fast}/{ema_slow}",
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
            "summary": "Baseline EMA crossover. Next: RSI/ATR + multi-timeframe confirmation + fees/slippage.",
            "confidence_pct": 70
        }
    }
