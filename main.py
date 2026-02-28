from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

class BacktestRequest(BaseModel):
    symbol: str
    timeframe: str
    days: int = 30


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/backtests/mock")
def backtests_mock():
    return {
        "strategy_name": "Mock Strategy",
        "symbol": "BTCUSDT",
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
        "ai_insights": {
            "summary": "Mock data.",
            "confidence_pct": 60
        }
    }


def fetch_binance_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    # Binance limit=1000 candles per request. For 1h and 30 days, 720 candles -> OK.
    end_time_ms = int(datetime.utcnow().timestamp() * 1000)
    start_time_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time_ms,
        "endTime": end_time_ms,
        "limit": 1000
    }

    r = requests.get(
        BINANCE_KLINES_URL,
        params=params,
        timeout=12,
        headers={"User-Agent": "lovable-trading/1.0"},
    )

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Binance HTTP {r.status_code}: {r.text}")

    data = r.json()

    # Binance errors come as dict like {"code":-1121,"msg":"Invalid symbol."}
    if isinstance(data, dict):
        msg = data.get("msg") or str(data)
        raise HTTPException(status_code=400, detail=f"Binance error: {msg}")

    if not isinstance(data, list) or len(data) == 0:
        raise HTTPException(status_code=400, detail="No OHLCV candles returned (check symbol/timeframe/days)")

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades",
        "taker_base_vol", "taker_quote_vol", "ignore"
    ])

    # Convert types safely
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)

    df = df.dropna(subset=["close"]).sort_values("time").reset_index(drop=True)

    if df.empty or len(df) < 10:
        raise HTTPException(status_code=400, detail="Not enough OHLCV data to run backtest")

    return df


@app.post("/backtests/run")
def run_backtest(req: BacktestRequest):
    """
    Simple EMA 20/50 crossover backtest on Binance OHLCV.
    Returns format expected by frontend (kpi + curves).
    """
    symbol = (req.symbol or "").upper().strip()
    interval = (req.timeframe or "").strip()

    if not symbol:
        raise HTTPException(status_code=400, detail="symbol is required")
    if not interval:
        raise HTTPException(status_code=400, detail="timeframe is required")
    if req.days < 1 or req.days > 365:
        raise HTTPException(status_code=400, detail="days must be between 1 and 365")

    df = fetch_binance_klines(symbol, interval, req.days)

    # Strategy params (hard-coded for v1)
    ema_fast = 20
    ema_slow = 50

    df["ema_fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()

    df["signal"] = (df["ema_fast"] > df["ema_slow"]).astype(int)

    df["returns"] = df["close"].pct_change().fillna(0.0)
    df["strategy_returns"] = (df["returns"] * df["signal"].shift(1).fillna(0)).replace([np.inf, -np.inf], 0).fillna(0)

    df["equity"] = (1.0 + df["strategy_returns"]).cumprod()
    df["equity"] = df["equity"].replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(1.0)

    running_max = df["equity"].cummax()
    drawdown = df["equity"] / running_max - 1.0

    # KPI
    initial_sek = 10000.0
    final_sek = float(df["equity"].iloc[-1] * initial_sek)

    total_return_pct = float((final_sek / initial_sek - 1.0) * 100.0)
    max_drawdown_pct = float(drawdown.min() * 100.0)

    std = float(df["strategy_returns"].std())
    sharpe = float(df["strategy_returns"].mean() / (std + 1e-9) * np.sqrt(252))

    # naive trade count: count signal changes
    trade_count = int((df["signal"].diff().fillna(0) != 0).sum())

    # Curves for charts
    equity_curve = [{"time": str(t), "equity_sek": float(e * initial_sek)} for t, e in zip(df["time"], df["equity"])]
    drawdown_curve = [{"time": str(t), "dd_pct": float(d * 100.0)} for t, d in zip(df["time"], drawdown)]

    return {
        "strategy_name": f"EMA {ema_fast}/{ema_slow}",
        "symbol": symbol,
        "execution_timeframe": interval,
        "capital": {
            "initial_sek": int(initial_sek),
            "final_sek": final_sek
        },
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
            "summary": "EMA crossover baseline. Next: add RSI/ATR risk, and multi-timeframe confirmation.",
            "confidence_pct": 70
        }
    }
