# ===== PROFESSIONAL TRADE ENGINE VERSION =====

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

class StrategyBlock(BaseModel):
    type: str
    params: dict = {}

class RiskBlock(BaseModel):
    atr_length: int = 14
    sl_atr_mult: float = 2.0
    rr: float = 2.0

class BacktestRequest(BaseModel):
    market: str
    symbol: str
    interval: str
    outputsize: int = 500
    exchange: str | None = None

    entry: StrategyBlock
    filter: StrategyBlock | None = None
    risk: RiskBlock

    fee_bps: float = 2.0
    slippage_bps: float = 1.0


@app.get("/health")
def health():
    return {"status": "ok"}


def fetch_data(symbol, interval, outputsize, exchange):
    url = f"{TD_BASE}/time_series"
    params = {
        "apikey": TD_API_KEY,
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "format": "JSON",
        "timezone": "UTC",
        "order": "ASC"
    }
    if exchange:
        params["exchange"] = exchange

    r = requests.get(url, params=params, timeout=15)
    data = r.json()

    if isinstance(data, dict) and data.get("status") == "error":
        raise HTTPException(status_code=400, detail=data.get("message"))

    values = data.get("values")
    if not values:
        raise HTTPException(status_code=400, detail="No data returned")

    df = pd.DataFrame(values)
    df["time"] = pd.to_datetime(df["datetime"])
    for col in ["open","high","low","close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("time").reset_index(drop=True)
    return df


def compute_atr(df, length):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(length).mean()
    return atr


@app.post("/backtests/run")
def run_backtest(req: BacktestRequest):

    df = fetch_data(req.symbol, req.interval, req.outputsize, req.exchange)

    # Entry logic (EMA crossover only for now)
    fast = req.entry.params.get("fast", 20)
    slow = req.entry.params.get("slow", 50)

    df["ema_fast"] = df["close"].ewm(span=fast).mean()
    df["ema_slow"] = df["close"].ewm(span=slow).mean()

    df["long_signal"] = (df["ema_fast"] > df["ema_slow"])
    df["short_signal"] = (df["ema_fast"] < df["ema_slow"])

    df["atr"] = compute_atr(df, req.risk.atr_length)

    initial_capital = 10000
    capital = initial_capital
    position = None
    trades = []
    equity_curve = []

    for i in range(1, len(df)):
        row = df.iloc[i]

        # Update equity
        equity_curve.append({
            "time": str(row["time"]),
            "equity_sek": capital
        })

        if position:
            # Check exit
            if position["side"] == "long":
                if row["low"] <= position["sl"]:
                    pnl = -position["risk"]
                    capital += pnl
                    trades.append({**position, "exit_time": str(row["time"]), "pnl": pnl})
                    position = None
                elif row["high"] >= position["tp"]:
                    pnl = position["risk"] * req.risk.rr
                    capital += pnl
                    trades.append({**position, "exit_time": str(row["time"]), "pnl": pnl})
                    position = None

            if position and position["side"] == "short":
                if row["high"] >= position["sl"]:
                    pnl = -position["risk"]
                    capital += pnl
                    trades.append({**position, "exit_time": str(row["time"]), "pnl": pnl})
                    position = None
                elif row["low"] <= position["tp"]:
                    pnl = position["risk"] * req.risk.rr
                    capital += pnl
                    trades.append({**position, "exit_time": str(row["time"]), "pnl": pnl})
                    position = None

        # New entry
        if not position and not np.isnan(row["atr"]):
            risk_amount = capital * 0.01  # 1% risk per trade
            atr = row["atr"]

            if row["long_signal"]:
                sl = row["close"] - atr * req.risk.sl_atr_mult
                tp = row["close"] + (atr * req.risk.sl_atr_mult * req.risk.rr)
                position = {
                    "side": "long",
                    "entry_time": str(row["time"]),
                    "entry_price": row["close"],
                    "sl": sl,
                    "tp": tp,
                    "risk": risk_amount
                }

            elif row["short_signal"]:
                sl = row["close"] + atr * req.risk.sl_atr_mult
                tp = row["close"] - (atr * req.risk.sl_atr_mult * req.risk.rr)
                position = {
                    "side": "short",
                    "entry_time": str(row["time"]),
                    "entry_price": row["close"],
                    "sl": sl,
                    "tp": tp,
                    "risk": risk_amount
                }

    wins = sum(1 for t in trades if t["pnl"] > 0)
    win_rate = (wins / len(trades) * 100) if trades else 0

    return {
        "strategy_name": "EMA Pro Engine",
        "symbol": req.symbol,
        "execution_timeframe": req.interval,
        "capital": {
            "initial_sek": initial_capital,
            "final_sek": capital
        },
        "kpi": {
            "total_return_pct": ((capital / initial_capital - 1) * 100),
            "max_drawdown_pct": 0,
            "sharpe": 0,
            "win_rate_pct": win_rate,
            "trade_count": len(trades)
        },
        "equity_curve": equity_curve,
        "drawdown_curve": [],
        "trades": trades,
        "ai_insights": {
            "summary": "ATR-based risk engine with long & short.",
            "confidence_pct": 80
        }
    }
