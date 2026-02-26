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
        "equity_curve": [],
        "drawdown_curve": [],
        "ai_insights": {
            "summary": "Mock data.",
            "confidence_pct": 60
        }
    }


@app.post("/backtests/run")
def run_backtest(req: BacktestRequest):

    try:
        # Fetch OHLCV
        end_time = int(datetime.utcnow().timestamp() * 1000)
        start_time = int((datetime.utcnow() - timedelta(days=req.days)).timestamp() * 1000)

        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": req.symbol,
            "interval": req.timeframe,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        }

       response = requests.get(
    url,
    params=params,
    timeout=10,
    headers={"User-Agent": "lovable-trading/1.0"},
)

# Om Binance svarar med felstatus, ge tillbaka texten
if response.status_code != 200:
    raise HTTPException(status_code=502, detail=f"Binance HTTP {response.status_code}: {response.text}")

data = response.json()

# Binance fel kommer ofta som dict: {"code": ..., "msg": "..."}
if isinstance(data, dict):
    msg = data.get("msg") or str(data)
    raise HTTPException(status_code=400, detail=f"Binance error: {msg}")

# Tom lista = inga candles
if not isinstance(data, list) or len(data) == 0:
    raise HTTPException(status_code=400, detail="No OHLCV candles returned (check symbol/timeframe)")

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "num_trades",
            "taker_base_vol", "taker_quote_vol", "ignore"
        ])

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["time"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.dropna(subset=["close"])

        if df.empty:
            raise HTTPException(status_code=400, detail="Empty dataset")

        # EMA Strategy
        df["ema_fast"] = df["close"].ewm(span=20).mean()
        df["ema_slow"] = df["close"].ewm(span=50).mean()

        df["signal"] = 0
        df.loc[df["ema_fast"] > df["ema_slow"], "signal"] = 1

        df["returns"] = df["close"].pct_change().fillna(0)
        df["strategy_returns"] = df["returns"] * df["signal"].shift(1).fillna(0)

        df["equity"] = (1 + df["strategy_returns"]).cumprod()

        total_return = float(df["equity"].iloc[-1] - 1) * 100

        running_max = df["equity"].cummax()
        drawdowns = df["equity"] / running_max - 1
        max_dd = float(drawdowns.min()) * 100

        sharpe = float(
            df["strategy_returns"].mean() /
            (df["strategy_returns"].std() + 1e-9) * np.sqrt(252)
        )

        equity_curve = [
            {"time": str(row["time"]), "equity_sek": float(row["equity"] * 10000)}
            for _, row in df.iterrows()
        ]

        drawdown_curve = [
            {
                "time": str(row["time"]),
                "dd_pct": float((row["equity"] / df["equity"].cummax().loc[row.name] - 1) * 100)
            }
            for _, row in df.iterrows()
        ]

        return {
            "strategy_name": "EMA 20/50",
            "symbol": req.symbol,
            "execution_timeframe": req.timeframe,
            "capital": {
                "initial_sek": 10000,
                "final_sek": float(df["equity"].iloc[-1] * 10000)
            },
            "kpi": {
                "total_return_pct": total_return,
                "max_drawdown_pct": max_dd,
                "sharpe": sharpe,
                "win_rate_pct": 50,
                "trade_count": int(len(df))
            },
            "equity_curve": equity_curve,
            "drawdown_curve": drawdown_curve,
            "ai_insights": {
                "summary": "EMA crossover strategy.",
                "confidence_pct": 70
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
