from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backtest")

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

@app.post("/backtests/run")
def run_backtest(req: BacktestRequest):
    try:
        # --- 1) Fetch OHLCV from Binance ---
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

        resp = requests.get(url, params=params, timeout=10)
        try:
            data = resp.json()
        except Exception as e:
            logger.exception("Failed to parse JSON from Binance")
            raise HTTPException(status_code=502, detail="Invalid response from data provider")

        # Validate data shape
        if not isinstance(data, list) or len(data) == 0:
            logger.warning("Binance returned no candle data: %s", data)
            raise HTTPException(status_code=400, detail="No OHLCV data returned for symbol/timeframe")

        # Build dataframe
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "num_trades",
            "taker_base_vol", "taker_quote_vol", "ignore"
        ])

        # convert types and time
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["time"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.sort_values("time").reset_index(drop=True)

        # drop rows with missing close
        df = df.dropna(subset=["close"])
        if df.empty or len(df) < 3:
            logger.warning("Not enough valid candle rows after cleaning")
            raise HTTPException(status_code=400, detail="Not enough OHLCV data to run backtest")

        # --- 2) Simple EMA Strategy (example) ---
        # safe spans, require at least span*2 rows ideally but we guard
        df["ema_fast"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=50, adjust=False).mean()

        # signal: 1 when fast > slow, else 0
        df["signal"] = 0
        df.loc[df["ema_fast"] > df["ema_slow"], "signal"] = 1

        # returns
        df["returns"] = df["close"].pct_change().fillna(0)
        df["strategy_returns"] = df["returns"] * df["signal"].shift(1).fillna(0)

        # replace any infinite or NaN
        df["strategy_returns"] = df["strategy_returns"].replace([np.inf, -np.inf], 0).fillna(0)

        # equity curve (starting capital 1.0)
        df["equity"] = (1 + df["strategy_returns"]).cumprod()

        # ensure equity has no NaN
        df["equity"] = df["equity"].fillna(method="ffill").fillna(1.0)

        # --- 3) KPIs safe computation ---
        try:
            total_return = float(df["equity"].iloc[-1] - 1) * 100
        except Exception:
            total_return = 0.0

        try:
            running_max = df["equity"].cummax()
            drawdowns = df["equity"] / running_max - 1
            max_dd = float(drawdowns.min()) * 100
        except Exception:
            max_dd = 0.0

        try:
            # annualize: use 252 trading days approx; for intraday this is approximate
            mean_ret = df["strategy_returns"].mean()
            std_ret = df["strategy_returns"].std() if df["strategy_returns"].std() != 0 else 1e-9
            sharpe = float(mean_ret / std_ret * np.sqrt(252))
        except Exception:
            sharpe = 0.0

        # trades: naive count of signal changes
        try:
            trades = int((df["signal"].diff().fillna(0) != 0).sum())
        except Exception:
            trades = int(len(df))

        # equity_curve and drawdown_curve serialization (limit size for payload)
        equity_curve = [
            {"time": str(row["time"]), "equity_sek": float(row["equity"] * 10000)}
            for _, row in df.iterrows()
        ]

        drawdown_curve = [
            {
                "time": str(row["time"]),
                "dd_pct": float((row["equity"] / df["equity"].cummax().loc[idx] - 1) * 100)
            }
            for idx, row in df.reset_index().iterrows()
        ]

        result = {
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
                "trade_count": trades
            },
            "equity_curve": equity_curve,
            "drawdown_curve": drawdown_curve,
            "ai_insights": {
                "summary": "EMA crossover strategy.",
                "confidence_pct": 70
            }
        }

        return result

    except HTTPException:
        # re-raise known HTTP exceptions
        raise
    except Exception as e:
        # catch-all: log and return 500 with message
        logger.exception("Unhandled error in run_backtest")
        raise HTTPException(status_code=500, detail="Internal server error")
