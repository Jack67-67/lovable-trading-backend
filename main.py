from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Lovable Trading Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # senare kan vi låsa till din domän
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "message": "Backend is running"}

@app.get("/backtests/mock")
def backtests_mock():
    return {
        "id": "bt_mock_001",
        "strategy_id": "strat_mock_001",
        "strategy_name": "Demo EMA + RSI",
        "symbol": "BTCUSDT",
        "market": "crypto",
        "execution_timeframe": "1h",
        "higher_timeframes": ["4h", "1D"],
        "period": {
            "from": "2026-01-20T00:00:00Z",
            "to": "2026-02-20T00:00:00Z"
        },
        "capital": {"initial_sek": 10000, "final_sek": 13250, "currency": "SEK"},
        "kpi": {
            "total_return_pct": 32.5,
            "total_return_sek": 3250,
            "cagr_pct": 120.0,
            "max_drawdown_pct": -8.7,
            "max_drawdown_sek": -870,
            "sharpe": 1.8,
            "sortino": 2.3,
            "win_rate_pct": 57.0,
            "trade_count": 43
        },
        "equity_curve": [
            {"time": "2026-01-20T00:00:00Z", "equity_sek": 10000},
            {"time": "2026-01-25T00:00:00Z", "equity_sek": 10400},
            {"time": "2026-02-01T00:00:00Z", "equity_sek": 11200},
            {"time": "2026-02-10T00:00:00Z", "equity_sek": 12050},
            {"time": "2026-02-20T00:00:00Z", "equity_sek": 13250}
        ],
        "drawdown_curve": [
            {"time": "2026-01-20T00:00:00Z", "dd_pct": 0},
            {"time": "2026-01-23T00:00:00Z", "dd_pct": -8.7},
            {"time": "2026-02-20T00:00:00Z", "dd_pct": -1.0}
        ],
        "trades": [
            {
                "id": "tr_001",
                "side": "long",
                "timeframe": "1h",
                "entry_time": "2026-01-22T10:00:00Z",
                "exit_time": "2026-01-22T16:00:00Z",
                "entry_price": 40000,
                "exit_price": 41200,
                "size_units": 0.05,
                "size_sek": 2000,
                "pnl_sek": 600,
                "pnl_pct": 6.0,
                "entry_reason": ["RSI_OVERSOLD", "EMA_TREND_UP"],
                "exit_reason": ["TAKE_PROFIT"]
            }
        ],
        "ai_insights": {
            "summary": "Strategin presterade starkt i upptrender men var känslig för snabba reversaler.",
            "top_issues": [
                "Relativt hög drawdown vid plötsliga prisfall.",
                "Positionsstorlek är aggressiv vid hög volatilitet.",
                "Inga filter för sidledes marknad."
            ],
            "top_improvements": [
                "Lägg till ATR-baserad volatilitetsfilter.",
                "Sänk max positionstorlek under hög volatilitet.",
                "Filtrera bort trades när priset konsoliderar kring EMA200."
            ],
            "confidence_pct": 80
        }
    }
