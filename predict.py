"""Interactive Binance spot forecasts using the existing Kronos model.

The application has two modes:
* an individual coin forecast with a Plotly chart; and
* a liquidity-filtered scan that ranks coins by Kronos's predicted final-close gain.

Forecasts are model estimates only. They are not trading advice or guaranteed
future prices.
"""

from __future__ import annotations

import html
import time
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import torch

from model import Kronos, KronosPredictor, KronosTokenizer


BINANCE_BASE = "https://api.binance.com"
LOOKBACK = 400
MAX_CONTEXT = 512
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_ATTEMPTS = 4
REQUEST_PACE_SECONDS = 0.10

# The individual-coin feature keeps its original three choices.
TIMEFRAME_OPTIONS = {
    "1": ("1h", "1 hour", 1),
    "2": ("4h", "4 hours", 4),
    "3": ("1d", "1 day", 24),
}

# A 4-hour interval keeps the 7-day forecast to 42 generated candles while
# retaining a 400-candle context, comfortably inside Kronos's 512 context limit.
TOP_GAINER_PERIOD_OPTIONS = {
    "1": ("1h", "1 hour", 1),
    "2": ("1h", "4 hours", 4),
    "3": ("1h", "12 hours", 12),
    "4": ("1h", "24 hours", 24),
    "5": ("1h", "48 hours", 48),
    "6": ("4h", "7 days", 42),
}

SCAN_SIZE_OPTIONS = {
    "1": 25,
    "2": 50,
    "3": 100,
    "4": None,
}

INTERVAL_FREQUENCIES = {
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}

# Binance does not expose a universal leveraged-token flag on every exchangeInfo
# payload. These standard leveraged-token suffixes are therefore excluded only
# from the scanner. Individual symbol lookup remains unchanged.
LEVERAGED_TOKEN_SUFFIXES = (
    "UP",
    "DOWN",
    "BULL",
    "BEAR",
    "2L",
    "2S",
    "3L",
    "3S",
    "5L",
    "5S",
)

ALIASES = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "eth": "ETH",
    "pepe": "PEPE",
    "dogecoin": "DOGE",
    "doge": "DOGE",
    "solana": "SOL",
    "sol": "SOL",
    "ripple": "XRP",
    "xrp": "XRP",
    "bnb": "BNB",
    "cardano": "ADA",
    "ada": "ADA",
    "shiba": "SHIB",
    "shib": "SHIB",
    "tron": "TRX",
    "trx": "TRX",
    "avalanche": "AVAX",
    "avax": "AVAX",
    "chainlink": "LINK",
    "link": "LINK",
}


def binance_get(path: str, params: dict[str, Any] | None = None) -> Any:
    """Fetch public Binance data with pacing and retry/backoff for rate limits."""
    url = f"{BINANCE_BASE}{path}"
    last_error: Exception | None = None

    for attempt in range(REQUEST_ATTEMPTS):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)

            if response.status_code in (418, 429):
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 2 ** (attempt + 1)
                except ValueError:
                    delay = 2 ** (attempt + 1)
                print(f"\nBinance rate limit reached; waiting {delay:.0f}s before retrying...")
                time.sleep(delay)
                last_error = requests.HTTPError(
                    f"Binance returned HTTP {response.status_code}."
                )
                continue

            # Transient server failures are retried; invalid symbols and other
            # normal client errors are allowed to fail immediately.
            if 500 <= response.status_code < 600:
                response.raise_for_status()

            response.raise_for_status()
            payload = response.json()

            if isinstance(payload, dict) and payload.get("code") == -1003:
                delay = 2 ** (attempt + 1)
                print(f"\nBinance rate limit reached; waiting {delay}s before retrying...")
                time.sleep(delay)
                last_error = RuntimeError(payload.get("msg", "Binance rate limit exceeded."))
                continue

            time.sleep(REQUEST_PACE_SECONDS)
            return payload

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as error:
            last_error = error
            if attempt == REQUEST_ATTEMPTS - 1:
                break
            delay = 2 ** attempt
            print(f"\nTemporary Binance connection issue; retrying in {delay}s...")
            time.sleep(delay)
        except requests.exceptions.HTTPError as error:
            last_error = error
            if error.response is None or not (500 <= error.response.status_code < 600):
                raise
            if attempt == REQUEST_ATTEMPTS - 1:
                break
            delay = 2 ** attempt
            print(f"\nBinance server error; retrying in {delay}s...")
            time.sleep(delay)

    raise RuntimeError(
        f"Binance request failed after {REQUEST_ATTEMPTS} attempts: {last_error}"
    ) from last_error


def is_leveraged_token(base: str) -> bool:
    """Return whether a base asset looks like a Binance leveraged token."""
    return base.upper().endswith(LEVERAGED_TOKEN_SUFFIXES)


def get_usdt_symbol_records() -> list[dict[str, str]]:
    """Get current active Binance USDT spot symbol metadata dynamically."""
    info = binance_get("/api/v3/exchangeInfo")
    records: list[dict[str, str]] = []

    for item in info.get("symbols", []):
        if (
            item.get("status") == "TRADING"
            and item.get("quoteAsset") == "USDT"
            and item.get("isSpotTradingAllowed", True)
        ):
            base = item.get("baseAsset", "").upper()
            symbol = item.get("symbol", "").upper()
            if base and symbol:
                records.append({"base": base, "symbol": symbol})

    return records


def get_usdt_symbols() -> dict[str, str]:
    """Return active USDT spot pairs for individual symbol resolution."""
    return {
        record["base"]: record["symbol"]
        for record in get_usdt_symbol_records()
    }


def get_24h_volume() -> dict[str, float]:
    """Return Binance's current 24-hour quote volume keyed by trading symbol."""
    tickers = binance_get("/api/v3/ticker/24hr")
    volumes: dict[str, float] = {}

    for ticker in tickers:
        symbol = str(ticker.get("symbol", "")).upper()
        try:
            volume = float(ticker.get("quoteVolume", 0.0))
        except (TypeError, ValueError):
            volume = 0.0
        if symbol:
            volumes[symbol] = max(volume, 0.0)

    return volumes


def get_liquid_usdt_symbols(scan_size: int | None) -> list[dict[str, Any]]:
    """Return current eligible USDT spot pairs ranked by 24-hour quote volume."""
    volumes = get_24h_volume()
    eligible = [
        record
        for record in get_usdt_symbol_records()
        if not is_leveraged_token(record["base"])
    ]
    ranked = sorted(
        eligible,
        key=lambda record: volumes.get(record["symbol"], 0.0),
        reverse=True,
    )
    return ranked if scan_size is None else ranked[:scan_size]


def resolve_token(user_input: str, symbols: dict[str, str]) -> tuple[str | None, str | None]:
    raw = user_input.strip().upper()
    if not raw:
        return None, None
    if raw.endswith("USDT") and raw in symbols.values():
        return raw[:-4], raw

    candidate = ALIASES.get(raw.lower(), raw)
    if candidate in symbols:
        return candidate, symbols[candidate]
    return None, None


def suggest_tokens(user_input: str, symbols: dict[str, str], limit: int = 5) -> list[str]:
    raw = user_input.strip().upper()
    candidates = list(symbols.keys())
    suggestions = get_close_matches(raw, candidates, n=limit, cutoff=0.35)

    if len(suggestions) < limit:
        for symbol in candidates:
            if raw and (raw in symbol or symbol in raw) and symbol not in suggestions:
                suggestions.append(symbol)
            if len(suggestions) >= limit:
                break
    return suggestions[:limit]


def get_binance_data(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Fetch closed OHLCV candles in the six-column format expected by Kronos."""
    data = binance_get(
        "/api/v3/klines",
        {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)},
    )
    if not data:
        raise RuntimeError(f"No market data returned for {symbol}.")

    columns = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "buy_base_volume", "buy_quote_volume", "ignore",
    ]
    df = pd.DataFrame(data, columns=columns)
    df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms")
    for column in ("open", "high", "low", "close", "volume", "quote_volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # The open kline is excluded: every input candle was fully available when
    # the model forecast was generated, with no future-candle leakage.
    now_ms = int(time.time() * 1000)
    df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce")
    df = df[df["close_time"] < now_ms].copy()
    df["amount"] = df["quote_volume"]
    return df[
        ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    ].dropna().reset_index(drop=True)


def load_kronos() -> KronosPredictor:
    """Load the tokenizer/model once, selecting CUDA only when available."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading Kronos on {device.upper()}...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    tokenizer.eval()
    model.eval()
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=MAX_CONTEXT)
    print("Kronos loaded successfully and will be reused for this session.")
    return predictor


def prepare_prediction_input(
    symbol: str,
    interval: str,
    pred_len: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Fetch a closed historical window and generate future timestamps."""
    df = get_binance_data(symbol, interval, LOOKBACK + 5)
    if len(df) < LOOKBACK:
        raise RuntimeError(
            f"Only {len(df)} closed candles were received for {symbol}; "
            f"{LOOKBACK} are required."
        )
    df = df.tail(LOOKBACK).reset_index(drop=True)
    x_timestamp = pd.Series(pd.to_datetime(df["timestamps"])).reset_index(drop=True)
    frequency = INTERVAL_FREQUENCIES[interval]
    future_timestamps = pd.Series(
        pd.date_range(
            start=x_timestamp.iloc[-1] + pd.tseries.frequencies.to_offset(frequency),
            periods=pred_len,
            freq=frequency,
        )
    )
    return df, x_timestamp, future_timestamps


def forecast_ohlcv(
    predictor: KronosPredictor,
    df: pd.DataFrame,
    x_timestamp: pd.Series,
    future_timestamps: pd.Series,
) -> pd.DataFrame:
    """Run Kronos and explicitly restore the future timestamp column.

    KronosPredictor.predict returns timestamps as its DataFrame index rather than
    as a ``timestamps`` column. Keeping this explicit assignment prevents the
    historic KeyError: 'timestamps' regression in the terminal output and chart.
    """
    x_df = df[["open", "high", "low", "close", "volume", "amount"]].copy()
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=future_timestamps,
        pred_len=len(future_timestamps),
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=False,
    )
    pred_df = pred_df.reset_index(drop=True).copy()
    # Do not remove this: Kronos may not return a timestamps column.
    pred_df["timestamps"] = future_timestamps.to_numpy()
    ordered_columns = ["timestamps"] + [
        column for column in pred_df.columns if column != "timestamps"
    ]
    return pred_df[ordered_columns]


def calculate_risk(predicted_high: float, predicted_low: float, current_price: float) -> str:
    """Classify forecast range volatility; this is not a guaranteed risk measure."""
    range_percent = (predicted_high - predicted_low) / current_price * 100
    downside_percent = (predicted_low - current_price) / current_price * 100
    if downside_percent <= -10 or range_percent >= 20:
        return "HIGH"
    if downside_percent <= -4 or range_percent >= 8:
        return "MEDIUM"
    return "LOW"


def calculate_metrics(
    base: str,
    symbol: str,
    current_price: float,
    pred_df: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate the ranking and range metrics from Kronos's projected candles."""
    predicted_close = float(pred_df["close"].iloc[-1])
    predicted_high = float(pred_df["high"].max())
    predicted_low = float(pred_df["low"].min())
    gain_percent = (predicted_close - current_price) / current_price * 100
    high_percent = (predicted_high - current_price) / current_price * 100
    downside_percent = (predicted_low - current_price) / current_price * 100
    direction = "UP" if gain_percent > 0 else "DOWN" if gain_percent < 0 else "FLAT"
    return {
        "base": base,
        "symbol": symbol,
        "current_price": current_price,
        "predicted_close": predicted_close,
        "predicted_high": predicted_high,
        "predicted_low": predicted_low,
        "gain_percent": gain_percent,
        "high_percent": high_percent,
        "downside_percent": downside_percent,
        "direction": direction,
        "risk": calculate_risk(predicted_high, predicted_low, current_price),
    }


def predict_coin(
    predictor: KronosPredictor,
    base: str,
    symbol: str,
    interval: str,
    forecast_candles: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.Series]:
    """Fetch one coin's data, forecast it, and calculate its metrics."""
    df, x_timestamp, future_timestamps = prepare_prediction_input(
        symbol, interval, forecast_candles
    )
    pred_df = forecast_ohlcv(predictor, df, x_timestamp, future_timestamps)
    current_price = float(df["close"].iloc[-1])
    metrics = calculate_metrics(base, symbol, current_price, pred_df)
    return metrics, df, pred_df, x_timestamp


def format_price(price: float) -> str:
    """Display both major and small-token prices without excessive zeroes."""
    if price >= 1000:
        return f"${price:,.2f}"
    if price >= 1:
        return f"${price:,.4f}".rstrip("0").rstrip(".")
    if price >= 0.01:
        return f"${price:,.6f}".rstrip("0").rstrip(".")
    if price >= 0.0001:
        return f"${price:,.8f}".rstrip("0").rstrip(".")
    return f"${price:,.10f}".rstrip("0").rstrip(".")


def ask_token(symbols: dict[str, str]) -> tuple[str, str]:
    while True:
        print("\n--------------------------------------")
        token = input("Enter token (e.g. BTC, ETH, PEPE): ").strip()
        base, pair = resolve_token(token, symbols)
        if pair:
            print(f"\nFound: {base} ({pair})")
            return base, pair

        print(f"\n'{token}' was not found on Binance.")
        suggestions = suggest_tokens(token, symbols)
        if not suggestions:
            continue
        print("\nDid you mean:")
        for index, suggestion in enumerate(suggestions, 1):
            print(f"  {index}. {suggestion}")
        choice = input("\nEnter suggestion number, or press Enter to search again: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(suggestions):
            selected = suggestions[int(choice) - 1]
            return selected, symbols[selected]


def ask_timeframe() -> tuple[str, str, int]:
    print("\n--------------------------------------")
    print("How far ahead do you want the prediction?")
    print("  1. 1 hour")
    print("  2. 4 hours")
    print("  3. 1 day")
    while True:
        choice = input("\nChoose 1, 2 or 3: ").strip()
        if choice in TIMEFRAME_OPTIONS:
            return TIMEFRAME_OPTIONS[choice]
        print("Please enter 1, 2 or 3.")


def ask_top_gainer_period() -> tuple[str, str, int]:
    print("\n======================================")
    print("          FUTURE TOP GAINERS")
    print("======================================")
    print("\nSelect prediction period:")
    print("  1. 1 hour")
    print("  2. 4 hours")
    print("  3. 12 hours")
    print("  4. 24 hours")
    print("  5. 48 hours")
    print("  6. 7 days")
    while True:
        choice = input("\nChoose 1-6: ").strip()
        if choice in TOP_GAINER_PERIOD_OPTIONS:
            return TOP_GAINER_PERIOD_OPTIONS[choice]
        print("Please enter a number from 1 to 6.")


def ask_scan_size() -> int | None:
    print("\nHow many coins should I scan?")
    print("  1. Top 25 liquid coins")
    print("  2. Top 50 liquid coins (default)")
    print("  3. Top 100 liquid coins")
    print("  4. All eligible coins")
    while True:
        choice = input("\nChoose 1-4 [default: 2]: ").strip() or "2"
        if choice in SCAN_SIZE_OPTIONS:
            return SCAN_SIZE_OPTIONS[choice]
        print("Please enter a number from 1 to 4.")


def run_prediction(
    base: str,
    symbol: str,
    interval: str,
    timeframe_name: str,
    hours: int,
    predictor: KronosPredictor | None = None,
) -> pd.DataFrame:
    """Keep the existing individual forecast and chart workflow intact."""
    print("\n======================================")
    print(f"          {base} FORECAST")
    print("======================================")
    print(f"\nPair:       {symbol}")
    print(f"Timeframe:  {timeframe_name}")
    print(f"Forecast:   {hours:g} hour(s)")
    print("\nGetting latest closed market data...")
    if predictor is None:
        predictor = load_kronos()

    candle_hours = {"1h": 1, "4h": 4, "1d": 24}[interval]
    forecast_candles = max(1, round(hours / candle_hours))
    print("Generating AI forecast...\n")
    metrics, df, pred_df, x_timestamp = predict_coin(
        predictor, base, symbol, interval, forecast_candles
    )
    high_time = pred_df.loc[pred_df["high"].idxmax(), "timestamps"]
    low_time = pred_df.loc[pred_df["low"].idxmin(), "timestamps"]

    print("=" * 38)
    print(f"        {base} - KRONOS FORECAST")
    print("=" * 38)
    print(f"\nCurrent Price : {format_price(metrics['current_price'])}")
    print(f"Forecast      : Next {hours:g} hour(s)")
    print("\nPREDICTED HIGH")
    print(f"  {format_price(metrics['predicted_high'])}")
    print(f"  Change: {metrics['high_percent']:+.2f}%")
    print(f"  Around: {high_time}")
    print("\nPREDICTED LOW")
    print(f"  {format_price(metrics['predicted_low'])}")
    print(f"  Change: {metrics['downside_percent']:+.2f}%")
    print(f"  Around: {low_time}")
    print("\nPREDICTED CLOSE")
    print(f"  {format_price(metrics['predicted_close'])}")
    print(f"  Change: {metrics['gain_percent']:+.2f}%")
    print("\nMODEL DIRECTION")
    print(f"  {metrics['direction']}")
    print("\nForecast generated by Kronos. It is not a guaranteed future price.")
    print("=" * 38)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x_timestamp, open=df["open"], high=df["high"], low=df["low"],
        close=df["close"], name="Historical",
    ))
    fig.add_trace(go.Candlestick(
        x=pred_df["timestamps"], open=pred_df["open"], high=pred_df["high"],
        low=pred_df["low"], close=pred_df["close"], name="Kronos Forecast",
    ))
    fig.update_layout(
        title=f"{base} - Kronos {timeframe_name} Forecast",
        xaxis_title="Time", yaxis_title="Price", xaxis_rangeslider_visible=False,
        template="plotly_dark",
    )
    output_file = Path(f"kronos_{base.lower()}_forecast.html")
    fig.write_html(str(output_file))
    print(f"\nChart saved: {output_file}")
    return pred_df


def progress_bar(completed: int, total: int, width: int = 20) -> str:
    filled = int(width * completed / total) if total else width
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def scan_top_gainers(
    predictor: KronosPredictor,
    coins: list[dict[str, Any]],
    interval: str,
    forecast_candles: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Forecast selected liquid pairs sequentially with one predictor instance."""
    results: list[dict[str, Any]] = []
    skipped: list[str] = []
    total = len(coins)
    print("\nScanning Binance market and generating Kronos forecasts...")

    for index, coin in enumerate(coins, 1):
        base = str(coin["base"])
        symbol = str(coin["symbol"])
        print(
            f"\r{progress_bar(index - 1, total)} {index - 1}/{total}  Current coin: {base:<12}",
            end="", flush=True,
        )
        try:
            metrics, _, _, _ = predict_coin(
                predictor, base, symbol, interval, forecast_candles
            )
            results.append(metrics)
        except Exception as error:  # One failure must never terminate a scan.
            skipped.append(base)
            print(f"\r{progress_bar(index, total)} {index}/{total}  {base} skipped: {type(error).__name__}")
        print(
            f"\r{progress_bar(index, total)} {index}/{total}  Current coin: {base:<12}",
            end="", flush=True,
        )

    print("\nScan complete.")
    return results, skipped


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank successful forecasts by final-close predicted percentage gain."""
    ranked = sorted(results, key=lambda result: result["gain_percent"], reverse=True)
    for rank, result in enumerate(ranked, 1):
        result["rank"] = rank
    return ranked


def save_results(results: list[dict[str, Any]]) -> tuple[Path, Path]:
    """Save all successful ranked forecasts in stable and timestamped CSV files."""
    csv_rows = [
        {
            "Rank": result["rank"],
            "Coin": result["base"],
            "Symbol": result["symbol"],
            "Current Price": result["current_price"],
            "Predicted Close": result["predicted_close"],
            "Predicted High": result["predicted_high"],
            "Predicted Low": result["predicted_low"],
            "Gain %": result["gain_percent"],
            "High %": result["high_percent"],
            "Downside %": result["downside_percent"],
            "Direction": result["direction"],
            "Risk": result["risk"],
        }
        for result in results
    ]
    result_df = pd.DataFrame(csv_rows)
    latest_file = Path("kronos_top_gainers.csv")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_file = Path(f"kronos_top_gainers_{timestamp}.csv")
    result_df.to_csv(latest_file, index=False)
    result_df.to_csv(timestamped_file, index=False)
    return latest_file, timestamped_file


def generate_html_report(
    results: list[dict[str, Any]],
    forecast_name: str,
    requested_scan_size: int | None,
    successful_count: int,
    skipped_count: int,
) -> Path:
    """Create a compact, readable HTML report for only the top ten rankings."""
    displayed_size = (
        "All eligible coins" if requested_scan_size is None
        else f"Top {requested_scan_size} liquid coins"
    )
    rows = "\n".join(
        "<tr>"
        f"<td>{result['rank']}</td>"
        f"<td><strong>{html.escape(str(result['base']))}</strong></td>"
        f"<td>{format_price(result['current_price'])}</td>"
        f"<td>{format_price(result['predicted_close'])}</td>"
        f"<td class=\"{'positive' if result['gain_percent'] >= 0 else 'negative'}\">{result['gain_percent']:+.2f}%</td>"
        f"<td>{format_price(result['predicted_high'])}</td>"
        f"<td>{html.escape(str(result['risk']).title())}</td>"
        "</tr>"
        for result in results[:10]
    )
    if not rows:
        rows = "<tr><td colspan=\"7\">No successful forecasts were available.</td></tr>"

    report = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Kronos Future Top Gainers</title>
  <style>
    body {{ background: #10151d; color: #edf2f7; font-family: Arial, sans-serif; margin: 0; padding: 32px; }}
    main {{ max-width: 1050px; margin: auto; }} h1 {{ margin-bottom: 8px; }}
    .meta, footer {{ color: #b8c3d1; line-height: 1.6; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 24px; background: #18212d; }}
    th, td {{ padding: 13px 12px; text-align: left; border-bottom: 1px solid #2a3746; }}
    th {{ background: #223044; color: #ffffff; }}
    .positive {{ color: #59d98e; font-weight: bold; }} .negative {{ color: #ff8383; font-weight: bold; }}
    footer {{ margin-top: 24px; font-size: 0.92rem; }}
  </style>
</head>
<body><main>
  <h1>Kronos Future Top Gainers</h1>
  <div class=\"meta\">Forecast period: Next {html.escape(forecast_name)}<br>Scan size: {html.escape(displayed_size)}<br>Successful forecasts: {successful_count} | Skipped: {skipped_count}</div>
  <table><thead><tr><th>Rank</th><th>Coin</th><th>Current price</th><th>Predicted price</th><th>Predicted gain</th><th>Predicted high</th><th>Risk</th></tr></thead><tbody>{rows}</tbody></table>
  <footer>Top gainers are ranked by Kronos's predicted percentage change. Actual market performance may differ significantly. These are model-generated forecasts, not guaranteed future prices. Risk labels reflect the model-predicted range only and are not guaranteed risk measurements.</footer>
</main></body>
</html>"""
    output_file = Path("kronos_top_gainers.html")
    output_file.write_text(report, encoding="utf-8")
    return output_file


def print_top_gainers(
    results: list[dict[str, Any]],
    forecast_name: str,
    successful_count: int,
    skipped_count: int,
) -> None:
    print("\n" + "=" * 42)
    print("       KRONOS FUTURE TOP GAINERS")
    print("=" * 42)
    print(f"Forecast: Next {forecast_name}")
    print(f"Coins scanned successfully: {successful_count}")
    print(f"Skipped: {skipped_count}")
    for result in results[:10]:
        print(f"\n#{result['rank']}  {result['base']}")
        print(f"Current:   {format_price(result['current_price'])}")
        print(f"Expected:  {format_price(result['predicted_close'])}")
        print(f"Gain:      {result['gain_percent']:+.2f}%")
        print(f"High:      {format_price(result['predicted_high'])}")
        print(f"Risk:      {result['risk'].title()}")
    print("\nTop gainers are ranked by Kronos's predicted percentage change.")
    print("Actual market performance may differ significantly.")
    print("These are model-generated forecasts, not guaranteed future prices.")
    print("=" * 42)


def main_menu() -> str:
    print("\n======================================")
    print("       KRONOS AI CRYPTO PREDICTOR")
    print("======================================")
    print("1. Predict a coin")
    print("2. Find future top gainers")
    print("3. Exit")
    while True:
        choice = input("\nChoose: ").strip()
        if choice in {"1", "2", "3"}:
            return choice
        print("Please enter 1, 2 or 3.")


def handle_individual_prediction(predictor: KronosPredictor | None) -> KronosPredictor:
    print("\nLoading current Binance coin list...")
    symbols = get_usdt_symbols()
    print(f"Found {len(symbols):,} active USDT spot coins.")
    base, symbol = ask_token(symbols)
    interval, timeframe_name, hours = ask_timeframe()
    if predictor is None:
        predictor = load_kronos()
    run_prediction(base, symbol, interval, timeframe_name, hours, predictor)
    return predictor


def handle_top_gainer_scan(predictor: KronosPredictor | None) -> KronosPredictor | None:
    interval, forecast_name, forecast_candles = ask_top_gainer_period()
    scan_size = ask_scan_size()
    print("\nLoading current Binance spot volumes and eligible USDT pairs...")
    coins = get_liquid_usdt_symbols(scan_size)
    if not coins:
        raise RuntimeError("No eligible Binance USDT spot pairs were found.")
    print(f"Selected {len(coins):,} liquid USDT spot coins.")
    if predictor is None:
        predictor = load_kronos()

    results, skipped = scan_top_gainers(predictor, coins, interval, forecast_candles)
    ranked_results = rank_results(results)
    if not ranked_results:
        print("\nNo coin forecasts completed, so no report was generated.")
        return predictor

    print_top_gainers(ranked_results, forecast_name, len(results), len(skipped))
    latest_csv, timestamped_csv = save_results(ranked_results)
    report_file = generate_html_report(
        ranked_results, forecast_name, scan_size, len(results), len(skipped)
    )
    print(f"\nSaved CSV: {latest_csv}")
    print(f"Saved timestamped CSV: {timestamped_csv}")
    print(f"Saved HTML report: {report_file}")
    return predictor


def main() -> None:
    predictor: KronosPredictor | None = None
    while True:
        choice = main_menu()
        if choice == "3":
            print("\nGoodbye.")
            return
        try:
            if choice == "1":
                predictor = handle_individual_prediction(predictor)
            else:
                predictor = handle_top_gainer_scan(predictor)
        except requests.exceptions.RequestException as error:
            print("\nNetwork/Binance error:")
            print(error)
        except Exception as error:
            print("\nError:")
            print(type(error).__name__, "-", error)
            print("The application is still running. You can choose another option.")


if __name__ == "__main__":
    main()
