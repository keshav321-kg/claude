#!/usr/bin/env python3
"""claude-cli: a small command-line agent with financial data and AI chat commands."""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

FMP_BASE_URL = "https://financialmodelingprep.com/stable"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-5"


def _http_get_json(url):
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode())


def _require_env(name):
    value = os.environ.get(name)
    if not value:
        print(f"error: {name} environment variable is not set", file=sys.stderr)
        sys.exit(1)
    return value


def cmd_quote(args):
    api_key = _require_env("FMP_API_KEY")
    url = f"{FMP_BASE_URL}/quote?symbol={urllib.parse.quote(args.ticker)}&apikey={api_key}"
    data = _http_get_json(url)
    if not data:
        print(f"no quote data found for {args.ticker}", file=sys.stderr)
        sys.exit(1)
    quote = data[0]
    print(f"{quote.get('symbol')} ({quote.get('name')})")
    print(f"  price:  {quote.get('price')}")
    print(f"  change: {quote.get('change')} ({quote.get('changePercentage')}%)")
    print(f"  volume: {quote.get('volume')}")


def cmd_news(args):
    api_key = _require_env("FMP_API_KEY")
    url = (
        f"{FMP_BASE_URL}/news/stock?symbols={urllib.parse.quote(args.ticker)}"
        f"&limit={args.limit}&apikey={api_key}"
    )
    data = _http_get_json(url)
    if not data:
        print(f"no news found for {args.ticker}", file=sys.stderr)
        sys.exit(1)
    for item in data:
        print(f"- [{item.get('publishedDate')}] {item.get('title')}")
        print(f"  {item.get('url')}")


def cmd_ask(args):
    api_key = _require_env("ANTHROPIC_API_KEY")
    payload = json.dumps(
        {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": args.prompt}],
        }
    ).encode()
    request = urllib.request.Request(
        ANTHROPIC_URL,
        data=payload,
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(request) as resp:
        result = json.loads(resp.read().decode())
    for block in result.get("content", []):
        if block.get("type") == "text":
            print(block["text"])


def build_parser():
    parser = argparse.ArgumentParser(prog="claude-cli", description="A small CLI agent.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    quote_parser = subparsers.add_parser("quote", help="Fetch a stock quote")
    quote_parser.add_argument("ticker", help="Stock ticker symbol, e.g. AAPL")
    quote_parser.set_defaults(func=cmd_quote)

    news_parser = subparsers.add_parser("news", help="Fetch recent news for a ticker")
    news_parser.add_argument("ticker", help="Stock ticker symbol, e.g. AAPL")
    news_parser.add_argument("--limit", type=int, default=5, help="Number of articles to show")
    news_parser.set_defaults(func=cmd_news)

    ask_parser = subparsers.add_parser("ask", help="Ask an AI agent a question")
    ask_parser.add_argument("prompt", help="The question or prompt to send")
    ask_parser.set_defaults(func=cmd_ask)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
