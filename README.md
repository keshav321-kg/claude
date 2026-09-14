# claude

A small CLI agent with financial data and AI chat commands.

## Setup

Requires Python 3.8+. No third-party dependencies.

- `FMP_API_KEY` — API key from [Financial Modeling Prep](https://financialmodelingprep.com/), required for `quote` and `news`.
- `ANTHROPIC_API_KEY` — API key from [Anthropic](https://console.anthropic.com/), required for `ask`.

## Usage

```
python3 cli.py quote AAPL
python3 cli.py news AAPL --limit 5
python3 cli.py ask "What is the capital of France?"
```
