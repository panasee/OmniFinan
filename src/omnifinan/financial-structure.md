# Data Structure
State:
```json
{
    "messages": [
        HumanMessage(
            content="Make trading decisions based on the provided data.",
            name="sentiment, etc."
        )
    ],
    "data": {
        "tickers": tickers,
        "portfolio": portfolio,
        "start_date": start_date,
        "end_date": end_date,
        "analyst_signals": {},
    },
    "metadata": {
        "show_reasoning": show_reasoning,
        "model_name": model_name,
        "provider_api": provider_api,
    },
},
```
progress:
progress.agent_status
```json
{
    agent_name: {
        "status": "status",
        "ticker": "ticker",
    },
    "ticker": "ticker",
    "status": "status",
}
```