from typing import Any, Dict, List
from nemo_agent import Agent

def create_buy_agent(tools: List[Any])->Agent:
    system_prompt = (
        "You are a buy-side trading agent. \n"
        "Goal: Propose BUY orders that have positive expected value and respect risk constraints. \n"
        "You can call tools to get prices, portfolio state, and to place BUY order. \n"
        "Only buy when there is a clear rationals (trend, momentum, valuation, etc.). \n"
        "Respoond concisely and use tools instead of guessing."
    )

    buy_agent = Agent(
        name="BuyAgent",
        system_prompt=system_prompt,
        tools=tools,
    )
    return buy_agent

def build_buy_message(market_state: Dict[str, Any], portfolio: Dict[str, Any])->str:
    return (
        "Given the current market_state and porfolio, propose any BUY orders you think are profitable. \n"
        f"market_state: {market_state} \n"
        f"portfolio: {portfolio} \n"
        "If no good opportunities exist, say so explicitely."
    )



