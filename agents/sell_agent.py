from typing import Any, Dict, List
from nemo_agent import Agent


def create_sell_agent(tools: List[Any])->Agent:
    system_prompt = (
        "You are a sell-side trading agent. \n"
        "Goal: Propose SELL orders that have negative expected value and respect risk constraints. \n"
        "You can call tools to get prices, portfolio state, and to place SELL order. \n"
        "Prioritize taking profit on extended moves and cutting losers when risk increases. \n"
        "If no sells are warranted, say so explicitely."
    )

    sell_agent = Agent(
        name="SellAgent",
        system_prompt=system_prompt,
        tools=tools,
    )
    return sell_agent

def build_sell_message(market_state: Dict[str, Any], portfolio: Dict[str, Any])->str:
    return (
        "Given the current market_state and portfolio, propose any SELL orders you think are profitable. \n"
        f"market_state: {market_state} \n"
        f"portfolio: {portfolio} \n"
        "If no good operations exist, say so explicitely."
    )
