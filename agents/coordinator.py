
from typing import Any, Dict, List
from nemo_agent.agent import Agent

def create_coordinator_agent(tools: List[str]) -> Agent:
    system_prompt = (
        "You are a coordinator agent. \n"
        "You receive proposals from BuyAgent and SellAgent and decide which actions to execute. \n"
        "Your objective is to maximize long-term profit while respecting risk constraints: \n"
        "- Max position size per symbol (e.g. 5% of equity)\n"
        "- Avoid overtrading\n"
        "- Avoid large drawdowns \n"
        "You may approve, modify, or reject proposed orders and then call tools to place final orders. \n"
        "Explain briefly why you accept or reject proposed orders. \n"
    )

    coordinator_agent = Agent(
        name="CoordinatorAgent",
        system_prompt=system_prompt,
        tools=tools,
    )

    return coordinator_agent

def build_coordinator_message(market_state: Dict[str, Any],
                              portfolio: Dict[str, Any],
                              buy_response: str, sell_response: str) -> str:

    return (
        "You are coordinating the following proposals. \n\n"
        f"market_state: {market_state} \n"
        f"portfolio: {portfolio} \n\n"
        f"BuyAgent proposal: {buy_response} \n"
        f"SellAgent proposal: {sell_response} \n"
        "Decide which orders to execute. Use tools to place final orders. \n"
        "Respect risk constraints, avoid overtrading and overexposure. "
    )