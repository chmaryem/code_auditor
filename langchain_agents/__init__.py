"""
langchain_agents — Multi-Agent System using LangChain + LangGraph.

Architecture:
  - LangChain: Agents (4 pillars: LLM, Tools, Memory, Planning)
  - LangGraph: Orchestrator only (StateGraph connecting agents)

Packages:
  tools/   — @tool wrappers around existing services
  agents/  — LangChain agents with 4 pillars
  graphs/  — LangGraph StateGraphs (WatchGraph, etc.)
  memory/  — Redis-backed memory for agents
"""
