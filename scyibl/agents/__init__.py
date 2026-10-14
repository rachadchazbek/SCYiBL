"""Agents: router, classifier, adjudicator and safety (issues #6, #7, #8).

Agents are plain async functions over :class:`scyibl.llm.LLMClient`; the LangGraph graph in
:mod:`scyibl.graph` wires them together (ADR-001).
"""
