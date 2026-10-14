"""LangGraph orchestration (ADR-001, issue #18).

Import :mod:`scyibl.graph.build` explicitly; this package does not import LangGraph eagerly so
that agents and config stay usable without it.
"""
