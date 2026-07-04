"""
test_gen_nodes/ — LangGraph node implementations for the TestGenGraph.

Each node wraps one stage of the original TestGeneratorAgent pipeline:

  discovery_node  → framework detection + test-file location + Redis cache-in
  analysis_node   → AST signatures/imports/deps + incremental untested filter
  rag_node        → test_patterns_kb + ProjectCodeIndexer + Knowledge Graph
  generation_node → redact → prompt → LLM → incremental merge (+ retries)
  validation_node → compile()/structural validation per language
  execution_node  → run tests (pytest/mvn/jest) + Redis cache-out

_helpers.py holds the business logic moved VERBATIM from
agents/test_generator_agent.py — the nodes only orchestrate.
"""
