# Code Auditor AI - Functionalities Summary

Based on the analysis of the `code_auditor` project structure and source code, here is a comprehensive summary of the system's functionalities.

## 1. Core Capabilities
**Code Auditor AI** is an intelligent code analysis tool that utilizes Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and Abstract Syntax Tree (AST) parsing to evaluate, audit, and provide feedback on source code.

* **Supported Languages**: Python, JavaScript, TypeScript, and Java (via `tree-sitter`).
* **LLM & RAG Integration**: Uses models like Google's Gemini alongside embeddings (Jina AI) and a Vector Store (ChromaDB) to ground analysis in code-specific context and established best practices.

## 2. Command-Line Interface (CLI) Features
The system provides a robust CLI (`main.py`) with several powerful commands:

* **Single File Analysis (`python main.py file <path>`)**
  * Parses a specific file and evaluates it using the LLM & RAG pipeline.
  * Outputs detailed analysis and cites the best practices referenced during the evaluation.

* **Full Project Analysis (`python main.py project <path>`)**
  * Performs an architectural breakdown of an entire directory.
  * **Architecture mapping**: Identifies application entry points, circular dependencies, and orphaned modules.
  * **Conflict detection**: Finds potential conflicts that might arise during refactoring.
  * **Reporting**: Generates a refactoring plan and provides metrics on project complexity (e.g., file criticality scores, dependency counts).

* **Real-Time Surveillance (`python main.py watch <path>`)**
  * Watches the file system for changes using `watchdog`.
  * Automatically triggers incremental code analysis on modified files.
  * Optionally runs a background "Smart Git Session Tracker" to monitor accumulated bugs during active development.

## 3. Smart Git Integration (`smart_git/`)
A significant portion of the system is dedicated to deeply integrating with Git workflows to catch issues before they are permanently committed:

* **Session Auditor (`git-status`)**: Tracks "uncommitted bugs" in real-time. It accumulates a session score based on ongoing modifications, warning the developer if they are hoarding too many issues without resolving them.
* **Pre-Merge Branch Analyzer (`git-branch <branch> --base <main>`)**: Analyzes an entire feature branch against its base (e.g., `main`). It provides a detailed report and a "merge verdict" to determine if the code is safe to merge. Can output the report as JSON.
* **Commit Analyzer (`git --commit <hash>`)**: Analyzes specifically the files modified in a given commit to ensure quality.
* **Pre-commit Hook Management (`hook / hook --uninstall`)**: Easily installs or uninstalls a Git pre-commit hook that uses the auditor to block problematic commits (can be run in strict or non-strict warning mode).

## 4. Multi-Agent Architecture (`agents/`)
The AI engine is divided into specialized agents working together:
* **Analysis Agent**: Focuses on finding flaws, security issues, and bad practices.
* **Code Agent**: Probably dedicated to understanding code syntax, variables, and structure.
* **Retriever Agent**: Interfaces with the Vector Store to fetch relevant documentation, previous bug fixes, or architecture rules.
* **Learning Agent**: Continuously refines the internal system knowledge base from feedback.

## 5. Caching and Knowledge Management
* **Aggressive Caching**: Uses a local SQLite cache database (`cache_service.py`) to store previous analyses and avoid hitting the LLM API repeatedly for unchanged files.
* **Knowledge Graph**: Maintains a graph of the codebase (`knowledge_graph.py`, `graph_service.py`) to understand relationships between modules, functions, and classes across the whole software architecture.
