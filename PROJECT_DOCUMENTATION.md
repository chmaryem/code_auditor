# Code Auditor AI — Project Documentation
### Final Year Project Reference Document

**Author:** Maryem Chalghoumi  
**Academic Year:** 2025–2026  
**Project Type:** AI-Powered Developer Assistant — Full-Stack Multi-Agent Platform

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [Objectives](#3-objectives)
4. [System Architecture](#4-system-architecture)
5. [Backend — FastAPI Server](#5-backend--fastapi-server)
6. [Core AI Module — WatchGraph](#6-core-ai-module--watchgraph)
7. [Smart Git Module — SmartGitGraph](#7-smart-git-module--smartgitgraph)
8. [CI/CD Intelligence — CIGraph](#8-cicd-intelligence--cigraph)
9. [Chat Module — ChatGraph](#9-chat-module--chatgraph)
10. [AI/ML Components](#10-aiml-components)
11. [Frontend — VS Code Extension](#11-frontend--vs-code-extension)
12. [Frontend — React Web Dashboard](#12-frontend--react-web-dashboard)
13. [API Reference](#13-api-reference)
14. [Data & Event Flow](#14-data--event-flow)
15. [Technology Stack](#15-technology-stack)
16. [Comparison with Existing Tools](#16-comparison-with-existing-tools)
17. [Performance Considerations](#17-performance-considerations)
18. [Project File Structure](#18-project-file-structure)
19. [Environment Configuration](#19-environment-configuration)
20. [Key Design Decisions](#20-key-design-decisions)
21. [Limitations & Future Work](#21-limitations--future-work)

---

## 1. Project Overview

**Code Auditor AI** is an intelligent developer assistant platform that covers the **entire software development lifecycle** in real time. Unlike existing tools that address isolated tasks (e.g., autocomplete or linting), Code Auditor AI integrates four interconnected modules into a single cohesive system:

| Module | What it does |
|---|---|
| **Watch Mode** | Continuously monitors file changes, analyzes code with a 14-node AI pipeline, and displays inline results in the editor |
| **Smart Git** | Multi-agent analysis of commit readiness, branch risk, diff review, PR preparation, and conflict resolution |
| **Test Generation** | Automatically detects test coverage gaps and generates unit tests contextual to the current file |
| **CI/CD Intelligence** | Monitors GitHub Actions pipeline runs, classifies failures, finds similar past fixes via RAG, generates automatic remediation |

The platform is delivered as two client interfaces:
- A **VS Code Extension** (`plugin_code_auditor`) for local, real-time in-editor analysis
- A **React Web Dashboard** (`webview-ui`) for project-wide visibility, CI/CD monitoring, and AI chat

The backend is a **Python FastAPI server** (`code_auditor`) orchestrated by **LangGraph StateGraphs** and powered by **LangChain agents**, **ChromaDB RAG**, and **Redis semantic memory**.

---

## 2. Problem Statement & Motivation

Modern developers use many separate tools during a workday:
- A linter for syntax and style errors
- A code review tool before merging
- A CI/CD dashboard for pipeline monitoring
- A test framework with manual coverage checks
- A Git client for commit and branch management
- An AI assistant (Copilot, Claude) for code completion

These tools are **disconnected** — they do not share context with each other, do not learn from past feedback, and do not cover the developer's full workflow in one place.

**Key gaps identified:**
1. No tool provides **real-time, project-aware AI analysis** triggered automatically on file save.
2. No tool **learns from patterns** across a project and promotes recurring issues into a permanent knowledge base.
3. No tool connects **CI/CD failures to their root cause in source code** and generates fixes automatically.
4. No tool gives the developer a **commit readiness score** based on code quality, test coverage, and risk analysis simultaneously.

**Code Auditor AI** addresses all four gaps in a single platform.

---

## 3. Objectives

### Primary Objectives
1. Build a **real-time Watch Mode** that analyzes code files on save using a full RAG + LLM pipeline without blocking the developer.
2. Implement a **self-improving learning mechanism** that promotes recurring code patterns into a permanent knowledge base.
3. Develop a **CI/CD Intelligence module** that autonomously diagnoses pipeline failures and posts fixes as PR comments.
4. Create a **Smart Git module** that evaluates commit readiness and guides the developer through safe commit and PR workflows.

### Secondary Objectives
5. Provide **inline VS Code decorations** (gutter indicators, CodeLens, hover cards) with one-click fix application.
6. Build a **WebSocket-based real-time communication** channel between the backend and all frontends.
7. Implement a **two-phase UX response**: immediate feedback (< 1ms) before the AI analysis completes (10–60s).
8. Deliver a **React Web Dashboard** that visualizes all AI events (issues, test gaps, git readiness, CI failures) in one place.

---

## 4. System Architecture

### 4.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    CLIENT INTERFACES                      │
│                                                          │
│  ┌─────────────────────┐    ┌──────────────────────────┐ │
│  │  VS Code Extension  │    │  React Web Dashboard     │ │
│  │  (TypeScript)       │    │  (React + Zustand)       │ │
│  │  WatchController    │    │  WatchPage               │ │
│  │  WatchInlineManager │    │  GitSmartPage            │ │
│  │  ChatPanel          │    │  CicdDashboardPage       │ │
│  │  WatchCodeLens      │    │  ChatAgentPage           │ │
│  └──────────┬──────────┘    └────────────┬─────────────┘ │
│             │ WebSocket /ws              │ REST + SSE     │
└─────────────┼──────────────────────────-┼───────────────-┘
              │                            │
┌─────────────▼────────────────────────────▼───────────────┐
│                  FASTAPI BACKEND (port 8765)              │
│                                                          │
│  ┌────────────┐ ┌───────────┐ ┌──────────┐ ┌─────────┐  │
│  │ /watch/*   │ │ /api/chat │ │ /api/git │ │ /api/ci │  │
│  │ WebSocket  │ │ SSE stream│ │          │ │         │  │
│  └─────┬──────┘ └─────┬─────┘ └────┬─────┘ └────┬────┘  │
│        │              │             │              │      │
│  ┌─────▼──────────────▼─────────────▼──────────────▼───┐ │
│  │              LangGraph Orchestration Layer            │ │
│  │   WatchGraph │ ChatGraph │ SmartGitGraph │ CIGraph   │ │
│  └─────┬──────────────┬────────────┬────────────┬──────┘ │
│        │              │            │            │        │
│  ┌─────▼──────┐ ┌─────▼────┐ ┌────▼────┐ ┌────▼──────┐  │
│  │ ChromaDB   │ │  Redis   │ │  GitHub │ │  SonarQube│  │
│  │ RAG + Jina │ │ Memory   │ │  API    │ │  MCP      │  │
│  └────────────┘ └──────────┘ └─────────┘ └───────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 4.2 Component Interaction Summary

| Component | Role | Communication |
|---|---|---|
| VS Code Extension | Primary user interface in editor | WebSocket (watch), REST (chat/git/tests), SSE (chat stream) |
| React Dashboard | Project-wide visualization | REST + SSE |
| FastAPI Server | API gateway, event broadcast | Handles all client connections |
| LangGraph Graphs | Orchestrate multi-step AI pipelines | Internal (Python function calls) |
| LangChain Agents | Execute specific AI tasks within graphs | Internal |
| ChromaDB | Vector store for RAG retrieval | Python client |
| Redis | Session memory, pattern frequency tracking | Python client |
| GitHub API | CI run logs, PR comments | REST via `langchain_agents/tools/ci_tools.py` |

---

## 5. Backend — FastAPI Server

**File:** `api/server.py` (~793 lines)

### 5.1 Server Initialization

The server starts with:
- A background file watcher thread (watchdog) monitoring the project directory
- An asyncio event loop for WebSocket broadcasting
- Lazy-loaded shared singletons (RAG system, dependency graph, ChromaDB indexer, Redis cache)
- Port: configurable via `--port` argument (default `8765`)

### 5.2 Two-Phase Watch Response

When a file change is detected, the server performs a **two-phase response**:

**Phase A (< 1ms):** Broadcasts `analysis_started` event immediately via WebSocket.
```python
asyncio.run_coroutine_threadsafe(
    _ws_manager.broadcast({
        "type":      "analysis_started",
        "file_path": str(fp),
        "file_name": fp.name,
        "language":  _ext_to_language(fp),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }),
    _loop,
)
```

**Phase B (10–60s):** Invokes the full 14-node `WatchGraph` pipeline. Results are broadcast as WebSocket events in the `ws_events` schema v2.0.

### 5.3 API Routers

| Router | File | Endpoints |
|---|---|---|
| Watch | `server.py` | `POST /watch/start`, `POST /watch/stop`, `GET /watch/status`, `GET /watch/events/latest` |
| WebSocket | `server.py` | `WS /ws` — real-time event broadcast |
| Chat | `api/chat_router.py` | `POST /api/chat`, `POST /api/chat/stream` (SSE), `GET /api/chat/history/{id}` |
| Git | `api/git_router.py` | `POST /api/git/status`, `POST /api/git/commit-msg`, `POST /api/git/smart` |
| CI/CD | `api/ci_router.py` | `POST /api/ci/run`, `POST /api/ci/analyze`, `GET /api/ci/history` |
| Tests | `server.py` | `POST /generate-tests` |
| Diagnostics | `api/diagnostics_router.py` | `POST /api/diagnostics/analyze` |
| Code Actions | `api/code_actions_router.py` | `POST /api/code-actions/apply` |
| Health | `server.py` | `GET /health`, `GET /stats` |

### 5.4 WebSocket Event Schema (v2.0)

All WebSocket events follow this structure:

```json
{
  "type": "analysis_result | analysis_started | dependency_impact | test_gap | git_recommendation | known_issue | error",
  "file_path": "...",
  "language": "python | java | typescript | javascript",
  "schema_version": "2.0",
  "issues": [
    {
      "id": "uuid",
      "title": "...",
      "message": "...",
      "severity": "critical | error | warning | info",
      "line": 42,
      "column": 8,
      "rule": "...",
      "fix_id": "uuid",
      "fix_available": true
    }
  ],
  "fixes": [
    {
      "id": "uuid",
      "issue_id": "uuid",
      "title": "...",
      "current_code": "...",
      "fixed_code": "...",
      "explanation": "...",
      "diff_hunks": [
        { "start_line": 40, "end_line": 45, "original_lines": [...], "new_lines": [...] }
      ],
      "apply_mode": "replace_snippet | replace_method | full_file"
    }
  ]
}
```

---

## 6. Core AI Module — WatchGraph

**File:** `langchain_agents/graphs/watch_graph.py` (~1587 lines)

The WatchGraph is the most sophisticated component of the system. It is a **LangGraph StateGraph** with 14 nodes that processes a file from raw content to structured AI results.

### 6.1 Graph Topology

```
hash_check ──► read_file ──► change_filter ──► parse_ast ──► index_chromadb
                                                              │
                                                         update_kg
                                                              │
                                                      update_dep_graph
                                                              │
                                                      test_gap_detect
                                                              │
                                                      get_neighborhood
                                                              │
                                                       rag_retrieve
                                                              │
                                                        git_session
                                                              │
                                                       build_context
                                                              │
                                                        llm_analyze
                                                              │
                                                       cache_results
                                                              │
                                                      learn_feedback
                                                         │       │
                                            has_deps?  YES      NO
                                                 │               │
                                         analyze_dependents  emit_ws_events
                                                 │               │
                                          emit_ws_events ────────►END
```

### 6.2 Node Descriptions

| Node | Function | Description |
|---|---|---|
| 1 `hash_check` | `node_hash_check` | Computes SHA-256 hash of file content. If hash matches last analysis → skip. Prevents unnecessary re-analysis. |
| 2 `read_file` | `node_read_file` | Detects programming language. Filters unsupported languages early. |
| 3 `change_filter` | `node_change_filter` | Uses `CodeAgent` to score the significance of the change (0–100). Minor changes (whitespace, comments) → skip. |
| 4 `parse_ast` | `node_parse_ast` | Parses the file into an AST (Abstract Syntax Tree). Extracts entities: classes, functions, imports, variables. |
| 5 `index_chromadb` | `node_index_chromadb` | Indexes the file content into ChromaDB vector store for future RAG retrieval. |
| 6 `update_kg` | `node_update_kg` | Updates the local Knowledge Graph JSON with new entities extracted from the file. |
| 7 `update_dep_graph` | `node_update_dep_graph` | Updates the NetworkX dependency graph with import relationships of the current file. |
| 8 `test_gap_detect` | `node_test_gap_detect` | Checks for test files covering the current source file. Emits a `test_gap` event if missing. |
| 9 `get_neighborhood` | `node_get_neighborhood` | Retrieves the file's dependency neighborhood from the graph: predecessors (files that import this) and successors. |
| 10 `rag_retrieve` | `node_rag_retrieve` | **2-pass RAG retrieval**: (1) semantic query on code content with Jina embeddings, (2) reranking by relevance score. Returns top 8 documents (threshold: 0.45). |
| 11 `git_session` | `node_git_session` | Reads the current Git session state: modified files, staged files, branch, last commit. Used to build context for the LLM. |
| 12 `build_context` | `node_build_context` | Assembles all gathered information (AST, RAG docs, knowledge base, Git state, dependency neighborhood) into a structured prompt context. |
| 13 `llm_analyze` | `node_llm_analyze` | **LLM cascade**: calls OpenRouter (primary) → Gemini (fallback). Exponential backoff: 10s/30s/60s/120s. Max 4 attempts. Produces structured `issues[]` + `fixes[]`. |
| 14 `cache_results` | `node_cache_results` | Caches analysis results to Redis with TTL. Emits `git_recommendation` event based on session state. |
| 15 `learn_feedback` | `node_learn_feedback` | **LearningAgent**: records detected patterns. If a pattern is seen 3+ times → promotes it to the permanent knowledge base. CRITICAL severity → auto-promote immediately. |
| 16 `analyze_dependents` | `node_analyze_dependents` | If the changed file has dependents (files that import it) → re-analyzes up to 5 dependent files recursively. |
| 17 `emit_ws_events` | `node_emit_ws_events` | Transforms all analysis results into the WebSocket schema v2.0 and broadcasts them to all connected clients. |

### 6.3 Conditional Edges

| From Node | Condition | Branch |
|---|---|---|
| `hash_check` | Content unchanged | → `END` (skip) |
| `change_filter` | Change score < threshold | → `END` (skip) |
| `learn_feedback` | Has dependent files? | `yes` → `analyze_dependents`, `no` → `emit_ws_events` |

### 6.4 LLM Cascade Strategy

```
Attempt 1: OpenRouter (minimax/minimax-m2.5:free)
    │ Failed? wait 10s
Attempt 2: OpenRouter retry
    │ Failed? wait 30s  
Attempt 3: Gemini (gemini-2.5-flash)
    │ Failed? wait 60s
Attempt 4: Gemini retry
    │ Failed? → emit error event
```

### 6.5 Analysis Strategies

The LLM produces results in one of three strategies based on file complexity:

| Strategy | Trigger | Output |
|---|---|---|
| `full_class` | Class-level change, large file | Full file refactoring suggestions |
| `targeted_methods` | Method-level change | Per-method issue list |
| `block_fix` | Small change, single block | Minimal targeted fix |

---

## 7. Smart Git Module — SmartGitGraph

**File:** `langchain_agents/graphs/smart_git_graph.py`

A 6-agent LangGraph StateGraph for intelligent Git operations.

### 7.1 Graph Topology

```
decide → session → diff → branch → pr → conflict → synthesize → END
```

### 7.2 Agent Descriptions

| Node | Agent | Responsibility |
|---|---|---|
| `decide` | `GitDecisionAgent` | Intent classification: `commit_message`, `pr_review`, `conflict_resolve`, `branch_analysis`, `git_status` |
| `session` | `GitSessionAgent` | Reads current git state: branch, modified files, staged files, last commit |
| `diff` | `GitDiffAgent` | Analyzes diffs. For `commit_message` intent: generates semantic commit message from diff |
| `branch` | `GitBranchAgent` | Evaluates branch risk: number of changes, divergence from main, dependency impact |
| `pr` | `GitPRAgent` | Prepares PR description, reviewers, labels based on diff and branch analysis |
| `conflict` | `GitConflictAgent` | Detects merge conflicts and suggests resolution strategies |
| `synthesize` | `GitSynthesisAgent` | Aggregates all agent results into a unified `SmartGitResult` response |

**Safety guarantees:** The graph never performs automatic commits, merges, or pushes. All operations are read-only and advisory only.

---

## 8. CI/CD Intelligence — CIGraph

**File:** `langchain_agents/graphs/ci_graph.py`

A 10-node LangGraph StateGraph that monitors GitHub Actions and provides automated diagnosis and remediation.

### 8.1 Graph Topology

```
fetch_run → classify_failure
                │         │
             success    failure
                │         │
         index_result  sonar_mcp_query
                         │
                    search_similar
                    │           │
               found_fix    needs_llm
                    │           │
           generate_comment  analyze_root_cause
                    │           │
                generate_fix ───┘
                    │
              post_pr_comment
                    │
               index_result
                    │
                  notify
                    │
                   END
```

### 8.2 Node Descriptions

| Node | Description |
|---|---|
| `fetch_run` | Fetches GitHub Actions run logs via GitHub API |
| `classify_failure` | Classifies failure type: build error, test failure, lint error, timeout, dependency error |
| `sonar_mcp_query` | Queries SonarQube MCP for static analysis findings related to the failure |
| `search_similar` | Searches ChromaDB for similar past failures and their fixes (RAG) |
| `analyze_root_cause` | Sends logs + SonarQube results to LLM for root cause analysis |
| `generate_fix` | Generates a code fix for the identified root cause |
| `generate_comment` | Formats the fix as a structured Markdown PR comment |
| `post_pr_comment` | Posts the generated comment to the GitHub PR via API |
| `index_result` | Stores the run result + fix in ChromaDB for future similarity searches |
| `notify` | Sends a summary notification via WebSocket |

---

## 9. Chat Module — ChatGraph

**File:** `langchain_agents/graphs/chat_graph.py`

A conversational AI graph with tool-calling capabilities for project-aware chat interactions.

### Key Agents

| Agent | File | Role |
|---|---|---|
| `ChatDecisionAgent` | `lc_chat_decision_agent.py` | Classifies user intent: code_question, file_explain, generate_code, debug, general |
| `ChatAgent` | `lc_chat_agent.py` | Main conversational agent with RAG context injection |
| `ProactiveAgent` | `lc_proactive_agent.py` | Generates unsolicited suggestions based on detected patterns |
| `ToolCallingAgent` | `lc_tool_calling_agent.py` | Executes tool calls: file read, search, code execution |
| `InlineCompletionAgent` | `lc_inline_completion_agent.py` | Serves inline code completions at cursor position |

### Chat Streaming

The chat endpoint uses **Server-Sent Events (SSE)** for real-time token streaming:

```
POST /api/chat/stream → text/event-stream

data: {"type": "status", "content": "Analyzing your question..."}
data: {"type": "plan", "intent": "code_question", "target_file": "..."}
data: {"type": "token", "content": "The issue is"}
data: {"type": "token", "content": " related to..."}
data: {"type": "code", "content": "def fixed_function():..."}
data: {"type": "done", "session_id": "...", "elapsed": 2.3}
```

---

## 10. AI/ML Components

### 10.1 RAG System (Retrieval-Augmented Generation)

**Embedding Model:** `jinaai/jina-embeddings-v2-base-code`
- Dimension: 768
- Optimized for source code (Python, Java, JavaScript, TypeScript)
- Auto-detects compute device: CUDA → MPS → CPU

**Vector Store:** ChromaDB
- Collection name: `code_kb_jina_v2`
- Distance metric: cosine similarity
- Chunk size: 800 characters, overlap: 150 characters
- Retrieval: top_k = 8, relevance threshold = 0.45

**2-Pass Retrieval Process:**
1. **Pass 1** — Semantic query using file content embeddings
2. **Pass 2** — Reranking by relevance score among top candidates

### 10.2 LearningAgent (Self-Improving KB)

**File:** `langchain_agents/agents/lc_learning_agent.py`

The LearningAgent implements a **4-pillar architecture**:

| Pillar | Implementation | Description |
|---|---|---|
| **LLM** | OpenRouter/Gemini | Generalizes a specific fix into a reusable KB rule |
| **Tools** | `tool_write_kb_rule`, `tool_reload_chromadb` | File operations on the knowledge base |
| **Memory** | Redis Sorted Set (PatternMemory) | Tracks pattern frequency per language |
| **Planning** | Conditional promotion logic | 3+ occurrences → auto-promote; CRITICAL → immediate promote |

**Promotion Flow:**
```
Analysis detects issue X in file.py
    │
Record pattern(language="python", pattern="unhandled_exception")
    │
Count = 1, 2 → store in Redis, no promotion
Count = 3   → LLM generalizes fix → writes KB rule → reloads ChromaDB
```

Once a rule is in the knowledge base, it is retrieved by the RAG system and injected into all future LLM prompts for the same language, achieving **project-specific self-improvement**.

### 10.3 Redis Semantic Memory

**File:** `langchain_agents/memory/lc_semantic_memory.py`

Uses Redis for:
- **Chat session history** — persists conversation turns across reconnections
- **Analysis results cache** — avoids re-analyzing unchanged files (TTL-based)
- **Pattern frequency tracking** — sorted sets for LearningAgent (promotion threshold: 3)
- **Agent working memory** — key-value store per agent instance

### 10.4 Knowledge Graph

A local JSON file (`data/knowledge_graph.json`) stores:
- File-to-entity relationships (classes, functions, imports)
- Cross-file dependency edges
- Historical analysis metadata

Used by the `get_neighborhood` node to inject relevant project context into the LLM prompt.

---

## 11. Frontend — VS Code Extension

**Directory:** `plugin_code_auditor/src/`  
**Language:** TypeScript  
**VS Code API:** `^1.85.0`

### 11.1 Architecture Overview

```
extension.ts (entry point)
    │
    ├── BackendClient ──────────────────── HTTP + REST calls
    │
    ├── WatchController ────────────────── WebSocket lifecycle
    │       ├── WatchEventNormalizer       Raw → NormalizedWatchEvent
    │       ├── WatchDiagnostics           VS Code Problems panel
    │       ├── WatchState                 In-memory event store
    │       └── WatchInlineManager ────── Editor decorations
    │               ├── WatchCodeLensProvider   CodeLens at file top
    │               └── WatchHoverProvider      Hover cards on issues
    │
    ├── StatusBarManager ───────────────── Status bar item
    ├── DiagnosticsManager ─────────────── Legacy REST diagnostics
    ├── IssuesTreeProvider ─────────────── Activity bar tree view
    │
    ├── ChatPanel (WebView) ────────────── AI chat interface
    ├── DashboardPanel (WebView) ────────── Project dashboard
    │
    └── commands.ts ────────────────────── All command handlers
```

### 11.2 Watch Mode — Real-Time Analysis Pipeline

**Phase A — Immediate feedback (< 1ms):**
1. Backend sends `analysis_started` event via WebSocket
2. `WatchController.handleRawEvent()` normalizes the event
3. `WatchInlineManager.setAnalyzing(filePath)` is called
4. Old issue decorations are cleared from the editor
5. `WatchCodeLensProvider` shows `$(loading~spin) Code Auditor: Analyzing…`
6. `StatusBarManager.setAnalyzing(fileName)` shows spinner in status bar

**Phase B — Full results (10–60s later):**
1. Backend sends `analysis_result` event with `issues[]` + `fixes[]`
2. `WatchInlineManager.handleEvent()` updates `dataByFile` map
3. Decorations are applied: 2px left border + gutter circle SVG by severity
4. `WatchCodeLensProvider` shows summary: `$(shield) Code Auditor: 4 issues 🔴 1 · 🟠 2 · 🔵 1`
5. `WatchDiagnostics.setFileIssues()` updates the Problems panel
6. `StatusBarManager.setWatchingWithCount(total, hasCritical)` updates status bar

### 11.3 Decoration System

| Severity | Gutter | Border | Overview Ruler |
|---|---|---|---|
| `critical` / `error` | Red circle `#f44747` | 2px left red | Right lane red |
| `warning` | Orange circle `#f5a623` | 2px left orange | Right lane orange |
| `medium` | Yellow circle `#e5c07b` | 2px left yellow | Right lane yellow |
| `info` | Blue circle `#61afef` | 2px left blue | Right lane blue |
| **Pending fix** | — | 3px green left | — |

Pending fix highlight: `rgba(78, 206, 122, 0.18)` green background on changed lines after fix applied.

### 11.4 Fix Application — 5-Strategy Algorithm

When a developer clicks "Apply fix", `smartApplyFix()` tries strategies in order:

```
Strategy 1: diff_hunks (preferred)
    Apply line-level hunks bottom-to-top to preserve line positions
    
Strategy 2: exact snippet match
    indexOf(current_code) in file text
    
Strategy 3: whitespace-normalized match
    Normalize all whitespace → indexOf → map back to original offsets
    
Strategy 4: line-number targeted
    Replace only the line at issue.line (single-line fallback)
    
Strategy 5: full_file replacement
    Safety guard: if fixedCode has < 50% of original line count → warn developer
```

After a successful fix:
- Green highlight shown on changed lines (inline diff visual)
- Old issue decoration removed immediately from `clearIssueAtLine()`
- Developer reviews with `$(check) Keep & Save` / `$(discard) Undo fix` CodeLens
- No auto-save — developer must explicitly confirm

### 11.5 VS Code Commands

| Command | Keybinding | Description |
|---|---|---|
| `codeAuditor.startWatch` | — | Start Watch Mode for workspace |
| `codeAuditor.stopWatch` | — | Stop Watch Mode |
| `codeAuditor.analyzeFile` | — | One-shot analysis of current file |
| `codeAuditor.browseFileIssues` | — | QuickPick with all issues + actions |
| `codeAuditor.applyInlineFix` | — | Apply AI fix with 5-strategy algorithm |
| `codeAuditor.applyAllFixes` | — | Apply all fixes in one atomic operation |
| `codeAuditor.keepInlineFix` | — | Confirm fix and save file |
| `codeAuditor.undoInlineFix` | — | Revert fix and restore issue decorations |
| `codeAuditor.generateTests` | — | Generate unit tests for current file |
| `codeAuditor.generateCommitMessage` | — | AI commit message → editable input box → clipboard |
| `codeAuditor.openChat` | — | Open AI chat panel |
| `codeAuditor.explainWatchIssue` | — | Send issue explain prompt to chat |
| `codeAuditor.fixWatchIssue` | — | Request fix via chat |
| `codeAuditor.navigateImpactedFiles` | — | QuickPick to open files impacted by dependency change |

### 11.6 Notification Throttling

Toast notifications are throttled to **10 minutes per file per event type** to prevent notification fatigue:
- `test_gap` — max 1 notification per file per 10 min
- `git_recommendation` — max 1 global per 10 min
- `dependency_impact` — max 1 per file per 10 min
- `known_issue` — max 1 per file per 10 min

### 11.7 WebSocket Connection Management

The `WatchController` manages the WebSocket lifecycle:
- **Start**: POST `/watch/start` with 5 retries × 3s delay
- **Connect**: WebSocket to `ws://127.0.0.1:8765/ws`
- **Heartbeat**: ping every 20 seconds
- **Reconnect**: exponential backoff, max 10s, after any disconnection
- **Warning**: shown after 3 consecutive failed reconnects
- **Stop**: POST `/watch/stop`, close socket, clear all diagnostics + decorations

### 11.8 Status Bar States

| State | Icon | Color | Trigger |
|---|---|---|---|
| Idle | `$(shield) Code Auditor` | Default | No activity |
| Starting | `$(loading~spin) Code Auditor starting...` | Default | Backend launch |
| Analyzing | `$(loading~spin) Analyzing {file}...` | Default | Phase A event |
| Watching | `$(eye) Code Auditor: Watching` | Default | Watch start |
| Watch + issues | `$(error/warning) Watching — N issues` | Red/Orange | After analysis result |
| Watch + clean | `$(shield-check) Watching — Clean` | Default | After clean analysis |
| Tests missing | `$(warning) Code Auditor: Tests missing` | Orange | test_gap event |
| Impact detected | `$(warning) Code Auditor: Impact detected` | Orange | dependency_impact |
| Error | `$(error) Code Auditor: {message}` | Red | Fatal error |

---

## 12. Frontend — React Web Dashboard

**Directory:** `plugin_code_auditor/webview-ui/src/`  
**Framework:** React 18 + TypeScript + Vite  
**State Management:** Zustand

### 12.1 Application Structure

```
App.tsx
└── MainLayout.tsx
    ├── Sidebar.tsx (navigation)
    ├── TopBar.tsx
    ├── [ActivePage] (routed via uiStore)
    │   ├── ChatAgentPage.tsx  (default)
    │   ├── GitSmartPage.tsx
    │   ├── CicdDashboardPage.tsx
    │   ├── WatchPage.tsx (standalone at /watch-plugin)
    │   ├── SettingsPage.tsx
    │   └── PlaceholderPage (Tests, Proactive, History)
    └── RightPanel.tsx
```

### 12.2 Pages

| Page | Route | Description |
|---|---|---|
| **ChatAgent** | Default | Main AI interface. Chat with context from CI/CD, git, and watch state |
| **Smart Git** | `git` | Displays commit readiness score, changed files risk, suggested commit message |
| **CI/CD Dashboard** | `cicd` | Pipeline steps visualization, issue list, fix application |
| **Watch Plugin** | `/watch-plugin` | Standalone page for VS Code extension install/access flow |
| **Settings** | `settings` | Backend URL, auth token, configuration |

### 12.3 State Stores (Zustand)

| Store | File | Managed State |
|---|---|---|
| `watchStore` | `store/watchStore.ts` | Issues, fixes, test gaps, dependency impacts, git recommendations from WebSocket events |
| `cicdStore` | `store/cicdStore.ts` | Pipeline steps, issues, scores, verdict, repo URL |
| `gitSmartStore` | `store/gitSmartStore.ts` | Readiness score, verdict, changed files risk signals, commit message |
| `backendStore` | `store/backendStore.ts` | Backend health status, version, services |
| `authStore` | `store/authStore.ts` | Authentication state, user info |
| `settingsStore` | `store/settingsStore.ts` | Backend URL, auth token |
| `uiStore` | `store/uiStore.ts` | Active page routing |

### 12.4 WatchStore Event Ingestion

The `watchStore.ingestMessage()` method processes all WebSocket event types:

| Event Type | Store Update |
|---|---|
| `analysis_result` | Upsert issues + fixes for file (replaces old data for same `filePath`) |
| `diagnostics_update` | Upsert issues from diagnostics |
| `test_gap` | Update `testGaps[filePath]` |
| `dependency_impact` | Update `dependencyImpacts[filePath]` |
| `git_recommendation` | Update `gitRecommendation` |

### 12.5 VS Code Bridge

```typescript
// vscodeApi.ts — communication with VS Code extension host
export function postToExtension(type: string, payload?: unknown) {
    vscodeApi = window.acquireVsCodeApi();
    vscodeApi?.postMessage({ type, payload });
}
```

This allows the React webview panel to trigger VS Code commands from within the webview.

### 12.6 Design System

The dashboard uses a custom dark-theme design system:

| Token | Value | Usage |
|---|---|---|
| `--bg` / `caBg` | `#0B1020` | Page background |
| `--surface` / `caSurface` | `#0f1629` | Card backgrounds |
| `--border` / `caBorder` | `#1e2d4a` | Card borders |
| `--primary` / `caPrimary` | `#7C3AED` | Buttons, active states |
| `caDanger` | `#ef4444` | Critical issues |
| `caWarning` | `#f59e0b` | Warnings |
| `caSuccess` | `#22c55e` | Clean state |
| `caInfo` | `#38bdf8` | Informational |

---

## 13. API Reference

### 13.1 Watch Mode Endpoints

```
POST /watch/start
Body: { "project_path": "/path/to/project" }
Response: { "status": "started", "project_path": "...", "files_analyzed": 0 }

POST /watch/stop
Body: { "project_path": "..." }
Response: { "status": "stopped" }

GET /watch/status
Response: { "is_running": true, "project_path": "...", "files_processed": 42 }

GET /watch/events/latest
Response: {
    "has_data": true,
    "events": [ ...NormalizedWatchEvent[] ]
}
```

### 13.2 Analysis Endpoints

```
POST /analyze/file
Body: { "file_path": "...", "project_path": "..." }
Response: AnalysisResult (synchronous, uses WatchGraph)

POST /analyze/project
Body: { "project_path": "..." }
Response: { "status": "complete", "files_analyzed": N }
```

### 13.3 Chat Endpoints

```
POST /api/chat
Body: { "message": "...", "project_path": "...", "session_id": "...", "target_file": "..." }
Response: ChatResponse { "response": "...", "session_id": "...", "code_blocks": [...] }

POST /api/chat/stream
Body: same as /api/chat
Response: text/event-stream (SSE tokens)

GET /api/chat/history/{session_id}
Response: { "session_id": "...", "turns": [ {role, content}... ] }
```

### 13.4 Git Endpoints

```
POST /api/git/status
Body: { "project_path": "..." }
Response: GitStatus object

POST /api/git/commit-msg
Body: { "project_path": "...", "diff": "..." }
Response: { "commit_message": "feat: ...", "type": "feat", "scope": "auth" }

POST /api/git/smart
Body: { "message": "...", "project_path": "...", "repo": "...", "owner": "..." }
Response: SmartGitResult from SmartGitGraph
```

### 13.5 CI/CD Endpoints

```
POST /api/ci/run
Body: { "run_id": "...", "repo": "...", "owner": "...", "pr_number": 42 }
Response: CIAnalysisResult from CIGraph

POST /generate-tests
Body: { "file_path": "...", "project_path": "..." }
Response: { "test_code": "...", "test_file": "...", "language": "..." }
```

---

## 14. Data & Event Flow

### 14.1 Watch Mode Complete Flow

```
Developer saves file
    │
[OS file system event — watchdog]
    │
_debounced(fp) — 4 second debounce
    │
Phase A: broadcast {type: "analysis_started", file_path: ...}
    │                           │
    │              ┌────────────▼────────────┐
    │              │   VS Code Extension     │
    │              │   spinner shows         │
    │              │   old decorations clear │
    │              └─────────────────────────┘
    │
Phase B: invoke_watch(fp, project_path, rag, dep_graph, ...)
    │
WatchGraph: hash → change_filter → AST → ChromaDB → KG → deps →
            RAG (2-pass) → git_session → build_context →
            LLM (cascade) → cache → learn → emit_ws_events
    │
broadcast ws_events (issues[], fixes[], test_gap?, git_recommendation?)
    │
[WebSocket to all clients]
    │
VS Code Extension: decorate editor + update Problems + update status bar
React Dashboard: update watchStore → re-render IssueCard, WatchSummary
```

### 14.2 Fix Application Flow

```
Developer hovers issue line → WatchHoverProvider shows hover card
Developer clicks CodeLens "Browse issues" → QuickPick opens
Developer selects "Apply fix"
    │
codeAuditor.applyInlineFix(event, issueIndex)
    │
smartApplyFix(editor, fix, issue, watchInlineManager)
    │
Strategy 1: diff_hunks? → applyDiffHunks() bottom-to-top
Strategy 2: exact match? → editor.edit(replace)
Strategy 3: normalized match? → findNormalizedMatch() → editor.edit()
Strategy 4: line number? → replace that line
Strategy 5: full_file? → safety check → editor.edit()
    │
handleFixSuccess()
    │
detectChangedLines() → green highlight on changed lines
clearIssueAtLine() → remove decoration for fixed issue
CodeLens changes to "$(check) Keep & Save" / "$(discard) Undo"
    │
Developer clicks "Keep & Save"
    │
clearPendingFix() → save file
File save → watch watcher triggers → new analysis starts
```

---

## 15. Technology Stack

### 15.1 Backend

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Core language |
| FastAPI | 0.115+ | REST API + WebSocket server |
| LangGraph | 0.2+ | Multi-agent graph orchestration |
| LangChain | 0.3+ | Agent + tool + memory abstractions |
| ChromaDB | 0.5+ | Local vector store |
| Sentence Transformers | 3.x | Jina embedding model runtime |
| Redis | 7.x | Semantic memory + caching |
| Watchdog | 5.x | OS file system monitoring |
| Pydantic | 2.x | Data validation and configuration |
| NetworkX | 3.x | Dependency graph |
| OpenRouter API | — | Primary LLM provider (MiniMax M2.5) |
| Google Gemini API | gemini-2.5-flash | Fallback LLM provider |
| GitHub REST API | v3 | CI run logs, PR comments |
| SonarQube | MCP | Static analysis queries |
| LangSmith | — | Optional LangGraph tracing |

### 15.2 VS Code Extension

| Technology | Purpose |
|---|---|
| TypeScript 5.x | Extension source language |
| VS Code Extension API `^1.85.0` | Editor integrations (decorations, CodeLens, hover, diagnostics, webview) |
| `ws` (WebSocket) | WebSocket client for watch mode |
| Node.js `child_process` | Backend server auto-start |
| esbuild | Extension bundler |

### 15.3 React Dashboard

| Technology | Purpose |
|---|---|
| React 18 | UI framework |
| TypeScript | Type safety |
| Vite | Build tool |
| Zustand | Global state management |
| Tailwind CSS | Utility-first styling |
| Lucide React | Icon library |
| TanStack Query | Data fetching (planned) |

---

## 16. Comparison with Existing Tools

| Feature | GitHub Copilot | Claude Code | **Code Auditor AI** |
|---|---|---|---|
| Inline autocomplete | ✅ Excellent | ❌ | ✅ (InlineCompletionAgent) |
| Real-time watch mode | ❌ | ❌ | ✅ WebSocket + LangGraph |
| RAG over project codebase | ⚠️ Limited | ✅ Context window | ✅ ChromaDB + Jina + reranking |
| Self-improving KB | ❌ | ❌ | ✅ LearningAgent (Redis patterns) |
| CI/CD intelligence | ❌ | ⚠️ Via shell | ✅ CIGraph (GitHub Actions) |
| Smart Git multi-agent | ❌ | ⚠️ Partial | ✅ 6-agent SmartGitGraph |
| Dependency cascade analysis | ❌ | ❌ | ✅ NetworkX dep graph |
| Test gap detection | ❌ | ❌ | ✅ Automated per file |
| Inline fix decorations | ⚠️ Limited | ❌ | ✅ 5-strategy algorithm |
| Commit readiness score | ❌ | ❌ | ✅ Git recommendation event |
| Session memory (Redis) | ❌ | ⚠️ Context only | ✅ Persistent Redis memory |
| Project-specific learning | ❌ | ❌ | ✅ Pattern → KB promotion |
| Setup complexity | ⭐ Zero config | ⭐ Zero config | ⚠️ Python + Redis + API keys |
| LLM response latency | ~200ms | Interactive | 10–60s (full pipeline) |

**Positioning:** Code Auditor AI is not a direct competitor to Copilot (autocomplete) or Claude Code (interactive sessions). It occupies the **continuous code quality guardian** niche — monitoring, learning, and advising across the full developer lifecycle.

---

## 17. Performance Considerations

### 17.1 Debounce Strategy

The file watcher uses a **4-second debounce** before triggering analysis. This prevents:
- Rapid keystroke events from queuing multiple analyses
- Incomplete file writes from being analyzed mid-save

### 17.2 Hash-Based Skip

Node 1 (`hash_check`) computes SHA-256 of file content. If the hash matches the previous analysis, the entire pipeline is skipped. This eliminates unnecessary LLM calls on repeated saves of unchanged files.

### 17.3 Change Score Filter

Node 3 (`change_filter`) scores the significance of the change (0–100). Changes below a threshold (whitespace reformatting, comment updates) are skipped before any expensive operations.

### 17.4 Two-Phase UX

The critical insight for developer experience: the **10–60s LLM latency** is tolerable if the UI responds immediately. Phase A broadcasts `analysis_started` in < 1ms, giving the developer instant visual feedback while the pipeline runs in the background.

### 17.5 LLM Cascade Latency Expectations

| Scenario | Expected Latency |
|---|---|
| OpenRouter (first attempt) | 5–20s |
| OpenRouter timeout → Gemini | 30–50s |
| Both fail → error event | 120s+ |
| Cache hit (unchanged file) | < 100ms |

### 17.6 ChromaDB Performance

- Cold start (first embed): 2–5s (Jina model load)
- Subsequent queries: 50–200ms (in-memory index)
- Index update (on file change): 100–300ms

---

## 18. Project File Structure

```
code_auditor/                         ← Backend root
├── api/
│   ├── server.py                     Main FastAPI server (~793 lines)
│   ├── chat_router.py                Chat endpoints
│   ├── git_router.py                 Git endpoints
│   ├── ci_router.py                  CI/CD endpoints
│   ├── diagnostics_router.py         Diagnostics endpoints
│   ├── code_actions_router.py        Code actions endpoints
│   ├── websocket_manager.py          WebSocket broadcast manager
│   ├── models.py                     Pydantic models
│   └── diff_utils.py                 Diff/patch utilities
├── langchain_agents/
│   ├── agents/
│   │   ├── lc_code_agent.py          File parsing, hashing, language detection
│   │   ├── lc_analysis_agent.py      LLM analysis with cascade
│   │   ├── lc_retriever_agent.py     RAG retrieval
│   │   ├── lc_learning_agent.py      Self-improving knowledge base
│   │   ├── lc_git_*.py (×6)          Git intelligence agents
│   │   ├── lc_chat_agent.py          Conversational agent
│   │   ├── lc_proactive_agent.py     Proactive suggestions
│   │   └── lc_inline_completion_agent.py  Code completions
│   ├── graphs/
│   │   ├── watch_graph.py            14-node WatchGraph (~1587 lines)
│   │   ├── smart_git_graph.py        6-node SmartGitGraph
│   │   ├── ci_graph.py               10-node CIGraph
│   │   ├── chat_graph.py             ChatGraph
│   │   ├── cd_graph.py               CD Graph (deployment)
│   │   ├── state.py                  WatchState, CIState TypedDicts
│   │   └── smart_git_state.py        SmartGitState TypedDict
│   ├── memory/
│   │   ├── redis_memory.py           AgentRedisMemory, PatternMemory
│   │   └── lc_semantic_memory.py     Semantic memory with embeddings
│   └── tools/
│       ├── rag_tools.py              ChromaDB retrieval tools
│       ├── cache_tools.py            Redis cache tools
│       ├── ci_tools.py               GitHub API tools
│       ├── git_session_tools.py      Git state tools
│       └── test_gap_tools.py         Test coverage tools
├── core/
│   └── project_analyzer.py          Project-level analysis
├── config.py                         All configuration (Pydantic)
├── data/
│   ├── knowledge_base/               Markdown KB rules
│   ├── vector_store/                 ChromaDB persistent storage
│   ├── cache/                        Analysis result cache
│   └── knowledge_graph.json          File dependency graph
└── .env                              API keys (gitignored)

plugin_code_auditor/                  ← Frontend root
├── src/
│   ├── extension.ts                  Entry point
│   ├── commands.ts                   All command handlers (~1267 lines)
│   ├── backendClient.ts              HTTP + REST API client
│   ├── statusBar.ts                  Status bar manager
│   ├── diagnostics.ts                Legacy REST diagnostics
│   ├── treeView.ts                   Activity bar tree view
│   ├── codeActions.ts                Legacy code action provider
│   ├── watch/
│   │   ├── WatchController.ts        WebSocket + lifecycle
│   │   ├── WatchInlineManager.ts     Decorations + CodeLens data
│   │   ├── WatchCodeLensProvider.ts  CodeLens provider
│   │   ├── WatchHoverProvider.ts     Hover card provider
│   │   ├── WatchDiagnostics.ts       Problems panel integration
│   │   ├── WatchEventNormalizer.ts   Raw → NormalizedWatchEvent
│   │   ├── WatchEventTypes.ts        TypeScript interfaces
│   │   └── WatchState.ts             In-memory event store
│   ├── webview/
│   │   ├── ChatPanel.ts              Chat WebView panel
│   │   ├── DashboardPanel.ts         Dashboard WebView panel
│   │   └── WebviewHost.ts            Shared WebView utilities
│   └── providers/
│       ├── InlineCompletionProvider.ts
│       ├── DiagnosticsProvider.ts
│       └── CodeActionsProvider.ts
├── media/
│   ├── chat.css                      Chat UI styling (navy + violet)
│   ├── chat.js                       Chat WebView JavaScript
│   ├── dashboard.css                 Dashboard styling
│   └── dashboard.js                  Dashboard JavaScript
├── webview-ui/                       React Dashboard
│   └── src/
│       ├── App.tsx
│       ├── components/Layout/
│       ├── features/
│       │   ├── watch/                WatchPage, IssueCard, WatchSummary
│       │   ├── git/                  GitSmartPage
│       │   ├── cicd/                 CicdDashboardPage
│       │   ├── chat/                 ChatAgentPage
│       │   ├── auth/                 LoginPage, ConnectBackendPage
│       │   └── settings/             SettingsPage
│       └── store/                    Zustand stores
└── package.json                      Extension manifest
```

---

## 19. Environment Configuration

### Required Environment Variables (`.env`)

```bash
# LLM Providers
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=minimax/minimax-m2.5:free
GOOGLE_API_KEY=AIza...
GEMINI_MODEL=gemini-2.5-flash

# GitHub Integration (CI/CD module)
GITHUB_TOKEN=ghp_...

# SonarQube (optional)
SONARQUBE_TOKEN=sqa_...

# Redis
REDIS_URL=redis://localhost:6379/0

# Server
SERVER_HOST=127.0.0.1
SERVER_PORT=8765

# LangSmith (optional tracing)
LANGSMITH_API_KEY=ls__...
LANGSMITH_TRACING=false
LANGSMITH_PROJECT=code-auditor
```

### VS Code Extension Settings

```json
{
  "codeAuditor.serverUrl": "http://127.0.0.1:8765",
  "codeAuditor.pythonPath": "python",
  "codeAuditor.backendPath": "",
  "codeAuditor.autoStart": true,
  "codeAuditor.analyzeOnSave": true
}
```

---

## 20. Key Design Decisions

### Decision 1: LangGraph over Custom Orchestration

**Chosen:** LangGraph StateGraph  
**Alternative:** Custom async pipeline with Python asyncio  
**Rationale:** LangGraph provides built-in conditional edges, typed state management, and LangSmith tracing integration. It makes the analysis pipeline declarative and testable at each node level, which is critical for a 14-step pipeline.

### Decision 2: Two-Phase WebSocket Response

**Chosen:** Phase A `analysis_started` (< 1ms) + Phase B full results  
**Alternative:** Single response after analysis completes  
**Rationale:** LLM pipeline latency (10–60s) makes a single-response approach unacceptable for developer UX. The two-phase approach gives immediate visual feedback without any changes to the underlying LLM pipeline.

### Decision 3: Local ChromaDB + Jina over Cloud Embeddings

**Chosen:** `jinaai/jina-embeddings-v2-base-code` running locally  
**Alternative:** OpenAI `text-embedding-ada-002` or Cohere embeddings  
**Rationale:** (1) Code-specialized model outperforms general text embeddings on source code similarity. (2) No per-token cost. (3) Works fully offline. (4) Auto-detects CUDA/MPS/CPU.

### Decision 4: 5-Strategy Fix Application

**Chosen:** `diff_hunks → exact → normalized → line-number → full_file`  
**Alternative:** Only full-file replacement  
**Rationale:** Full-file replacement is destructive for large files and LLM-generated fixes can be truncated. Precise diff_hunks preserve code not in the fix scope. The fallback chain ensures maximum fix success rate even when code drifts between analysis and application.

### Decision 5: Self-Improving Knowledge Base

**Chosen:** LearningAgent with Redis pattern frequency + KB promotion  
**Alternative:** Static knowledge base only  
**Rationale:** Code projects accumulate project-specific patterns (recurring architectural errors, team conventions, known vulnerabilities). A static KB becomes stale. The LearningAgent makes Code Auditor AI a system that improves as it is used.

### Decision 6: No Auto-Save After Fix

**Chosen:** Pending fix state with explicit Keep/Undo by developer  
**Alternative:** Auto-save immediately after fix  
**Rationale:** LLM-generated fixes can be incorrect. Auto-saving without review can silently corrupt working code. The green highlight + Keep/Undo CodeLens gives the developer full control while still providing immediate visual feedback.

---

## 21. Limitations & Future Work

### Current Limitations

| Limitation | Impact | Notes |
|---|---|---|
| LLM pipeline latency (10–60s) | Developer waits for full results | Mitigated by two-phase UX but still high for autocomplete use case |
| Local deployment required | No cloud version | Backend runs on developer's machine; Redis + ChromaDB must be installed |
| 4 languages supported | Python, JavaScript, TypeScript, Java only | Go, Rust, C# partially supported but not tested |
| React dashboard not WebSocket-connected | Dashboard shows demo data, no real-time | Zustand stores ready, WebSocket connection not yet implemented |
| No real authentication | WatchPage uses hardcoded `246810` demo code | OAuth or API key auth needed for multi-user deployment |
| ChatAgentPage send button inactive | Chat input in React dashboard has no `onSubmit` handler | Backend `/api/chat/stream` endpoint exists and works via VS Code extension |

### Planned Enhancements

| Feature | Description | Priority |
|---|---|---|
| Docker Compose deployment | `redis + chromadb + fastapi` in one `docker-compose.yml` | High |
| Connect React dashboard to backend | Wire `watchStore.ingestMessage` to a real WebSocket | High |
| Wire ChatAgentPage to `/api/chat/stream` | Implement SSE handler in React | High |
| Extend language support | Go, Rust, C#, Ruby | Medium |
| GitHub Actions webhook trigger | React to pushes/PRs automatically (no polling) | Medium |
| Multi-project support | Watch multiple projects simultaneously | Medium |
| Keyboard shortcut for quick fix | `Ctrl+Shift+.` applies first available fix at cursor | Medium |
| Conflict resolution UI | Visual merge conflict editor in VS Code | Low |
| Team knowledge base sharing | Export/import KB rules across team members | Low |
| Metrics dashboard | Issue trends, fix acceptance rate, KB growth chart | Low |

---

## Appendix A — LangGraph State Types

### WatchState (TypedDict)
```python
class WatchState(TypedDict):
    file_path:              str
    project_path:           str
    code:                   str
    language:               str
    content_hash:           str
    skip_reason:            Optional[str]
    parsed:                 Dict
    change_info:            Dict
    neighborhood:           Dict
    rag_docs:               List[Dict]
    context:                Dict
    analysis:               Dict
    strategy:               str
    ws_events:              List[Dict]
    dependents_to_analyze:  List[str]
    post_solution_mode:     bool
    stats:                  Dict
    # Injected services (prefixed with _)
    _project_indexer:       Any
    _rag_system:            Any
    _dep_graph:             Any
    _cache:                 Any
    _learning_agent:        Any
```

### SmartGitState (TypedDict)
```python
class SmartGitState(TypedDict):
    user_message:       str
    project_path:       str
    repo:               str
    owner:              str
    branch:             str
    pr_number:          int
    intent:             str
    confidence:         float
    selected_agents:    List[str]
    safe_mode:          bool
    session_snapshot:   Dict
    diff_result:        Dict
    branch_analysis:    Dict
    pr_result:          Dict
    conflict_result:    Dict
    synthesis:          Dict
```

---

## Appendix B — WebSocket Event Types Reference

| Event Type | Producer | Consumer | Payload Fields |
|---|---|---|---|
| `analysis_started` | `server.py` Phase A | VS Code spinner, CodeLens | `file_path`, `file_name`, `language`, `timestamp` |
| `analysis_result` | `node_emit_ws_events` | VS Code decorations, React watchStore | `file_path`, `language`, `issues[]`, `fixes[]`, `strategy`, `schema_version` |
| `diagnostics_update` | `api/diagnostics_router.py` | VS Code Problems panel | `file_path`, `diagnostics[]` |
| `dependency_impact` | `node_emit_ws_events` | VS Code toast, React watchStore | `file_path`, `impacted_files[]`, `risk`, `reason` |
| `test_gap` | `node_test_gap_detect` | VS Code toast + status bar, React watchStore | `file_path`, `missing_tests`, `recommendation`, `related_test_files[]` |
| `git_recommendation` | `node_cache_results` | VS Code git status bar, React watchStore | `should_commit`, `message`, `risk`, `blocking_reasons[]`, `suggested_commit_message` |
| `known_issue` | `node_learn_feedback` | VS Code toast | `file_path`, `issue_title`, `similarity`, `previous_fix`, `seen_count` |
| `connected` | WebSocket on open | VS Code silent | `version`, `clients` |
| `error` | Any node | VS Code error toast | `detail`, `message` |

---

*Document generated: June 2026*  
*Project: Code Auditor AI — Final Year Project*  
*Author: Maryem Chalghoumi*
