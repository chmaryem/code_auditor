# Code Auditor AI - Description Complète de l'Application

> **Version**: v6.2 - RAG-Enhanced Pipeline  
> **Date**: Mai 2026  
> **Auteur**: Analyse Système Complète

---

## Table des Matières

1. [Résumé Exécutif](#résumé-exécutif)
2. [Architecture Générale](#architecture-générale)
3. [Modes de Fonctionnement](#modes-de-fonctionnement)
4. [Stack Technique](#stack-technique)
5. [Composants Principaux](#composants-principaux)
6. [Pipeline d'Analyse](#pipeline-danalyse)
7. [Système RAG](#système-rag)
8. [Knowledge Graph](#knowledge-graph)
9. [Smart Git Integration](#smart-git-integration)
10. [CI/CD Intelligence](#cicd-intelligence)
11. [MCP Code Mode](#mcp-code-mode)
12. [LangGraph Multi-Agent](#langgraph-multi-agent)
13. [Configuration](#configuration)
14. [Limitations et Améliorations](#limitations-et-améliorations)

---

## Résumé Exécutif

**Code Auditor AI** est un système multi-agent d'analyse de code intelligent qui combine l'intelligence artificielle, le RAG (Retrieval-Augmented Generation) et l'intégration Git profonde pour fournir une analyse continue et automatisée du code source.

### Objectifs Principaux

- **Analyse statique augmentée par IA**: Détection de bugs, vulnérabilités et anti-patterns
- **Surveillance temps réel**: Mode watch pour analyse automatique à chaque sauvegarde
- **Intégration Git native**: Hooks pre-commit/merge, session tracking, branch analysis
- **Automatisation GitHub**: Agents autonomes pour PR review, résolution de conflits
- **CI/CD Intelligence**: Analyse des failures GitHub Actions et génération de workflows
- **Self-Improving RAG**: Système qui apprend et enrichit sa base de connaissances

### Points Forts

- Architecture RAG sophistiquée avec Knowledge Graph et reranking
- Intégration Git profonde (hooks, session tracking, branch analysis)
- Pipeline self-improving (Learning Agent)
- MCP Code Mode pour automatisation GitHub
- Support multi-langages (Python, Java, JavaScript, TypeScript)
- Résolution de conflits multi-niveaux (déterministe + LLM)

---

## Architecture Générale

```
┌──────────────────────────────────────────────────────────────────────┐
│                           main.py (CLI)                              │
│                    Point d'entrée - 10 commandes                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │
│  │ Orchestrator  │  │  Smart Git   │  │   MCP Code Mode           │  │
│  │ (core/)       │  │ (smart_git/) │  │  (agents/ + services/)    │  │
│  │               │  │              │  │                           │  │
│  │ • FileWatcher │  │ • hooks      │  │ • CodeModeAgent           │  │
│  │ • PriorityQ   │  │ • session    │  │ • SandboxExecutor         │  │
│  │ • Debounce    │  │ • branch     │  │ • MCPGitHubService        │  │
│  │ • Cancel      │  │ • conflict   │  │ • GitHubClient (wrapper)  │  │
│  └──────┬───────┘  │ • merge      │  └────────────┬──────────────┘  │
│         │          └──────────────┘               │                  │
│         ▼                                          ▼                  │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                    Services Layer                             │    │
│  │                                                              │    │
│  │  ┌────────────┐ ┌────────────┐ ┌──────────────┐             │    │
│  │  │ LLM Service │ │ RAG/ChromaDB│ │Knowledge Graph│            │    │
│  │  │ (Gemini/   │ │ (Jina emb.) │ │ (NetworkX)    │            │    │
│  │  │  Groq)     │ │ + Reranker  │ │ + auto-rules  │            │    │
│  │  └────────────┘ └────────────┘ └──────────────┘             │    │
│  │                                                              │    │
│  │  ┌────────────┐ ┌────────────┐ ┌──────────────┐             │    │
│  │  │Cache SQLite │ │Code Parser │ │Graph Service  │            │    │
│  │  │ (analyses, │ │ (AST Java, │ │ (NetworkX     │            │    │
│  │  │  patterns) │ │  Python...)│ │  dépendances) │            │    │
│  │  └────────────┘ └────────────┘ └──────────────┘             │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                    Agents Layer                               │    │
│  │                                                              │    │
│  │  CodeAgent   AnalysisAgent   RetrieverAgent   LearningAgent  │    │
│  │  (parsing,   (LLM analysis,  (RAG retrieval,  (self-improve, │    │
│  │   diff)       strategy)       neighborhood)    KB enrichment)│    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Modes de Fonctionnement

### Mode Local

Analyse de code sur le système de fichiers local.

| Commande | Description | LLM ? |
|---|---|---|
| `python main.py file <fichier>` | Analyse un seul fichier avec RAG | Oui |
| `python main.py project <dossier>` | Analyse un projet complet (architecture + fichiers) | Oui |
| `python main.py watch <dossier>` | Surveillance temps réel — analyse à chaque sauvegarde | Oui |

### Mode Git Local

Intégration profonde avec Git pour analyse des commits et branches.

| Commande | Description | LLM ? |
|---|---|---|
| `python main.py git <dossier>` | Analyse les fichiers modifiés dans le dernier commit | Oui |
| `python main.py git-status <dossier>` | Score de session — bugs accumulés non commités | Non |
| `python main.py git-branch <dossier>` | Analyse une branche vs sa base avant merge | Oui |
| `python main.py hook <dossier>` | Installe/désinstalle le pre-commit hook | Non |
| `python main.py resolve-conflicts <dossier>` | Résout les conflits de merge locaux via LLM | Oui |
| `python main.py merge-hook <dossier>` | Installe le pre-merge hook (bloque si score ≥ 35) | Non |

### Mode MCP (GitHub)

Agents autonomes pour automatisation GitHub via Model Context Protocol.

| Commande | Description | LLM ? | MCP ? |
|---|---|---|---|
| `python main.py pr-check --repo owner/repo --pr N` | Revue de PR via agent autonome | Oui (Gemini) | Oui |
| `python main.py pr-resolve --repo owner/repo --pr N` | Résolution de conflits PR | Conditionnel | Oui |
| `python main.py pr-merge-check --repo owner/repo --pr N` | Vérification merge readiness | Non (0 token) | Oui |
| `python main.py ci-deploy --repo owner/repo` | Déploiement workflow CI/CD | Non | Oui |
| `python main.py ci-poll --repo owner/repo` | Polling CI Intelligence | Oui | Oui |

---

## Stack Technique

### Intelligence Artificielle

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| **LLM Principal** | OpenRouter (MiniMax) | Analyse de code primaire |
| **LLM Fallback** | Gemini 2.5/2.0/1.5 Flash | Backup si quota épuisé |
| **Embeddings** | Jina v2 base-code (768 dims) | Vectorisation code et règles |
| **Vector Store** | ChromaDB | Stockage et recherche sémantique |
| **Reranker** | Cross-Encoder | Re-classement résultats RAG |
| **Knowledge Graph** | NetworkX | Modélisation relations code |

### Analyse de Code

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| **Parsing** | Tree-sitter | Extraction AST multi-langages |
| **Langages** | Python, Java, JavaScript, TypeScript | Support principal |
| **Validation** | Pylint, ASTroid | Analyse statique Python |
| **Dépendances** | NetworkX | Graphe de dépendances |

### Infrastructure

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| **Cache** | SQLite | Stockage analyses et patterns |
| **File Watcher** | Watchdog | Surveillance fichiers |
| **MCP Client** | mcp >= 1.0.0 | Communication serveurs MCP |
| **MCP GitHub** | @modelcontextprotocol/server-github | 26 outils GitHub |
| **MCP Redis** | redis-mcp-server | Cache distribué |
| **MCP SonarQube** | Custom service | Intégration SonarCloud |

### Orchestration

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| **Multi-Agent** | LangGraph | Orchestration agents autonomes |
| **Tracing** | LangSmith | Monitoring et debugging |
| **Async** | asyncio | Exécution parallèle |
| **Events** | PriorityQueue + Debounce | Gestion événements |

---

## Composants Principaux

### 1. Point d'Entrée (main.py)

Fichier principal avec 10 commandes CLI:
- 3 commandes mode local (file, project, watch)
- 6 commandes mode git (git, git-status, git-branch, hook, resolve-conflicts, merge-hook)
- 3 commandes mode MCP (pr-check, pr-resolve, pr-merge-check)
- 2 commandes CI/CD (ci-deploy, ci-poll)
- 1 commande génération tests (generate-tests)

### 2. Services Layer

**services/llm_service.py**
- CodeRAGSystemAPI: Interface principale RAG
- Gestion embeddings Jina v2
- ChromaDB vector store
- Fallback LLM (OpenRouter → Gemini)
- Analyse avec contexte enrichi

**services/knowledge_graph.py**
- Knowledge Graph basé sur NetworkX
- Détection automatique de patterns (SQL injection, resource leaks, etc.)
- Nœuds: FILE, CLASS, METHOD, CONCEPT, RULE
- Edges: CONTAINS, CALLS, IMPORTS, MATCHES, DETECTS

**services/knowledge_loader.py**
- Chargement base de connaissances depuis data/knowledge_base/
- Indexation dans ChromaDB
- Support YAML et Markdown
- Règles de best practices

**services/cache_service.py**
- Cache SQLite pour analyses
- Stockage patterns détectés
- Mémoire épisodique LearningAgent

**services/code_parser.py**
- Parsing AST via Tree-sitter
- Support Python, Java, JavaScript, TypeScript
- Extraction entités, imports, méthodes
- Détection structure code

**services/graph_service.py**
- Graphe de dépendances NetworkX
- Extraction dépendances
- Analyse voisinage
- Détection cycles

**services/mcp_github_service.py**
- Client MCP GitHub
- 26 outils GitHub disponibles
- Gestion PR, commits, files, reviews
- Wrapper GitHubClient

**services/mcp_redis_service.py**
- Client MCP Redis
- Cache distribué
- Stockage état agents
- Synchronisation multi-process

**services/mcp_sonarqube_service.py**
- Intégration SonarCloud
- Récupération métriques qualité
- Analyse debt technique

### 3. Agents Layer

**agents/code_agent.py**
- Parsing et filtrage code
- Analyse changements (score 0-100)
- Détection modifications mineures
- Diff parsing

**agents/analysis_agent.py**
- Construction contexte LLM
- Appel LLM avec stratégie
- Validation résultats
- Parsing réponse agentique

**agents/retriever_agent.py**
- RAG retrieval + Knowledge Graph
- Analyse voisinage
- Cross-encoder reranking
- Contexte enrichi

**agents/learning_agent.py**
- Self-Improving RAG
- Enregistrement patterns
- Auto-promotion règles (3+ occurrences)
- Enrichissement KB

**agents/code_mode_agent.py**
- Génération scripts MCP
- SandboxExecutor
- Agents autonomes GitHub

**agents/test_gap_agent.py**
- Détection tests manquants
- Analyse couverture tests
- Propositions tests

**agents/test_generator_agent.py**
- Génération tests unitaires
- RAG sur patterns de test
- Validation syntaxe

### 4. Smart Git Layer

**smart_git/git_diff_parser.py**
- Parsing diffs Git
- Détection fichiers modifiés
- Extraction changements

**smart_git/git_session_tracker.py**
- Surveillance accumulation bugs
- Score de session (0-100+)
- Thread daemon (check toutes les 3 min)
- Notifications hystérésis

**smart_git/git_branch_analyzer.py**
- Analyse branche vs base
- Verdict merge
- Rapport JSON

**smart_git/git_hook.py**
- Pre-commit hook
- Bloque si score ≥ 35
- Installation/désinstallation

**smart_git/git_merge_hook.py**
- Pre-merge hook
- Bloque si code critique
- Vérification readiness

**smart_git/git_conflict_resolver.py**
- Résolution conflits locaux
- Pipeline 3 niveaux
- 3-way merge déterministe
- Merge conservateur
- Fallback LLM

**smart_git/conflict_resolution_agent.py**
- Résolution conflits PR
- Détection multi-stratégies
- Contexte RAG
- Push branche auto-resolve

**smart_git/pr_analyzer.py**
- Analyse PR via MCP
- Coordination agents
- Gestion workflow

**smart_git/pr_review_agent.py**
- Revue PR structurée
- Calcul score
- Verdict (APPROVE/COMMENT/REQUEST_CHANGES)

**smart_git/merge_automation_agent.py**
- Vérification merge readiness
- 0 token LLM (100% factuel)
- Rapport Markdown

### 5. CI/CD Layer

**ci_cd/workflow_generator.py**
- Génération YAML GitHub Actions
- Détection profil projet (Java/Python/JS/TS)
- Support Maven, Gradle, npm, pip, Poetry
- Jobs: build-test, sonar-scan

**ci_cd/ci_deploy_agent.py**
- Déploiement workflow via MCP
- Détection langage
- Push sur GitHub
- Configuration secrets

**ci_cd/ci_runner.py**
- Exécution workflows locaux
- Simulation CI/CD
- Tests integration

**ci_cd/pipeline_failure_analyzer.py**
- Analyse failures GitHub Actions
- Root cause detection
- Propositions fixes

**ci_cd/ci_logs_indexer.py**
- Indexation logs CI
- Recherche patterns
- Analyse historique

**ci_cd/ci_status_reporter.py**
- Rapports statut CI
- Agrégation résultats
- Notifications

### 6. Core Layer

**core/orchestrator.py**
- Orchestration async
- PriorityQueue avec priorités
- Debounce coalesce
- Cancellation événements
- Gestion dépendants

**core/events.py**
- Système d'événements
- file_changed_event
- git_commit_event
- Priorités événements

**core/project_analyzer.py**
- Analyse projet complet
- Détection architecture
- Points d'entrée
- Dépendances circulaires
- Modules orphelins

### 7. LangChain Agents Layer

**langchain_agents/graphs/watch_graph.py**
- LangGraph StateGraph
- Orchestration agents LangChain
- Workflow watch mode
- Tracing LangSmith

**langchain_agents/graphs/ci_graph.py**
- LangGraph pour CI Intelligence
- Analyse failures
- Root cause detection

**langchain_agents/agents/**
- CodeAgent (LangChain)
- RetrieverAgent (LangChain)
- AnalysisAgent (LangChain)
- LearningAgent (LangChain)

**langchain_agents/tools/**
- Outils LangChain
- CI tools
- Git tools
- RAG tools

**langchain_agents/memory/**
- Redis memory
- Agent state
- Conversation history

### 8. Watchers Layer

**watchers/file_watcher.py**
- Surveillance fichiers via Watchdog
- Callback on_change
- Gestion suppressions
- Debounce intégré

---

## Pipeline d'Analyse

### Mode Watch - Pipeline en 12 Étapes

```
Fichier sauvegardé
    │
    ▼
┌─ Étape 1: Hash Check ─────────────────────────┐
│  Si le hash est identique → SKIP (0 token)     │
└────────────────────────────────────────────────┘
    │
    ▼
┌─ Étape 2: Lecture du fichier ──────────────────┐
└────────────────────────────────────────────────┘
    │
    ▼
┌─ Étape 3: Filtre intelligent ──────────────────┐
│  CodeAgent.analyze_change() calcule un score   │
│  de changement (0-100). Si < seuil → SKIP     │
│  Détecte : whitespace, comments, imports only  │
└────────────────────────────────────────────────┘
    │
    ▼
┌─ Étape 4: Parsing AST ────────────────────────┐
│  CodeParser extrait entités, imports, méthodes │
│  Supporte : Java, Python, JS/TS, Go, C#       │
└────────────────────────────────────────────────┘
    │
    ▼
┌─ Étape 4.5: ProjectCodeIndexer ───────────────┐
│  Indexe le fichier dans ChromaDB (embeddings)  │
│  timeout=4s, non-bloquant                      │
└────────────────────────────────────────────────┘
    │
    ▼
┌─ Étape 4.6: KG Update Incrémental ───────────┐
│  Met à jour le Knowledge Graph pour ce fichier │
└────────────────────────────────────────────────┘
    │
    ▼
┌─ Étape 5: Graphe de dépendances ─────────────┐
│  NetworkX — met à jour imports/exports         │
└────────────────────────────────────────────────┘
    │
    ▼
┌─ Étape 6: Voisinage ─────────────────────────┐
│  RetrieverAgent calcule : prédécesseurs,       │
│  successeurs, impact indirect, criticité       │
└────────────────────────────────────────────────┘
    │
    ▼
┌─ Étape 7: SystemAwareRAG ────────────────────┐
│  ChromaDB retrieval + Cross-Encoder reranker   │
│  Récupère les règles KB + code projet similaire│
└────────────────────────────────────────────────┘
    │
    ▼
┌─ Étape 8: Contexte enrichi ──────────────────┐
│  Assemble : voisinage + index projet +         │
│  change_info + post_solution flag              │
└────────────────────────────────────────────────┘
    │
    ▼
┌─ Étape 9: LLM Analysis ─────────────────────┐
│  Gemini/Groq analyse le code avec tout le      │
│  contexte RAG. Choisit une stratégie :         │
│  • block_fix : corrections ciblées             │
│  • targeted_methods : méthodes spécifiques     │
│  • full_class : réécriture complète            │
└────────────────────────────────────────────────┘
    │
    ▼
┌─ Étape 10: Cache SQLite ─────────────────────┐
│  Sauvegarde l'analyse pour réutilisation       │
└────────────────────────────────────────────────┘
    │
    ▼
┌─ Étape 11: Self-Improving RAG ───────────────┐
│  LearningAgent collecte les patterns détectés  │
│  Si un pattern est vu 3+ fois → nouvelle       │
│  règle ajoutée à la KB automatiquement         │
└────────────────────────────────────────────────┘
    │
    ▼
┌─ Étape 12: Analyse proactive dépendants ─────┐
│  Si le fichier a des dépendants (appelants),   │
│  les analyser aussi (asyncio.gather, max 2)    │
└────────────────────────────────────────────────┘
```

### Architecture Async

L'Orchestrator utilise une architecture asynchrone avancée:

- **Event Loop**: `asyncio` dans un thread daemon
- **PriorityQueue**: les événements sont priorisés (git commit > file change)
- **Debounce Coalesce**: N événements en <1s → 1 seul batch (évite les analyses redondantes)
- **Cancellation**: si un fichier est re-modifié pendant l'analyse, l'ancienne tâche est annulée
- **Parallel**: `asyncio.gather()` pour les dépendants

---

## Système RAG

### Architecture RAG

```
Code du fichier → Embedding Jina → Recherche ChromaDB
                                        │
                    ┌───────────────────┼──────────────────┐
                    │                   │                  │
              KB Rules           Project Code        KG Concepts
              (best practices)   (code similaire)    (patterns)
                    │                   │                  │
                    └───────────────────┼──────────────────┘
                                        │
                                   Reranker (Cross-Encoder)
                                        │
                                    Top-K docs
                                        │
                                  Contexte → LLM
```

### Composants RAG

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| **Embeddings** | Jina v2 (HuggingFace) | Vectorisation du code et des règles |
| **Vector Store** | ChromaDB | Stockage et recherche sémantique |
| **Knowledge Base** | `data/knowledge_base/` | Règles de bonnes pratiques (YAML/MD) |
| **Reranker** | Cross-Encoder | Re-classement des résultats RAG |
| **Project Code Indexer** | `services/knowledge_loader.py` | Indexe le code du projet dans ChromaDB |

### Fallback 429

Quand le quota Gemini est épuisé (`RESOURCE_EXHAUSTED` / `429`), le système:
1. Détecte l'erreur dans le texte retourné par le RAG
2. Active le `_StaticFallbackAnalyzer` (regex-based)
3. Retourne un score basé sur des patterns connus (SQL injection, hardcoded credentials, etc.)
4. Aucun faux négatif (score=0) n'est retourné

---

## Knowledge Graph

### Architecture

Le Knowledge Graph utilise **NetworkX** pour modéliser les relations entre:

```
┌─────────────────────────────────────────────────┐
│              Knowledge Graph (NetworkX)          │
│                                                  │
│  Nodes :                                         │
│    • FILE    : chaque fichier du projet          │
│    • CLASS   : classes détectées                 │
│    • METHOD  : méthodes/fonctions                │
│    • CONCEPT : patterns détectés (SqlInjection,  │
│                MissingResourceClose, etc.)        │
│    • RULE    : règles de la KB                   │
│                                                  │
│  Edges :                                         │
│    • CONTAINS : file → class → method            │
│    • CALLS    : method → method                  │
│    • IMPORTS  : file → file                      │
│    • MATCHES  : concept → rule                   │
│    • DETECTS  : file → concept                   │
└─────────────────────────────────────────────────┘
```

### Pattern Detection

Le KG détecte automatiquement des patterns dans le code:

- `SqlInjectionVulnerability` — concaténation SQL
- `MissingResourceClose` — connexions JDBC non fermées
- `PlainTextPasswordComparison` — comparaison de mots de passe en clair
- `UnboundedQuery` — SELECT * sans WHERE/LIMIT
- `DefaultFieldVisibility` — champs sans modificateur d'accès
- `UnclosedResultSet` / `UnclosedJdbcResource`
- `JdbcPreparedStatementNotClosed`

### Auto-Rules

Les concepts détectés sont mappés aux règles de la KB via le fichier `PATTERN_TO_KG_NODE`, permettant une détection contextuelle et un enrichissement automatique des requêtes RAG.

---

## Smart Git Integration

### Pré-commit Hook

```
git commit
    │
    ▼
Pre-commit hook activé
    │
    ▼
Lecture du cache SQLite → score des fichiers staged
    │
    ├── Score < 35 → ✅ Commit autorisé
    │
    └── Score ≥ 35 → ❌ Commit BLOQUÉ
        └── Message : "Fix les bugs critiques avant de commiter"
```

### Git Session Tracker

Surveille en arrière-plan (thread daemon, toutes les 3 min):

```
Session Tracker (daemon thread)
    │
    ├── Lit le cache SQLite (analyses Watch)
    ├── Calcule le score de session cumulé
    ├── Si le score monte → alerte GitNotifier
    └── git-status affiche le rapport complet
```

### Calcul du Score

**Poids par sévérité**:
- CRITICAL → 10 pts
- HIGH → 3 pts
- MEDIUM → 1 pt
- LOW → 0 pt (informatif seulement)

**Facteur temps (multiplicateur)**:
- < 30 min → ×1.0 (normal)
- 30–60 min → ×1.2 (attention)
- 60–120 min → ×1.5 (risque)
- > 120 min → ×2.0 (critique — commits trop espacés)

**Seuils de niveau**:
- CLEAN: score == 0 → tout va bien
- WATCH: 0 < score < 15 → information légère
- WARN: 15 ≤ score < 35 → rapport intermédiaire recommandé
- CRITICAL: score ≥ 35 → correction urgente avant commit

### Résolution de Conflits (3 Niveaux)

```
┌─────────────────────────────────────────────────┐
│  Niveau 1 : 3-way Merge Déterministe            │
│  (difflib.SequenceMatcher)                       │
│                                                  │
│  • Compare base ↔ ours ↔ theirs                 │
│  • Si pas de chevauchement → merge automatique   │
│  • Couvre ~70% des conflits                      │
│  • 0 token LLM                                   │
└─────────────────────┬───────────────────────────┘
                      │ échec
                      ▼
┌─────────────────────────────────────────────────┐
│  Niveau 2 : Merge Conservateur                   │
│                                                  │
│  • Garde OURS comme base                         │
│  • Détecte les nouvelles méthodes de THEIRS      │
│  • Les ajoute avant le dernier "}"               │
│  • Couvre ~20% des cas supplémentaires            │
│  • 0 token LLM                                   │
└─────────────────────┬───────────────────────────┘
                      │ échec
                      ▼
┌─────────────────────────────────────────────────┐
│  Niveau 3 : Gemini + RAG Context (v6.2)          │
│                                                  │
│  • Extrait seulement les blocs différents         │
│  • Injecte les patterns RAG dans le prompt        │
│  • Budget : ~200-500 tokens d'input               │
│  • Cascade : Gemini → Groq                        │
│  • Fallback ultime : OURS                         │
└─────────────────────────────────────────────────┘
```

---

## CI/CD Intelligence

### Workflow Generator

Génération automatique de workflows GitHub Actions adaptés au projet:

**Détection automatique**:
- Langage: Java (Maven/Gradle), Python (pip/Poetry), JavaScript/TypeScript (npm)
- Version: Java 17/11/8, Python 3.11/3.10, Node 20/18
- Tests: Détection répertoires tests/
- Docker: Détection Dockerfile

**Jobs générés**:
1. **build-test**: Compilation + tests unitaires
2. **sonar-scan**: Analyse SonarCloud (optionnel)

### CI Deploy Agent

Déploiement automatique de workflows sur GitHub via MCP:

1. Détection profil projet
2. Génération YAML adapté
3. Push via MCP GitHub
4. Configuration secrets (SONAR_TOKEN, etc.)
5. Activation workflow

### CI Poll Mode

Surveillance des GitHub Actions runs:

```
Polling (toutes les 2 min)
    │
    ├── Récupère derniers runs complétés
    ├── Détection nouveaux runs
    ├── Pour chaque run failure:
    │   ├── Invoque CIGraph (LangGraph)
    │   ├── Analyse logs
    │   ├── Root cause detection
    │   └── Propose fixes
    └── Marque run comme vu (Redis)
```

### Pipeline Failure Analyzer

Analyse des failures CI/CD:

- Extraction logs depuis GitHub
- Indexation dans ChromaDB
- Recherche patterns de failure
- Corrélation avec code modifié
- Propositions fixes basées sur RAG

---

## MCP Code Mode

### Architecture MCP

```
┌────────────────────────────────────────────────────────────────┐
│                      Code Auditor                              │
│                                                                │
│  ┌──────────────────┐    ┌───────────────────────────────┐    │
│  │  main.py (CLI)    │    │  code_mode_client.py           │    │
│  │                   │    │  (GitHubClient wrapper)         │    │
│  │  pr-check ───────►│    │                                 │    │
│  │  pr-resolve ─────►│    │  github.get_pr_info()           │    │
│  │  pr-merge-check ─►│    │  github.get_pr_files()          │    │
│  └──────────────────┘    │  github.get_file_content()       │    │
│                           │  github.post_review()            │    │
│                           │  github.push_file()              │    │
│                           │  github.create_pull_request()    │    │
│                           │  github.get_pr_mergeable_status()│    │
│                           └──────────────┬──────────────────┘    │
│                                          │                       │
│                                          ▼                       │
│                    ┌─────────────────────────────────────┐       │
│                    │    MCPGitHubService                  │       │
│                    │    (services/mcp_github_service.py)  │       │
│                    │                                     │       │
│                    │    MCP Client Session (stdio)        │       │
│                    └──────────────┬──────────────────────┘       │
│                                  │ stdin/stdout                  │
│                                  ▼                               │
│                    ┌─────────────────────────────────────┐       │
│                    │  @modelcontextprotocol/server-github │       │
│                    │  (npm, serveur MCP GitHub)           │       │
│                    │                                     │       │
│                    │  26 tools : get_pull_request,        │       │
│                    │  get_pull_request_files, push_files,  │       │
│                    │  create_pull_request, etc.            │       │
│                    └──────────────┬──────────────────────┘       │
│                                  │ HTTPS                         │
│                                  ▼                               │
│                         GitHub REST API                          │
└────────────────────────────────────────────────────────────────┘
```

### pr-check — Revue de PR

```
main.py pr-check
    │
    ▼
pr_analyzer.py → CodeModeAgent
    │
    ▼
CodeModeAgent génère un script Python via Gemini
    │   (system prompt impose les noms de méthodes exacts)
    │
    ▼
SandboxExecutor exécute le script dans un subprocess isolé
    │
    │   Le script sandbox a accès à :
    │   • github.* (wrappers MCP)
    │   • rag.analyze() (pipeline RAG complet)
    │   • cache.read_analysis() (cache SQLite)
    │   • kg.detect_patterns() (Knowledge Graph)
    │
    ▼
Le script :
    1. Récupère les fichiers de la PR
    2. Pour chaque fichier : cache → RAG → patterns KG
    3. Calcule le score total
    4. Poste un review structuré :
       • APPROVE si score < 15
       • COMMENT si 15 ≤ score < 35
       • REQUEST_CHANGES si score ≥ 35 ou critical > 0

Cascade LLM :
    gemini-2.5-flash → gemini-2.0-flash → gemini-1.5-flash
    (si quota 429 → backoff 15s → 45s → modèle suivant)
```

### pr-resolve — Résolution de Conflits

```
main.py pr-resolve
    │
    ▼
conflict_resolution_agent.py
    │
    ▼
1. Détection des conflits
   │
   ├── Stratégie 0 : REST API direct (urllib)
   │   → mergeable=True/False (résultat instantané, 100% fiable)
   │
   ├── [fallback] Stratégie 1 : get_pull_request_status (MCP)
   │   → mergeableState: "dirty" / "clean" / "unknown"
   │
   ├── [fallback] Stratégie 2 : Polling get_pull_request (MCP)
   │   → champ "mergeable" (6 tentatives × 5s)
   │
   ├── [fallback] Stratégie 3 : Inspection des patches
   │   → recherche de marqueurs <<<<<<< dans les patches
   │
   └── [fallback] Stratégie 4 : Comparaison contenu base vs patch
       → si >30% des lignes supprimées n'existent plus sur main
       → = divergence significative → conflit probable
    │
    ▼
2. Chargement contexte RAG (0 token LLM)
   │
   ├── RAGAnalyzer() — ChromaDB + Knowledge Graph
   └── CacheClient() — cache SQLite des analyses précédentes
    │
    ▼
3. Résolution fichier par fichier (3 niveaux)
   │
   │  Pour chaque fichier :
   │    a) Requête RAG : cache SQLite → ChromaDB/KG
   │    b) Niveau 1 : 3-way merge déterministe (difflib) — 0 token
   │    c) Niveau 2 : Merge conservateur (OURS + new methods THEIRS) — 0 token
   │    d) Niveau 3 : Gemini + RAG context — budget minimal
   │       Le prompt Gemini inclut les patterns du Knowledge Graph
   │
    ▼
4. RESOLVE_README.md
   │  Généré automatiquement (0 token) :
   │  • Liste des fichiers résolus + méthode
   │  • Patterns RAG détectés et appliqués
   │  • Instructions pour le reviewer
    │
    ▼
5. Push sur GitHub
   │  • Branche : auto-resolve/pr-{N}
   │  • Fichiers résolus pushés
   │  • RESOLVE_README.md pushé
   │  • PR auto-resolve créée → main
   │  • Commentaire sur la PR originale
```

### pr-merge-check — Vérification Merge

```
main.py pr-merge-check
    │
    ▼
merge_automation_agent.py     ← 0 token LLM (100% factuel)
    │
    ├── 1. get_pr_info() → titre, SHA
    │
    ├── 2. get_pr_mergeable_status()
    │      → Stratégie 0 (REST) ou fallback MCP
    │      → has_conflicts: true/false
    │
    ├── 3. get_check_runs() → CI/CD status
    │
    ├── 4. get_pr_reviews() → approvals / changes requested
    │
    └── 5. Verdict : PRÊTE ✓ ou PAS PRÊTE ✗
           │
           └── Poste un rapport Markdown sur la PR :
               • ✅/❌ Mergeable
               • ✅/❌ CI/CD
               • ✅/❌ Reviews
               • Verdict global
```

---

## LangGraph Multi-Agent

### WatchGraph

Orchestration LangGraph pour le mode watch:

```
WatchGraph (StateGraph)
    │
    ├── CodeAgent (parsing + filtrage)
    │   ↓
    ├── RetrieverAgent (RAG + KG + voisinage)
    │   ↓
    ├── AnalysisAgent (LLM + stratégie)
    │   ↓
    ├── LearningAgent (self-improving)
    │   ↓
    └── Output (rendering)
```

**Avantages**:
- Orchestration déclarative
- Tracing LangSmith
- État partagé entre agents
- Parallélisation automatique

### CIGraph

Orchestration LangGraph pour CI Intelligence:

```
CIGraph (StateGraph)
    │
    ├── Fetch CI Logs
    │   ↓
    ├── Index Logs (ChromaDB)
    │   ↓
    ├── Detect Patterns
    │   ↓
    ├── Correlate with Code
    │   ↓
    ├── Root Cause Analysis
    │   ↓
    └── Propose Fixes
```

### Memory

**Redis Memory**:
- Stockage état agents
- Conversation history
- Cache distribué
- Synchronisation multi-process

---

## Configuration

### Fichier config.py

Configuration centralisée avec Pydantic:

```python
class APIConfig(BaseModel):
    provider: str = "openrouter"
    openrouter_api_key: str
    openrouter_model: str = "minimax/minimax-m2.5:free"
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_tokens: int = 16384

class RAGConfig(BaseModel):
    embedding_model: str = "jinaai/jina-embeddings-v2-base-code"
    embedding_dimension: int = 768
    embedding_device: str = None  # auto-detect (cuda/mps/cpu)
    vector_store: str = "chromadb"
    distance_metric: str = "cosine"
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 8
    relevance_threshold: float = 0.45

class AnalysisConfig(BaseModel):
    supported_languages: List[str] = ["python", "javascript", "typescript", "java"]
    max_file_size_mb: int = 5
    max_code_chars: int = 10_000
    max_knowledge_chars: int = 2_000
    max_context_chars: int = 1_500
    exclude_patterns: List[str] = [...]
    analysis_depth: str = "medium"

class WatcherConfig(BaseModel):
    enabled: bool = True
    debounce_seconds: float = 4.0
    analyze_impacted: bool = True
    max_impacted_files: int = 5
    watched_extensions: List[str] = [".py", ".js", ".jsx", ".ts", ".tsx", ".java"]
    excluded_dirs: List[str] = [...]

class RedisConfig(BaseModel):
    url: str = "redis://localhost:6379/0"
    prefix: str = "ca:"  # Namespace isolation

class LangGraphConfig(BaseModel):
    enabled: bool = False
    langsmith_tracing: bool = False
    langsmith_api_key: str
    langsmith_project: str = "code-auditor"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
```

### Variables d'Environnement

Fichier `.env` (non versionné):
```
OPENROUTER_API_KEY=sk-...
OPENROUTER_MODEL=minimax/minimax-m2.5:free
GOOGLE_API_KEY=AIza...
GEMINI_MODEL=gemini-2.5-flash
REDIS_URL=redis://localhost:6379/0
USE_LANGGRAPH=false
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=code-auditor
```

---

## Limitations et Améliorations

### Problèmes Critiques Identifiés

| # | Problème | Composant | Impact |
|---|----------|-----------|--------|
| 1 | **Orchestrateur monothread** | `core/orchestrator.py` | 10 fichiers = 250s de blocage séquentiel |
| 2 | **Analyse dépendants bloquante** | `_analyze_dependents()` | Modifier GlobalConstants.java bloque plusieurs minutes |
| 3 | **Pas de filtrage token count** | `llm_service.py` | Fichier 8000 lignes = explosion coût API |
| 4 | **Latence 28.8s moyenne** | Orchestrateur + LLM | UX inacceptable pour IDE |
| 5 | **Pas de coalesce batch** | `watchers/file_watcher.py` | 20 fichiers = 20 analyses simultanées |
| 6 | **ANSI hardcodé, pas de JSON** | `output/console_renderer.py` | Incompatible plugin IDE |

### Améliorations Requises

#### 1. Architecture Asynchrone (Priorité 🔴)

**Problème**: L'Orchestrateur est synchrone et bloquant.

**Solution**:
```python
# Remplacer threading par asyncio
async def _worker_loop(self):
    while self._running:
        event = await self._priority_queue.get()
        await self._process_event_async(event)

# Analyse dépendants avec semaphore
async def _analyze_dependents(self, file_path):
    async with self._semaphore:
        await asyncio.gather(*tasks, limit=2)
```

#### 2. Lazy Loading (Priorité 🟠)

**Problème**: Initialisation au import module.

**Solution**:
```python
# Remplacer instance globale par factory
class CodeRAGSystemAPI:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

#### 3. Tests Automatisés (Priorité 🟠)

**Actions**:
- Créer dossier `tests/`
- Ajouter `pytest` + `pytest-asyncio`
- Mocks pour Gemini/ChromaDB
- Tests unitaires pour chaque agent

#### 4. API Structurée pour IDE (Priorité 🔴)

**Problème**: Sortie console ANSI uniquement.

**Solution**:
```python
# Nouveau module output/json_renderer.py
class JSONRenderer:
    def render_analysis(self, result) -> dict:
        return {
            "file": result.file_path,
            "score": result.criticality_score,
            "issues": [issue.to_dict() for issue in result.issues],
            "fixes": [fix.to_dict() for fix in result.fixes]
        }
```

#### 5. Priorisation des Événements (Priorité 🟠)

**Solution**:
```python
class EventPriority(Enum):
    CRITICAL = 0    # Bug sécurité détecté
    HIGH = 1        # Changement méthode critique
    MEDIUM = 2      # Changement standard
    LOW = 3         # Commentaire/modification mineure
```

### Feuille de Route

#### Phase 1: Fondations (Semaines 1-2)
- [ ] Ajouter suite de tests (`pytest`, mocks)
- [ ] Implémenter lazy loading pour LLM/embeddings
- [ ] Créer API JSON pour IDE

#### Phase 2: Performance (Semaines 3-4)
- [ ] Migrer Orchestrateur vers `asyncio`
- [ ] Ajouter coalesce batch au FileWatcher
- [ ] Implémenter cancellation d'événements

#### Phase 3: Robustesse (Semaines 5-6)
- [ ] Connection pool SQLite
- [ ] Cache avec TTL
- [ ] Validation multi-langages

#### Phase 4: Features (Semaines 7-8)
- [ ] Parsing incrémental Tree-sitter
- [ ] Knowledge Graph distribué
- [ ] Amélioration Web Search (cache, backoff)

### Vision Long Terme

```
┌─────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE CIBLE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐        ┌──────────────────────┐       │
│  │ ORCHESTRATOR     │        │ LANGGRAPH (PR Mode)  │       │
│  │ asyncio + Queue  │        │ Agents autonomes     │       │
│  │ < 3s latence     │        │ ~30s latence OK      │       │
│  └──────────────────┘        └──────────────────────┘       │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              API IDE (JSON/LSP)                    │    │
│  │  • DiagnosticsCollection                           │    │
│  │  • Code Actions                                     │    │
│  │  • Progress notifications                           │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Conclusion

**Code Auditor AI** est un système d'analyse de code sophistiqué qui combine:

- **RAG avancé** avec Knowledge Graph et reranking
- **Intégration Git profonde** (hooks, session tracking, branch analysis)
- **Pipeline self-improving** (Learning Agent)
- **MCP Code Mode** pour automatisation GitHub
- **CI/CD Intelligence** pour analyse des failures
- **LangGraph** pour orchestration multi-agent

**Points forts**:
- Architecture conceptuelle solide
- Pipeline RAG multi-passe sophistiqué
- Intégration Git innovante
- Agents autonomes GitHub

**Axes d'amélioration critiques**:
1. **Async/Performance**: Migrer vers `asyncio` pour réduire latence
2. **Tests**: Ajouter couverture de tests avant refactoring
3. **API IDE**: Sortie JSON structurée pour plugin VS Code
4. **Lazy Loading**: Réduire empreinte mémoire au démarrage

Le système est prêt pour une transition vers une architecture asynchrone et une intégration IDE réussie, une fois les tests ajoutés et l'API JSON implémentée.

---

*Document généré automatiquement - Analyse Système Complète*
