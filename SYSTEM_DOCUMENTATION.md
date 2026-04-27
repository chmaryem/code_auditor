# Code Auditor AI — Documentation Système Complète

> **Version** : v6.2 — RAG-Enhanced Pipeline  
> **Date** : Avril 2026  
> **Auteur** : Code Auditor Team

---

## Table des Matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture Globale](#2-architecture-globale)
3. [Commandes CLI](#3-commandes-cli)
4. [Mode Watch — Surveillance Temps Réel](#4-mode-watch--surveillance-temps-réel)
5. [Pipeline RAG (Retrieval-Augmented Generation)](#5-pipeline-rag)
6. [Knowledge Graph](#6-knowledge-graph)
7. [Self-Improving RAG (Learning Agent)](#7-self-improving-rag)
8. [Smart Git — Analyse Git Locale](#8-smart-git--analyse-git-locale)
9. [MCP Code Mode — Agents Autonomes GitHub](#9-mcp-code-mode--agents-autonomes-github)
10. [Détection de Conflits — Stratégie Multi-niveaux](#10-détection-de-conflits)
11. [Résolution de Conflits — Pipeline RAG-Enhanced](#11-résolution-de-conflits)
12. [Merge Readiness Check](#12-merge-readiness-check)
13. [Arborescence du Projet](#13-arborescence-du-projet)
14. [Configuration](#14-configuration)
15. [Limitations et Roadmap](#15-limitations-et-roadmap)

---

## 1. Vue d'ensemble

**Code Auditor AI** est un système multi-agent d'analyse de code qui combine :
- **Analyse statique** augmentée par IA (Gemini / Groq)
- **RAG** (Retrieval-Augmented Generation) avec ChromaDB + Knowledge Graph
- **Intégration Git** profonde (hooks, session tracking, branch analysis)
- **MCP Code Mode** pour l'automatisation GitHub (PR review, conflict resolution, merge check)

Le système fonctionne en 3 modes principaux :

```
┌─────────────────────────────────────────────────────────────────┐
│                    Code Auditor AI v6.2                         │
├─────────────────┬─────────────────────┬─────────────────────────┤
│  Mode Local     │  Mode Git           │  Mode MCP (GitHub)      │
│                 │                     │                         │
│  • file         │  • git              │  • pr-check             │
│  • project      │  • git-status       │  • pr-resolve           │
│  • watch        │  • git-branch       │  • pr-merge-check       │
│                 │  • hook             │                         │
│                 │  • resolve-conflicts│                         │
│                 │  • merge-hook       │                         │
└─────────────────┴─────────────────────┴─────────────────────────┘
```

---

## 2. Architecture Globale

```
┌──────────────────────────────────────────────────────────────────────┐
│                           main.py (CLI)                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │
│  │ Orchestrator  │  │  Smart Git   │  │   MCP Code Mode           │  │
│  │ (core/)       │  │ (smart_git/) │  │  (agents/ + services/)    │  │
│  │               │  │              │  │                           │  │
│  │ • FileWatcher │  │ • hooks      │  │ • CodeModeAgent           │  │
│  │ • PriorityQ   │  │ • session    │  │ • SandboxExecutor         │  │
│  │ • Debounce    │  │ • branch     │  │ • MCPGitHubService        │  │
│  │ • Cancel      │  │ • diff       │  │ • GitHubClient (wrapper)  │  │
│  └──────┬───────┘  │ • conflict   │  └────────────┬──────────────┘  │
│         │          │ • merge      │               │                  │
│         ▼          └──────────────┘               ▼                  │
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

## 3. Commandes CLI

### Mode Local

| Commande | Description | LLM ? |
|---|---|---|
| `python main.py file <fichier>` | Analyse un seul fichier avec RAG | Oui |
| `python main.py project <dossier>` | Analyse un projet complet (architecture + fichiers) | Oui |
| `python main.py watch <dossier>` | Surveillance temps réel — analyse à chaque sauvegarde | Oui |

### Mode Git Local

| Commande | Description | LLM ? |
|---|---|---|
| `python main.py git <dossier>` | Analyse les fichiers modifiés dans le dernier commit | Oui |
| `python main.py git-status <dossier>` | Score de session — bugs accumulés non commités | Non |
| `python main.py git-branch <dossier>` | Analyse une branche vs sa base avant merge | Oui |
| `python main.py hook <dossier>` | Installe/désinstalle le pre-commit hook | Non |
| `python main.py resolve-conflicts <dossier>` | Résout les conflits de merge locaux via LLM | Oui |
| `python main.py merge-hook <dossier>` | Installe le pre-merge hook (bloque si score ≥ 35) | Non |

### Mode MCP (GitHub)

| Commande | Description | LLM ? | MCP ? |
|---|---|---|---|
| `python main.py pr-check --repo owner/repo --pr N` | Revue de PR via agent autonome | Oui (Gemini) | Oui |
| `python main.py pr-resolve --repo owner/repo --pr N` | Résolution de conflits PR | Conditionnel | Oui |
| `python main.py pr-merge-check --repo owner/repo --pr N` | Vérification merge readiness | Non (0 token) | Oui |

---

## 4. Mode Watch — Surveillance Temps Réel

Le mode `watch` est le coeur du système. Il surveille un projet et analyse chaque fichier modifié en temps réel.

### Pipeline en 12 étapes

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

L'Orchestrator utilise une architecture asynchrone avancée :

- **Event Loop** : `asyncio` dans un thread daemon
- **PriorityQueue** : les événements sont priorisés (git commit > file change)
- **Debounce Coalesce** : N événements en <1s → 1 seul batch (évite les analyses redondantes)
- **Cancellation** : si un fichier est re-modifié pendant l'analyse, l'ancienne tâche est annulée
- **Parallel** : `asyncio.gather()` pour les dépendants

---

## 5. Pipeline RAG

### Composants

| Composant | Fichier | Rôle |
|---|---|---|
| **Embeddings** | Jina v2 (HuggingFace) | Vectorisation du code et des règles |
| **Vector Store** | ChromaDB | Stockage et recherche sémantique |
| **Knowledge Base** | `data/knowledge_base/` | Règles de bonnes pratiques (YAML/MD) |
| **Reranker** | Cross-Encoder | Re-classement des résultats RAG |
| **Project Code Indexer** | `services/knowledge_loader.py` | Indexe le code du projet dans ChromaDB |

### Flux RAG

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

### Fallback 429

Quand le quota Gemini est épuisé (`RESOURCE_EXHAUSTED` / `429`), le système :
1. Détecte l'erreur dans le texte retourné par le RAG
2. Active le `_StaticFallbackAnalyzer` (regex-based)
3. Retourne un score basé sur des patterns connus (SQL injection, hardcoded credentials, etc.)
4. Aucun faux négatif (score=0) n'est retourné

---

## 6. Knowledge Graph

### Architecture

Le Knowledge Graph utilise **NetworkX** pour modéliser les relations entre :

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

Le KG détecte automatiquement des patterns dans le code :

- `SqlInjectionVulnerability` — concaténation SQL
- `MissingResourceClose` — connexions JDBC non fermées
- `PlainTextPasswordComparison` — comparaison de mots de passe en clair
- `UnboundedQuery` — SELECT * sans WHERE/LIMIT
- `DefaultFieldVisibility` — champs sans modificateur d'accès
- `UnclosedResultSet` / `UnclosedJdbcResource`
- `JdbcPreparedStatementNotClosed`

---

## 7. Self-Improving RAG

### LearningAgent

Le `LearningAgent` enrichit automatiquement la Knowledge Base :

```
Analyse LLM → Fix blocks détectés
                   │
                   ▼
            record_pattern() → SQLite (mémoire épisodique)
                   │
                   ▼
            Pattern vu 3+ fois ?
                   │
            ┌──────┴──────┐
            │ OUI         │ NON
            ▼             ▼
    Promotion auto    Continuer
    → nouvelle règle    observation
      dans la KB
```

- **Mémoire épisodique** : chaque pattern détecté est enregistré dans SQLite avec son fichier, sa sévérité, et la session
- **Patterns récurrents** : si un pattern est vu ≥3 fois → alerte au développeur
- **Auto-promotion** : les patterns validés deviennent des règles permanentes dans la KB

---

## 8. Smart Git — Analyse Git Locale

### Composants

| Fichier | Rôle |
|---|---|
| `git_diff_parser.py` | Parse les diffs Git, détecte les fichiers modifiés |
| `git_branch_analyzer.py` | Analyse une branche feature vs sa base |
| `git_session_tracker.py` | Surveille l'accumulation de bugs non commités |
| `git_notifier.py` | Notifications console (alertes de score) |
| `git_commit_msg.py` | Génération de messages de commit intelligents |
| `git_hook.py` | Pre-commit hook (bloque si bugs critiques) |
| `git_merge_hook.py` | Pre-merge hook (bloque si score ≥ 35) |
| `git_conflict_resolver.py` | Résolution de conflits merge locaux via LLM |
| `git_report.py` | Rapports formatés (session, branche) |
| `conflict_context_builder.py` | Construit le contexte pour la résolution de conflits |

### Pre-commit Hook

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

Surveille en arrière-plan (thread daemon, toutes les 3 min) :

```
Session Tracker (daemon thread)
    │
    ├── Lit le cache SQLite (analyses Watch)
    ├── Calcule le score de session cumulé
    ├── Si le score monte → alerte GitNotifier
    └── git-status affiche le rapport complet
```

---

## 9. MCP Code Mode — Agents Autonomes GitHub

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

### pr-resolve — Résolution de Conflits (v6.2)

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

## 10. Détection de Conflits

### Stratégie Multi-niveaux (get_pr_mergeable_status)

Le serveur MCP npm (`@modelcontextprotocol/server-github`) ne transmet pas le champ `mergeable` de l'API GitHub REST. Pour contourner cette limitation, 5 stratégies sont implémentées :

| # | Stratégie | Source | Fiabilité | Latence |
|---|---|---|---|---|
| **0** | REST API direct (urllib) | GitHub REST API | 100% | ~1s |
| 1 | get_pull_request_status | MCP tool | Moyenne | ~2s |
| 2 | Polling get_pull_request | MCP tool × 6 | Faible | ~30s |
| 3 | Inspection patches | MCP tool | Faible* | ~3s |
| 4 | Comparaison contenu base | MCP tool | Heuristique | ~5s |

> \* La Stratégie 3 ne détecte que les conflits avec marqueurs `<<<<<<<`. La Stratégie 4 peut produire des faux positifs sur les PRs avec beaucoup de changements.

**La Stratégie 0 (REST direct)** est la plus fiable et la plus rapide. Elle est utilisée en priorité. Les stratégies 1-4 ne sont utilisées que si le token GitHub n'est pas disponible ou si l'API REST est inaccessible.

---

## 11. Résolution de Conflits

### Pipeline à 3 Niveaux

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

### RESOLVE_README.md

Chaque branche `auto-resolve/pr-{N}` contient un `RESOLVE_README.md` automatique :

```markdown
# Auto-Resolve Report — PR #7

> **PR originale** : #7 — Test analyzer conflict2
> **Branche base** : main | **Branche source** : test-analyzer-conflict2
> **Généré par** : Code Auditor v6.2 (RAG-enhanced conflict resolution)

## Fichiers résolus

| Fichier | Méthode | Status |
|---|---|---|
| UserRepository.java | 3way | Resolved |
| UserService.java | 3way | Resolved |

## Patterns détectés (Knowledge Graph)

### UserRepository.java
- SqlInjectionVulnerability
- MissingResourceClose
- UnboundedQuery

## Instructions pour le reviewer
1. Vérifier que les résolutions préservent la logique métier
2. Vérifier qu'aucune vulnérabilité n'a été réintroduite
3. Exécuter les tests unitaires avant de merger
```

---

## 12. Merge Readiness Check

Le `pr-merge-check` vérifie 3 critères sans utiliser de LLM :

| Critère | Source | Méthode |
|---|---|---|
| **Mergeable** | REST API / MCP | `get_pr_mergeable_status()` |
| **CI/CD** | MCP | `get_check_runs()` → conclusion ≠ failure |
| **Reviews** | MCP | `get_pr_reviews()` → au moins 1 APPROVED |

**Verdict** :
- ✅ PRÊTE : mergeable + CI OK + pas de changes_requested
- ⚠️ PAS PRÊTE : raisons listées (conflits, CI, reviews)

---

## 13. Arborescence du Projet

```
code_auditor/
│
├── main.py                          # Point d'entrée CLI (10 commandes)
├── config.py                        # Configuration centralisée
├── .env                             # Variables d'environnement (API keys)
├── requirements.txt                 # Dépendances Python
│
├── core/                            # Orchestration
│   ├── orchestrator.py              # Chef d'orchestre async (12 étapes)
│   ├── project_analyzer.py          # Analyse de projet complète
│   └── events.py                    # Système d'événements (FileChanged, GitCommit)
│
├── agents/                          # Agents IA
│   ├── code_agent.py                # Parsing AST + détection de changements
│   ├── analysis_agent.py            # Analyse LLM (stratégie block_fix/targeted/full_class)
│   ├── retriever_agent.py           # RAG retrieval + voisinage + reranker
│   ├── learning_agent.py            # Self-Improving RAG (auto-promotion de règles)
│   ├── code_mode_agent.py           # Agent MCP Code Mode (Gemini génère des scripts)
│   └── tools/                       # Outils des agents
│
├── services/                        # Services techniques
│   ├── llm_service.py               # LLM Service (Gemini/Groq, cascade 429)
│   ├── cache_service.py             # Cache SQLite (analyses, patterns, sessions)
│   ├── code_parser.py               # Parsing AST multi-langage
│   ├── graph_service.py             # Graphe de dépendances NetworkX
│   ├── knowledge_graph.py           # Knowledge Graph (concepts, patterns, rules)
│   ├── knowledge_loader.py          # Chargement KB + ProjectCodeIndexer (ChromaDB)
│   ├── project_indexer.py           # Index structurel du projet
│   ├── code_mode_client.py          # GitHubClient wrapper (sync → async MCP)
│   ├── mcp_github_service.py        # Client MCP GitHub (26 tools + REST fallback)
│   ├── sandbox_executor.py          # Exécution sandboxée des scripts Gemini
│   ├── feedback_processor.py        # Traitement du feedback développeur
│   └── web_search_client.py         # Recherche web (complément RAG)
│
├── smart_git/                       # Intégration Git
│   ├── git_diff_parser.py           # Parse des diffs Git
│   ├── git_branch_analyzer.py       # Analyse de branche (feature vs base)
│   ├── git_session_tracker.py       # Session Tracker (bugs accumulés)
│   ├── git_notifier.py              # Notifications console
│   ├── git_commit_msg.py            # Génération de messages de commit
│   ├── git_hook.py                  # Pre-commit hook
│   ├── git_merge_hook.py            # Pre-merge hook
│   ├── git_conflict_resolver.py     # Résolution conflits locaux
│   ├── conflict_context_builder.py  # Contexte pour résolution
│   ├── conflict_resolution_agent.py # Résolution conflits PR (v6.2 RAG-enhanced)
│   ├── merge_automation_agent.py    # Vérification merge readiness (0 token)
│   ├── pr_analyzer.py               # Routeur PR (check/resolve/merge-check)
│   ├── pr_review_agent.py           # Agent de review PR
│   └── git_report.py               # Rapports formatés
│
├── validators/                      # Validateurs de code
├── watchers/                        # FileWatcher (watchdog)
├── output/                          # Renderers console
├── data/                            # Knowledge Base (règles YAML/MD)
├── benchmarks/                      # Benchmarks de performance
└── .codeaudit/                      # Données runtime (cache, sandbox, ChromaDB)
```

---

## 14. Configuration

### Variables d'environnement (.env)

```bash
GOOGLE_API_KEY=...                    # Clé API Gemini
GITHUB_PERSONAL_ACCESS_TOKEN=...      # Token GitHub (MCP + REST)
GROQ_API_KEY=...                      # Clé API Groq (fallback LLM)
GEMINI_MODEL=gemini-2.0-flash         # Modèle par défaut
```

### Cascade LLM

Si un modèle retourne une erreur 429 (quota épuisé) :

```
gemini-2.5-flash → gemini-2.0-flash → gemini-1.5-flash → Groq Llama3.3-70B
      429?              429?               429?              429?
       │                 │                  │                  │
       └──►backoff 15s   └──►backoff 15s    └──►backoff 15s   └──►Static Fallback
```

---

## 15. Limitations et Roadmap

### Limitations actuelles

| Limitation | Impact | Contournement |
|---|---|---|
| Quota Gemini Free Tier | Analyse tronquée si quota épuisé | Cascade LLM + Static Fallback |
| MCP ne transmet pas `mergeable` | Nécessite REST fallback | Stratégie 0 (REST direct) |
| `get_file_content` tronque à 8000 chars | Grands fichiers partiellement analysés | Budget token optimization |
| Résolution conflits limitée à Java/Python | Autres langages non supportés pour merge conservateur | Fallback OURS |
| Knowledge Graph statique au démarrage | Ne détecte pas les nouveaux patterns en cours de session | KG update incrémental (étape 4.6) |

### Roadmap

- [ ] **MCP Code Mode pour les 3 commandes PR** : réunifier sous CodeModeAgent
- [ ] **Tests automatisés** : suite de tests pour chaque stratégie de détection
- [ ] **Support multi-langage** : étendre le merge conservateur à JS/TS/Python
- [ ] **Dashboard web** : visualisation des analyses et du Knowledge Graph
- [ ] **GitHub Actions** : intégration CI/CD native

---

*Documentation générée pour Code Auditor AI v6.2 — Avril 2026*
