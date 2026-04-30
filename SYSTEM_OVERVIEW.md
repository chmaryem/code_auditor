# Code Auditor AI - Vue d'Ensemble Complète

> **Version**: v6.2 - RAG-Enhanced Pipeline  
> **Date**: Avril 2026  
> **Projet**: Analyse intelligente de code avec IA multi-modale

---

## Table des Matières

1. [Vue d'Ensemble](#1-vue-densemble)
2. [Fonctionnalités Principales](#2-fonctionnalités-principales)
3. [Architecture & Workflows](#3-architecture--workflows)
4. [Problèmes Identifiés](#4-problèmes-identifiés)
5. [Améliorations Requises](#5-améliorations-requises)
6. [Feuille de Route](#6-feuille-de-route)

---

## 1. Vue d'Ensemble

**Code Auditor AI** est un système d'analyse de code qui combine l'intelligence artificielle (Gemini/Groq), le RAG (Retrieval-Augmented Generation) et l'intégration Git profonde pour fournir une analyse continue et intelligente du code source.

### 3 Modes de Fonctionnement

```
┌──────────────────────────────────────────────────────────────┐
│                  Code Auditor AI v6.2                        │
├─────────────────┬─────────────────────┬────────────────────────┤
│  Mode Local     │  Mode Git           │  Mode MCP (GitHub)     │
│                 │                     │                        │
│  • file         │  • git              │  • pr-check            │
│  • project      │  • git-status       │  • pr-resolve          │
│  • watch        │  • git-branch       │  • pr-merge-check      │
│                 │  • hook             │  • ci-deploy           │
│                 │  • resolve-conflicts │                        │
│                 │  • merge-hook       │                        │
└─────────────────┴─────────────────────┴────────────────────────┘
```

### Stack Technique

| Composant | Technologie |
|-----------|-------------|
| LLM | Gemini (Google), Groq via OpenRouter |
| Embeddings | Jina v2 (768 dims) |
| Vector Store | ChromaDB |
| Knowledge Graph | NetworkX |
| Cache | SQLite |
| Parsing | Tree-sitter (Python, Java, JS/TS) |
| MCP | @modelcontextprotocol/server-github |

---

## 2. Fonctionnalités Principales

### 2.1 Analyse de Code avec RAG

- **Vectorisation** du code source avec Jina embeddings
- **Recherche sémantique** dans la base de connaissances (best practices)
- **Reranking** avec Cross-Encoder pour affiner les résultats
- **Détection de patterns** via Knowledge Graph (SQL injection, ressources non fermées, etc.)

### 2.2 Mode Watch (Surveillance Temps Réel)

```
Fichier sauvegardé
    │
    ├─→ Hash Check (skip si identique)
    ├─→ Filtre intelligent (score de changement)
    ├─→ Parsing AST (Tree-sitter)
    ├─→ Indexation ChromaDB
    ├─→ Mise à jour Knowledge Graph
    ├─→ Calcul du voisinage (dépendances)
    ├─→ RAG retrieval + reranking
    ├─→ Analyse LLM (Gemini/Groq)
    ├─→ Cache SQLite
    └─→ Analyse proactive des dépendants
```

### 2.3 Smart Git Integration

| Commande | Description |
|----------|-------------|
| `git` | Analyse les fichiers modifiés dans un commit |
| `git-status` | Score de session (bugs accumulés non commités) |
| `git-branch` | Analyse d'une branche vs sa base avant merge |
| `hook` | Pre-commit hook (bloque si score ≥ 35) |
| `resolve-conflicts` | Résolution automatique des conflits via LLM |
| `merge-hook` | Pre-merge hook (bloque si code critique) |

### 2.4 MCP Code Mode (Agents GitHub)

- **pr-check**: Revue automatique de Pull Request
- **pr-resolve**: Résolution des conflits de merge PR
- **pr-merge-check**: Vérification de readiness (0 token LLM)
- **ci-deploy**: Déploiement automatique de workflows CI/CD

### 2.5 Self-Improving RAG (Learning Agent)

- **Mémoire épisodique**: Patterns détectés enregistrés dans SQLite
- **Auto-promotion**: Patterns vus 3+ fois → nouvelle règle KB
- **Feedback processing**: Enrichissement continu de la base de connaissances

---

## 3. Architecture & Workflows

### 3.1 Structure du Projet

```
code_auditor/
├── main.py                    # Point d'entrée CLI (10 commandes)
├── config.py                  # Configuration centralisée (Pydantic)
├── core/
│   ├── orchestrator.py        # Orchestration async (PriorityQueue, debounce)
│   ├── project_analyzer.py    # Analyse projet complet
│   └── events.py              # Système d'événements
├── agents/
│   ├── code_agent.py          # Parsing et filtrage
│   ├── analysis_agent.py      # Appel LLM avec stratégie
│   ├── retriever_agent.py     # RAG retrieval + KG
│   ├── learning_agent.py      # Self-improving RAG
│   └── code_mode_agent.py     # Génération de scripts MCP
├── services/
│   ├── llm_service.py         # Client Gemini/Groq
│   ├── knowledge_graph.py     # Knowledge Graph (NetworkX)
│   ├── knowledge_loader.py    # ChromaDB ingestion
│   ├── cache_service.py       # Cache SQLite
│   ├── code_parser.py         # Tree-sitter parsing
│   ├── graph_service.py       # Analyse de dépendances
│   └── mcp_github_service.py  # Client MCP GitHub
├── smart_git/
│   ├── git_diff_parser.py     # Parse des diffs Git
│   ├── git_session_tracker.py # Surveillance bugs non commités
│   ├── git_branch_analyzer.py # Analyse branche feature
│   ├── git_conflict_resolver.py # Résolution conflits locaux
│   ├── conflict_resolution_agent.py # Résolution conflits PR
│   └── git_hook.py            # Pre-commit / pre-merge hooks
└── watchers/
    └── file_watcher.py        # Surveillance fichiers (watchdog)
```

### 3.2 Pipeline d'Analyse (12 Étapes)

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE WATCH                          │
├─────────────────────────────────────────────────────────────┤
│ 1. Hash Check           │ Skip si fichier inchangé          │
│ 2. Lecture fichier      │ Chargement UTF-8                  │
│ 3. Filtre intelligent   │ Score changement (0-100)          │
│ 4. Parsing AST          │ Tree-sitter extraction            │
│ 4.5. Index ChromaDB     │ Vectorisation code                │
│ 4.6. Update KG          │ Mise à jour incrémentale          │
│ 5. Graphe dépendances   │ NetworkX update                   │
│ 6. Voisinage            │ Prédécesseurs/successeurs         │
│ 7. SystemAwareRAG       │ Retrieval + reranking             │
│ 8. Contexte enrichi     │ Assembly contexte complet         │
│ 9. LLM Analysis         │ Gemini/Groq + stratégie           │
│ 10. Cache SQLite        │ Sauvegarde analyse                │
│ 11. Self-Improving      │ LearningAgent patterns            │
│ 12. Analyse dépendants  │ asyncio.gather(max 2)             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Résolution de Conflits (3 Niveaux)

```
┌────────────────────────────────────────────────────────────┐
│ Niveau 1: 3-way Merge Déterministe (difflib)              │
│ • 0 token LLM                                             │
│ • Couvre ~70% des conflits                                │
└──────────────────────┬─────────────────────────────────────┘
                       │ échec
                       ▼
┌────────────────────────────────────────────────────────────┐
│ Niveau 2: Merge Conservateur                              │
│ • 0 token LLM                                             │
│ • Garde OURS + nouvelles méthodes THEIRS                  │
│ • Couvre ~20% supplémentaires                             │
└──────────────────────┬─────────────────────────────────────┘
                       │ échec
                       ▼
┌────────────────────────────────────────────────────────────┐
│ Niveau 3: Gemini + RAG Context                            │
│ • ~200-500 tokens input                                   │
│ • Fallback ultime: OURS                                   │
└────────────────────────────────────────────────────────────┘
```

### 3.4 Configuration (config.py)

```python
# Clés API
OPENROUTER_API_KEY      # Primary provider
GOOGLE_API_KEY          # Fallback Gemini

# RAG Settings
embedding_model: jinaai/jina-embeddings-v2-base-code
embedding_dimension: 768
top_k: 8
relevance_threshold: 0.45

# Analysis Limits
max_file_size_mb: 5
max_code_chars: 10_000
supported_languages: [python, javascript, typescript, java]

# Watcher
debounce_seconds: 4.0
max_impacted_files: 5
```

---

## 4. Problèmes Identifiés

### 4.1 Problèmes Critiques (🔴)

| # | Problème | Composant | Impact |
|---|----------|-----------|--------|
| 1 | **Orchestrateur monothread** | `core/orchestrator.py` | 10 fichiers = 250s de blocage séquentiel |
| 2 | **Analyse dépendants bloquante** | `_analyze_dependents()` | Modifier GlobalConstants.java bloque plusieurs minutes |
| 3 | **Pas de filtrage token count** | `llm_service.py` | Fichier 8000 lignes = explosion coût API |
| 4 | **Latence 28.8s moyenne** | Orchestrateur + LLM | UX inacceptable pour IDE |
| 5 | **Pas de coalesce batch** | `watchers/file_watcher.py` | 20 fichiers = 20 analyses simultanées |
| 6 | **ANSI hardcodé, pas de JSON** | `output/console_renderer.py` | Incompatible plugin IDE |

### 4.2 Problèmes Haute Priorité (🟠)

| # | Problème | Composant | Impact |
|---|----------|-----------|--------|
| 7 | **Pas de priorité ni cancellation** | `core/events.py` | Ressources gaspillées |
| 8 | **Init au import, pas lazy** | `llm_service.py:942` | 2-3 Go RAM au démarrage |
| 9 | **Zéro test automatisé** | Tout le système | Régressions silencieuses |
| 10 | **Lock global SQLite** | `cache_service.py` | Contention threads |
| 11 | **Stdin bloquant** | `feedback_processor.py` | Incompatible IDE |

### 4.3 Problèmes Moyenne Priorité (🟡)

| # | Problème | Composant | Impact |
|---|----------|-----------|--------|
| 12 | **Cycles O(N!)** | `graph_service.py` | Faux positifs sur gros projets |
| 13 | **Full-rebuild only** | `knowledge_loader.py` | Lenteur rechargement KB |
| 14 | **Chunks orphelins** | `ProjectCodeIndexer` | Résultats obsolètes |
| 15 | **Event loop hack** | `web_search_client.py` | Fuites threads Windows |
| 16 | **Score regex fragile** | `git_hook.py` | Faux scores |

### 4.4 Problèmes Faible Priorité (🔵)

| # | Problème | Composant | Impact |
|---|----------|-----------|--------|
| 17 | **Parsing non-incrémental** | `code_parser.py` | Overhead parsing |
| 18 | **RAM O(N) Knowledge Graph** | `knowledge_graph.py` | Lenteur grosses KB |

---

## 5. Améliorations Requises

### 5.1 Architecture Asynchrone (Priorité 🔴)

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

### 5.2 Lazy Loading (Priorité 🟠)

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

### 5.3 Tests Automatisés (Priorité 🟠)

**Actions**:
- Créer dossier `tests/`
- Ajouter `pytest` + `pytest-asyncio`
- Mocks pour Gemini/ChromaDB
- Tests unitaires pour chaque agent

### 5.4 API Structurée pour IDE (Priorité 🔴)

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

### 5.5 Priorisation des Événements (Priorité 🟠)

**Solution**:
```python
class EventPriority(Enum):
    CRITICAL = 0    # Bug sécurité détecté
    HIGH = 1        # Changement méthode critique
    MEDIUM = 2      # Changement standard
    LOW = 3         # Commentaire/modification mineure
```

### 5.6 Cache Amélioré (Priorité 🟠)

**Actions**:
- Remplacer Lock global par connection pool
- Ajouter TTL (Time To Live) aux entrées
- Hash incrémental (ne pas relire fichier entier)

### 5.7 Validation Multi-Langages (Priorité 🟡)

**Problème**: Validation syntaxique Python-only.

**Solution**:
```python
# Étendre validators/fix_validator.py
class MultiLanguageValidator:
    def validate(self, code: str, language: str) -> bool:
        validators = {
            "python": self._validate_python,
            "java": self._validate_java,
            "javascript": self._validate_javascript,
        }
        return validators[language](code)
```

---

## 6. Feuille de Route

### Phase 1: Fondations (Semaines 1-2)

- [ ] Ajouter suite de tests (`pytest`, mocks)
- [ ] Implémenter lazy loading pour LLM/embeddings
- [ ] Créer API JSON pour IDE

### Phase 2: Performance (Semaines 3-4)

- [ ] Migrer Orchestrateur vers `asyncio`
- [ ] Ajouter coalesce batch au FileWatcher
- [ ] Implémenter cancellation d'événements

### Phase 3: Robustesse (Semaines 5-6)

- [ ] Connection pool SQLite
- [ ] Cache avec TTL
- [ ] Validation multi-langages

### Phase 4: Features (Semaines 7-8)

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

## Résumé Exécutif

**Points Forts**:
- Architecture RAG avancée (multi-passe, KG, reranking)
- Intégration Git profonde (hooks, session tracking, branch analysis)
- Pipeline self-improving (Learning Agent)
- MCP Code Mode pour automatisation GitHub

**Axes d'Amélioration Critiques**:
1. **Async/Performance**: Migrer vers `asyncio` pour réduire latence
2. **Tests**: Ajouter couverture de tests avant refactoring
3. **API IDE**: Sortie JSON structurée pour plugin VS Code
4. **Lazy Loading**: Réduire empreinte mémoire au démarrage

**Verdict**: Le système est conceptuellement solide avec un RAG sophistiqué et une intégration Git innovante. La transition vers une architecture asynchrone et l'ajout de tests sont prérequis pour une intégration IDE réussie.

---

*Document généré automatiquement - Code Auditor AI Analysis*
