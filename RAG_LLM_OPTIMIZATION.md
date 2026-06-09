# Optimisation RAG + LLM — État des lieux & plan d'amélioration

> Analyse de l'intelligence déjà en place pour rendre le RAG/LLM **précis tout en
> économisant les tokens**, puis recommandations concrètes pour injecter les
> techniques manquantes.
>
> Périmètre analysé : [services/llm_service.py](services/llm_service.py),
> [services/llm_factory.py](services/llm_factory.py),
> [agents/retriever_agent.py](agents/retriever_agent.py),
> [services/cache_service.py](services/cache_service.py),
> [config.py](config.py),
> [langchain_agents/](langchain_agents/) (agents chat, inline, analysis, mémoire sémantique),
> [services/knowledge_loader.py](services/knowledge_loader.py).

---

## 1. Tableau de bord — ce qui est déjà là

| Technique | Statut | Où dans le code |
|---|---|---|
| **Reranking cross-encoder** | ✅ Implémenté | [retriever_agent.py:434](agents/retriever_agent.py#L434) (`_rerank`, ms-marco-MiniLM-L-6-v2) |
| **Multi-query retrieval** | ✅ Implémenté | [retriever_agent.py:523](agents/retriever_agent.py#L523) (`_build_queries`) |
| **Query expansion (Knowledge Graph)** | ✅ Implémenté | [retriever_agent.py:315-328](agents/retriever_agent.py#L315) (`expand_queries` depth=2, `n_hop_retrieval` depth=3) |
| **Seuil de pertinence + filtrage** | ✅ Implémenté | [llm_service.py:140](services/llm_service.py#L140), [retriever_agent.py:365](agents/retriever_agent.py#L365) |
| **Boost par langage** | ✅ Implémenté | [llm_service.py:156](services/llm_service.py#L156), [retriever_agent.py:404](agents/retriever_agent.py#L404) |
| **Déduplication des chunks** | ✅ Implémenté (exacte) | [retriever_agent.py:367-374](agents/retriever_agent.py#L367) (`seen` dict) |
| **Chunking structurel (par méthode)** | ✅ Implémenté | [llm_service.py:827](services/llm_service.py#L827) (`_chunk_code_by_methods`) |
| **Cache d'analyse (Redis, par content-hash)** | ✅ Implémenté | [cache_service.py:76-105](services/cache_service.py#L76) (`has_file_changed`, `get_cached_analysis`) |
| **Cache exact inline completion** | ✅ Implémenté | [lc_inline_completion_agent.py:167](langchain_agents/agents/lc_inline_completion_agent.py#L167) |
| **Cache RAG par fichier (anti-latence)** | ✅ Implémenté | [lc_inline_completion_agent.py:212](langchain_agents/agents/lc_inline_completion_agent.py#L212) (TTL 60s) |
| **Cascade / fallback de modèles** | ✅ Implémenté | [llm_factory.py:233](services/llm_factory.py#L233) (OpenRouter→Groq→Gemini) |
| **max_tokens par tâche** | ✅ Partiel | inline=64, fact=128, profil=200, analyse=8192/16384 |
| **Récupération conditionnelle (needs_rag)** | ✅ Implémenté | [chat_graph.py:280-295](langchain_agents/graphs/chat_graph.py#L280), decision agent |
| **Sortie structurée + plafonds** | ✅ Implémenté | [lc_analysis_agent.py:116](langchain_agents/agents/lc_analysis_agent.py#L116) (cap 20 issues / 10 fixes) |
| **Stratégie de réparation (block_fix par défaut)** | ✅ Implémenté | [llm_service.py:590-611](services/llm_service.py#L590) (évite la réécriture complète) |
| **Mémoire sémantique de session** | ✅ Implémenté | [lc_semantic_memory.py](langchain_agents/memory/lc_semantic_memory.py) |
| **Chargement de contexte parallèle** | ✅ Implémenté | [chat_graph.py:251](langchain_agents/graphs/chat_graph.py#L251) (`node_parallel_context`) |
| **Streaming des réponses** | ✅ Implémenté | [chat_graph.py:911](langchain_agents/graphs/chat_graph.py#L911) (`stream_chat`) |
| — | — | — |
| **Prompt caching (préfixe statique)** | ❌ Absent | — |
| **Recherche hybride (BM25 + vectoriel)** | ❌ Absent | — |
| **Compression contextuelle (vs troncature brute)** | ✅ Implémenté (P4) | [context_compression.py](services/context_compression.py) (NumPy natif) + câblé dans [llm_service.py](services/llm_service.py) `analyze_code_with_rag` |
| **Cache sémantique des réponses** | ❌ Absent | (seul l'inline a un cache *exact*) |
| **MMR / diversité des résultats** | ❌ Absent | dédup exacte seulement |
| **Routage par complexité (petit vs gros modèle)** | ✅ Implémenté (P2) | [config.py](config.py) `model_for_level` + [llm_factory.py](services/llm_factory.py) `build_llm_for_level` + chat agent câblé sur `context_level` |
| **Budget en tokens (vs caractères)** | ❌ Absent | `code[:10000]`, etc. |

**Lecture rapide :** le pipeline de *récupération* est déjà mûr (reranking, multi-query, KG, filtrage). Les gains restants se situent surtout côté **coût des appels LLM** (prompt caching, routage par complexité, cache sémantique) et **qualité de la compression du contexte** (hybride, compression vs troncature).

---

## 2. Détail de l'existant

### 2.1 Pipeline de récupération — `SystemAwareRAG` (déjà excellent)

Le cœur RAG est dans [agents/retriever_agent.py](agents/retriever_agent.py). C'est un pipeline **2 passes** :

**Passe 1 — large (top-20 candidats)** [retriever_agent.py:346](agents/retriever_agent.py#L346)
- Source A : queries structurelles (code brut + signatures des appelants + des dépendances)
- Source B : `KG.expand_queries(depth=2)` — patterns détectés → règles connexes
- Source C : `KG.n_hop_retrieval(depth=3)` — voisinage NetworkX + KG
- Source D : `project_code_index` — code réel du projet similaire
- Filtrage par seuil L2 (`THRESHOLD=1.2`) + déduplication (meilleur score par chunk)

**Passe 2 — précise (top-8)** [retriever_agent.py:434](agents/retriever_agent.py#L434)
- Cross-encoder `ms-marco-MiniLM-L-6-v2` (local, ~80 MB)
- Score final = `0.7 × ce_norm + 0.3 × l2_norm`
- Query de rerank enrichie des patterns KG [retriever_agent.py:494](agents/retriever_agent.py#L494)
- **Fallback automatique** en tri L2 si `sentence-transformers` absent

> 💡 C'est exactement la technique « récupérer large → reranker → garder peu » qui
> donne le meilleur rapport qualité/tokens. **Déjà en place.**

### 2.2 Budgets de tokens — par **troncature de caractères**

Les plafonds sont dans [config.py:84-89](config.py#L84) :
```
max_code_chars      = 10_000
max_knowledge_chars =  2_000
max_context_chars   =  1_500
```
Appliqués par troncature directe : `code[:max_code]`, `project_context[:max_ctx]`,
`analysis_text[:3000]`, etc. ([llm_service.py:433](services/llm_service.py#L433)).

⚠️ **Limite :** la troncature est *char-based* et aveugle — elle peut couper au milieu
d'une méthode ou d'un bloc JSON. De plus l'injection des fichiers dépendants
([llm_service.py:457-468](services/llm_service.py#L457), 3 fichiers × 2000 chars) s'ajoute
au budget **sans comptage global** → le prompt réel peut dépasser silencieusement.

### 2.3 Caching — réel mais partiel

| Cache | Type | Fichier |
|---|---|---|
| Analyse par fichier | Redis, clé = `content_hash` (skip si fichier inchangé) | [cache_service.py:76](services/cache_service.py#L76) |
| Inline completion | Redis, clé = `md5(lang+prefix+suffix)`, TTL 5 min | [lc_inline_completion_agent.py:167](langchain_agents/agents/lc_inline_completion_agent.py#L167) |
| Snippet RAG inline | mémoire, par fichier, TTL 60 s (anti-latence ChromaDB) | [lc_inline_completion_agent.py:212](langchain_agents/agents/lc_inline_completion_agent.py#L212) |
| Analyse (LangChain) | `AnalysisCacheMemory` Redis | [lc_analysis_agent.py:240](langchain_agents/agents/lc_analysis_agent.py#L240) |

✅ Le cache par content-hash évite de réanalyser un fichier non modifié — **gros gain**.
❌ Mais **aucun cache sémantique** : deux questions chat quasi identiques relancent
tout le LLM. Et le cache inline est *exact* (un caractère de différence = miss).

### 2.4 Cascade de modèles — failover, pas (encore) routage

[llm_factory.py:233](services/llm_factory.py#L233) — `invoke_with_fallback` :
`OpenRouter → Groq → Gemini`, en HTTP direct (évite le crash PyTorch Windows), avec
backoff sur quota. Les agents LangChain utilisent `RunnableWithFallbacks`
([lc_analysis_agent.py:153](langchain_agents/agents/lc_analysis_agent.py#L153)).

L'inline completion a sa **propre cascade rapide** (Groq d'abord, `max_tokens=64`)
[lc_inline_completion_agent.py:68](langchain_agents/agents/lc_inline_completion_agent.py#L68).

⚠️ C'est du **failover** (si A tombe → B), pas du **routage par complexité**
(question simple → petit modèle, question complexe → gros). Pourtant le
`decision_agent` calcule déjà un `context_level` (`fast`/`context`/`deep`)
[chat_graph.py:672](langchain_agents/graphs/chat_graph.py#L672) — le signal existe mais
n'est pas utilisé pour choisir la **taille du modèle**.

### 2.5 Contrôle de la sortie — bien fait

- **Sortie structurée** `<STRUCTURED_OUTPUT>` JSON, parsée et **plafonnée**
  (20 issues, 10 fixes) [lc_analysis_agent.py:116](langchain_agents/agents/lc_analysis_agent.py#L116).
- **Arbre de décision** `block_fix | targeted_methods | full_class` avec `block_fix`
  par défaut [llm_service.py:590](services/llm_service.py#L590) — évite de réécrire toute
  la classe (énorme économie de tokens en sortie).
- **post_solution_mode** interdit une 2ᵉ réécriture complète
  [lc_analysis_agent.py:304](langchain_agents/agents/lc_analysis_agent.py#L304).

### 2.6 Récupération conditionnelle — déjà un levier majeur

Le `decision_agent` pose des drapeaux `needs_rag / needs_git / needs_ci`
([chat_graph.py:148](langchain_agents/graphs/chat_graph.py#L148)). Si `needs_rag=False`,
la recherche ChromaDB est **sautée** ([chat_graph.py:281](langchain_agents/graphs/chat_graph.py#L281)).
Le `context_level=fast` court-circuite tout le contexte projet
([chat_graph.py:665](langchain_agents/graphs/chat_graph.py#L665)). C'est exactement
l'idée « agentic RAG : ne récupérer que si nécessaire ».

---

## 3. Ce qui manque & comment l'injecter

Classé par **rapport impact / effort**. Chaque point indique le fichier à toucher.

### 🥇 P1 — Prompt caching du préfixe statique d'analyse
**Problème.** Le prompt d'analyse ([llm_service.py:524-749](services/llm_service.py#L524))
contient un énorme bloc d'instructions **identique à chaque fichier** (RÈGLES,
LANGUAGE-SPECIFIC RULES, STEP 1/2/3, format STRUCTURED_OUTPUT) — plusieurs milliers de
tokens répétés à chaque appel.

**Comment.**
1. **Séparer** le préfixe stable (instructions) du suffixe variable (code + RAG).
   Mettre les instructions dans un **system prompt** constant, et seulement
   `code + knowledge_context + contexte` dans le message user.
2. Activer le caching selon le provider :
   - **Gemini** → *context caching* (`cached_content`) ; idéal car déjà dans la cascade.
   - **OpenRouter** → caching implicite/`cache_control` selon le modèle sous-jacent
     (Anthropic/DeepSeek le supportent).
   - **Groq** → pas de cache, mais bénéficie quand même d'un préfixe stable.

> ⚠️ Vous n'utilisez pas l'API Anthropic directement → le `cache_control: ephemeral`
> classique ne s'applique pas tel quel. Le gain vient surtout de Gemini context
> caching + d'un préfixe **stable et identique** (octet pour octet) entre appels.

**Gain.** Jusqu'à −70/90 % du coût d'entrée sur la partie instructions, **sans rien
retirer** du contexte. Le levier #1 pour un outil qui analyse beaucoup de fichiers.

### 🥈 P2 — Routage par complexité ✅ FAIT
**Problème (résolu).** Avant, tout passait par le même gros modèle, même pour une
question triviale.

**Ce qui a été implémenté.**
1. **Un seul endroit à éditer** : [config.py](config.py) → `APIConfig` expose un
   mapping `niveau → (provider, model)` (`fast_*`, `context_*`, `deep_*`) + un flag
   `route_by_complexity`. Pilotable aussi par `.env`
   (`FAST_PROVIDER`/`FAST_MODEL`/`CONTEXT_*`/`DEEP_*`, `ROUTE_BY_COMPLEXITY`).
2. **Fabrique par niveau** : [llm_factory.py](services/llm_factory.py) →
   `build_llm_for_level(level)` construit un LLM dont le **modèle primaire** dépend du
   niveau, les autres providers restant **en fallback** (fiabilité inchangée). Budget
   `max_tokens` de sortie réduit pour `fast` (1024) vs `deep` (8192).
3. **Câblage** : [lc_chat_agent.py](langchain_agents/agents/lc_chat_agent.py) →
   `_llm_for_level(context_level)` est utilisé par `answer`/`aanswer` et le mode rapide
   (`fast` → modèle rapide). Le `context_level` vient déjà du Decision Agent.

Mapping par défaut (modifiable) : `fast → groq`, `context → openrouter`, `deep → openrouter`.

**Pour changer de modèle :** éditer les valeurs dans [config.py](config.py) (ou poser
les variables `.env`). Aucune autre partie du code à toucher. Mettre
`ROUTE_BY_COMPLEXITY=false` pour revenir à « un seul modèle pour tout ».

**Gain.** Les questions simples (souvent la majorité) coûtent une fraction du prix.

### 🥉 P3 — Recherche hybride BM25 + vectoriel
**Problème.** La recherche est 100 % vectorielle. Pour du **code**, le match exact de
noms (fonctions, API, classes) est crucial et l'embedding seul le rate parfois.

**Comment.** Ajouter un `BM25Retriever` et fusionner avec le vectoriel via
`EnsembleRetriever` (LangChain), **en amont** de la passe-2 cross-encoder existante :
```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
# bm25 sur les mêmes documents que la collection Chroma
ensemble = EnsembleRetriever(retrievers=[bm25, chroma_retriever], weights=[0.4, 0.6])
```
À brancher dans `SystemAwareRAG.retrieve` ([retriever_agent.py:274](agents/retriever_agent.py#L274))
comme source supplémentaire avant le reranking (le reranker tranche ensuite).

**Gain.** Meilleur rappel sur les requêtes « par mot-clé » → moins besoin de gonfler
`top_k`, donc tokens stables et qualité en hausse.

### P4 — Compression contextuelle ✅ FAIT
**Problème (résolu).** Avant, on tronquait par caractères
([llm_service.py:433](services/llm_service.py#L433), `_build_knowledge_context`) — coupure
aveugle qui pouvait jeter l'info utile et garder des doublons.

**Ce qui a été implémenté (sans coût LLM, sans nouvelle dépendance).**
1. **Module natif** : [context_compression.py](services/context_compression.py) →
   `compress_documents()` applique deux filtres en **NumPy pur** (les packages
   `langchain` / `langchain_community` ne sont pas installés, donc implémentation
   maison via le modèle d'embeddings déjà chargé) :
   - **anti-doublons** : supprime les chunks quasi identiques (cosinus chunk↔chunk
     ≥ `redundant_threshold`, défaut 0.95) ;
   - **filtre de pertinence** : ne garde que les chunks proches de la requête
     (cosinus ≥ `compression_threshold`), **mais jamais moins de `min_keep`** docs
     (sécurité anti sur-filtrage).
2. **Câblage** : [llm_service.py](services/llm_service.py) `analyze_code_with_rag`,
   juste **après le reranking, avant** `_build_knowledge_context`.
3. **Config** : [config.py](config.py) → `RAGConfig.compression_enabled` /
   `compression_threshold` / `compression_min_keep`. Désactivable via
   `RAG_COMPRESSION=false`, seuil ajustable via `RAG_COMPRESSION_THRESHOLD`.

**Gain.** Même information utile, moins de tokens — et supprime les doublons que le
multi-query / KG fait remonter. Testé : 4 docs (1 doublon + 1 hors-sujet) → 2 docs
pertinents conservés.

### P5 — Cache sémantique des réponses
**Problème.** Seul l'inline a un cache, et il est **exact**. Les questions chat
récurrentes (« explique ce fichier ») relancent tout.

**Comment.** Vous avez déjà ChromaDB + embeddings. Avant l'appel LLM du chat :
1. embed la question (+ `target_file`),
2. chercher dans une collection `chat_response_cache`,
3. si similarité > seuil → renvoyer la réponse cachée, sinon appeler le LLM et stocker.
À brancher dans [chat_graph.py](langchain_agents/graphs/chat_graph.py) avant
`answer_question`, ou réutiliser l'infra de
[lc_semantic_memory.py](langchain_agents/memory/lc_semantic_memory.py).

**Gain.** Court-circuite des appels LLM entiers sur les questions répétées.

### P6 — MMR / diversité dans la passe-1
**Problème.** La dédup est *exacte* (clé) [retriever_agent.py:367](agents/retriever_agent.py#L367) ;
3 règles KB quasi identiques peuvent occuper le top-8.

**Comment.** Utiliser `max_marginal_relevance_search` de Chroma, ou appliquer une
pénalité de redondance avant le tri top-20 [retriever_agent.py:404](agents/retriever_agent.py#L404).

**Gain.** Plus de variété d'information dans le même budget de docs.

### P7 — Budget en **tokens** plutôt qu'en caractères
**Problème.** `code[:10000]` ≠ budget token réel ; l'ajout des fichiers dépendants
peut faire dépasser la fenêtre sans contrôle.

**Comment.** Compter avec un tokenizer (`tiktoken` ou l'API du provider) et assembler
le prompt jusqu'à un budget token global dans `_build_prompt`
([llm_service.py:396](services/llm_service.py#L396)), en priorisant : code > RAG > deps.

**Gain.** Évite les troncatures provider-side imprévues et les coupures en plein code.

---

## 4. Feuille de route suggérée

| Priorité | Action | Effort | Gain tokens | Gain qualité |
|---|---|---|---|---|
| **P1** | Prompt caching (préfixe stable + Gemini cache) | Moyen | ⭐⭐⭐ | = |
| **P2** | Routage `context_level` → modèle | Faible | ⭐⭐⭐ | = |
| **P3** | Hybride BM25 + vectoriel | Moyen | ⭐ | ⭐⭐ |
| **P4** | Compression contextuelle (embeddings) | Faible | ⭐⭐ | ⭐ |
| **P5** | Cache sémantique des réponses | Moyen | ⭐⭐ | = |
| **P6** | MMR / diversité | Faible | = | ⭐ |
| **P7** | Budget en tokens | Moyen | ⭐ | ⭐ |

**Démarrage conseillé : P2 puis P4** (faible effort, gain immédiat, zéro risque sur la
qualité), puis **P1** (le plus rentable mais demande de restructurer le prompt et de
gérer le caching par provider).

---

## 5. Pièges à éviter

- **Ne pas baisser brutalement `top_k` ou les `max_*_chars`** pour économiser : c'est
  ce qui dégrade les réponses. Préférer P3/P4/P6 qui réduisent les tokens **à qualité
  égale**.
- **Prompt caching** : le préfixe doit être **strictement identique** entre appels
  (même les espaces). Toute interpolation dynamique dans la partie « cachée » casse le
  cache.
- **Compression LLM** : éviter `LLMChainExtractor` (coûte un appel LLM par doc) ici —
  préférer les filtres *embeddings* (P4), gratuits en tokens.
- **Cache sémantique** : bien choisir le seuil de similarité, sinon risque de renvoyer
  une réponse proche mais pas exacte. Inclure `target_file`/langage dans la clé.
