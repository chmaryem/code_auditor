# Couches d'agents — `agents/` vs `langchain_agents/agents/`

Le projet a **deux** paquets d'agents. Ce n'est pas de la duplication accidentelle :
c'est une architecture en couches. Lire ceci avant d'ajouter ou de corriger un agent.

## Les deux couches

| Couche | Rôle | Dépend de LangChain ? |
|---|---|---|
| `agents/` | **Cœur métier** : la logique réelle (parsing, RAG system-aware, KG, learning, détection de gaps de test). Framework-agnostique. | Non |
| `langchain_agents/agents/` (`lc_*`) | **Adaptateurs** LangChain/LangGraph : exposent le cœur sous forme de `BaseRetriever`, `@tool`, `Runnable`, nœuds de graphe. | Oui |

## Règle d'or

- **Logique métier → `agents/`** (ou `services/`). C'est la source de vérité.
- **Intégration LangChain/LangGraph → `langchain_agents/agents/lc_*`**, qui **délègue** au cœur.
- **Corriger un bug de logique** : le faire dans `agents/` ; le wrapper `lc_*` en hérite automatiquement.
- **Ne pas réimplémenter** la logique du cœur dans un `lc_*` → ça crée de la dérive (cf. exception ci-dessous).

## Carte des `lc_*`

### Wrappers qui délèguent au cœur (✅ source de vérité unique)
| Adaptateur | Délègue à |
|---|---|
| `lc_analysis_agent` | `agents.analysis_agent` (fonctions ré-exportées) |
| `lc_code_agent` | `agents.code_agent` (via `tools/code_tools`) |
| `lc_retriever_agent` | `agents.retriever_agent` (via `tool_rag_retrieve`) |
| `lc_test_gap_agent` | `agents.test_gap_agent` |
| `lc_test_proposal_notifier` | `agents.test_proposal_notifier` |
| `lc_proactive_agent` | `agents.test_gap_agent` |

### ⚠️ Exception — réimplémentation parallèle (risque de dérive)
| Adaptateur | Concept dupliqué |
|---|---|
| `lc_learning_agent` | `agents.learning_agent` — **n'importe pas** le cœur ; il a ses propres `should_promote` / `promote_to_kb`. Un correctif de logique de promotion doit être appliqué **aux deux** jusqu'à consolidation. |

### Agents LangChain natifs (pas de doublon — aucun équivalent dans `agents/`)
`lc_chat_agent`, `lc_chat_decision_agent`, `lc_apply_agent`,
`lc_inline_completion_agent`, `lc_tool_calling_agent`,
`lc_ci_notifier`, `lc_cd_notifier`,
`lc_git_branch_agent`, `lc_git_conflict_agent`, `lc_git_decision_agent`,
`lc_git_diff_agent`, `lc_git_pr_agent`, `lc_git_session_agent`,
`lc_git_synthesis_agent`.

### Cœur sans wrapper `lc_*` (encore)
`agents/code_mode_agent`, `agents/test_generator_agent`.

## Convention d'import

Quand un `lc_*` doit ré-exporter des helpers du cœur, importer en **haut** du fichier.
Éviter l'import en bas de fichier (après le singleton) — il fonctionne mais masque la
dépendance et complique l'ordre d'import.