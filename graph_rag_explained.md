# Graph RAG dans Code Auditor AI — Explication Complète

---

## 1. RAG Classique vs. Graph RAG

### RAG Classique (ce que 90% des projets font)

```
Code modifié → Embedding → Recherche ChromaDB → Top-5 docs → LLM
```

C'est une recherche **textuelle** : on cherche les documents dont le texte ressemble le plus au code. Problème : si le code contient un `executeQuery()`, le RAG trouve les règles qui mentionnent `executeQuery` — mais il **rate** les règles liées (Resource Leak, Connection Pool) parce que ces mots ne sont pas dans le code.

### Graph RAG (ce que votre système fait)

```
Code modifié → Embedding → Recherche ChromaDB
                ↓
             + Knowledge Graph (traversal N-hop)
                ↓
             + Graphe de dépendances (fichiers impactés)
                ↓
             + Code projet similaire (même package)
                ↓
             Fusion → Reranking cross-encoder → Top-8 docs → LLM
```

C'est une recherche **sémantique + structurelle**. Le graphe permet de trouver des documents que la recherche textuelle ne trouverait **jamais**.

---

## 2. Les 3 Graphes du Système

Votre système utilise **3 structures de graphe** qui travaillent ensemble :

```
┌────────────────────────────────────────────────────────────────┐
│                     3 GRAPHES                                  │
│                                                                │
│  ┌──────────────────────┐                                      │
│  │ GRAPHE 1 : NetworkX  │  Qui importe qui ?                   │
│  │ (Dépendances)        │  UserController → UserService → User │
│  │ graph_service.py     │  "Si je casse User, qui est impacté?"│
│  └──────────┬───────────┘                                      │
│             │                                                  │
│  ┌──────────▼───────────┐                                      │
│  │ GRAPHE 2 : KG        │  Quel concept ↔ quelle règle ?       │
│  │ (Knowledge Graph)    │  login() → Authentication → OWASP    │
│  │ knowledge_graph.py   │  "Quelles règles de sécurité          │
│  └──────────┬───────────┘   s'appliquent à ce code ?"           │
│             │                                                  │
│  ┌──────────▼───────────┐                                      │
│  │ GRAPHE 3 : ChromaDB  │  Quel texte ↔ quel texte ?           │
│  │ (Vector Store)       │  Recherche par similarité sémantique  │
│  │ Embeddings Jina      │  "Trouvez les règles similaires"      │
│  └──────────────────────┘                                      │
└────────────────────────────────────────────────────────────────┘
```

### Comment ils s'interconnectent

```
  Auth.java modifié
       │
       ▼
  GRAPHE 1 (NetworkX) : "Qui dépend de Auth.java ?"
  → UserController.java, LoginService.java
       │
       ▼
  GRAPHE 2 (KG) : "Quels concepts dans Auth.java ?"
  → Auth.java contient authenticate()
  → authenticate HANDLES Authentication
  → Authentication IS_CONCEPT_OF Security
  → Security connecté à : OWASP_Auth, Password_Storage, Session_Management
       │
       ▼
  GRAPHE 3 (ChromaDB) : "Trouvez les règles pour Security + Authentication"
  → sql_injection.md (score 0.85)
  → password_hashing.md (score 0.91)
  → session_management.md (score 0.78)
```

---

## 3. Le Knowledge Graph — Construction Automatique

Le KG se construit **tout seul** depuis 3 sources. On n'écrit jamais le graphe à la main.

### Source 1 — Front-matter des fichiers .md (Knowledge Base)

Chaque règle `.md` déclare ses nœuds et relations KG :

```markdown
# sql_injection.md

---
kg_nodes:
  - name: SQL_Injection
    type: vulnerability
    severity: CRITICAL
    languages: [java, python]
    kb_queries:
      java: "SQL injection PreparedStatement JDBC"
      python: "SQL injection cursor parameterized"

  - name: PreparedStatement
    type: fix

kg_relations:
  - [SQL_Injection, FIXED_BY, PreparedStatement]
  - [SQL_Injection, OFTEN_WITH, Resource_Leak]

pattern_map:
  java:
    "executeQuery": SQL_Injection
    "Statement":    SQL_Injection
  python:
    "cursor.execute": SQL_Injection
---

## Description
L'injection SQL se produit quand...
```

**Ce que le KG crée** :
```
  ┌─────────────┐  FIXED_BY   ┌───────────────────┐
  │SQL_Injection │────────────▶│PreparedStatement   │
  │(vulnerability│             │(fix)               │
  │ CRITICAL)    │             └───────────────────┘
  └──────┬───────┘
         │ OFTEN_WITH
         ▼
  ┌──────────────┐
  │Resource_Leak │
  │(vulnerability│
  │ HIGH)        │
  └──────────────┘
```

### Source 2 — AST du code projet (via code_parser.py)

Le KG parse le code du projet et crée des nœuds pour chaque classe et méthode :

```java
// Auth.java
public class TokenManager {
    public boolean authenticate(String username, String password) { ... }
    public void logout(String token) { ... }
}
```

**Ce que le KG crée** :
```
  ┌───────────┐ CONTAINS  ┌─────────────────────┐
  │ Auth.java │──────────▶│ Auth::TokenManager   │
  │ (file)    │           │ (entity_class)       │
  └───────────┘           └──────┬──────┬────────┘
                                 │      │
                          HAS_METHOD  HAS_METHOD
                                 │      │
                                 ▼      ▼
                    ┌──────────────┐  ┌──────────┐
                    │ authenticate │  │ logout   │
                    │ (entity_     │  │ (entity_ │
                    │  method)     │  │  method) │
                    └──────────────┘  └──────────┘
```

### Source 3 — Liaison sémantique (heuristiques + LLM)

Le KG analyse les noms des méthodes et les relie aux concepts de sécurité :

```
authenticate(username, password)
  → HANDLES → "Authentication"     (heuristique : nom de méthode)
  
"Authentication"
  → IS_CONCEPT_OF → "Security"     (heuristique : catégorie)
  
"Security"
  → les règles .md tagguées Security sont accessibles
```

### Le KG complet (les 3 sources fusionnées)

```
  ┌───────────┐         ┌──────────────────┐        ┌──────────────┐
  │ Auth.java │─CONTAINS─▶│ TokenManager     │─HAS_METHOD─▶│ authenticate │
  └───────────┘         └──────────────────┘        └──────┬───────┘
                                                           │
                                                      HANDLES
                                                           │
                                                    ┌──────▼───────┐
                              ┌─────────────────────│Authentication│
                              │                     └──────────────┘
                         IS_CONCEPT_OF
                              │
                        ┌─────▼────┐
                        │ Security │
                        └──┬───┬───┘
                           │   │
              ┌────────────┘   └────────────┐
              ▼                             ▼
     ┌─────────────┐               ┌────────────────┐
     │SQL_Injection│──FIXED_BY──▶  │PreparedStatement│
     │ (CRITICAL)  │               └────────────────┘  
     └──────┬──────┘
            │ OFTEN_WITH
            ▼
     ┌──────────────┐
     │Resource_Leak │
     │ (HIGH)       │
     └──────────────┘
```

---

## 4. Le Pipeline Graph RAG — 2 Passes

Voici exactement ce qui se passe dans `RetrieverAgent.retrieve()` :

### Passe 1 — Collecte large (4 sources → 20 candidats)

```
SOURCE A : Queries structurelles
│  → Le code brut du fichier modifié
│  → Les signatures des fichiers appelants (predecessors)
│  → Les noms des dépendances (successors)
│  → Chaque query = une recherche ChromaDB
│
SOURCE B : KG expand_queries (depth=2)
│  → Détecte les patterns dans le code (executeQuery → SQL_Injection)
│  → Parcourt le KG à 2 niveaux de profondeur
│  → SQL_Injection → FIXED_BY → PreparedStatement
│  → Génère des queries ChromaDB ciblées :
│    "SQL injection PreparedStatement JDBC java"
│
SOURCE C : KG n_hop_retrieval (depth=3)
│  → Part du fichier modifié dans NetworkX
│  → Traverse 3 niveaux : fichier → entité → concept → règle
│  → Auth.java → authenticate → Authentication → Security
│  → Génère des queries pour les règles Security
│
SOURCE D : Project Code Index
│  → Cherche du code SIMILAIRE dans le projet
│  → "Y a-t-il d'autres fichiers avec executeQuery ?"
│  → Trouve UserRepository.java (même pattern)
│
└──→ FUSION : union de toutes les sources
     Déduplication par clé unique (source_file + chunk_index)
     Tri : boost langage > boost KB > score L2
     → TOP 20 candidats
```

### Passe 2 — Reranking précis (20 → 8 documents)

```
Cross-encoder (ms-marco-MiniLM-L-6-v2)
│
│  Pour chaque candidat :
│    Query enrichie = code modifié + patterns KG détectés
│    Score CE = cross-encoder.predict(query, document)
│    Score L2 = distance ChromaDB (inversée)
│    Score final = 0.7 × CE + 0.3 × L2
│
│  Tri par score final décroissant
│  → TOP 8 documents les plus pertinents
│
└──→ Ces 8 docs sont injectés dans le prompt LLM
```

### Pourquoi 2 passes ?

| | Passe 1 (ChromaDB) | Passe 2 (Cross-encoder) |
|---|---|---|
| **Vitesse** | ~20ms | ~200ms |
| **Précision** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Rôle** | Ratisser large | Trier finement |
| **Volume** | 20 candidats | 8 retenus |

La passe 1 est comme un filet de pêche (attrape beaucoup). La passe 2 est comme un expert qui trie les poissons (garde les meilleurs).

---

## 5. Scénario Réel : Auth.java Modifié

```
Le développeur modifie Auth.java :
  + public boolean authenticate(String user, String pass) {
  +     String query = "SELECT * FROM users WHERE name='" + user + "'";
  +     ResultSet rs = stmt.executeQuery(query);  // ← SQL Injection !

═══════════════════════════════════════════════════════════════

ÉTAPE 1 — Détection des patterns (KG)
  Code contient "executeQuery" + "ResultSet"
  pattern_map → SQL_Injection, Resource_Leak
  → Le KG sait AVANT le LLM qu'il y a un risque

ÉTAPE 2 — Expansion KG (depth=2)
  SQL_Injection → FIXED_BY → PreparedStatement
  SQL_Injection → OFTEN_WITH → Resource_Leak
  Resource_Leak → FIXED_BY → TryWithResources
  → Queries : "SQL injection PreparedStatement JDBC"
               "Resource leak try-with-resources ResultSet"

ÉTAPE 3 — N-hop (depth=3)
  Auth.java → authenticate → Authentication → Security
  → Query : "Authentication Security best practices"
  NetworkX : Auth.java ← UserController.java (predecessor)
  → Le LLM saura que UserController sera impacté

ÉTAPE 4 — ChromaDB + Reranking
  20 documents trouvés → cross-encoder → 8 retenus :
    1. sql_injection.md (score 0.95)         ← Source B (KG expand)
    2. resource_leak.md (score 0.88)         ← Source B (KG expand)
    3. prepared_statement_examples.md (0.85) ← Source A (texte)
    4. authentication_owasp.md (0.82)        ← Source C (N-hop !)
    5. UserRepository.java (0.79)            ← Source D (code projet)
    6. connection_pool.md (0.75)             ← Source C (N-hop !)
    7. password_hashing.md (0.71)            ← Source C (N-hop !)
    8. input_validation.md (0.68)            ← Source A (texte)

ÉTAPE 5 — Le LLM reçoit tout
  Prompt :
    [CODE] Auth.java + les modifications
    [IMPACT] UserController.java dépend de ce fichier
    [RÈGLES] 8 documents RAG dont OWASP Auth et Resource Leak
    [PATTERNS KG] SQL_Injection détecté, Resource_Leak détecté
    
  → Le LLM produit : 1 CRITICAL (SQL Injection), 1 HIGH (Resource Leak)
  → Il suggère PreparedStatement ET try-with-resources
  → Il prévient que UserController.java doit être vérifié

═══════════════════════════════════════════════════════════════

SANS Graph RAG, le LLM aurait trouvé :
  ✅ SQL Injection (le mot "executeQuery" suffit)
  ❌ Resource Leak (pas dans le code, trouvé par KG OFTEN_WITH)
  ❌ OWASP Authentication (trouvé par N-hop 3 niveaux)
  ❌ Impact sur UserController (trouvé par NetworkX)
```

---

## 6. Le Self-Improving Graph RAG

Votre système **apprend et enrichit** le Graph RAG automatiquement :

```
  Développeur corrige un bug
       │
       ▼
  LearningAgent observe la correction
       │
       ▼
  LLM généralise : "Ce pattern est un bug récurrent"
       │
       ▼
  Nouvelle règle .md créée dans auto_learned/
       │
       ├──▶ ChromaDB : rechargé (nouveaux embeddings)
       │
       └──▶ Knowledge Graph : mis à jour
            (nouveaux nœuds, nouvelles relations)
       │
       ▼
  Prochaine analyse : le Graph RAG trouve
  cette nouvelle règle via KG traversal
```

La boucle est **fermée** : détection → correction → apprentissage → meilleure détection.

---

## 7. Ce que c'est vs. ce que ce n'est pas

| | Ce que c'est | Ce que ce n'est pas |
|---|---|---|
| **Type** | Graph RAG hybride (KG + Vector + Graphe code) | Pas un RAG classique simple |
| **KG** | NetworkX automatisé (pas Neo4j) | Pas une base de données graphe lourde |
| **Traversal** | N-hop heuristique (depth=3) | Pas du GraphQL ou SPARQL |
| **Reranking** | Cross-encoder local (80 MB) | Pas un LLM comme reranker |
| **Auto-construction** | 3 sources (KB + AST + sémantique) | Pas de construction manuelle |
| **Self-improving** | Feedback → nouvelle règle → KG mis à jour | Pas du fine-tuning LLM |

### Pour votre encadrante, en une phrase :

> *"Notre système implémente un Graph RAG qui combine un Knowledge Graph auto-construit (vulnérabilités + entités code + concepts), un graphe de dépendances NetworkX, et un vector store ChromaDB. Le traversal N-hop à 3 niveaux de profondeur permet de trouver des règles de sécurité sémantiquement liées que la recherche vectorielle seule ne trouverait jamais — et le système s'auto-améliore en apprenant des corrections du développeur."*
