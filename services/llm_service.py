from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import config

load_dotenv()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE GRAPH — Phase 1
#
# Remplace _SECURITY_PATTERNS hardcodé par le KnowledgeGraph.
#
# Avant : dict Python statique dans ce fichier
#         → impossible à modifier sans toucher au code
#         → retourne juste True/False (code risqué ou non)
#
# Après : KnowledgeGraph chargé depuis data/knowledge_graph.json
#         → modifiable via les fichiers .md et PATTERN_TO_KG_NODE
#         → retourne les NŒUDS KG précis (SQL_Injection, Resource_Leak...)
#         → ces nœuds seront utilisés pour enrichir les requêtes RAG
# ─────────────────────────────────────────────────────────────────────────────

from services.knowledge_graph import knowledge_graph


def _has_security_patterns(code: str, language: str) -> bool:
    """
    Retourne True si le code contient des patterns à risque.
    Délègue au KnowledgeGraph (remplace l'ancien dict hardcodé).
    """
    return knowledge_graph.has_pattern(code, language)


# ─────────────────────────────────────────────────────────────────────────────
# CodeRAGSystemAPI
# ─────────────────────────────────────────────────────────────────────────────

class CodeRAGSystemAPI:

    def __init__(self) -> None:
        self.embeddings:   HuggingFaceEmbeddings  | None = None
        self.vector_store: Chroma                 | None = None
        self.llm:          ChatGoogleGenerativeAI | None = None
        self._initialize()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _initialize(self) -> None:
        """Initialise les 3 composants : Embeddings → ChromaDB → Google Gemini."""
        logger.info("Initialisation CodeRAGSystemAPI...")

        # 1. Embeddings Jina v2
        logger.info("Chargement embeddings : %s (device=%s)",
                    config.rag.embedding_model, config.rag.embedding_device)
        self.embeddings = HuggingFaceEmbeddings(
            model_name   = config.rag.embedding_model,
            model_kwargs = {
                "device":            config.rag.embedding_device,
                "trust_remote_code": True,
            },
            encode_kwargs = {
                "normalize_embeddings": True,
                "batch_size":           32,
            },
        )

        # 2. ChromaDB
        self.vector_store = Chroma(
            persist_directory  = str(config.VECTOR_STORE_DIR),
            embedding_function = self.embeddings,
            collection_name    = config.CHROMA_COLLECTION,
        )

        chunk_count = self.vector_store._collection.count()
        if chunk_count == 0:
            logger.warning(
                "Collection ChromaDB '%s' est vide ! "
                "Lancez : python knowledge_loader.py",
                config.CHROMA_COLLECTION,
            )
            print("\n⚠️  ATTENTION : Knowledge Base vide !")
            print("   Lancez : python knowledge_loader.py")
            print("   (sans ça, le RAG ne trouvera aucune règle)\n")
        elif chunk_count < 50:
            # KB anormalement petite — probablement seules les règles auto_learned
            # sont présentes, la KB manuelle (java/, python/...) est absente.
            logger.warning(
                "Collection ChromaDB '%s' : seulement %d chunks — KB incomplète ! "
                "Les règles manuelles sont absentes. Lancez : python knowledge_loader.py",
                config.CHROMA_COLLECTION, chunk_count,
            )
            print(f"\n⚠️  KB INCOMPLÈTE : {chunk_count} chunks seulement (normal = 200+)")
            print("   Les règles manuelles sont absentes du vector store.")
            print("   Lancez : python knowledge_loader.py  pour restaurer la KB complète")
            print("   (le système fonctionne mais le RAG sera dégradé)\n")
        else:
            logger.info("Collection '%s' : %d chunks disponibles",
                        config.CHROMA_COLLECTION, chunk_count)

        # 3. Google Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY non défini — les analyses LLM échoueront")

        self.llm = ChatGoogleGenerativeAI(
            model                           = config.api.model,
            temperature                     = config.api.temperature,
            google_api_key                  = api_key or "placeholder",
            max_output_tokens               = config.api.max_tokens,
            convert_system_message_to_human = True,
        )

        logger.info("Google Gemini initialisé ✓")

    #  Retrieval RAG 

    def _retrieve_relevant_knowledge(
        self,
        query: str,
        language: str,
        k: int | None = None,
    ) -> tuple[list[Document], list[float]]:
        """
        Recherche et filtre les documents pertinents de la knowledge base.

        Stratégie :
          1. Recherche vectorielle large (top_k x 2)
          2. Filtre par seuil cosinus (score > 0.75 = ignoré)
          3. Boost langage : documents du même langage en premier
        """
        search_k  = (k or config.rag.top_k) * 2
        threshold = config.rag.relevance_threshold

        results_with_scores = self.vector_store.similarity_search_with_score(
            query, k=search_k
        )

        if not results_with_scores:
            logger.debug("Aucun résultat de recherche pour le code")
            return [], []

        # Filtre 1 : seuil de pertinence
        filtered = [
            (doc, score)
            for doc, score in results_with_scores
            if score <= threshold
        ]

        n_before = len(results_with_scores)
        n_after  = len(filtered)
        if n_before != n_after:
            logger.debug(
                "RAG filtrage : %d/%d docs conservés (seuil=%.2f)",
                n_after, n_before, threshold
            )

        # Filtre 2 : boost langage
        lang_lower = language.lower()
        filtered.sort(
            key=lambda pair: (
                0 if pair[0].metadata.get("language", "") == lang_lower else 1,
                pair[1],
            )
        )

        final_k  = k or config.rag.top_k
        filtered = filtered[:final_k]

        docs   = [doc   for doc, _ in filtered]
        scores = [score for _, score in filtered]

        for doc, score in zip(docs, scores):
            logger.debug(
                "  [%.3f] %s (%s/%s)",
                score,
                doc.metadata.get("source_file", "?"),
                doc.metadata.get("language",    "?"),
                doc.metadata.get("category",    "?"),
            )

        return docs, scores

    def _build_knowledge_context(
        self,
        docs: list[Document],
        scores: list[float],
    ) -> str:
        """Formate les chunks RAG pour injection dans le prompt."""
        if not docs:
            return ""

        parts: list[str] = []
        total_chars      = 0
        max_chars        = config.analysis.max_knowledge_chars

        for doc, score in zip(docs, scores):
            meta    = doc.metadata
            source  = meta.get("source_file", "unknown")
            lang    = meta.get("language",    "general")
            cat     = meta.get("category",    "general")
            sev     = meta.get("severity",    "")
            sev_tag = f" | severity: {sev}" if sev else ""

            header  = f"[Source: {source} | {lang}/{cat}{sev_tag} | score: {score:.2f}]"
            content = doc.page_content.strip()
            block   = f"{header}\n{content}"

            if total_chars + len(block) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 200:
                    parts.append(block[:remaining] + "\n... [tronqué]")
                break

            parts.append(block)
            total_chars += len(block) + 2

        return "\n\n".join(parts)

    #Security Section ──────────────────────────────────────────────────────

    def _build_security_section(self, code: str, language: str) -> str:
        """
        OPTIMISE — checklist exhaustive couvrant :
          - Sécurité (SQL injection, passwords, credentials)
          - Ressources (Statement/ResultSet/Connection non fermés)
          - Architecture (SRP, DI, N+1, pagination)
          - Qualité (exception swallowing, static mutable state, magic numbers)

        
        """
        if not _has_security_patterns(code, language):
            return ""

        # Extraire toutes les méthodes déclarées dans le fichier
        if language.lower() == "java":
            method_re = re.compile(
                r'(public|private|protected)\s+[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)',
                re.MULTILINE,
            )
            methods = [m.group(2) for m in method_re.finditer(code)]
        else:
            method_re = re.compile(
                r'def\s+(\w+)\s*\('
                r'|function\s+(\w+)\s*\('
                r'|const\s+(\w+)\s*=\s*(?:async\s*)?\(',
                re.MULTILINE,
            )
            methods = []
            for m in method_re.finditer(code):
                name = m.group(1) or m.group(2) or m.group(3)
                if name:
                    methods.append(name)

        method_list = ", ".join(methods) if methods else "all methods in the file"

        return f"""
SECURITY SCAN MODE — EXHAUSTIVE AUDIT REQUIRED

You MUST inspect each of these {len(methods)} methods individually: {method_list}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE RULE — READ BEFORE ANYTHING ELSE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A compilation error (undeclared variable, missing import, wrong brace) in
method A does NOT exempt methods B, C, D... from being fully analyzed.
Treat EVERY method as a standalone unit. Report and fix EACH independently.

UNDECLARED VARIABLE PATTERN (extremely common in Java):
  If `connection` is used but not declared → the fix for EVERY method that
  uses it is the same: replace with dataSource.getConnection() inside
  try-with-resources. You MUST produce a separate ---FIX START--- block
  for EVERY affected method showing the complete rewritten method body.
  DO NOT produce one single block saying "declare connection field" and stop.
  One method with undeclared connection + SQL injection = TWO blocks for
  that method (one for undeclared var, one for SQL injection).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MANDATORY: One separate ---FIX START--- block per issue per method.
NEVER group issues from different methods into one block.
NEVER stop before all {len(methods)} methods are fully checked.

SECURITY checklist — CRITICAL per occurrence:
  - SQL built with string concatenation (+, format, f-string)
  - Password stored or compared as plain text
  - Hardcoded credentials or secrets in static fields
  - Missing authentication or authorization on sensitive operations

RESOURCE MANAGEMENT checklist — HIGH per occurrence:
  - Statement / PreparedStatement not in try-with-resources
  - ResultSet not in try-with-resources
  - Connection obtained outside try-with-resources (leak on exception)
  - FileWriter / InputStream / Stream not in try-with-resources
  - Transaction (setAutoCommit) without rollback in catch block

ARCHITECTURE checklist — HIGH per occurrence:
  - Method violating Single Responsibility (DB + email + logging + stats)
  - Direct DriverManager.getConnection() instead of injected DataSource
  - Business logic mixed with data access in the same method
  - N+1 query: DB call or query inside a loop (nested query in while/for)
  - Unbounded query: SELECT * with no LIMIT on list-returning method

QUALITY checklist — MEDIUM or LOW:
  - Exception swallowed: empty catch / return null / return false in catch   MEDIUM
  - e.printStackTrace() instead of proper logger                             MEDIUM
  - Package name typo or wrong package declaration                           MEDIUM
  - Static mutable state (static List, static counter)                       MEDIUM
  - Magic numbers or magic strings not extracted to constants                LOW

If 4 methods have SQL injection → 4 separate CRITICAL blocks.
If 5 methods have resource leaks → 5 separate HIGH blocks.
If 3 methods use undeclared `connection` → 3 separate blocks, each with
  the FULL rewritten method using dataSource.getConnection() in try-with-resources.
"""

    # ── Prompt builder ────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        code: str,
        context: dict[str, Any],
        knowledge_context: str,
    ) -> str:
        """
        OPTIMISE — prompt exhaustif.

        Differences vs version precedente :
          1. "Be concise" -> "EXHAUSTIVE audit, report EVERY issue"
          2. issue_limit  -> aucun plafond artificiel
          3. Regle 1      -> liste explicite de tout ce qu'il faut signaler
          4. Regle 9      -> checklist Java specifique
          5. Regle 10     -> interdiction de s'arreter avant la fin
          6. Security section -> checklist etendue (archi + qualite)

        Budget token (inchange) :
          Code source     -> max_code_chars    (10 000 chars)
          Contexte projet -> max_context_chars (1 500 chars)
          Knowledge RAG   -> max_knowledge_chars (2 000 chars)
        """
        # Extraction du contexte
        file_path       = context.get("file_path",         "unknown")
        language        = context.get("language",          "unknown")
        criticality     = context.get("criticality_score", 0)
        dependencies    = context.get("dependencies",      [])
        dependents      = context.get("dependents",        [])
        is_entry_point  = context.get("is_entry_point",    False)
        change_type     = context.get("change_type",       "unknown")
        lines_changed   = context.get("lines_changed",     0)
        project_context = context.get("project_context",   "")

        # Budget token
        max_code = config.analysis.max_code_chars
        max_ctx  = config.analysis.max_context_chars

        code_to_send = code[:max_code]
        if len(code) > max_code:
            code_to_send += f"\n// ... [TRONQUE — {len(code) - max_code} chars restants]"

        project_ctx_compressed = project_context[:max_ctx]

        # Dependency info
        dependency_info = ""
        if dependencies or dependents:
            status = (
                "CRITICAL"  if criticality > 5
                else "IMPORTANT" if criticality > 0
                else "ISOLATED"
            )
            dependency_info = (
                f"\nDEPENDENCY CONTEXT:\n"
                f"- Status: {status} ({criticality} files depend on this)\n"
                f"- Entry point: {'Yes' if is_entry_point else 'No'}\n"
                f"- Uses: {len(dependencies)} file(s) | Used by: {len(dependents)} file(s)\n"
                + (f"- Breaking changes will affect {criticality} other files!\n"
                   if criticality > 0 else "")
            )
            
            # Include dependent files content for cross-file analysis
            if dependents:
                dependency_info += "\nDEPENDENT FILES CONTENT (files that call this one):\n"
                for dep_path in dependents[:3]:  # Limit to first 3 dependents
                    try:
                        dep_file = Path(dep_path)
                        if dep_file.exists():
                            dep_content = dep_file.read_text(encoding="utf-8", errors="replace")[:2000]
                            dependency_info += f"\n--- {dep_file.name} ---\n{dep_content}\n"
                    except Exception:
                        pass
                if len(dependents) > 3:
                    dependency_info += f"\n... and {len(dependents) - 3} more dependent files\n"

        # Flag post-solution — interdit une seconde réécriture full_class
        post_solution_hint = ""
        if context.get("post_solution_mode"):
            post_solution_hint = f"""
╔══════════════════════════════════════════════════════════════════╗
║  POST-SOLUTION MODE — IMPORTANT                                  ║
╚══════════════════════════════════════════════════════════════════╝
{context.get("post_solution_hint", "")}
CONSEQUENCE: In STEP 1 (DECIDE YOUR REPAIR STRATEGY), you MUST choose
block_fix or targeted_methods. full_class is FORBIDDEN for this analysis.
If the code looks mostly correct, respond with block_fix and 0 issues.
"""

        # Contexte upstream — quand ce fichier est un dépendant d'un fichier qui vient de changer
        upstream_hint = ""
        if context.get("upstream_change"):
            upstream_hint = f"""
╔══════════════════════════════════════════════════════════════════╗
║  UPSTREAM CHANGE DETECTED — COMPATIBILITY CHECK                  ║
╚══════════════════════════════════════════════════════════════════╝
{context["upstream_change"]}

YOUR TASK FOR THIS FILE:
1. Check that every call to {context.get("upstream_change", "").split()[1] if context.get("upstream_change") else "the changed file"} still uses the correct method signatures.
2. Check that every import from that file still resolves correctly.
3. Check for any compilation errors caused by the upstream change.
4. If this file is unaffected, say so explicitly with 0 fix blocks.
DO NOT rewrite the entire class. Use block_fix only for real incompatibilities found.
"""

        # Focus area
        focus_map = {
            "new_function":       "New function added — check logic, edge cases, resource management",
            "function_signature": f"Signature changed — verify compatibility with {criticality} dependents",
            "logic_change":       "Logic modified — check all impacted code paths",
            "import_change":      "Import change — check for unused or missing dependencies",
        }
        focus_area = ""
        if change_type in focus_map:
            focus_area = f"\nFOCUS: {focus_map[change_type]}\n"

        # Security scan section (etendue)
        security_section = self._build_security_section(code, language)

        # Regle breaking changes
        breaking_changes_rule = (
            "NO breaking changes allowed (do not rename public methods/classes) — high criticality file."
            if criticality > 3
            else "Breaking changes acceptable with justification."
        )
        
        

        # ── Prompt final ──────────────────────────────────────────────────────
        prompt = f"""You are a SENIOR code reviewer performing an EXHAUSTIVE audit.
Your mission: find and report EVERY issue in the code below — do not skip, do not group, do not stop early.
{post_solution_hint}{upstream_hint}{project_ctx_compressed}
{context.get("system_impact_section", "")}
{dependency_info}{focus_area}{security_section}
CODE TO ANALYZE:
File: {file_path}
Language: {language}
Change: {lines_changed} line(s) modified

```{language}
{code_to_send}
```

BEST PRACTICES FROM KNOWLEDGE BASE:
{knowledge_context if knowledge_context else "(no relevant rules found for this code)"}

RULES:
1. Report ALL issues including: SQL injection (every vulnerable method = separate block),
   plain text passwords, resource leaks (Statement/ResultSet/Connection/FileWriter not closed),
   hardcoded credentials, missing error handling, architectural violations (SRP, DI),
   N+1 queries, unbounded queries, package/import errors, static mutable state,
   magic numbers, utility class design issues.
2. Focus on: COMPILATION ERRORS, security vulnerabilities, critical bugs,
   architecture violations, resource leaks, performance problems.
3. SYNTAX/IMPORTS: Check imports against Existing Internal Packages in PROJECT CONTEXT.
   Missing internal import = CRITICAL compilation error.
4. DO NOT suggest creating files/classes that already exist in the project context.
5. {breaking_changes_rule}
6. NO artificial limit on number of issues — report every single one found.
   Each method with SQL injection = its own block. Each resource leak = its own block.
6b. DEPENDENT FILES: If this file is used by other files (see DEPENDENCY CONTEXT above),
    you MUST also check if your fix for this file requires changes in those dependent files.
    If a public method signature changes, or if you fix a bug that callers were working around,
    generate ---FIX START--- blocks for the DEPENDENT files too, with file path in LOCATION.
    Example: LOCATION: UserController.java:42 (calling method, line 42)
7. Only suggest libraries already present in the project imports. Never invent dependencies.
7b. FIXED CODE must ALWAYS contain real, compilable source code — never only comments.
    BAD:  // Use PreparedStatement instead  // String query = "SELECT..."
    GOOD: try (PreparedStatement stmt = conn.prepareStatement("SELECT ... WHERE id = ?")) {{
              stmt.setInt(1, id);
              ...
          }}
    If the fix is architectural and cannot fit in one block, show the MINIMUM compilable
    change that demonstrates the fix — even if incomplete, it must be real code.
8. Only respond with "Code quality is good, no major issues." if there are literally
   zero issues. If you find even one issue, report it. Never truncate your analysis.
9. For Java specifically, always check:
   - try-with-resources for ALL Statement / PreparedStatement / ResultSet / Connection / FileWriter
   - password hashing (BCrypt / Argon2) — plain text comparison is always CRITICAL
   - Single Responsibility: one method must not do DB + email + logging + stats
   - DataSource injection instead of DriverManager.getConnection()
   - Pagination on every method that returns a List from DB
   - Package declaration correctness
10. Do not stop until every method listed in the SECURITY SCAN section has been checked.
    If you have not finished all methods, continue — do not truncate.
11. CRITICAL — compilation errors do NOT stop the analysis of other issues.
    If a variable is undeclared (e.g. `connection` not declared), report it as CRITICAL,
    THEN CONTINUE analyzing every other method independently for:
    SQL injection, resource leaks, N+1 queries, transaction issues, etc.
    These issues exist regardless of whether the code compiles.
    A file with 1 compilation error + 10 resource leaks = 11 separate blocks.
    NEVER use a compilation error as a reason to skip analyzing individual methods.
12. Treat each method as INDEPENDENT. Even if method A has a fatal error,
    analyze method B, C, D... as if each stands alone.
    Report every distinct issue in every method.

═══════════════════════════════════════════════════════════════
STEP 1 — DECIDE YOUR REPAIR STRATEGY
═══════════════════════════════════════════════════════════════
Before writing any fix, reason about the nature and distribution of the problems.
Then output EXACTLY this block:

---DECISION---
STRATEGY: full_class | targeted_methods | block_fix
SCOPE: [for full_class: "entire file" | for targeted_methods: list the method names | for block_fix: "N isolated issues"]
REASON: [one sentence explaining why this strategy is the most effective]
---DECISION END---

Use this decision tree — in ORDER of priority:

  → full_class       if ANY of these is true:
                       • An undeclared variable is used in 3+ methods (systemic)
                       • The class uses a wrong pattern throughout (e.g. every method
                         needs dataSource.getConnection() but uses `connection` instead)
                       • 5+ methods need structural rewriting (not just small patches)
                       • Fixing method A correctly requires changing method B and C
                       → When in doubt between full_class and targeted_methods,
                         CHOOSE full_class. A complete rewrite is always safer.

  → targeted_methods if ALL of these are true:
                       • Only 2-4 specific methods have problems
                       • The other methods are clean and correct
                       • Each affected method can be fixed independently
                       • No systemic undeclared variable or wrong class design
                       IMPORTANT: If you choose targeted_methods, you MUST produce
                       a ---METHOD START: name--- block for EVERY method you listed
                       in SCOPE. Zero method blocks = wrong choice, use full_class.

  → block_fix        if: problems are genuinely small (1-2 issues, localized patches)
                         Never use for undeclared variables or resource leaks across
                         multiple methods.

═══════════════════════════════════════════════════════════════
STEP 2 — GENERATE THE FIX MATCHING YOUR DECISION
═══════════════════════════════════════════════════════════════

IF STRATEGY = full_class:
  Rewrite the COMPLETE class. Every method, every line. No ellipsis, no omissions.

  HARD CONSTRAINTS — violating any of these makes the solution worse than the original:

  A. NEVER change any public method signature — name, return type, OR parameter names.
     • Same method name, same parameter types and names, same return type.
     • If authenticate(String username, String password) → keep EXACTLY those parameter names.
       NEVER rename password to Stringpassword, pwd, rawPassword or anything else.
     • If findById() returns User → still returns User (not Optional<User>).
     • If findAll() takes no args → still takes no args.
     • This applies to EVERY parameter of EVERY public method, no exceptions.

  B. NEVER create new classes inside this file.
     • If a feature requires a class that does not exist (e.g. Order), SKIP that feature.
     • A missing dependency is not your problem to invent — leave a TODO comment instead.
     • One file = one public class. Period.

  C. ONLY use fields and methods you can SEE in the original code.
     • If User has fields: id, username, email, password → use EXACTLY those names.
     • NEVER write user.passwordHash, user.orders, user.getEmail() unless they appear
       in the original source. Wrong field names = compilation errors.

  D. NEVER add new imports for classes that don't exist in the project.
     • No java.util.Optional unless the original already imports it.
     • No Order, no DTO classes, no new framework annotations.

  E. Fix ALL of these without inventing new architecture:
     → Replace 'connection' (undeclared) with 'dataSource.getConnection()' in try-with-resources
     → Wrap every Statement/PreparedStatement/ResultSet/Connection in try-with-resources
     → Replace string-concatenated SQL with PreparedStatement + setXxx()
     → Add rollback in batchInsert: catch(SQLException e){{ conn.rollback(); throw e; }}

  F. Write CLEAN code — minimal inline comments.
     • DO NOT add // PROBLEM 4: ..., // CRITICAL:, // Fixed by try-with-resources, // MEDIUM: etc.
     • These explanatory comments pollute the production code.
     • One short Javadoc per public method is acceptable. No inline fix-annotations.
     • The CHANGES MADE section (after ---SOLUTION END---) is the right place for explanations.

  Format — use EXACTLY these markers, no variation:
  ---SOLUTION START---
  ```{language}
  [complete class here]
  ```
  ---SOLUTION END---
  CHANGES MADE:
  - methodName: one-line description of what was fixed

IF STRATEGY = targeted_methods:
  For EACH affected method, output ONE rewrite block:
  ---METHOD START: [methodName]---
  ```{language}
  [complete rewritten method — every line]
  ```
  ---METHOD END---
  WHY: [one sentence per method]
  Then list remaining issues in other methods as ---FIX START--- blocks (problem only, no code fix needed).

IF STRATEGY = block_fix:
  Output one ---FIX START--- block per issue (existing format):
  ---FIX START---
  **PROBLEM**: [issue]
  **SEVERITY**: CRITICAL | HIGH | MEDIUM | LOW
  **LOCATION**: [method], line [N]
  **CURRENT CODE**:
  ```{language}
  [exact broken lines]
  ```
  **FIXED CODE**:
  ```{language}
  [compilable fix — no comments only, no pseudo-code]
  ```
  **WHY**: [one sentence]
  ---FIX END---

ANALYZE NOW — START WITH ---DECISION---:"""

        return prompt

    # ── Analyse principale ────────────────────────────────────────────────────

    def analyze_code_with_rag(
        self,
        code: str,
        context: dict[str, Any],
        precomputed_docs:   list | None = None,
        precomputed_scores: list | None = None,
    ) -> dict[str, Any]:
        """
        Analyse un fichier de code avec RAG filtré et prompt exhaustif.

        Args:
            code               : Code source complet du fichier
            context            : Dictionnaire de contexte (file_path, language,
                                 criticality, system_impact_section, ...)
            precomputed_docs   : Docs RAG déjà calculés par SystemAwareRAG
                                 (si fournis, évite une 2e recherche ChromaDB)
            precomputed_scores : Scores correspondants

        Returns:
            {
                "analysis"          : str  — texte de l'analyse LLM
                "relevant_knowledge": list — métadonnées des docs RAG utilisés
                "rag_scores"        : list — scores cosinus (pour debug)
                "docs_used"         : int  — nombre de docs RAG passés au LLM
                "security_mode"     : bool — True si security scan activé
                "code"              : str
                "context"           : dict
            }
        """
        language = context.get("language", "unknown")

        # 1. Retrieval RAG — utilise les docs précomputés si fournis par SystemAwareRAG
        #    sinon fait la recherche standard (rétro-compatible avec project_analyzer.py)
        if precomputed_docs is not None:
            relevant_docs = precomputed_docs
            rag_scores    = precomputed_scores or []
        else:
            relevant_docs, rag_scores = self._retrieve_relevant_knowledge(
                query    = code,
                language = language,
            )

        # 2. Formatage du contexte de connaissance
        knowledge_context = self._build_knowledge_context(relevant_docs, rag_scores)

        # 3. Construction du prompt
        prompt = self._build_prompt(code, context, knowledge_context)

        # 4. Appel LLM
        try:
            response = self.llm.invoke(prompt)
            analysis = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
        except Exception as e:
            logger.error("Erreur LLM lors de l'analyse de %s : %s",
                         context.get("file_path", "?"), e)
            analysis = f"Erreur: {e}\n\nVerifiez votre GOOGLE_API_KEY."

        return {
            "analysis":           analysis,
            "relevant_knowledge": [doc.metadata for doc in relevant_docs],
            "rag_scores":         rag_scores,
            "docs_used":          len(relevant_docs),
            "security_mode":      bool(self._build_security_section(code, language)),
            "code":               code,
            "context":            context,
        }

    # ── Plan de refactoring ───────────────────────────────────────────────────


    # ── Solution Generator ────────────────────────────────────────────────────

    def generate_complete_solution(
        self,
        code:              str,
        context:           dict[str, Any],
        analysis_text:     str,
        knowledge_context: str = "",
    ) -> str:
        """
        Génère la classe entière réécrite et fonctionnelle en utilisant
        TOUT le contexte disponible : RAG, KG, dépendances, project indexer.

        Différence vs analyze_code_with_rag() :
          audit mode    → liste les problèmes, propose des diffs
          solution mode → réécrit la classe COMPLÈTE, compilable, prête à copier

        Appelé depuis Orchestrator quand l'analyse détecte >= 3 CRITICAL ou >= 5 HIGH.
        """
        file_path       = context.get("file_path",         "unknown")
        language        = context.get("language",          "java")
        criticality     = context.get("criticality_score", 0)
        dependencies    = context.get("dependencies",      [])
        dependents      = context.get("dependents",        [])
        project_context = context.get("project_context",   "")
        system_impact   = context.get("system_impact_section", "")

        max_code = config.analysis.max_code_chars
        code_to_send = code[:max_code]

        dep_info = ""
        if dependencies or dependents:
            dep_info = (
                f"\nDEPENDENCIES (DO NOT break these contracts):\n"
                f"  This class is used by: {[Path(d).name for d in dependents[:5]]}\n"
                f"  This class uses:       {[Path(d).name for d in dependencies[:5]]}\n"
            )

        prompt = f"""You are a SENIOR {language.upper()} developer.
Your task: produce the COMPLETE, FULLY REWRITTEN version of this class.
The rewritten class must compile immediately and fix EVERY problem listed below.

{system_impact}
{dep_info}
FILE: {file_path}
CRITICALITY: {criticality} files depend on this class.

═══════════════════════════════════════════════════════════════
ORIGINAL CODE (broken):
═══════════════════════════════════════════════════════════════
```{language}
{code_to_send}
```

═══════════════════════════════════════════════════════════════
PROBLEMS FOUND BY AUDIT (fix ALL of them):
═══════════════════════════════════════════════════════════════
{analysis_text[:3000]}

═══════════════════════════════════════════════════════════════
BEST PRACTICES FROM KNOWLEDGE BASE:
═══════════════════════════════════════════════════════════════
{knowledge_context[:1500] if knowledge_context else "(standard Java best practices apply)"}

═══════════════════════════════════════════════════════════════
PROJECT CONTEXT (do not create classes that already exist):
═══════════════════════════════════════════════════════════════
{project_context[:800] if project_context else ""}

═══════════════════════════════════════════════════════════════
REWRITING RULES — MANDATORY:
═══════════════════════════════════════════════════════════════
1. Output the COMPLETE class from `package` declaration to closing `}}`.
   No ellipsis (...), no "// rest of code", no omissions.
   Every method must be fully written.

2. Every method that accesses the database MUST use:
   try (Connection conn = dataSource.getConnection();
        PreparedStatement stmt = conn.prepareStatement("...")) {{
       // ...
   }}  // auto-closed

3. Every SQL query with user input MUST use PreparedStatement with setXxx().
   Never concatenate user input into SQL strings.

4. Passwords MUST be compared with BCrypt.checkpw() or similar.
   Never compare plain text passwords.

5. Methods returning List<T> from DB MUST include pagination (LIMIT/OFFSET or
   accept page/size parameters).

6. Transactions MUST have rollback in catch:
   conn.setAutoCommit(false);
   try {{ ... conn.commit(); }}
   catch (SQLException e) {{ conn.rollback(); throw e; }}
   finally {{ conn.setAutoCommit(true); }}

7. Keep the same class name, package, and public method signatures.
   Do NOT rename or remove public methods (other classes depend on them).

8. Only use libraries already imported in the original file.
   Add only standard Java imports (java.sql.*, javax.sql.*) if missing.

9. Fix ALL compilation errors, ALL resource leaks, ALL SQL injections,
   ALL plain-text password comparisons, ALL N+1 queries.

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT — STRICT:
═══════════════════════════════════════════════════════════════
---SOLUTION START---
```{language}
[Complete rewritten class here — EVERY method, EVERY line]
```
---SOLUTION END---

After the block, add a short summary:
CHANGES MADE:
- [method name]: [what was fixed]
- ...

WRITE THE COMPLETE SOLUTION NOW:"""

        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error("generate_complete_solution erreur : %s", e)
            return f"Erreur génération solution : {e}"

    def generate_refactoring_plan(self, analysis_results: list[dict]) -> str:
        """
        Génère un plan de refactoring global à partir de toutes les analyses.
        Prend en compte la criticité des fichiers pour prioriser les phases.
        """
        if not analysis_results:
            return (
                "\n PLAN DE REFACTORING\n\n"
                "AUCUN FICHIER ANALYSE\n\n"
                "Verifiez que le projet contient des fichiers .py, .js, .ts, .java\n"
                "et qu'ils ne sont pas dans des dossiers exclus.\n"
            )

        summaries: list[str] = []
        total_critical = total_high = total_medium = 0

        for i, result in enumerate(analysis_results, 1):
            ctx         = result.get("context", {})
            file_name   = Path(ctx.get("file_path", f"File_{i}")).name
            criticality = ctx.get("criticality_score", 0)
            text        = result.get("analysis", "")
            text_upper  = text.upper()

            c_count = text_upper.count("CRITICAL")
            h_count = text_upper.count("HIGH")
            m_count = text_upper.count("MEDIUM")
            total_critical += c_count
            total_high     += h_count
            total_medium   += m_count

            rag_info = ""
            if result.get("docs_used", 0) > 0:
                rag_info = f"\n  RAG docs utilises: {result['docs_used']}"

            summaries.append(
                f"\nFichier {i}: {file_name}\n"
                f"  Criticite: {criticality} | CRITICAL: {c_count} | HIGH: {h_count} | MEDIUM: {m_count}"
                f"{rag_info}\n"
                f"  Extrait:\n{text[:1000]}"
                f"{'...' if len(text) > 1000 else ''}"
            )

        analyses_text = "\n".join(summaries)

        prompt = f"""Vous etes un architecte logiciel expert. Creez un plan de refactoring GLOBAL et COHERENT.

ANALYSES ({len(analysis_results)} fichiers)

{analyses_text[:7000]}

STATISTIQUES
CRITICAL: {total_critical} | HIGH: {total_high} | MEDIUM: {total_medium}

MISSION: Creez un plan qui :
1. Priorise par impact reel (CRITICAL > HIGH > MEDIUM)
2. Identifie les dependances entre corrections (corriger X avant Y)
3. Estime l'effort en heures/jours pour chaque phase
4. Organise les phases pour ne pas casser le projet en cours de refactoring

FORMAT REQUIS:

PHASE 1: SECURITE CRITIQUE (faire IMMEDIATEMENT)
Impact: [fichiers/fonctionnalites affectes]
Corrections:
1. [Fichier.ext] - [Probleme exact]
   Raison: [pourquoi c'est critique]
   Effort: [estimation temps]

PHASE 2: BUGS et ARCHITECTURE (cette semaine)
Dependances: [phases prealables]

PHASE 3: QUALITE et PERFORMANCE (prochain sprint)

ORDRE D'EXECUTION OBLIGATOIRE
1. [Correction A] AVANT [B] car [raison technique]

RISQUES et MITIGATION
Risque: [description] -> Mitigation: [comment l'eviter]

RECOMMANDATIONS FINALES
[Conseils pour l'equipe]

PLAN:"""

        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error("Erreur generation plan refactoring : %s", e)
            return f"Erreur lors de la generation du plan: {e}"


assistant_agent = CodeRAGSystemAPI()