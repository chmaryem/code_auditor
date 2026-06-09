# Proposition — Prompt d'analyse condensé (à valider avant remplacement)

> Réécriture de `_build_prompt()` et `_build_security_section()` de
> [services/llm_service.py](services/llm_service.py).
> **Objectif : mêmes fonctionnalités, ~6× moins de répétitions.**
>
> ⚠️ RIEN n'est remplacé tant que tu n'as pas validé ce fichier. Seul le **texte**
> du prompt change — toute la logique Python autour (extraction des méthodes,
> calcul de `dependency_info`, `post_solution_hint`, `code_to_send`, etc.) reste
> **identique**.

---

## 1. Garanties

**Marqueurs du parseur préservés mot pour mot** (lus par [agents/analysis_agent.py](agents/analysis_agent.py)) :
`---DECISION--- / ---DECISION END---`, `---FIX START--- / ---FIX END---`
(+ `PROBLEM / SEVERITY / LOCATION / CURRENT CODE / FIXED CODE`),
`---SOLUTION START--- / ---SOLUTION END---` + `CHANGES MADE:`,
`---METHOD START: nom--- / ---METHOD END---` + `WHY:`,
`<STRUCTURED_OUTPUT> … </STRUCTURED_OUTPUT>` (champs `strategy, strategy_reason, issues[], fixes[]`).

**Placeholders dynamiques préservés** : `{post_solution_hint}`, `{upstream_hint}`,
`{project_ctx_compressed}`, `{system_impact_section}`, `{dependency_info}`,
`{focus_area}`, `{security_section}`, `{file_path}`, `{language}`, `{lines_changed}`,
`{code_to_send}`, `{knowledge_context}`, `{breaking_changes_rule}`,
`{self._build_language_rules(language)}`.

---

## 2. Table de couverture (rien n'est perdu)

| Fonctionnalité du prompt initial | Conservée ? | Où dans la version condensée |
|---|---|---|
| Audit exhaustif / 1 problème = 1 bloc / ne pas s'arrêter / ne pas tronquer | ✅ | En-tête (dit **1 fois** au lieu de 6) |
| « erreur de compil n'arrête pas l'analyse » + méthodes indépendantes | ✅ | En-tête + section sécurité (1 fois au lieu de 4) |
| Catalogue des problèmes + sévérités (SQL, passwords, leaks, SRP, N+1…) | ✅ | Bloc `AUDIT CHECKLIST` (1 fois au lieu de 3) |
| Import interne manquant = CRITICAL | ✅ | RULE 1 |
| Ne pas recréer fichiers/classes existants | ✅ | RULE 1 |
| Seulement libs déjà présentes / FIXED CODE = vrai code | ✅ | RULE 2 |
| `breaking_changes_rule` (criticité) | ✅ | RULE 3 |
| Fichiers dépendants → blocs FIX avec LOCATION | ✅ | RULE 4 |
| Règles spécifiques au langage | ✅ | `_build_language_rules()` (inchangé) |
| Décision full_class / targeted_methods / block_fix + arbre | ✅ | STEP 1 |
| Contraintes anti-hallucination A–F (full_class) | ✅ | STEP 2 |
| Formats SOLUTION / METHOD / FIX | ✅ | STEP 2 (marqueurs intacts) |
| Schéma `<STRUCTURED_OUTPUT>` + règles JSON | ✅ | STEP 3 (intact) |
| Scan sécurité par méthode (liste + undeclared var pattern) | ✅ | `_build_security_section` condensé |
| Modes post-solution / upstream / focus / dépendances / system impact | ✅ | Placeholders inchangés |

---

## 3. `_build_security_section()` — version condensée

> ~58 lignes de texte → ~10. La logique d'extraction des méthodes (`method_re`,
> `methods`, `method_list`) **ne change pas** — seul le `return f"""..."""` est réécrit.

```python
        return f"""
SECURITY SCAN — inspect EACH of these {len(methods)} methods individually: {method_list}
Audit every method as a standalone unit: a compilation error in one method (e.g. an undeclared
variable like `connection`) never exempts the others. Produce ONE ---FIX START--- block per issue
per method — a method with an undeclared variable AND a SQL injection = TWO blocks; 4 methods with
SQL injection = 4 separate CRITICAL blocks. For an undeclared/leaked resource the fix is the same
everywhere: acquire it safely (Java: dataSource.getConnection() inside try-with-resources;
Python: a `with` block) and show the FULL corrected method body. Never group issues from different
methods; never stop before all {len(methods)} methods are checked.
"""
```

---

## 4. `_build_prompt()` — version condensée (le `return prompt` complet)

> Tout le code AVANT le `prompt = f"""..."""` (extraction du contexte, budgets,
> `dependency_info`, `post_solution_hint`, `upstream_hint`, `focus_area`,
> `security_section`, `breaking_changes_rule`, `code_to_send`,
> `project_ctx_compressed`) **reste identique**. On ne remplace que la chaîne.

```python
        prompt = f"""You are a SENIOR {language} code reviewer doing an EXHAUSTIVE audit.
Find and report EVERY issue. One issue = one ---FIX START--- block, even within the same method.
A compilation error never stops the audit: report it, then keep auditing every other method
independently. Only answer "Code quality is good, no major issues." if there are literally zero
issues. Never truncate your analysis.
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

AUDIT CHECKLIST (report EVERY occurrence; severity in brackets):
- SQL built with string concatenation / f-string / .format()                       [CRITICAL]
- Plain-text password stored or compared; hardcoded secrets or credentials         [CRITICAL]
- eval/exec/__import__, pickle.loads on untrusted data, shell=True / os.system      [CRITICAL/HIGH]
- Missing authentication or authorization on a sensitive operation                 [CRITICAL]
- Unclosed resource: Statement/PreparedStatement/ResultSet/Connection/File/Stream  [HIGH]
- Transaction (setAutoCommit) without rollback in catch                            [HIGH]
- SRP violation, business logic mixed with data access, DriverManager instead of injected DataSource [HIGH]
- N+1 query (DB call inside a loop); unbounded query (list returned with no LIMIT) [HIGH]
- Swallowed exception / printStackTrace; static mutable state                       [MEDIUM]
- Wrong or typo'd package declaration; magic numbers / magic strings               [MEDIUM/LOW]

RULES:
1. Imports: a missing INTERNAL import (see PROJECT CONTEXT) is a CRITICAL compilation error.
   Never invent files, classes, or libraries; never suggest creating something that already exists.
2. Use only libraries already imported in the project. FIXED CODE is ALWAYS real compilable code,
   never comments only — if the fix is architectural, show the MINIMUM compilable change.
3. {breaking_changes_rule}
4. DEPENDENT FILES: if you change a public signature or fix a bug callers relied on, also emit
   ---FIX START--- blocks for those dependent files (put the file + line in LOCATION).
{self._build_language_rules(language)}

═══════════════════════════════════════════════════════════════
STEP 1 — DECIDE YOUR REPAIR STRATEGY
═══════════════════════════════════════════════════════════════
Output EXACTLY this block:
---DECISION---
STRATEGY: full_class | targeted_methods | block_fix
SCOPE: [full_class: "entire file" | targeted_methods: list the method names | block_fix: "N isolated issues"]
REASON: [one sentence explaining why this strategy is the most effective]
---DECISION END---

Choose, in ORDER of priority:
  → block_fix     DEFAULT — prefer this for real-time / per-save review. Each problem maps to a few
                  specific lines. Produce ONE targeted fix PER issue, current_code = ONLY the exact
                  broken lines (2-6), NEVER the whole file. Different issues → different locations.
  → targeted_methods  2-4 specific methods each need a full rewrite AND the others are clean. Produce
                  one ---METHOD START: name--- block for EVERY method listed in SCOPE.
  → full_class    ONLY when the problem is systemic and CANNOT be expressed as per-line patches (same
                  broken pattern across 5+ methods, or fixing one forces rewriting most others). When
                  in doubt, choose block_fix. Never rewrite code that isn't broken — it adds
                  regressions and hallucinated references to fields/files that don't exist here.

═══════════════════════════════════════════════════════════════
STEP 2 — GENERATE THE FIX MATCHING YOUR DECISION
═══════════════════════════════════════════════════════════════

IF STRATEGY = full_class — rewrite the COMPLETE class (every method, every line, no ellipsis).
  HARD CONSTRAINTS (violating any makes the result worse than the original):
  A. NEVER change a public method signature — same name, same parameter names AND types, same return type.
  B. NEVER create new classes in this file; a missing dependency → leave a TODO, don't invent it.
  C. ONLY use fields/methods visible in the original code (no invented field names, getters, or files).
  D. NEVER add imports for classes that don't exist in the project.
  E. Fix every issue with {language} idioms ONLY (Python → `with`; Java → try-with-resources; JS/TS →
     try/finally; parameterized queries via the language's real DB API). Never import another
     language's patterns.
  F. Clean code: no inline // PROBLEM / // CRITICAL / // Fixed annotations (explanations go in CHANGES MADE).
  Format — use EXACTLY these markers:
  ---SOLUTION START---
  ```{language}
  [complete rewritten class]
  ```
  ---SOLUTION END---
  CHANGES MADE:
  - methodName: one-line description of what was fixed

IF STRATEGY = targeted_methods — for EACH affected method output ONE block:
  ---METHOD START: [methodName]---
  ```{language}
  [complete rewritten method — every line]
  ```
  ---METHOD END---
  WHY: [one sentence]
  Then list remaining issues in other methods as ---FIX START--- blocks.

IF STRATEGY = block_fix — output one block PER issue:
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

═══════════════════════════════════════════════════════════════
STEP 3 — STRUCTURED OUTPUT (MANDATORY — append after your analysis)
═══════════════════════════════════════════════════════════════
Use EXACTLY the markers below — the API parser depends on them:

<STRUCTURED_OUTPUT>
{{
  "strategy": "full_class|targeted_methods|block_fix",
  "strategy_reason": "one sentence max 100 chars explaining why",
  "issues": [
    {{
      "title": "short issue title",
      "message": "detailed description",
      "line": <integer or null>,
      "column": <integer or null>,
      "end_line": <integer or null>,
      "severity": "critical|error|warning|info",
      "rule": "category.rule_name",
      "suggestion": "how to fix in one sentence"
    }}
  ],
  "fixes": [
    {{
      "title": "fix title",
      "explanation": "why this change is needed",
      "line": <integer or null>,
      "apply_mode": "replace_snippet|replace_method|full_file",
      "current_code": "exact code to replace (copy from CURRENT CODE block)",
      "fixed_code": "replacement code (copy from FIXED CODE block)"
    }}
  ]
}}
</STRUCTURED_OUTPUT>

Rules for STRUCTURED_OUTPUT:
- strategy must match your ---DECISION---. issues = EVERY problem (one entry each). fixes parallel
  issues ONE-TO-ONE (issue[i] is fixed by fixes[i]).
- block_fix (DEFAULT): apply_mode="replace_snippet"; current_code = the exact 2-6 broken lines copied
  verbatim from the source (never the whole file, never lines from another issue); each fix.line points
  to its own location; fixes target DIFFERENT lines.
- full_class only (rare): a SINGLE fix, apply_mode="full_file", current_code/fixed_code = first 3 lines
  of the original / of your solution.
- current_code MUST be an exact substring of the source. Valid JSON only (escape quotes, use \\n, no raw
  newlines inside strings). If there are no issues: "issues": [], "fixes": [].

ANALYZE NOW — START WITH ---DECISION---:"""

        return prompt
```

---

## 5. Estimation du gain

| | Instructions statiques par appel |
|---|---|
| Avant | ~3 000 – 3 800 tokens |
| Après | ~1 800 – 2 200 tokens |
| **Économie** | **~1 200 – 1 600 tokens / analyse** (× nombre de chunks sur les gros fichiers) |

Indépendant du modèle, sur 100 % des analyses.

---

## 6. Validation suggérée avant remplacement

1. Relis surtout la **table de couverture** (§2) — confirme qu'aucune règle utile ne manque.
2. Si OK → je remplace `_build_prompt` et `_build_security_section` dans
   [services/llm_service.py](services/llm_service.py) par les versions ci-dessus.
3. Optionnel : test avant/après sur un fichier réel (ex. un `.java` avec SQL injection +
   resource leak) pour comparer le nombre d'issues détectées.
