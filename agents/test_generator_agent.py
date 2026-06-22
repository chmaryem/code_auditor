"""
test_generator_agent.py — Génération on-demand de tests unitaires via LLM + RAG.

Architecture v2 (refactored) :
  1. Discovery   → convention de test + framework détecté
  2. RAG         → ChromaDB test_patterns_kb (patterns par langage)
                   + ProjectCodeIndexer (tests similaires dans le projet)
                   + Knowledge Graph (relations entre entités)
  3. Generation  → LLM direct via invoke_with_fallback() (pas analyze_code_with_rag)
  4. Validation  → SandboxExecutor pour vérifier la compilabilité (Python)

Différences vs v1 :
  - RAG sémantique au lieu de rglob("*test*") naïf
  - Appel LLM direct (pas le prompt d'audit)
  - Collection ChromaDB séparée test_patterns_kb
  - Prompt enrichi avec imports, dépendances, contexte KG
  - Validation post-génération via SandboxExecutor
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.ast_extractor import extract_signatures_ast
from services.test_discovery import TestDiscoveryService
from services.test_generation_cache import test_generation_cache
from services.test_runner import TestRunner, TestRunResult

logger = logging.getLogger(__name__)


class TestGeneratorAgent:
    """
    Agent de génération de tests unitaires avec RAG sémantique.
    """

    def __init__(
        self,
        project_path: Path,
        test_kb=None,
        project_code_indexer=None,
        knowledge_graph=None,
    ):
        """
        Args:
            project_path: Racine du projet à tester.
            test_kb: TestKnowledgeLoader — collection ChromaDB des patterns de test.
            project_code_indexer: ProjectCodeIndexer — code réel du projet (tests existants).
            knowledge_graph: KnowledgeGraph — relations entre entités pour context.
        """
        self.project_path = project_path
        self._discovery = TestDiscoveryService(project_path)
        self._test_kb = test_kb
        self._project_code_indexer = project_code_indexer
        self._knowledge_graph = knowledge_graph
        self._runner = TestRunner(project_path)

    # ── API publique ─────────────────────────────────────────────────────────

    def generate_for_file(
        self,
        source_path: Path,
        write: bool = False,
        incremental: bool = False,
    ) -> Dict[str, Any]:
        """
        Génère un fichier de test pour `source_path`.

        Args:
            source_path: Fichier source à tester.
            write: Écrire le résultat sur disque.
            incremental: Si True et qu'un fichier de test existe déjà,
                génère uniquement les méthodes manquantes et les insère
                à la fin du fichier existant plutôt que de l'écraser.

        Returns:
            {
                "test_file":   Path or None,
                "test_code":   str,
                "framework":   str,
                "error":       str or None,
                "rag_docs_used": int,
                "validated":   bool,
                "incremental": bool,
            }
        """
        if not source_path.exists():
            return {"error": f"Fichier introuvable : {source_path}", "test_file": None}

        # 0. Cache Redis — éviter un appel LLM si le fichier source n'a pas changé
        cached = test_generation_cache.get(source_path, incremental)
        if cached:
            return cached

        # 1. Discovery — conventions
        test_info = self._discovery.find_test_for(source_path)
        # Fallback: if discovery returns unknown, infer from file extension
        _ext_framework = {
            ".py":   "pytest",
            ".java": "junit",
            ".js":   "jest",
            ".ts":   "jest",
            ".jsx":  "jest",
            ".tsx":  "jest",
        }
        framework = (
            test_info.test_framework
            if test_info.test_framework and test_info.test_framework != "unknown"
            else _ext_framework.get(source_path.suffix.lower(), "pytest")
        )
        convention = test_info.convention_used or "{name}_test.py"

        # 2. Lecture du code source
        try:
            source_code = source_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return {"error": f"Lecture impossible : {e}", "test_file": None}

        # 3. Parsing — toutes les signatures (public + private pour la validation)
        all_signatures = self._extract_signatures(source_code, source_path)

        # 4. Extraction des imports du fichier source
        source_imports = self._extract_imports(source_code, source_path)

        # 5. Déterminer le chemin cible tôt pour le mode incrémentiel
        target_path = self._build_target_path(source_path, convention)
        language = self._detect_language(source_path)

        # ── Mode incrémentiel : détecter les entités déjà testées ────────────
        is_incremental = False
        existing_test_code = ""
        untested_signatures = all_signatures

        if incremental and target_path.exists():
            try:
                existing_test_code = target_path.read_text(encoding="utf-8", errors="replace")
                untested_signatures = self._filter_untested_signatures(
                    all_signatures, existing_test_code
                )
                if not untested_signatures:
                    logger.info("Mode incrémentiel : toutes les entités sont déjà testées dans %s", target_path.name)
                    return {
                        "test_file": target_path,
                        "test_code": existing_test_code,
                        "framework": framework,
                        "error": None,
                        "rag_docs_used": 0,
                        "validated": True,
                        "incremental": True,
                    }
                is_incremental = True
                logger.info(
                    "Mode incrémentiel : %d entité(s) sans test dans %s",
                    len(untested_signatures), source_path.name,
                )
            except Exception as e:
                logger.warning("Mode incrémentiel : lecture du fichier de test existant échouée (%s), génération complète.", e)
                is_incremental = False
                untested_signatures = all_signatures

        # 6. RAG — patterns de test + exemples du projet
        rag_context = self._retrieve_rag_context(source_path, source_code, untested_signatures)

        # 7. Contexte Knowledge Graph (dépendances, classes liées)
        kg_context = self._get_kg_context(source_path)

        # Security gate (Phase A): redact before source_code enters the LLM prompt.
        # Signatures/imports extraction above already used the real source_code.
        safe_source_code = source_code
        try:
            from services.secret_redactor import redact_secrets
            safe_source_code, _n = redact_secrets(source_code)
        except Exception:
            pass

        # 8. Prompt LLM enrichi
        prompt = self._build_prompt(
            source_path=source_path,
            source_code=safe_source_code,
            signatures=untested_signatures,
            imports=source_imports,
            framework=framework,
            rag_context=rag_context,
            kg_context=kg_context,
            language=language,
            incremental=is_incremental,
            existing_test_code=existing_test_code,
        )

        # 9. Génération LLM directe
        test_code = self._call_llm(prompt, source_path.name)
        if not test_code:
            return {"error": "Le LLM n'a pas généré de code de test.", "test_file": None}

        # 10. En mode incrémentiel : fusionner avec le fichier existant
        if is_incremental:
            test_code = self._merge_incremental(existing_test_code, test_code, language)

        # 11. Validation post-génération (Python + Java + JS/TS)
        # _run_validation retourne (code_final, is_valid) — le code peut avoir été corrigé par le retry
        test_code, validated = self._run_validation(test_code, source_path, all_signatures, language, prompt)

        # 12. Écriture + boucle d'exécution runtime
        run_result: Optional[TestRunResult] = None
        if write:
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(test_code, encoding="utf-8")
            except Exception as e:
                return {"error": f"Écriture impossible : {e}", "test_file": None}

            # Exécution des tests générés — détecte les erreurs runtime (ImportError, etc.)
            run_result = self._runner.run(target_path, language)
            if not run_result.success:
                logger.info(
                    "Exécution échouée pour %s : %s",
                    target_path.name, run_result.error_summary[:200],
                )
                # Retry LLM avec l'erreur d'exécution comme contexte
                fixed_code = self._retry_with_runtime_error(
                    test_code, source_path, prompt, run_result
                )
                if fixed_code:
                    test_code = fixed_code
                    target_path.write_text(test_code, encoding="utf-8")
                    run_result = self._runner.run(target_path, language)
                    if run_result.success:
                        validated = True
                        logger.info("Retry runtime réussi : tests passent maintenant")
                    else:
                        logger.info(
                            "Retry runtime échoué : %s", run_result.error_summary[:150]
                        )

        result = {
            "test_file":          target_path,
            "test_code":          test_code,
            "framework":          framework,
            "error":              None,
            "rag_docs_used":      rag_context.get("total_docs", 0),
            "validated":          validated,
            "incremental":        is_incremental,
            "run_success":        run_result.success if run_result else None,
            "run_error_summary":  run_result.error_summary if run_result and not run_result.success else None,
        }

        # Mettre en cache uniquement si le test est valide ET passe à l'exécution
        cache_ok = validated and (run_result is None or run_result.success)
        if cache_ok:
            test_generation_cache.set(source_path, incremental, result)

        return result

    # ── Helpers pour le mode incrémentiel ────────────────────────────────────

    @staticmethod
    def _filter_untested_signatures(
        signatures: List[Dict[str, Any]], existing_test_code: str
    ) -> List[Dict[str, Any]]:
        """
        Retourne uniquement les signatures non encore couvertes dans le fichier de test existant.
        Utilise la même logique que TestDiscoveryService.check_coverage pour la cohérence :
          1. Extraire les noms des fonctions de test déclarées
          2. Vérifier si le nom de l'entité est référencé dans ces noms
          3. Fallback : appel direct name( dans le corps du test
        """
        # Extraire les noms de fonctions de test (Python, Java, JS/TS)
        test_func_names: set = set(re.findall(
            r"(?:def\s+|function\s+|it\s*\(|test\s*\()([A-Za-z_]\w*)",
            existing_test_code,
        ))
        test_func_lower = {n.lower() for n in test_func_names}

        untested = []
        for sig in signatures:
            name = sig.get("name", "")
            if not name:
                continue
            name_lower = name.lower()

            # Entité couverte si un nom de méthode de test la référence
            is_covered = any(
                name_lower in fn_lower
                or fn_lower.startswith(("test_" + name_lower, name_lower + "test"))
                for fn_lower in test_func_lower
            )
            # Fallback : appel direct dans le corps du test (hors commentaires)
            if not is_covered:
                is_covered = bool(re.search(
                    rf"(?<!#)(?<!//)(?<!\*)\b{re.escape(name)}\s*[\(\.]",
                    existing_test_code,
                ))
            if not is_covered:
                untested.append(sig)
        return untested

    @staticmethod
    def _merge_incremental(existing: str, new_methods: str, language: str) -> str:
        """
        Insère les nouvelles méthodes de test dans le fichier existant.
        Stratégie : insérer avant la dernière accolade fermante (Java/JS/TS)
        ou à la fin du fichier (Python).
        """
        new_clean = new_methods.strip()

        if language == "python":
            return existing.rstrip() + "\n\n\n" + new_clean + "\n"

        # Java / JS / TS : insérer avant la dernière `}`
        last_brace = existing.rfind("}")
        if last_brace == -1:
            return existing + "\n\n" + new_clean
        return (
            existing[:last_brace].rstrip()
            + "\n\n"
            + new_clean
            + "\n"
            + existing[last_brace:]
        )

    def _run_validation(
        self,
        test_code: str,
        source_path: Path,
        signatures: List[Dict[str, Any]],
        language: str,
        original_prompt: str = "",
    ) -> tuple:
        """
        Valide le code généré et retente une fois en cas d'échec.
        Retourne (code_final, is_valid) — code_final est la version corrigée si le retry réussit,
        l'original sinon. Le code retourné DOIT remplacer test_code chez l'appelant.
        """
        validated = self._validate_generated_test(test_code, source_path, signatures=signatures)
        if not validated:
            logger.info("Test non valide, retry avec feedback d'erreur...")
            test_code_v2 = self._retry_with_error(test_code, source_path, original_prompt)
            if test_code_v2:
                validated = self._validate_generated_test(
                    test_code_v2, source_path, signatures=signatures
                )
                if validated:
                    logger.info("Retry réussi : test valide après correction")
                    return test_code_v2, True
        return test_code, validated

    # ── Extraction de signatures ─────────────────────────────────────────────

    @staticmethod
    def _extract_signatures(code: str, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extrait les signatures avec info de visibilité.

        Stratégie : AST tree-sitter en priorité (précis, gère les edge cases),
        fallback regex si tree-sitter est indisponible ou si le parse échoue.
        """
        # ── Tentative AST (tree-sitter) ──────────────────────────────────────
        ast_sigs = extract_signatures_ast(code, file_path)
        if ast_sigs:
            return ast_sigs

        # ── Fallback regex (comportement d'origine) ───────────────────────────
        logger.debug("AST indisponible pour %s — fallback regex", file_path.name)
        sigs = []
        lang = file_path.suffix.lower()

        if lang == ".py":
            py_func_pat = re.compile(
                r"^(?P<indent>[ \t]*)(?P<async>async\s+)?def\s+(?P<name>[A-Za-z_]\w*)"
                r"\s*\((?P<args>[^)]*)\)",
                re.M,
            )
            seen_names: set = set()
            for m in py_func_pat.finditer(code):
                name = m.group("name")
                if name in seen_names:
                    continue
                seen_names.add(name)
                is_async = bool(m.group("async"))
                visibility = "private" if name.startswith("_") else "public"
                start = m.start()
                preceding = code[max(0, start - 120): start]
                decorator_lines = [
                    ln.strip() for ln in preceding.splitlines()
                    if ln.strip().startswith("@")
                ]
                method_type = "function"
                if decorator_lines:
                    last_dec = decorator_lines[-1]
                    if "@property" in last_dec:
                        method_type = "property"
                    elif "@classmethod" in last_dec or "@staticmethod" in last_dec:
                        method_type = "classmethod"
                sig = f"{'async ' if is_async else ''}def {name}({m.group('args')})"
                sigs.append({
                    "type": method_type, "name": name,
                    "signature": sig, "visibility": visibility, "async": is_async,
                })
            for m in re.finditer(r"^class\s+([A-Za-z_]\w*)", code, re.M):
                sigs.append({"type": "class", "name": m.group(1), "visibility": "public"})

        elif lang == ".java":
            java_pat = re.compile(
                r"^\s*(?:(public|private|protected)\s+)?"
                r"(?:(static)\s+)?"
                r"(?:(\w[\w<>\[\],\s]*)\s+)?"
                r"([A-Za-z_]\w*)\s*\([^)]*\)",
                re.M,
            )
            for m in java_pat.finditer(code):
                vis = m.group(1) or "package-private"
                name = m.group(4)
                if name in ("if", "while", "for", "switch", "catch", "try"):
                    continue
                sigs.append({
                    "type": "method", "name": name,
                    "visibility": vis, "static": bool(m.group(2)),
                    "return_type": (m.group(3) or "").strip(),
                    "signature": m.group(0).strip(),
                })

        elif lang in (".js", ".ts", ".jsx", ".tsx"):
            seen_js: set = set()
            _KW = {"if", "while", "for", "switch", "catch", "return", "typeof", "instanceof"}
            for m in re.finditer(
                r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_]\w*)\s*\(",
                code, re.M,
            ):
                name = m.group(1)
                if name not in _KW and name not in seen_js:
                    seen_js.add(name)
                    sigs.append({"type": "function", "name": name, "visibility": "public"})
            for m in re.finditer(
                r"(?:export\s+)?(?:const|let)\s+([A-Za-z_]\w*)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[A-Za-z_]\w*)\s*=>",
                code, re.M,
            ):
                name = m.group(1)
                if name not in _KW and name not in seen_js:
                    seen_js.add(name)
                    sigs.append({"type": "arrow_function", "name": name, "visibility": "public"})
            for m in re.finditer(
                r"^\s+(?:async\s+)?([A-Za-z_]\w*)\s*\([^)]*\)\s*\{", code, re.M,
            ):
                name = m.group(1)
                if name not in _KW and name not in seen_js and not name[0].isupper():
                    seen_js.add(name)
                    sigs.append({"type": "method", "name": name, "visibility": "public"})

        return sigs

    @staticmethod
    def _extract_imports(code: str, file_path: Path) -> List[str]:
        """Extrait les imports du fichier source."""
        imports = []
        lang = file_path.suffix.lower()

        if lang == ".py":
            for m in re.finditer(r"^(?:from\s+\S+\s+)?import\s+.+$", code, re.M):
                imports.append(m.group(0).strip())
        elif lang == ".java":
            for m in re.finditer(r"^import\s+.+;$", code, re.M):
                imports.append(m.group(0).strip())
        elif lang in (".js", ".ts", ".jsx", ".tsx"):
            for m in re.finditer(r"^import\s+.+$", code, re.M):
                imports.append(m.group(0).strip())

        return imports

    @staticmethod
    def _detect_language(file_path: Path) -> str:
        """Détecte le langage depuis l'extension."""
        ext_map = {
            ".py": "python", ".java": "java",
            ".js": "javascript", ".jsx": "javascript",
            ".ts": "typescript", ".tsx": "typescript",
        }
        return ext_map.get(file_path.suffix.lower(), "unknown")

    # ── RAG : récupérer le contexte de test ──────────────────────────────────

    def _retrieve_rag_context(
        self,
        source_path: Path,
        source_code: str,
        signatures: List[Dict],
    ) -> Dict[str, Any]:
        """
        Récupère le contexte RAG en combinant 3 sources :
          1. test_patterns_kb — patterns de test par langage
          2. ProjectCodeIndexer — tests existants dans le projet
          3. Filesystem fallback — si aucun indexeur n'est disponible
        """
        context = {
            "test_patterns": [],
            "project_examples": [],
            "total_docs": 0,
        }
        language = self._detect_language(source_path)

        # Construire la query = signatures + nom du fichier
        sig_names = [s["name"] for s in signatures[:10]]
        query = f"{source_path.stem} {' '.join(sig_names)} unit test {language}"

        # ── Source 1 : test_patterns_kb (patterns de test par langage) ────────
        if self._test_kb:
            try:
                docs, scores = self._test_kb.search(
                    query=query,
                    language=language,
                    k=5,
                    threshold=0.75,
                )
                for doc, score in zip(docs, scores):
                    context["test_patterns"].append({
                        "content": doc.page_content[:2000],
                        "language": doc.metadata.get("language", ""),
                        "score": round(score, 3),
                        "source": doc.metadata.get("source_file", "?"),
                    })
                context["total_docs"] += len(docs)
            except Exception as e:
                logger.debug("test_patterns_kb search erreur: %s", e)

        # ── Source 2 : ProjectCodeIndexer (tests existants dans le projet) ────
        if self._project_code_indexer:
            try:
                results = self._project_code_indexer.search(
                    query=source_code[:2000],
                    k=5,
                    exclude_file=source_path.name,
                    threshold=0.80,
                )
                for doc, score in results:
                    # Ne garder que les fichiers de test
                    fname = doc.metadata.get("source_file", "")
                    if "test" in fname.lower():
                        context["project_examples"].append({
                            "content": doc.page_content[:1500],
                            "file": fname,
                            "score": round(score, 3),
                        })
                        context["total_docs"] += 1
            except Exception as e:
                logger.debug("ProjectCodeIndexer search erreur: %s", e)

        # ── Source 3 : Filesystem fallback (si aucun indexeur) ────────────────
        if context["total_docs"] == 0:
            examples = self._filesystem_fallback(source_path)
            for ex in examples:
                context["project_examples"].append({
                    "content": ex[:1500],
                    "file": "filesystem_scan",
                    "score": 0.0,
                })
                context["total_docs"] += 1

        return context

    def _filesystem_fallback(self, source_path: Path) -> List[str]:
        """Fallback : scan naïf du filesystem pour trouver des tests existants."""
        examples = []
        try:
            for candidate in self.project_path.rglob("*test*"):
                if candidate.is_file() and candidate.suffix == source_path.suffix:
                    try:
                        text = candidate.read_text(encoding="utf-8", errors="replace")
                        if len(text) < 5000:
                            examples.append(text)
                        if len(examples) >= 2:
                            break
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("filesystem_fallback erreur: %s", e)
        return examples

    # ── Knowledge Graph context ──────────────────────────────────────────────

    def _get_kg_context(self, source_path: Path) -> str:
        """Extrait le contexte du Knowledge Graph pour le fichier source."""
        if not self._knowledge_graph:
            return ""

        try:
            # Chercher les nœuds liés au fichier
            file_node = f"file:{source_path}"
            graph = getattr(self._knowledge_graph, "_graph", None)
            if not graph or not graph.has_node(file_node):
                return ""

            neighbors = list(graph.neighbors(file_node))
            if not neighbors:
                return ""

            parts = ["Entités liées dans le Knowledge Graph :"]
            for n in neighbors[:8]:
                node_data = graph.nodes[n]
                node_type = node_data.get("type", "unknown")
                parts.append(f"  - {n} ({node_type})")

            return "\n".join(parts)
        except Exception as e:
            logger.debug("KG context erreur: %s", e)
            return ""

    # ── Construction du prompt ────────────────────────────────────────────────

    def _build_prompt(
        self,
        source_path: Path,
        source_code: str,
        signatures: List[Dict],
        imports: List[str],
        framework: str,
        rag_context: Dict[str, Any],
        kg_context: str,
        language: str,
        incremental: bool = False,
        existing_test_code: str = "",
    ) -> str:
        """Construit un prompt LLM structuré et enrichi pour la génération de tests."""

        # Séparer les signatures par visibilité
        public_sigs = [s for s in signatures if s.get("visibility") in ("public", None)]
        private_sigs = [s for s in signatures if s.get("visibility") == "private"]
        other_sigs = [s for s in signatures if s.get("visibility") in ("protected", "package-private")]

        def _fmt_sig(s: Dict) -> str:
            vis = s.get('visibility', 'public')
            ret = s.get('return_type', '')
            static = ' static' if s.get('static') else ''
            return f"  - [{vis}]{static} {ret+' ' if ret else ''}{s['name']}()"

        sig_text = ""
        if public_sigs:
            sig_text += "### Méthodes PUBLIC (à tester directement) :\n"
            sig_text += "\n".join(_fmt_sig(s) for s in public_sigs[:15])
        if other_sigs:
            sig_text += "\n\n### Méthodes PROTECTED/PACKAGE-PRIVATE (tester via sous-classe ou même package) :\n"
            sig_text += "\n".join(_fmt_sig(s) for s in other_sigs[:10])
        if private_sigs:
            sig_text += "\n\n### Méthodes PRIVATE (NE PAS tester directement — tester via les méthodes publiques) :\n"
            sig_text += "\n".join(_fmt_sig(s) for s in private_sigs[:10])

        imports_text = "\n".join(f"  {imp}" for imp in imports[:15])

        # Extraire les classes de dépendances à mocker
        dep_classes = self._extract_dependency_classes(source_code, file_path=source_path)
        deps_text = ""
        if dep_classes:
            deps_text = "\n## Dépendances à mocker :\n"
            for cls_name, usage in dep_classes[:8]:
                deps_text += f"  - {cls_name} (utilisé comme : {usage})\n"

        # Patterns de test RAG
        patterns_text = ""
        if rag_context["test_patterns"]:
            patterns_text = "\n\n## Patterns de test de référence (Knowledge Base) :\n"
            for i, pat in enumerate(rag_context["test_patterns"][:3], 1):
                patterns_text += f"\n--- Pattern {i} ({pat['language']}, score: {pat['score']}) ---\n"
                patterns_text += pat["content"][:1800] + "\n"

        # Exemples de tests existants dans le projet
        examples_text = ""
        if rag_context["project_examples"]:
            examples_text = "\n\n## Exemples de tests existants dans CE projet :\n"
            for i, ex in enumerate(rag_context["project_examples"][:2], 1):
                examples_text += f"\n--- Exemple {i} ({ex['file']}) ---\n"
                examples_text += ex["content"][:1500] + "\n"

        # Budget dynamique pour le code source
        max_code = min(len(source_code), 6000)
        code_block = source_code[:max_code]
        if len(source_code) > max_code:
            code_block += f"\n// ... [tronqué — {len(source_code) - max_code} chars restants]"

        # Construire le bloc d'avertissement pour les méthodes private
        private_warning = ""
        private_names = [s["name"] for s in private_sigs]
        if private_names:
            private_warning = f"""
## REGLE CRITIQUE — METHODES PRIVATE
Les méthodes suivantes sont PRIVATE : {', '.join(private_names)}
- Tu NE PEUX PAS appeler ces méthodes directement depuis les tests (erreur de compilation).
- Tu DOIS les tester INDIRECTEMENT via les méthodes publiques qui les utilisent.
- NE PAS utiliser la réflexion (setAccessible, getDeclaredMethod) pour contourner la visibilité.
Exemple : si hashPassword() est private et appelé dans save(), teste hashPassword via save() :
  - test_save_shouldHashPasswordBeforeStoring() → vérifie que le password stocké != password brut
"""

        # Déterminer JUnit version
        junit_version_hint = ""
        if language == "java":
            junit_version_hint = """
Note: Préfère JUnit 5 (org.junit.jupiter.api) avec @ExtendWith(MockitoExtension.class)
plutôt que JUnit 4 (@RunWith). Utilise les assertions de org.junit.jupiter.api.Assertions."""

        # Contexte du fichier de test existant pour le mode incrémentiel
        incremental_block = ""
        if incremental and existing_test_code:
            preview = existing_test_code[:3000]
            if len(existing_test_code) > 3000:
                preview += f"\n// ... [{len(existing_test_code) - 3000} chars tronqués]"
            incremental_block = f"""
## Fichier de test EXISTANT (NE PAS réécrire ce qui est déjà testé) :
```{language}
{preview}
```
"""

        mode_instruction = (
            "Génère UNIQUEMENT les NOUVELLES méthodes de test manquantes listées dans les signatures "
            "ci-dessus. Ne répète PAS les méthodes de test déjà présentes dans le fichier existant. "
            "Génère uniquement les corps de méthodes, sans redéclarer la classe ni les imports déjà présents."
            if incremental
            else "Génère un fichier de tests COMPLET et EXÉCUTABLE."
        )

        prompt = f"""Tu es un expert en tests unitaires. {mode_instruction}

Fichier source : {source_path.name}
Langage : {language}
Framework de test détecté : {framework}
{junit_version_hint}

## Imports du fichier source :
{imports_text or "  (aucun import détecté)"}

## Signatures (avec visibilité) :
{sig_text or "  (aucune signature détectée)"}
{deps_text}
{private_warning}
{kg_context}

## Code source :
```{language}
{code_block}
```
{incremental_block}{patterns_text}{examples_text}

## Instructions STRICTES :
1. Génère UNIQUEMENT le code {"des nouvelles méthodes de test" if incremental else "du fichier de test"} (pas d'explications, pas de markdown fences).
2. Utilise le framework {framework} avec les conventions du projet.
3. Teste TOUTES les méthodes PUBLIC listées ci-dessus.
4. INTERDIT : appeler directement une méthode private ou protected depuis les tests. Teste-les UNIQUEMENT via les méthodes publiques qui les appellent.
5. INTERDIT : utiliser la réflexion Java (setAccessible, getDeclaredMethod, getDeclaredField) pour accéder aux méthodes private.
6. Inclus des cas edge cases : null/None, valeurs vides, exceptions attendues.
7. Utilise des mocks (@Mock + @ExtendWith(MockitoExtension.class) pour Java, patch() pour Python) pour les dépendances.
8. Le code DOIT être compilable/exécutable sans erreur.
9. CHAQUE variable utilisée dans un test DOIT être déclarée DANS ce test. Ne référence JAMAIS une variable d'un autre test.
10. Nomme les tests : test_<method>_<scenario> ou <method>_should<Behavior>.
11. N'invente PAS de méthodes ou classes qui n'existent pas dans le code source.
12. Inclus tous les imports nécessaires (framework + mocks + classes du source).

GÉNÈRE LE CODE {"DES NOUVELLES MÉTHODES DE TEST" if incremental else "DU FICHIER DE TEST"} MAINTENANT :"""

        return prompt

    # ── Appel LLM direct ─────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, file_name: str = "") -> Optional[str]:
        """
        Appelle le LLM directement via invoke_with_fallback().
        N'utilise PAS analyze_code_with_rag() pour éviter le prompt d'audit.
        """
        try:
            from services.llm_factory import invoke_with_fallback
            text = invoke_with_fallback(
                prompt,
                temperature=0.1,
                max_tokens=8192,
                label=f"test_gen:{file_name}",
            )
            if not text:
                return None

            return self._extract_code_from_response(text)
        except Exception as e:
            logger.error("LLM test generation erreur: %s", e)
            return None

    @staticmethod
    def _extract_code_from_response(text: str) -> str:
        """
        Extrait le code de la réponse LLM.
        Gère les cas : bloc unique, blocs multiples, texte brut.
        """
        # Chercher tous les blocs de code
        blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.S)
        if blocks:
            # Si un seul bloc → le retourner
            if len(blocks) == 1:
                return blocks[0].strip()
            # Si plusieurs → fusionner (le LLM a parfois splitté le code)
            biggest = max(blocks, key=len)
            if len(biggest) > sum(len(b) for b in blocks) * 0.7:
                # Un bloc domine → c'est le fichier complet
                return biggest.strip()
            # Sinon fusionner tous les blocs
            return "\n\n".join(b.strip() for b in blocks).strip()

        # Pas de blocs markdown → retourner le texte brut nettoyé
        # Retirer les lignes d'explication avant/après le code
        lines = text.strip().split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            if not in_code:
                # Détecter le début du code
                if line.strip().startswith(("package ", "import ", "from ", "class ", "def ",
                                            "@", "#!", "//", "/*", "public ", "private ")):
                    in_code = True
            if in_code:
                code_lines.append(line)
        return "\n".join(code_lines).strip() if code_lines else text.strip()

    # ── Validation post-génération ────────────────────────────────────────────

    @staticmethod
    def _extract_dependency_classes(code: str, file_path: Path = None) -> List[tuple]:
        """
        Extrait les classes de dépendances à mocker depuis le code source.
        Retourne [(class_name, usage_type), ...].
        """
        deps = []
        lang = file_path.suffix.lower() if file_path else ".py"

        if lang == ".java":
            # Champs d'instance : private DataSource dataSource;
            for m in re.finditer(
                r"(?:private|protected)\s+(\w+(?:<[^>]+>)?)\s+\w+\s*;",
                code, re.M,
            ):
                cls = m.group(1).split("<")[0]  # Strip generics
                if cls[0].isupper():
                    deps.append((cls, "field"))

            # Paramètres de constructeur
            for m in re.finditer(
                r"(?:public|protected)\s+\w+\s*\(([^)]+)\)",
                code, re.M,
            ):
                params = m.group(1)
                for p in re.findall(r"(\w+(?:<[^>]+>)?)\s+\w+", params):
                    cls = p.split("<")[0]
                    if cls[0].isupper() and cls not in ("String", "Integer", "Long",
                                                         "Boolean", "Double", "Float",
                                                         "List", "Map", "Set", "Optional"):
                        deps.append((cls, "constructor_param"))

        elif lang == ".py":
            # Attributs self.xxx = XxxService(...)
            for m in re.finditer(r"self\.\w+\s*=\s*(\w+)\s*\(", code, re.M):
                cls = m.group(1)
                if cls[0].isupper():
                    deps.append((cls, "attribute"))

        # Dédupliquer par nom de classe
        seen = set()
        unique = []
        for cls, usage in deps:
            if cls not in seen:
                seen.add(cls)
                unique.append((cls, usage))
        return unique

    def _validate_generated_test(
        self, test_code: str, source_path: Path,
        signatures: List[Dict] = None,
    ) -> bool:
        """
        Valide le test généré selon le langage :
          - Python : compile() pour vérifier la syntaxe
          - Java   : vérifications structurelles + private method calls
        """
        language = self._detect_language(source_path)

        if language == "python":
            try:
                compile(test_code, f"test_{source_path.stem}.py", "exec")
                logger.info("Test Python genere: syntaxe valide")
                return True
            except SyntaxError as e:
                logger.warning("Test Python genere: erreur syntaxe -- %s", e)
                return False

        elif language == "java":
            return self._validate_java_structural(test_code, signatures=signatures)

        elif language in ("javascript", "typescript"):
            return self._validate_js_structural(test_code)

        # Autres langages : pas de validation
        return True

    def _validate_java_structural(
        self, test_code: str, signatures: List[Dict] = None,
    ) -> bool:
        """
        Validation structurelle Java (sans javac).
        Checks :
          1. Accolades équilibrées
          2. Parenthèses équilibrées
          3. Strings non fermées
          4. Appels directs aux méthodes private
          5. Utilisation de la réflexion Java
        Note : la vérification de variables non déclarées est supprimée car son
        regex supposait une indentation fixe à 4 espaces → faux rejets systématiques.
        """
        errors = []

        # 1. Accolades équilibrées (hors chaînes et commentaires)
        opens = test_code.count("{")
        closes = test_code.count("}")
        if opens != closes:
            errors.append(f"Accolades desequilibrees : {opens} ouvertes, {closes} fermees")

        # 2. Parenthèses équilibrées
        opens_p = test_code.count("(")
        closes_p = test_code.count(")")
        if opens_p != closes_p:
            errors.append(f"Parentheses desequilibrees : {opens_p} ouvertes, {closes_p} fermees")

        # 3. Strings non fermées — parcours caractère par caractère
        in_string = False
        escaped = False
        for ch in test_code:
            if escaped:
                escaped = False
                continue
            if ch == '\\' and in_string:
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
        if in_string:
            errors.append("String non fermee detectee")

        # 4. Appels directs aux méthodes private
        if signatures:
            private_methods = [
                s["name"] for s in signatures
                if s.get("visibility") == "private"
            ]
            for pm in private_methods:
                pattern = re.compile(rf"\w+\.{re.escape(pm)}\s*\(")
                if pattern.search(test_code):
                    errors.append(
                        f"Appel direct a la methode PRIVATE '{pm}()' — "
                        f"doit etre testee via les methodes publiques"
                    )

        # 5. Utilisation de la réflexion Java
        reflection_patterns = [
            ("setAccessible", "setAccessible() utilise pour contourner la visibilite"),
            ("getDeclaredMethod", "getDeclaredMethod() — reflexion interdite"),
            ("getDeclaredField", "getDeclaredField() — reflexion interdite"),
        ]
        for rp, msg in reflection_patterns:
            if rp in test_code:
                errors.append(msg)

        if errors:
            for err in errors:
                logger.warning("Validation Java: %s", err)
            return False

        logger.info("Test Java genere: validation structurelle OK")
        return True

    @staticmethod
    def _validate_js_structural(test_code: str) -> bool:
        """
        Validation structurelle JS/TS (sans node/tsc) :
          1. Accolades équilibrées
          2. Parenthèses équilibrées
          3. Crochets équilibrés
          4. Strings non fermées (simple et double quotes)
          5. Au moins une fonction de test déclarée (it/test/describe)
        """
        errors = []

        # 1-3. Équilibrage des délimiteurs
        for open_c, close_c, label in [("{", "}", "Accolades"), ("(", ")", "Parenthèses"), ("[", "]", "Crochets")]:
            if test_code.count(open_c) != test_code.count(close_c):
                errors.append(
                    f"{label} desequilibre(e)s : "
                    f"{test_code.count(open_c)} ouvrant(s), {test_code.count(close_c)} fermant(s)"
                )

        # 4. Strings non fermées (simple et double quotes, hors template literals)
        for quote in ('"', "'"):
            count = sum(
                1 for i, ch in enumerate(test_code)
                if ch == quote and (i == 0 or test_code[i - 1] != '\\')
            )
            if count % 2 != 0:
                errors.append(f"String non fermee detectee (quote: {quote})")

        # 5. Au moins une fonction de test
        has_test = bool(re.search(r"\b(?:it|test|describe)\s*\(", test_code))
        if not has_test:
            errors.append("Aucune fonction de test detectee (it/test/describe)")

        if errors:
            for err in errors:
                logger.warning("Validation JS/TS: %s", err)
            return False

        logger.info("Test JS/TS genere: validation structurelle OK")
        return True

    def _retry_with_error(
        self,
        failed_code: str,
        source_path: Path,
        original_prompt: str,
    ) -> Optional[str]:
        """Retente la génération en incluant le message d'erreur."""
        language = self._detect_language(source_path)

        if language == "python":
            try:
                compile(failed_code, "test.py", "exec")
                return failed_code
            except SyntaxError as e:
                error_msg = f"Ligne {e.lineno}: {e.msg}"
        elif language == "java":
            # Collecter les erreurs structurelles
            error_parts = []
            opens = failed_code.count("{")
            closes = failed_code.count("}")
            if opens != closes:
                error_parts.append(f"Accolades desequilibrees ({opens} vs {closes})")
            # Vérifier les variables
            test_methods = re.findall(
                r"(?:void|@Test[^{]*)\s+\w+\s*\([^)]*\)[^{]*\{(.*?)\n    \}",
                failed_code, re.S,
            )
            for body in test_methods:
                used_vars = set(re.findall(r"\b([a-z]\w*)\.", body))
                declared = set(re.findall(r"(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*=", body))
                mock_fields = set(re.findall(r"@Mock\s+.*?\s+(\w+)\s*;", failed_code, re.S))
                mock_fields.update(re.findall(r"private\s+\w+\s+(\w+)\s*;", failed_code))
                declared.update(mock_fields)
                declared.update({"this", "super", "System"})
                undeclared = used_vars - declared
                if undeclared:
                    error_parts.append(f"Variable(s) non declaree(s) : {undeclared}")
            error_msg = " | ".join(error_parts) if error_parts else "Erreur structurelle Java"

        elif language in ("javascript", "typescript"):
            error_parts = []
            for open_c, close_c, label in [("{", "}", "Accolades"), ("(", ")", "Parenthèses"), ("[", "]", "Crochets")]:
                if failed_code.count(open_c) != failed_code.count(close_c):
                    error_parts.append(
                        f"{label} desequilibres ({failed_code.count(open_c)} vs {failed_code.count(close_c)})"
                    )
            for quote in ('"', "'"):
                count = sum(
                    1 for i, ch in enumerate(failed_code)
                    if ch == quote and (i == 0 or failed_code[i - 1] != '\\')
                )
                if count % 2 != 0:
                    error_parts.append(f"String non fermee (quote: {quote})")
            if not re.search(r"\b(?:it|test|describe)\s*\(", failed_code):
                error_parts.append("Aucune fonction de test (it/test/describe) detectee")
            error_msg = " | ".join(error_parts) if error_parts else "Erreur structurelle JS/TS"

        else:
            return None

        retry_prompt = f"""{original_prompt}

## ATTENTION -- Le code precedent avait des erreurs :
{error_msg}

Voici le code problematique :
```
{failed_code[:2000]}
```

Corrige les erreurs et regenere le fichier de test complet.
RAPPEL : chaque variable utilisee dans un test DOIT etre declaree dans CE test.
Code uniquement, pas d'explications."""

        return self._call_llm(retry_prompt, f"retry:{source_path.name}")

    def _retry_with_runtime_error(
        self,
        failed_code: str,
        source_path: Path,
        original_prompt: str,
        run_result: "TestRunResult",
    ) -> Optional[str]:
        """
        Retente la génération LLM en incluant l'erreur d'exécution exacte.
        Utilisé quand le code passe la validation structurelle mais échoue à l'exécution.
        """
        error_summary = run_result.error_summary[:2000]

        # Conseil ciblé selon le type d'erreur
        hint = ""
        if run_result.has_import_error:
            hint = ("HINT: Le chemin d'import est probablement incorrect. "
                    "Vérifie les noms de modules et de classes dans le code source fourni.")
        elif run_result.has_attribute_error:
            hint = ("HINT: Une méthode ou attribut n'existe pas dans la classe. "
                    "N'invente pas de méthodes — utilise uniquement celles listées dans les signatures.")

        retry_prompt = f"""{original_prompt}

## ERREUR D'EXÉCUTION — Le test généré a échoué à l'exécution :

```
{error_summary}
```

{hint}

## Code problématique :
```
{failed_code[:3000]}
```

Corrige les erreurs et régénère le fichier de test complet.
- Corrige les imports incorrects
- N'utilise que les méthodes qui existent réellement dans le code source
- Chaque variable doit être déclarée dans le test qui l'utilise
Code uniquement, sans explication."""

        logger.info("Retry runtime pour %s", source_path.name)
        return self._call_llm(retry_prompt, f"runtime_retry:{source_path.name}")

    # ── Utilitaires ──────────────────────────────────────────────────────────

    def _build_target_path(self, source_path: Path, convention: str) -> Path:
        """Construit le chemin du fichier de test selon la convention."""
        name = source_path.stem
        name_cap = name[:1].upper() + name[1:] if name else name

        # Gestion Maven/Gradle : src/main/java → src/test/java
        try:
            rel = source_path.relative_to(self.project_path)
            rel_str = str(rel).replace("\\", "/")
            if "src/main/" in rel_str:
                test_rel = rel_str.replace("src/main/", "src/test/", 1)
                test_dir = Path(test_rel).parent
                # Appliquer la convention au nom du fichier
                if "/" in convention:
                    parts = convention.split("/")
                    filename = parts[-1].format(name=name, name_cap=name_cap)
                else:
                    filename = convention.format(name=name, name_cap=name_cap)
                return self.project_path / test_dir / filename
        except (ValueError, IndexError):
            pass

        # Convention standard
        try:
            rel_dir = source_path.parent.relative_to(self.project_path)
        except ValueError:
            rel_dir = Path(".")

        if "/" in convention:
            parts = convention.split("/")
            test_dir = self.project_path / parts[0]
            filename = parts[1].format(name=name, name_cap=name_cap)
            return test_dir / rel_dir / filename

        filename = convention.format(name=name, name_cap=name_cap)
        return source_path.parent / filename

