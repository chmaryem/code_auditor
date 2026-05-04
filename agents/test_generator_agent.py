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

from services.test_discovery import TestDiscoveryService

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

    # ── API publique ─────────────────────────────────────────────────────────

    def generate_for_file(self, source_path: Path, write: bool = False) -> Dict[str, Any]:
        """
        Génère un fichier de test pour `source_path`.

        Returns:
            {
                "test_file":  Path or None,
                "test_code":  str,
                "framework":  str,
                "error":      str or None,
                "rag_docs_used": int,
                "validated": bool,
            }
        """
        if not source_path.exists():
            return {"error": f"Fichier introuvable : {source_path}", "test_file": None}

        # 1. Discovery — conventions
        test_info = self._discovery.find_test_for(source_path)
        framework = test_info.test_framework or "pytest"
        convention = test_info.convention_used or "{name}_test.py"

        # 2. Lecture du code source
        try:
            source_code = source_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return {"error": f"Lecture impossible : {e}", "test_file": None}

        # 3. Parsing — signatures publiques
        public_signatures = self._extract_signatures(source_code, source_path)

        # 4. Extraction des imports du fichier source
        source_imports = self._extract_imports(source_code, source_path)

        # 5. RAG — patterns de test + exemples du projet
        rag_context = self._retrieve_rag_context(source_path, source_code, public_signatures)

        # 6. Contexte Knowledge Graph (dépendances, classes liées)
        kg_context = self._get_kg_context(source_path)

        # 7. Prompt LLM enrichi
        language = self._detect_language(source_path)
        prompt = self._build_prompt(
            source_path=source_path,
            source_code=source_code,
            signatures=public_signatures,
            imports=source_imports,
            framework=framework,
            rag_context=rag_context,
            kg_context=kg_context,
            language=language,
        )

        # 8. Génération LLM directe
        test_code = self._call_llm(prompt, source_path.name)
        if not test_code:
            return {"error": "Le LLM n'a pas généré de code de test.", "test_file": None}

        # 9. Validation post-génération (Python + Java)
        validated = False
        if language in ("python", "java"):
            validated = self._validate_generated_test(
                test_code, source_path, signatures=public_signatures,
            )
            if not validated:
                # Retry avec le message d'erreur
                logger.info("Test non valide, retry avec feedback d'erreur...")
                test_code_v2 = self._retry_with_error(test_code, source_path, prompt)
                if test_code_v2:
                    validated = self._validate_generated_test(
                        test_code_v2, source_path, signatures=public_signatures,
                    )
                    if validated:
                        test_code = test_code_v2
                        logger.info("Retry reussi: test valide apres correction")
        else:
            validated = True  # Pas de validation pour les autres langages

        # 10. Déterminer le chemin cible
        target_path = self._build_target_path(source_path, convention)

        # 11. Écriture si demandé
        if write:
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(test_code, encoding="utf-8")
            except Exception as e:
                return {"error": f"Écriture impossible : {e}", "test_file": None}

        return {
            "test_file": target_path,
            "test_code": test_code,
            "framework": framework,
            "error": None,
            "rag_docs_used": rag_context.get("total_docs", 0),
            "validated": validated,
        }

    # ── Extraction de signatures ─────────────────────────────────────────────

    @staticmethod
    def _extract_signatures(code: str, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extrait les signatures avec info de visibilité.
        Pour Java, distingue public/private/protected/package-private
        afin que le LLM sache quelles méthodes il peut tester directement.
        """
        sigs = []
        lang = file_path.suffix.lower()

        if lang == ".py":
            # functions
            for m in re.finditer(r"^def\s+([A-Za-z_]\w*)\s*\([^)]*\)", code, re.M):
                name = m.group(1)
                visibility = "private" if name.startswith("_") else "public"
                sigs.append({
                    "type": "function", "name": name,
                    "signature": m.group(0), "visibility": visibility,
                })
            # classes
            for m in re.finditer(r"^class\s+([A-Za-z_]\w*)", code, re.M):
                sigs.append({"type": "class", "name": m.group(1), "visibility": "public"})

        elif lang == ".java":
            # Java : capturer la visibilité et le type de retour
            java_pat = re.compile(
                r"^\s*(?:(public|private|protected)\s+)?"
                r"(?:(static)\s+)?"
                r"(?:(\w[\w<>\[\],\s]*)\s+)?"
                r"([A-Za-z_]\w*)\s*\([^)]*\)",
                re.M,
            )
            for m in java_pat.finditer(code):
                vis = m.group(1) or "package-private"
                is_static = bool(m.group(2))
                ret_type = (m.group(3) or "").strip()
                name = m.group(4)
                if name in ("if", "while", "for", "switch", "catch", "try"):
                    continue
                sigs.append({
                    "type": "method", "name": name,
                    "visibility": vis,
                    "static": is_static,
                    "return_type": ret_type,
                    "signature": m.group(0).strip(),
                })

        elif lang in (".js", ".ts", ".jsx", ".tsx"):
            for m in re.finditer(
                r"(?:export\s+)?(?:async\s+)?(?:function\s+)?([A-Za-z_]\w*)\s*\([^)]*\)",
                code, re.M,
            ):
                name = m.group(1)
                if name not in ("if", "while", "for", "switch", "catch"):
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

        prompt = f"""Tu es un expert en tests unitaires. Génère un fichier de tests COMPLET et EXÉCUTABLE pour le code source ci-dessous.

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
{patterns_text}{examples_text}

## Instructions STRICTES :
1. Génère UNIQUEMENT le code du fichier de test (pas d'explications, pas de markdown fences).
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

GÉNÈRE LE CODE DU FICHIER DE TEST MAINTENANT :"""

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

        # Autres langages : pas de validation pour l'instant
        return True

    def _validate_java_structural(
        self, test_code: str, signatures: List[Dict] = None,
    ) -> bool:
        """
        Validation structurelle Java (sans javac) :
          1. Accolades équilibrées
          2. Parenthèses équilibrées
          3. Strings non fermées
          4. Variables non déclarées
          5. Appels directs aux méthodes private
          6. Utilisation de la réflexion Java
        """
        errors = []

        # 1. Accolades équilibrées
        opens = test_code.count("{")
        closes = test_code.count("}")
        if opens != closes:
            errors.append(f"Accolades desequilibrees : {opens} ouvertes, {closes} fermees")

        # 2. Parenthèses équilibrées
        opens_p = test_code.count("(")
        closes_p = test_code.count(")")
        if opens_p != closes_p:
            errors.append(f"Parentheses desequilibrees : {opens_p} ouvertes, {closes_p} fermees")

        # 3. Strings non fermées (détection basique)
        in_string = False
        for i, ch in enumerate(test_code):
            if ch == '"' and (i == 0 or test_code[i-1] != '\\'):
                in_string = not in_string
        if in_string:
            errors.append("String non fermee detectee")

        # 4. Variables potentiellement non déclarées
        test_methods = re.findall(
            r"(?:void|@Test[^{]*)\s+\w+\s*\([^)]*\)[^{]*\{(.*?)\n    \}",
            test_code,
            re.S,
        )
        for method_body in test_methods:
            used_vars = set(re.findall(r"\b([a-z]\w*)\.", method_body))
            declared = set(re.findall(r"(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*=", method_body))
            declared.update(re.findall(r"(\w+)\s*=\s*\w+", method_body))
            declared.update({"this", "super", "System", "Assert", "mock", "when",
                           "verify", "any", "anyString", "anyInt", "anyLong",
                           "assertEquals", "assertNotNull", "assertTrue",
                           "assertFalse", "assertThrows", "assertDoesNotThrow"})
            undeclared = used_vars - declared
            if undeclared:
                mock_fields = set(re.findall(r"@Mock\s+.*?\s+(\w+)\s*;", test_code, re.S))
                mock_fields.update(re.findall(r"private\s+\w+\s+(\w+)\s*;", test_code))
                undeclared -= mock_fields
                if undeclared:
                    errors.append(f"Variables potentiellement non declarees : {undeclared}")

        # 5. Appels directs aux méthodes private
        if signatures:
            private_methods = [
                s["name"] for s in signatures
                if s.get("visibility") == "private"
            ]
            for pm in private_methods:
                # Chercher objectInstance.privateMethod( dans le test
                pattern = re.compile(rf"\w+\.{re.escape(pm)}\s*\(")
                matches = pattern.findall(test_code)
                if matches:
                    errors.append(
                        f"Appel direct a la methode PRIVATE '{pm}()' — "
                        f"doit etre testee via les methodes publiques"
                    )

        # 6. Utilisation de la réflexion Java (contournement de visibilité)
        reflection_patterns = [
            ("setAccessible", "setAccessible() utilise pour contourner la visibilite"),
            ("getDeclaredMethod", "getDeclaredMethod() — reflexion interdite"),
            ("getDeclaredField", "getDeclaredField() — reflexion interdite"),
        ]
        for rp, msg in reflection_patterns:
            if rp in test_code:
                errors.append(msg)

        if errors:
            for e in errors:
                logger.warning("Validation Java: %s", e)
            return False

        logger.info("Test Java genere: validation structurelle OK")
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
            error_msg = " | ".join(error_parts) if error_parts else "Erreur structurelle"
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

