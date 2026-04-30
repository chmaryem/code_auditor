"""
test_generator_agent.py — Génération on-demand de tests unitaires via LLM + RAG.

Rôle :
  - Reçoit un fichier source à tester
  - Utilise TestDiscoveryService pour comprendre les conventions du projet
  - Interroge ChromaDB / Knowledge Graph pour récupérer des exemples de tests similaires
  - Construit un prompt ciblé et appelle le LLM
  - Retourne le code de test généré (preview ou écriture sur disque)

Architecture :
  1. Discovery   → convention de test + framework détecté
  2. Retrieval   → RAG : tests similaires dans le projet + patterns KB
  3. Generation  → LLM avec prompt structuré
  4. Validation  → (optionnel) syntax check via sandbox
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.test_discovery import TestDiscoveryService

logger = logging.getLogger(__name__)


class TestGeneratorAgent:
    """
    Agent de génération de tests unitaires.
    """

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self._discovery = TestDiscoveryService(project_path)

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
            }
        """
        if not source_path.exists():
            return {"error": f"Fichier introuvable : {source_path}", "test_file": None}

        # 1. Discovery — conventions
        test_info = self._discovery.find_test_for(source_path)
        framework = test_info.test_framework or "pytest"
        convention = test_info.convention_used or "{name}_test.py"

        # 2. Parsing source (extrait signatures publiques)
        try:
            source_code = source_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return {"error": f"Lecture impossible : {e}", "test_file": None}

        public_signatures = self._extract_signatures(source_code, source_path)

        # 3. RAG — récupérer des exemples de tests dans le projet
        similar_tests = self._retrieve_similar_tests(source_path)

        # 4. Prompt LLM
        prompt = self._build_prompt(
            source_path=source_path,
            source_code=source_code,
            signatures=public_signatures,
            framework=framework,
            similar_tests=similar_tests,
        )

        # 5. Génération LLM
        test_code = self._call_llm(prompt)
        if not test_code:
            return {"error": "Le LLM n'a pas généré de code de test.", "test_file": None}

        # 6. Déterminer le chemin cible
        target_path = self._build_target_path(source_path, convention)

        # 7. Écriture si demandé
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
        }

    # ── Extraction de signatures ─────────────────────────────────────────────

    @staticmethod
    def _extract_signatures(code: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extrait rapidement les signatures publiques sans parser lourd."""
        import re

        sigs = []
        lang = file_path.suffix.lower()

        if lang == ".py":
            # functions
            for m in re.finditer(r"^def\s+([A-Za-z_]\w*)\s*\([^)]*\)", code, re.M):
                name = m.group(1)
                if not name.startswith("_"):
                    sigs.append({"type": "function", "name": name, "signature": m.group(0)})
            # classes
            for m in re.finditer(r"^class\s+([A-Za-z_]\w*)", code, re.M):
                sigs.append({"type": "class", "name": m.group(1)})

        elif lang in (".java", ".js", ".ts", ".jsx", ".tsx"):
            for m in re.finditer(r"(?:public\s+)?(?:static\s+)?(?:\w+\s+)?([A-Za-z_]\w*)\s*\([^)]*\)", code, re.M):
                name = m.group(1)
                if name not in ("if", "while", "for", "switch", "catch"):
                    sigs.append({"type": "method", "name": name})

        return sigs

    # ── RAG : récupérer tests similaires ────────────────────────────────────

    def _retrieve_similar_tests(self, source_path: Path) -> List[str]:
        """Cherche des fichiers de test dans le projet pour exemples de style."""
        examples = []
        try:
            for candidate in self.project_path.rglob("*test*"):
                if candidate.is_file() and candidate.suffix == source_path.suffix:
                    try:
                        text = candidate.read_text(encoding="utf-8", errors="replace")
                        if len(text) < 4000:  # éviter les fichiers énormes
                            examples.append(text)
                        if len(examples) >= 3:
                            break
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("retrieve_similar_tests erreur : %s", e)
        return examples

    # ── Construction du prompt ────────────────────────────────────────────────

    def _build_prompt(
        self,
        source_path: Path,
        source_code: str,
        signatures: List[Dict],
        framework: str,
        similar_tests: List[str],
    ) -> str:
        """Construit un prompt LLM structuré pour la génération de tests."""

        sig_text = "\n".join(
            f"  - {s['type']} {s['name']}" for s in signatures[:15]
        )

        examples_text = ""
        if similar_tests:
            examples_text = "\n\nExemples de tests existants dans ce projet :\n"
            for i, ex in enumerate(similar_tests[:2], 1):
                examples_text += f"\n--- Exemple {i} ---\n{ex[:1500]}\n"

        prompt = f"""Tu es un expert en tests unitaires. Génère un fichier de tests complet pour le code source ci-dessous.

Fichier source : {source_path.name}
Framework de test détecté : {framework}

Signatures publiques à tester :
{sig_text}

Code source :
```
{source_code[:3000]}
```
{examples_text}

Instructions :
1. Génère UNIQUEMENT le code du fichier de test (pas d'explications).
2. Utilise le framework {framework} avec les conventions du projet.
3. Teste les fonctions/méthodes publiques principales.
4. Inclus des cas edge cases et des mocks si nécessaire.
5. Le code doit être compilable / exécutable sans erreur de syntaxe.
6. Nomme les tests de manière descriptive : test_<nom_de_la_fonction>_<scenario>.
"""
        return prompt

    # ── Appel LLM ────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Appelle le LLM via le service existant."""
        try:
            from services.llm_service import assistant_agent
            result = assistant_agent.analyze_code_with_rag(
                code=prompt,
                context={"mode": "test_generation", "task": "generate_unit_tests"},
            )
            text = result.get("analysis", "")
            # Extrait le code entre ``` ou retourne tout
            import re
            blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.S)
            if blocks:
                return blocks[-1].strip()
            return text.strip()
        except Exception as e:
            logger.error("LLM test generation erreur : %s", e)
            return None

    # ── Utilitaires ──────────────────────────────────────────────────────────

    def _build_target_path(self, source_path: Path, convention: str) -> Path:
        """Construit le chemin du fichier de test selon la convention."""
        name = source_path.stem
        name_cap = name[:1].upper() + name[1:] if name else name

        rel_dir = source_path.parent.relative_to(self.project_path)

        if "/" in convention:
            parts = convention.split("/")
            test_dir = self.project_path / parts[0]
            filename = parts[1].format(name=name, name_cap=name_cap)
            return test_dir / rel_dir / filename

        filename = convention.format(name=name, name_cap=name_cap)
        return source_path.parent / filename
