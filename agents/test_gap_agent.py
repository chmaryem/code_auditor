"""
test_gap_agent.py — Détection intelligente des tests manquants (0 token LLM).

Rôle dans le pipeline Watch :
  Étape 4.7 de l'Orchestrator — après le parsing AST, avant le RAG.

Logique :
  1. Parse le fichier source → liste d'entités publiques
  2. Cherche le fichier de test correspondant (via TestDiscoveryService)
  3. Si pas de test OU couverture < seuil :
     • Calcule un "impact score" (0–100) selon le type de changement
     • Retourne un TestGapStatus avec recommandation

Déclenchement :
  - Watch mode : affichage discret inline (pas de popup)
  - GitSessionTracker : agrégation dans le snapshot de session

Pas de génération ici — uniquement détection et notification.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from services.test_discovery import TestDiscoveryService, TestDiscoveryResult

logger = logging.getLogger(__name__)


@dataclass
class TestGapStatus:
    """Statut de couverture de tests pour un fichier source."""

    source_file:       Path
    test_file:         Optional[Path] = None
    missing:           bool = True      # True si pas de test du tout
    coverage_ratio:    float = 0.0     # 0.0–1.0
    untested_entities: List[str] = field(default_factory=list)
    tested_entities:   List[str] = field(default_factory=list)
    framework:         Optional[str] = None

    # Score d'impact calculé (0–100)
    impact_score:      int = 0
    reason:            str = ""

    @property
    def needs_attention(self) -> bool:
        """True si le développeur devrait être notifié."""
        if self.missing and self.impact_score >= 30:
            return True
        if self.coverage_ratio < 0.5 and len(self.untested_entities) >= 2:
            return True
        return False


class TestGapAgent:
    """
    Agent de détection de gap de tests.
    Analyse rapide, coût 0 token, pas de LLM.
    """

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self._discovery = TestDiscoveryService(project_path)

    # ── API publique ─────────────────────────────────────────────────────────

    def check(
        self,
        source_file: Path,
        parsed_entities: List[Dict[str, any]],
        change_info: Optional[Dict[str, any]] = None,
    ) -> TestGapStatus:
        """
        Vérifie si le fichier source a des tests à jour.

        Args:
            source_file : chemin du fichier modifié
            parsed_entities : entités extraites par CodeParser
            change_info : dict du CodeAgent (score, type de changement…)
        """
        result = self._discovery.check_coverage(source_file, parsed_entities)

        status = TestGapStatus(
            source_file=source_file,
            test_file=result.test_file,
            missing=result.test_file is None or not result.test_file.exists(),
            coverage_ratio=result.coverage_ratio,
            untested_entities=result.entities_untested,
            tested_entities=result.entities_tested,
            framework=result.test_framework,
        )

        status.impact_score = self._compute_impact_score(
            status, parsed_entities, change_info
        )
        status.reason = self._build_reason(status)

        # Log visible pour débogage (sera supprimé en production)
        print(f"  [TEST GAP DEBUG] {source_file.name}: missing={status.missing}, coverage={status.coverage_ratio*100:.0f}%, impact={status.impact_score}, needs_attention={status.needs_attention}")

        logger.debug(
            "TestGap %s : missing=%s coverage=%.0f%% impact=%d",
            source_file.name, status.missing, status.coverage_ratio * 100,
            status.impact_score,
        )
        return status

    # ── Score d'impact ───────────────────────────────────────────────────────

    def _compute_impact_score(
        self,
        status: TestGapStatus,
        parsed_entities: List[Dict[str, any]],
        change_info: Optional[Dict[str, any]],
    ) -> int:
        """
        Calcule un score 0–100 indiquant l'urgence d'avoir des tests.
        Plus le score est haut, plus la notification doit être visible.
        """
        score = 0

        # ── 1. Nouvelles entités publiques ───────────────────────────────────
        new_entities = len(status.untested_entities)
        if new_entities > 0:
            score += min(new_entities * 10, 40)

        # ── 2. Pas de fichier de test du tout ──────────────────────────────
        if status.missing:
            score += 25

        # ── 3. Type de changement (depuis change_info) ───────────────────────
        if change_info:
            ctype = change_info.get("change_type", "")
            if ctype in ("new_function", "new_method", "new_class"):
                score += 20
            elif ctype in ("signature_change", "logic_change"):
                score += 15
            elif ctype == "new_file":
                score += 30

            # Score brut du CodeAgent
            raw_score = change_info.get("score", 0)
            if raw_score > 70:
                score += 10

        # ── 4. Complexité du fichier ───────────────────────────────────────
        entity_count = len(parsed_entities)
        if entity_count > 5:
            score += 5
        if entity_count > 10:
            score += 5

        # ── 5. Fichier critique (dépendants présents) ───────────────────────
        # Ce champ est injecté par l'orchestrator dans change_info si besoin
        has_dependents = change_info and change_info.get("has_dependents", False)
        if has_dependents:
            score += 10

        return min(score, 100)

    def _build_reason(self, status: TestGapStatus) -> str:
        """Construit une phrase explicative courte pour l'affichage."""
        parts = []
        if status.missing:
            parts.append(f"aucun test pour {status.source_file.name}")
        else:
            missing = len(status.untested_entities)
            total = missing + len(status.tested_entities)
            if missing > 0:
                parts.append(f"couverture {total - missing}/{total}")
            else:
                parts.append("couverture complète")

        if status.untested_entities:
            ents = ", ".join(status.untested_entities[:3])
            if len(status.untested_entities) > 3:
                ents += f" +{len(status.untested_entities) - 3}"
            parts.append(f"sans test : {ents}")

        return " — ".join(parts)


# Singleton (ré-initialisé dans Orchestrator.initialize())
test_gap_agent = TestGapAgent(Path("."))
