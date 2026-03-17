"""
LearningAgent — Apprend du feedback développeur de façon asynchrone.

Rôle UNIQUE : collecter les corrections acceptées/rejetées et enrichir
la base de connaissances automatiquement, SANS bloquer les analyses.

Différence avec l'ancien FeedbackProcessor :
  AVANT → tournait dans le thread d'analyse, bloquait 10 secondes.
  APRÈS → thread de fond indépendant, totalement non-bloquant.
"""
import logging
import threading
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LearningAgent:
    """
    Agent d'apprentissage non-bloquant.

    Fonctionnement :
      1. submit_feedback()  → non bloquant, met en queue
      2. Thread de fond     → traite la queue tranquillement
      3. Fix validé         → nouvelle règle .md dans knowledge_base/
      4. KB rechargée       → disponible immédiatement dans ChromaDB
    """

    def __init__(self, kb_dir: Path, llm=None, vector_store=None):
        self.kb_dir      = kb_dir
        self._llm        = llm
        self._store      = vector_store
        self._queue      = Queue()
        self._running    = False
        self._thread: Optional[threading.Thread] = None
        self._stats      = {"received": 0, "accepted": 0, "rejected": 0, "promoted": 0}

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._worker, name="LearningAgent", daemon=True)
        self._thread.start()
        logger.info("LearningAgent démarré.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def submit_feedback(self, block: Dict[str, Any], action: str, language: str, modified_code: str = None):
        """Soumet un feedback. Non bloquant — retourne immédiatement."""
        self._stats["received"] += 1
        self._queue.put({
            "block": block, "action": action, "language": language,
            "modified_code": modified_code, "timestamp": datetime.now().isoformat(),
        })

    # ── Thread de fond ────────────────────────────────────────────────────────

    def _worker(self):
        while self._running:
            try:
                event = self._queue.get(timeout=1.0)
                self._process(event)
            except Empty:
                continue
            except Exception as e:
                logger.error("LearningAgent erreur : %s", e)

    def _process(self, event: Dict[str, Any]):
        action = event["action"]
        if action == "accepted":
            self._stats["accepted"] += 1
            self._try_promote_to_kb(event)
        elif action == "rejected":
            self._stats["rejected"] += 1
            logger.debug("Correction rejetée : %s", event["block"].get("problem", ""))
        elif action == "modified":
            self._stats["accepted"] += 1
            event["block"]["fixed_code"] = event.get("modified_code", "")
            self._try_promote_to_kb(event)

    def _try_promote_to_kb(self, event: Dict[str, Any]):
        if self._llm is None or self._store is None:
            return
        block    = event["block"]
        language = event["language"]
        if self._rule_already_exists(block.get("problem", ""), language):
            logger.debug("Règle similaire déjà dans KB — pas de doublon.")
            return
        rule_md = self._generalise_to_rule(block, language)
        if not rule_md:
            return
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        rule_file = self.kb_dir / f"auto_learned_{language}_{ts}.md"
        try:
            rule_file.write_text(rule_md, encoding="utf-8")
            self._stats["promoted"] += 1
            logger.info("Nouvelle règle KB : %s", rule_file.name)
            print(f"\n  KB enrichie : {rule_file.name}")
        except Exception as e:
            logger.error("Erreur écriture règle KB : %s", e)

    def _rule_already_exists(self, problem: str, language: str) -> bool:
        if not problem or self._store is None:
            return False
        try:
            results = self._store.similarity_search_with_score(f"{problem} {language}", k=1)
            if results:
                _, score = results[0]
                return score < 0.15
        except Exception:
            pass
        return False

    def _generalise_to_rule(self, block: Dict[str, Any], language: str) -> Optional[str]:
        prompt = f"""Un développeur a accepté cette correction de code.
Transforme-la en une RÈGLE GÉNÉRIQUE et RÉUTILISABLE pour la base de connaissances.

CORRECTION ACCEPTÉE :
Problème   : {block.get('problem', '')}
Langage    : {language}
Code cassé : {block.get('current_code', '')[:300]}
Code corrigé: {block.get('fixed_code', '')[:300]}
Explication: {block.get('why', '')}

Écris UN fichier .md avec ce format exact :
---
title: [nom court de la règle]
language: {language}
severity: [CRITICAL/HIGH/MEDIUM/LOW]
kg_nodes:
  - name: [NomDuNoeudKG]
    type: vulnerability
---

## Problème
[2 phrases sur le danger de ce pattern]

## Code à éviter
```{language}
[exemple minimal du mauvais pattern]
```

## Code correct
```{language}
[exemple minimal du bon pattern]
```

## Pourquoi
[Une phrase sur les conséquences si non corrigé]

Réponds UNIQUEMENT avec le contenu .md, rien d'autre."""
        try:
            response = self._llm.invoke(prompt)
            content  = response.content if hasattr(response, "content") else str(response)
            return content.strip().lstrip("```markdown").lstrip("```").rstrip("```").strip()
        except Exception as e:
            logger.error("Erreur génération règle KB : %s", e)
            return None

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)


learning_agent = LearningAgent(kb_dir=Path("data/knowledge_base"))