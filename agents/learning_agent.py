"""
LearningAgent — Apprend du feedback développeur de façon asynchrone.

Deux modes :
  Mode auto   : blocs CRITIQUE → auto-promus immédiatement (silencieux)
  Mode batch  : tout le reste → accumulé → bilan unique à Ctrl+C

Séquence de fin de session :
  Ctrl+C → orchestrator.stop() → learning_agent.stop()
         → feedback_processor.flush_session() → bilan terminal
"""
import logging
import threading
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Optional
from services.feedback_processor import FeedbackProcessor

logger = logging.getLogger(__name__)


class LearningAgent:

    def __init__(self, kb_dir: Path = None, llm=None, vector_store=None,
                 kb_loader=None, knowledge_graph=None):
        self.kb_dir              = kb_dir or Path("data/knowledge_base")
        self._llm                = llm
        self._store              = vector_store
        self._kb_loader          = kb_loader
        self._knowledge_graph    = knowledge_graph
        self._feedback_processor = None
        self._queue   = Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stats   = {"received": 0, "auto_promoted": 0,
                         "batch_promoted": 0, "rejected": 0}

    def initialize(self, llm, vector_store, kb_dir: Path,
                   kb_loader=None, knowledge_graph=None):
        self._llm             = llm
        self._store           = vector_store
        self.kb_dir           = kb_dir
        self._kb_loader       = kb_loader
        self._knowledge_graph = knowledge_graph

        try:
            
            if kb_loader:
                self._feedback_processor = FeedbackProcessor(
                    llm             = llm,
                    vector_store    = vector_store,
                    kb_dir          = kb_dir,
                    kb_loader       = kb_loader,
                    knowledge_graph = knowledge_graph,
                )
                logger.info("FeedbackProcessor initialisé — Self-Improving RAG actif.")
        except ImportError as e:
            logger.debug("FeedbackProcessor absent : %s", e)
        except Exception as e:
            logger.debug("FeedbackProcessor non initialisé : %s", e)

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._worker,
                                         name="LearningAgent", daemon=True)
        self._thread.start()

    def stop(self):
        """
        Arrêt propre :
          1. Vide la queue (traite tous les blocs reçus)
          2. Appelle flush_session() → bilan terminal pour le dev
          3. Arrête le thread
        """
        # Laisser la queue se vider avant d'afficher le bilan
        self._queue.join()

        # Bilan de fin de session
        if self._feedback_processor:
            try:
                self._feedback_processor.flush_session()
                fp_stats = self._feedback_processor.get_stats()
                self._stats["auto_promoted"]  = fp_stats.get("auto_promoted",  0)
                self._stats["batch_promoted"] = fp_stats.get("batch_promoted", 0)
                self._stats["rejected"]       = fp_stats.get("rejected",       0)
            except Exception as e:
                logger.error("flush_session erreur : %s", e)

        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def collect_feedback(self, blocks: list, code_before: str, language: str,
                         file_name: str, project_indexer=None, dependency_graph=None):
        """Non-bloquant — enfile et retourne immédiatement."""
        self._stats["received"] += 1
        self._queue.put({
            "blocks":      blocks,
            "code_before": code_before,
            "language":    language,
            "file_name":   file_name,
            "timestamp":   datetime.now().isoformat(),
        })

    def submit_feedback(self, block: Dict[str, Any], action: str,
                        language: str, modified_code: str = None):
        """Feedback utilisateur direct (API externe ou tests)."""
        self._stats["received"] += 1
        self._queue.put({"block": block, "action": action,
                         "language": language, "modified_code": modified_code})

 

    def _worker(self):
        while self._running:
            try:
                event = self._queue.get(timeout=0.5)
                self._process(event)
                self._queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error("LearningAgent erreur : %s", e)
                try:
                    self._queue.task_done()
                except Exception:
                    pass

    def _process(self, event: Dict[str, Any]):
        # Chemin A : FeedbackProcessor (triage auto + accumulation batch)
        if "blocks" in event and self._feedback_processor:
            try:
                self._feedback_processor.collect_feedback(
                    blocks       = event["blocks"],
                    code_before  = event["code_before"],
                    language     = event["language"],
                    file_name    = event["file_name"],
                )
            except Exception as e:
                logger.debug("FeedbackProcessor.collect_feedback erreur : %s", e)
            return

        # Chemin B : feedback direct sans FeedbackProcessor
        action = event.get("action", "")
        block  = event.get("block", {})
        if action in ("accepted", "modified"):
            if action == "modified":
                block["fixed_code"] = event.get("modified_code", "")
            self._try_promote_to_kb(block, event.get("language", ""))
        elif action == "rejected":
            self._stats["rejected"] += 1

    # ── Promotion directe (chemin B) ─────────────────────────────────────────

    def _try_promote_to_kb(self, block: Dict, language: str):
        if not self._llm or not self._store:
            return
        if self._rule_already_exists(block.get("problem", ""), language):
            return

        rule_md = self._generalise_to_rule(block, language)
        if not rule_md:
            return

        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        lang_dir = Path(self.kb_dir) / "auto_learned" / language
        lang_dir.mkdir(parents=True, exist_ok=True)
        rule_file = lang_dir / f"rule_{ts}.md"
        try:
            rule_file.write_text(rule_md, encoding="utf-8")
            self._stats["auto_promoted"] += 1
            logger.info("KB enrichie (chemin B) : %s", rule_file.name)
        except Exception as e:
            logger.error("Écriture règle KB : %s", e)
            return

        self._reload_chromadb(rule_file)
        self._reload_kg(rule_file)

    def _reload_chromadb(self, md_path: Path):
        if self._kb_loader is None or self._store is None:
            return
        try:
            docs = self._kb_loader.process_file(md_path)
            if docs:
                try:
                    self._store._collection.delete(
                        where={"source_file": md_path.name})
                except Exception:
                    pass
                self._store.add_documents(docs)
        except Exception as e:
            logger.error("ChromaDB reload : %s", e)

    def _reload_kg(self, md_path: Path):
        if self._knowledge_graph is None:
            return
        try:
            if hasattr(self._knowledge_graph, "reload_kb_file"):
                self._knowledge_graph.reload_kb_file(md_path)
            else:
                from config import config
                self._knowledge_graph._builder.build_from_kb(config.KNOWLEDGE_BASE_DIR)
                self._knowledge_graph._save()
        except Exception as e:
            logger.error("KG reload : %s", e)

    def _rule_already_exists(self, problem: str, language: str) -> bool:
        if not problem or not self._store:
            return False
        try:
            results = self._store.similarity_search_with_score(
                f"{problem} {language}", k=1)
            return bool(results) and results[0][1] < 0.35
        except Exception:
            return False

    def _generalise_to_rule(self, block: Dict, language: str) -> Optional[str]:
        prompt = f"""Génère une règle KB réutilisable depuis ce fix.

Problème : {block.get('problem','')}
Langage  : {language}
Code cassé  : {block.get('current_code','')[:300]}
Code correct: {block.get('fixed_code','')[:300]}
Pourquoi : {block.get('why','')}

Format .md (réponds uniquement avec le contenu, sans balises ```) :
---
title: [nom court]
language: {language}
severity: {block.get('severity','MEDIUM')}
kg_nodes:
  - name: [NomConcept]
    type: vulnerability
kg_relations:
  - [NomConcept, FIXED_BY, NomDuFix]
---
## Problème
## Code à éviter
## Code correct
## Pourquoi"""
        try:
            response = self._llm.invoke(prompt)
            content  = response.content if hasattr(response, "content") else str(response)
            content  = content.strip()
            if content.startswith("```"):
                lines   = content.splitlines()
                content = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```"
                                    else lines[1:])
            return content.strip()
        except Exception as e:
            logger.error("Erreur génération règle KB : %s", e)
            return None

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)


learning_agent = LearningAgent()