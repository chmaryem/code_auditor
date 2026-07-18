"""
backfill_chat_history_events.py — one-shot data backfill.

Contexte
--------
Le producteur d'événements chat (services/persistent_chat_memory_service.py::
save_exchange_sync) n'émettait aucune ligne `history_events` — contrairement à
ci_router.py / git_router.py qui en émettent une par analyse CI/CD ou revue
Git. Résultat : la page dashboard History > Timeline ne trouvait jamais rien
pour le module "chat", même avec des dizaines de conversations existantes
dans `conversations` (le compteur "Conversations" du header lit directement
cette table, pas `history_events` — d'où l'incohérence : compteur > 0 mais
Timeline vide).

Le producteur est maintenant corrigé pour les NOUVELLES conversations. Ce
script rattrape les conversations déjà existantes en créant, pour chacune, la
ligne `history_events` manquante (event_type="chat_started", source_module=
"chat", source_id=conversation.id).

Usage
-----
    # Aperçu (aucune écriture) — par défaut :
    python -m scripts.backfill_chat_history_events

    # Écriture réelle :
    python -m scripts.backfill_chat_history_events --apply
"""
from __future__ import annotations

import argparse
import sys
import uuid

from sqlalchemy import text

from database.connection import SyncSessionLocal

_SELECT_MISSING = text(
    """
    SELECT c.id, c.user_id, c.title, c.intent, c.session_id, c.created_at
    FROM conversations AS c
    WHERE NOT EXISTS (
        SELECT 1 FROM history_events AS h
        WHERE h.source_module = 'chat' AND h.source_id = c.id
    )
    ORDER BY c.created_at DESC
    """
)

_INSERT_EVENT = text(
    """
    INSERT INTO history_events
        (id, user_id, event_type, source_module, source_id, title, summary,
         severity, status, metadata_, created_at)
    VALUES
        (:id, :user_id, 'chat_started', 'chat', :source_id, :title, :summary,
         'info', 'completed', CAST(:metadata_ AS JSONB), :created_at)
    """
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Effectue l'insertion. Sans ce flag : dry-run (aperçu seulement).",
    )
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    with SyncSessionLocal() as db:
        rows = db.execute(_SELECT_MISSING).fetchall()

        if not rows:
            print("Aucune conversation sans history_event : rien a rattraper.")
            return 0

        print(f"{len(rows)} conversation(s) sans entree Timeline trouvee(s) :\n")
        for r in rows[:50]:
            title = (r.title or "(sans titre)")[:70]
            print(f"   - {r.session_id[:16]:<16}  {title}  ({r.created_at})")
        if len(rows) > 50:
            print(f"   ... et {len(rows) - 50} autre(s).")

        if not args.apply:
            print(
                "\nDRY-RUN — aucune donnee ecrite.\n"
                "    Relance avec --apply pour creer reellement ces events."
            )
            return 0

        import json

        for r in rows:
            db.execute(
                _INSERT_EVENT,
                {
                    "id": uuid.uuid4().hex,
                    "user_id": r.user_id,
                    "source_id": r.id,
                    "title": r.title or "Conversation",
                    "summary": (r.intent or "")[:300] or None,
                    "metadata_": json.dumps({"session_id": r.session_id, "intent": r.intent}),
                    "created_at": r.created_at,
                },
            )
        db.commit()
        print(f"\nBackfill effectue : {len(rows)} history_event(s) cree(s).")
        return 0


if __name__ == "__main__":
    sys.exit(main())
