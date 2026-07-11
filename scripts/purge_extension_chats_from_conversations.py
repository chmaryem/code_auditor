"""
purge_extension_chats_from_conversations.py — one-shot data cleanup.

Contexte
--------
Avant le correctif « scope-gate » de node_memory_save (chat_graph.py), chaque
conversation de l'EXTENSION VS Code (endpoint /api/chat/stream, scope=extension)
était aussi persistée dans la table `conversations` — la table réservée au chat
du DASHBOARD. Résultat : /api/chat/sessions (qui lit `conversations` sans filtre
de scope) affichait les chats de l'extension dans l'historique du dashboard.

Ce script supprime de `conversations` les lignes polluées, c.-à-d. les sessions
dont le `session_id` existe AUSSI dans `extension_chat_sessions` — discriminant
fiable : seule l'extension écrit dans cette table. Les messages liés tombent en
CASCADE (messages.conversation_id → conversations.id, ondelete=CASCADE).

Le chat de l'extension n'est PAS touché : ses données vivent dans
extension_chat_sessions / extension_chat_messages, qui ne sont jamais modifiées.

Usage
-----
    # Aperçu (aucune écriture) — par défaut :
    python -m scripts.purge_extension_chats_from_conversations

    # Suppression réelle :
    python -m scripts.purge_extension_chats_from_conversations --apply
"""
from __future__ import annotations

import argparse
import sys

from sqlalchemy import text

from database.connection import SyncSessionLocal


# session_id présents à la fois dans `conversations` et `extension_chat_sessions`
_SELECT_POLLUTED = text(
    """
    SELECT c.session_id, c.title, c.turn_count
    FROM conversations AS c
    WHERE c.session_id IN (SELECT session_id FROM extension_chat_sessions)
    ORDER BY c.updated_at DESC
    """
)

_DELETE_POLLUTED = text(
    """
    DELETE FROM conversations
    WHERE session_id IN (SELECT session_id FROM extension_chat_sessions)
    """
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Effectue la suppression. Sans ce flag : dry-run (aperçu seulement).",
    )
    args = parser.parse_args()

    # Console Windows (cp1252) : évite un UnicodeEncodeError sur les emojis/accents.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    with SyncSessionLocal() as db:
        rows = db.execute(_SELECT_POLLUTED).fetchall()

        if not rows:
            print("✅ Aucune conversation polluée : la table `conversations` est déjà propre.")
            return 0

        print(f"🔎 {len(rows)} conversation(s) issue(s) de l'extension trouvée(s) dans `conversations` :\n")
        for r in rows[:50]:
            title = (r.title or "(sans titre)")[:70]
            print(f"   • {r.session_id[:16]:<16}  {title}  ({r.turn_count} tours)")
        if len(rows) > 50:
            print(f"   … et {len(rows) - 50} autre(s).")

        if not args.apply:
            print(
                "\n⚠️  DRY-RUN — aucune donnée supprimée.\n"
                "    Relance avec --apply pour purger réellement ces lignes."
            )
            return 0

        result = db.execute(_DELETE_POLLUTED)
        db.commit()
        print(f"\n🧹 Purge effectuée : {result.rowcount} conversation(s) supprimée(s) de `conversations`.")
        print("    (Les messages liés sont supprimés en CASCADE. L'extension n'est pas affectée.)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
