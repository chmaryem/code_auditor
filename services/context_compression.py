"""
context_compression.py — P4 · Compression contextuelle des documents RAG.

Objectif : réduire les tokens envoyés au LLM **sans perdre l'information utile**,
en remplaçant la troncature aveugle par un filtrage intelligent.

Deux filtres, basés sur les embeddings déjà calculés → **aucun appel LLM**, donc
gratuit en tokens (juste un peu de CPU) :

  1. Anti-doublons     — supprime les chunks quasi identiques que le multi-query /
                         KG fait remonter (similarité cosinus entre chunks).
  2. Filtre pertinence — ne garde que les chunks proches de la requête
                         (similarité cosinus chunk ↔ requête), jette le bruit.

Implémentation NATIVE (NumPy) : on n'utilise QUE le modèle d'embeddings déjà
chargé (Jina), pas de dépendance à `langchain`/`langchain_community` (absents de
l'environnement). Se branche APRÈS le reranking, AVANT le knowledge_context :
il complète le pipeline, il ne le remplace pas.

Sécurité (jamais de perte silencieuse de contexte) :
  - NumPy/embeddings absents, ou docs vides → renvoie les docs d'origine ;
  - le filtre de pertinence garde TOUJOURS au moins `min_keep` docs (les plus
    pertinents), même si aucun n'atteint le seuil → on ne perd jamais tout.
"""
from __future__ import annotations

import logging
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)


def _cosine_matrix(vectors):
    """Retourne la matrice de similarité cosinus (n×n) pour une liste de vecteurs."""
    import numpy as np

    mat = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    unit = mat / norms
    return unit @ unit.T


def _cosine_to_query(doc_vectors, query_vector):
    """Similarité cosinus de chaque doc vers la requête → vecteur (n,)."""
    import numpy as np

    docs = np.asarray(doc_vectors, dtype="float32")
    q = np.asarray(query_vector, dtype="float32")
    dn = np.linalg.norm(docs, axis=1)
    dn[dn == 0] = 1e-8
    qn = np.linalg.norm(q) or 1e-8
    return (docs @ q) / (dn * qn)


def compress_documents(
    docs: List[Any],
    query: str,
    embeddings: Any,
    scores: List[float] | None = None,
    *,
    similarity_threshold: float = 0.70,
    redundant_threshold: float = 0.95,
    min_keep: int = 2,
) -> Tuple[List[Any], List[float]]:
    """Compresse une liste de Documents RAG (anti-doublons + pertinence).

    Args:
        docs                 : liste de Documents (avec .page_content).
        query                : requête (le code analysé ou la question).
        embeddings           : modèle d'embeddings (ex. assistant_agent.embeddings).
        scores               : scores alignés sur docs (optionnel, pour le réaffichage).
        similarity_threshold : seuil de pertinence cosinus (0-1). Plus haut = plus strict.
        redundant_threshold  : 2 chunks au-dessus de ce seuil = doublons → un seul gardé.
        min_keep             : nombre minimum de docs à conserver (anti sur-filtrage).

    Returns:
        (docs_compressés, scores_alignés). Renvoie l'entrée telle quelle si la
        compression est indisponible ou n'apporte rien.
    """
    scores = scores or []
    if not docs or embeddings is None or len(docs) <= 1:
        return docs, scores

    try:
        import numpy as np  # noqa: F401  (vérifie la disponibilité)
    except Exception as e:
        logger.debug("P4 compression indisponible (numpy) : %s", e)
        return docs, scores

    contents = [getattr(d, "page_content", "") or "" for d in docs]

    # ── Embeddings (une seule passe) ────────────────────────────────────────────
    try:
        doc_vecs = embeddings.embed_documents(contents)
        query_vec = embeddings.embed_query(query) if query else None
    except Exception as e:
        logger.debug("P4 compression : échec embeddings : %s", e)
        return docs, scores

    # Index des docs conservés (on travaille en indices pour garder docs+scores alignés)
    kept = list(range(len(docs)))

    # ── 1. Anti-doublons — garde le 1er d'un groupe de chunks quasi identiques ───
    try:
        sim = _cosine_matrix(doc_vecs)
        deduped: List[int] = []
        for i in kept:
            if all(sim[i][j] < redundant_threshold for j in deduped):
                deduped.append(i)
        if deduped:
            kept = deduped
    except Exception as e:
        logger.debug("P4 anti-doublons : %s", e)

    # ── 2. Filtre de pertinence — garde >= min_keep, priorité aux plus proches ──
    if query_vec is not None and len(kept) > min_keep:
        try:
            rel = _cosine_to_query([doc_vecs[i] for i in kept], query_vec)
            ranked = sorted(zip(kept, rel), key=lambda p: -p[1])  # plus pertinent d'abord
            above = [i for i, s in ranked if s >= similarity_threshold]
            if len(above) >= min_keep:
                kept = above
            else:
                kept = [i for i, _ in ranked[:min_keep]]  # garde au moins min_keep
        except Exception as e:
            logger.debug("P4 filtre pertinence : %s", e)

    kept_sorted = sorted(set(kept))
    if len(kept_sorted) == len(docs):
        return docs, scores  # rien filtré

    new_docs = [docs[i] for i in kept_sorted]
    new_scores = [scores[i] for i in kept_sorted if i < len(scores)]

    logger.debug("P4 compression : %d → %d docs", len(docs), len(new_docs))
    return new_docs, new_scores
