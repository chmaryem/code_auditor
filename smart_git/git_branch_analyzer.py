"""
git_branch_analyzer.py — Analyse complète d'une branche feature vs sa base.

Répond à la question : "Est-ce que ma branche est prête à merger ?"

Stratégie d'analyse :
  1. Trouver le merge-base (ancêtre commun entre feature et main)
  2. Lister tous les fichiers modifiés dans la branche depuis ce merge-base
  3. Pour chaque fichier, chercher l'analyse dans SQLite (Watch cache)
     Si absent du cache → lancer une analyse LLM via l'Orchestrator
  4. Calculer le score de risque global + détecter les conflits potentiels
  5. Produire le BranchReport avec le verdict final

Verdict de merge :
  MERGE_OK        : 0 CRITICAL, < 3 HIGH  → merge recommandé
  MERGE_WARN      : 0 CRITICAL, 3-5 HIGH  → merge possible avec attention
  MERGE_BLOCKED   : ≥ 1 CRITICAL ou > 5 HIGH → corrections requises avant merge
"""
from __future__ import annotations

import logging
from services.mcp_redis_service import get_mcp_redis, key_hash, KEY_PREFIX
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SEVERITY_WEIGHTS  = {"CRITICAL": 10, "HIGH": 3, "MEDIUM": 1, "LOW": 0}
WATCHED_EXTENSIONS = {".java", ".py", ".ts", ".js", ".tsx", ".jsx"}


# ─────────────────────────────────────────────────────────────────────────────
# Structures de données
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FileAnalysis:
    """Résultat d'analyse pour un fichier de la branche."""
    path:          str
    status:        str         # M / A / D
    bugs_critical: int = 0
    bugs_high:     int = 0
    bugs_medium:   int = 0
    bugs_low:      int = 0
    score:         float = 0.0
    from_cache:    bool = False  # True = lu depuis SQLite Watch, False = analysé maintenant
    analysis_text: str = ""

    @property
    def total_bugs(self) -> int:
        return self.bugs_critical + self.bugs_high + self.bugs_medium

    @property
    def max_severity(self) -> str:
        if self.bugs_critical: return "CRITICAL"
        if self.bugs_high:     return "HIGH"
        if self.bugs_medium:   return "MEDIUM"
        return "CLEAN"


@dataclass
class BranchReport:
    """Rapport complet d'une branche — produit par GitBranchAnalyzer.analyze()."""
    branch:          str
    base:            str
    merge_base_hash: str
    commits:         List[Dict]                    # commits exclusifs à la branche
    files:           List[FileAnalysis]
    conflict_risks:  List[str]                     # fichiers modifiés des deux côtés
    total_score:     float = 0.0
    verdict:         str = "MERGE_OK"              # MERGE_OK / MERGE_WARN / MERGE_BLOCKED
    recommendation:  str = ""

    @property
    def total_critical(self) -> int:
        return sum(f.bugs_critical for f in self.files)

    @property
    def total_high(self) -> int:
        return sum(f.bugs_high for f in self.files)

    @property
    def files_clean(self) -> List[FileAnalysis]:
        return [f for f in self.files if f.total_bugs == 0]

    @property
    def files_with_issues(self) -> List[FileAnalysis]:
        return [f for f in self.files if f.total_bugs > 0]


# ─────────────────────────────────────────────────────────────────────────────
# GitBranchAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class GitBranchAnalyzer:
    """
    Analyse une branche feature par rapport à sa branche de base.

    Usage :
        analyzer = GitBranchAnalyzer(
            project_path = Path("C:/monprojet"),
            cache_db     = Path("data/cache/analysis_cache.db"),
        )
        report = analyzer.analyze(branch="feature/auth", base="main")
    """

    def __init__(self, project_path: Path, cache_db: Path, orchestrator=None):
        self.project_path = project_path
        self.cache_db     = cache_db
        self._orchestrator = orchestrator  # injecté optionnellement pour analyse LLM

    # ── Point d'entrée principal ──────────────────────────────────────────────

    def analyze(self, branch: str = "HEAD", base: str = "main") -> BranchReport:
        """
        Lance l'analyse complète de la branche.

        Séquence :
          1. Trouver le merge-base
          2. Lister les commits et fichiers de la branche
          3. Analyser chaque fichier (cache puis LLM si nécessaire)
          4. Détecter les conflits potentiels
          5. Calculer le score et déterminer le verdict
          6. Retourner le BranchReport
        """
        from smart_git.git_diff_parser import (
            get_merge_base, get_branch_commits,
            get_branch_diff_files, is_git_repo,
        )

        if not is_git_repo(self.project_path):
            raise RuntimeError(f"Pas un dépôt git : {self.project_path}")

        # Étape 1 : merge-base
        merge_base = get_merge_base(branch, base, self.project_path)
        if not merge_base:
            logger.warning("merge-base introuvable entre %s et %s", branch, base)
            merge_base = "unknown"

        logger.info("Analyse branche %s vs %s (merge-base: %s)", branch, base, merge_base[:8])

        # Étape 2 : commits et fichiers
        commits = get_branch_commits(branch, base, self.project_path)
        diff_files = get_branch_diff_files(branch, base, self.project_path)

        # Filtrer sur les extensions surveillées (pas les fichiers de config, etc.)
        code_files = [
            f for f in diff_files
            if Path(f["path"]).suffix.lower() in WATCHED_EXTENSIONS
            and f["status"] != "D"
        ]

        logger.info("%d fichiers de code modifiés dans la branche", len(code_files))

        # Étape 3 : analyser chaque fichier
        file_analyses = []
        for file_info in code_files:
            fa = self._analyze_file(file_info, branch)
            file_analyses.append(fa)

        # Étape 4 : conflits potentiels
        conflicts = self._detect_conflict_risks(diff_files, base)

        # Étape 5 : score et verdict
        total_score = sum(f.score for f in file_analyses)
        verdict, recommendation = self._determine_verdict(
            total_score, file_analyses, conflicts
        )

        return BranchReport(
            branch          = branch,
            base            = base,
            merge_base_hash = merge_base,
            commits         = commits,
            files           = sorted(file_analyses, key=lambda f: f.score, reverse=True),
            conflict_risks  = conflicts,
            total_score     = round(total_score, 1),
            verdict         = verdict,
            recommendation  = recommendation,
        )

    # ── Analyse d'un fichier ──────────────────────────────────────────────────

    def _analyze_file(self, file_info: Dict, branch: str) -> FileAnalysis:
        """
        Analyse un fichier de la branche.
        Priorité : cache SQLite (Watch) → analyse LLM si absent.

        Pourquoi cette priorité ?
        Si le développeur a utilisé le mode Watch pendant le développement
        de sa branche, les analyses sont déjà dans SQLite. On réutilise ces
        résultats plutôt que de reconsommer du quota Gemini.
        Si le fichier n'a pas été analysé en Watch (fichier ajouté hors session)
        → on lance une analyse LLM ciblée sur le contenu du fichier à la branche.
        """
        path = file_info["path"]
        abs_path = str(self.project_path / path)

        fa = FileAnalysis(path=path, status=file_info["status"])

        # Tentative 1 : cache SQLite
        analysis_text = self._read_from_cache(abs_path)
        if analysis_text:
            fa.from_cache    = True
            fa.analysis_text = analysis_text
            self._populate_bugs(fa, analysis_text)
            return fa

        # Tentative 2 : analyse LLM via Orchestrator (si disponible)
        if self._orchestrator:
            analysis_text = self._analyze_with_llm(path, branch)
            if analysis_text:
                fa.analysis_text = analysis_text
                self._populate_bugs(fa, analysis_text)
                return fa

        # Pas d'analyse disponible → fichier marqué non-analysé
        logger.debug("Pas d'analyse disponible pour %s", path)
        return fa

    def _populate_bugs(self, fa: FileAnalysis, text: str) -> None:
        """
        Fix 5 — Extraction structurée des compteurs de bugs.

        v1 utilisait re.findall(r"severity.*?CRITICAL") qui comptait aussi
        les commentaires du LLM ("// CRITICAL: SHA-256 is not suitable").
        
        v2 utilise _count_severity_from_blocks() du git_hook.py qui parse
        uniquement les blocs ---FIX START--- / ---FIX END--- structurés.
        
        Fallback : si pas de blocs structurés, compte les patterns
        [CRITICAL], [HIGH], etc. avec des marqueurs stricts.
        """
        from smart_git.git_hook import _count_severity_from_blocks

        c, h, m, score = _count_severity_from_blocks(text)
        fa.bugs_critical = c
        fa.bugs_high     = h
        fa.bugs_medium   = m
        fa.score = score

    # ── Cache Redis MCP ────────────────────────────────────────────────────────

    def _read_from_cache(self, abs_path: str) -> Optional[str]:
        """Lit l'analyse depuis Redis MCP."""
        try:
            redis = get_mcp_redis()
            redis_key = f"{KEY_PREFIX}fc:{key_hash(abs_path)}"
            analysis = redis.hget(redis_key, "analysis_text")
            return analysis if analysis else None
        except Exception as e:
            logger.debug("Redis cache read erreur %s : %s", abs_path, e)
            return None

    # ── Analyse LLM (fallback) ────────────────────────────────────────────────

    def _analyze_with_llm(self, file_path: str, branch: str) -> Optional[str]:
        """
        Lance une analyse LLM sur le fichier à l'état de la branche.
        Utilisé uniquement si le cache ne contient pas d'analyse récente.
        """
        from smart_git.git_diff_parser import get_file_at_commit

        code = get_file_at_commit(file_path, branch, self.project_path)
        if not code:
            return None

        try:
            from services.llm_service import assistant_agent
            result = assistant_agent.analyze_code_with_rag(
                code    = code,
                context = {
                    "file_path": file_path,
                    "language":  Path(file_path).suffix.lstrip("."),
                    "git_context": f"Analyse branche {branch}",
                },
            )
            return result.get("analysis", "")
        except Exception as e:
            logger.debug("LLM analyse erreur pour %s : %s", file_path, e)
            return None

    # ── Détection de conflits ─────────────────────────────────────────────────

    def _detect_conflict_risks(self, branch_files: List[Dict], base: str) -> List[str]:
        """
        Détecte les fichiers modifiés des deux côtés depuis le merge-base.
        Un fichier modifié dans la branche ET dans main depuis le merge-base
        est un candidat probable à un conflit de merge.

        Utilise 'git diff merge-base..main --name-only' pour la liste côté base.
        """
        from smart_git.git_diff_parser import get_merge_base, _run_git

        merge_base = get_merge_base("HEAD", base, self.project_path)
        if not merge_base:
            return []

        # Fichiers modifiés dans main depuis le merge-base
        base_output = _run_git(
            ["diff", f"{merge_base}..{base}", "--name-only"],
            cwd=self.project_path,
        ) or ""
        base_files = set(base_output.strip().splitlines())

        # Intersection avec les fichiers de la branche
        branch_paths = {f["path"] for f in branch_files}
        conflicts    = sorted(branch_paths & base_files)

        if conflicts:
            logger.info("Conflits potentiels détectés : %s", conflicts)

        return conflicts

    # ── Verdict ───────────────────────────────────────────────────────────────

    def _determine_verdict(
        self,
        score: float,
        files: List[FileAnalysis],
        conflicts: List[str],
    ) -> Tuple[str, str]:
        """
        Détermine le verdict de merge et génère une recommandation textuelle.

        Règles :
          MERGE_BLOCKED  : au moins 1 CRITICAL  OU  plus de 5 HIGH
          MERGE_WARN     : 0 CRITICAL et 3-5 HIGH  OU  conflits potentiels
          MERGE_OK       : 0 CRITICAL, < 3 HIGH, pas de conflits
        """
        total_critical = sum(f.bugs_critical for f in files)
        total_high     = sum(f.bugs_high     for f in files)

        if total_critical >= 1 or total_high > 5:
            verdict = "MERGE_BLOCKED"
            parts   = []
            if total_critical:
                parts.append(f"{total_critical} bug(s) CRITICAL à corriger")
            if total_high > 5:
                parts.append(f"{total_high} bugs HIGH (seuil : 5)")
            recommendation = (
                "Merge non recommandé. " + ", ".join(parts) + ". "
                "Appliquez les corrections proposées par le mode Watch et re-analysez."
            )

        elif total_high >= 3 or conflicts:
            verdict = "MERGE_WARN"
            parts   = []
            if total_high >= 3:
                parts.append(f"{total_high} bugs HIGH détectés")
            if conflicts:
                parts.append(f"{len(conflicts)} fichier(s) à risque de conflit")
            recommendation = (
                "Merge possible avec attention. " + ", ".join(parts) + ". "
                "Revue de code recommandée avant de merger."
            )

        else:
            verdict = "MERGE_OK"
            recommendation = (
                "Branche prête à merger. "
                f"Score de risque : {score:.0f}. "
                + (f"{total_high} bug(s) HIGH mineur(s) à surveiller." if total_high else "Aucun problème critique détecté.")
            )

        return verdict, recommendation