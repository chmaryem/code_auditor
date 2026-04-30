"""
output/json_renderer.py — Rendu JSON pour API IDE

Remplace les sorties ANSI console_renderer.py par une API JSON structurée.
Compatible LSP (Language Server Protocol) pour diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path
import json


@dataclass
class Issue:
    """Problème détecté dans le code."""
    severity: str  # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    message: str
    line: Optional[int]
    column: Optional[int]
    code: str  # code snippet
    suggestion: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Fix:
    """Correction suggérée."""
    location: str
    current_code: str
    fixed_code: str
    explanation: str
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnalysisResult:
    """Résultat complet d'une analyse."""
    file_path: str
    language: str
    score: int  # 0-100 (importance du changement)
    issues: list[Issue]
    fixes: list[Fix]
    elapsed_seconds: float
    strategy: str  # "full_class" | "targeted_methods" | "block_fix"
    
    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "language": self.language,
            "score": self.score,
            "issues": [i.to_dict() for i in self.issues],
            "fixes": [f.to_dict() for f in self.fixes],
            "elapsed_seconds": self.elapsed_seconds,
            "strategy": self.strategy,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class JSONRenderer:
    """
    Rendu JSON pour communication IDE.
    
    Usage:
        renderer = JSONRenderer()
        result = renderer.render_analysis(text, file_path, context, elapsed)
        print(result.to_json())  # → VS Code
    """
    
    def render_analysis(
        self,
        text: str,
        file_path: Path,
        context: dict,
        elapsed: float,
        score: int = 0,
    ) -> AnalysisResult:
        """
        Parse la réponse LLM et retourne un objet AnalysisResult.
        
        Args:
            text: Réponse texte de l'analyse (avec FIX blocks)
            file_path: Chemin du fichier analysé
            context: Contexte du projet
            elapsed: Temps d'analyse en secondes
            score: Score d'importance du changement
        """
        from output.console_renderer import parse_fix_blocks
        
        blocks = parse_fix_blocks(text)
        issues = []
        fixes = []
        
        for block in blocks:
            issue = Issue(
                severity=block.get("severity", "MEDIUM"),
                message=block.get("problem", ""),
                line=block.get("line_number"),
                column=None,  # À extraire si disponible
                code=block.get("current_code", ""),
                suggestion=block.get("why", ""),
            )
            issues.append(issue)
            
            if block.get("fixed_code"):
                fix = Fix(
                    location=block.get("location", ""),
                    current_code=block.get("current_code", ""),
                    fixed_code=block.get("fixed_code", ""),
                    explanation=block.get("why", ""),
                )
                fixes.append(fix)
        
        # Détecter stratégie depuis le texte
        strategy = "block_fix"
        if "--- SOLUTION START ---" in text:
            strategy = "full_class"
        elif "targeted" in text.lower():
            strategy = "targeted_methods"
        
        language = context.get("language", "unknown")
        
        return AnalysisResult(
            file_path=str(file_path),
            language=language,
            score=score,
            issues=issues,
            fixes=fixes,
            elapsed_seconds=elapsed,
            strategy=strategy,
        )
    
    def render_empty(self, file_path: Path, reason: str) -> dict:
        """Résultat vide (fichier inchangé ou RAS)."""
        return {
            "file_path": str(file_path),
            "score": 0,
            "issues": [],
            "fixes": [],
            "message": reason,
            "strategy": "none",
        }
