"""
git_secret_scanner.py — F1 : Détection de secrets/credentials dans le staged diff.

Rôle :
  Scanne les fichiers stagés avant chaque commit pour détecter les secrets
  accidentellement inclus : tokens, clés API, mots de passe, certificats.

Architecture :
  - Détection 100% locale (regex + heuristiques) — 0 token LLM
  - Intégré dans git_hook.py (hard block) et accessible via le LangGraph
  - BLOCK immédiat : un seul secret détecté bloque le commit

Patterns couverts :
  AWS, GitHub, Google, JWT, SSH private keys, passwords dans du code,
  .env variables, connexions DB, tokens Bearer génériques.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_R  = "\033[0m"
_B  = "\033[1m"
_GR = "\033[92m"
_YL = "\033[93m"
_RD = "\033[91m"
_CY = "\033[96m"
_DM = "\033[2m"


# ── Patterns de secrets ───────────────────────────────────────────────────────

SECRET_PATTERNS: Dict[str, str] = {
    "AWS Access Key":       r"AKIA[0-9A-Z]{16}",
    "AWS Secret Key":       r"(?i)aws[_\-]?secret[_\-]?access[_\-]?key\s*[=:]\s*['\"]?[A-Za-z0-9/+]{40}",
    "GitHub Token":         r"gh[pousr]_[A-Za-z0-9]{36,255}",
    "GitHub OAuth":         r"github_pat_[A-Za-z0-9_]{82}",
    "Google API Key":       r"AIza[0-9A-Za-z\-_]{35}",
    "Google OAuth":         r"[0-9]+-[0-9A-Za-z_]{32}\.apps\.googleusercontent\.com",
    "Private RSA Key":      r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY( BLOCK)?-----",
    "JWT Secret":           r"(?i)(jwt|json.?web.?token)[_\-]?(secret|key|token)\s*[=:]\s*['\"][A-Za-z0-9+/=._\-]{20,}",
    "Generic API Key":      r"(?i)(api[_\-]?key|apikey)\s*[=:]\s*['\"][A-Za-z0-9\-_]{16,64}['\"]",
    "Generic Token":        r"(?i)(access[_\-]?token|auth[_\-]?token|bearer[_\-]?token)\s*[=:]\s*['\"][A-Za-z0-9\-_.]{20,}['\"]",
    "DB Password":          r"(?i)(db|database|mysql|postgres|mongodb|redis)[_\-]?(password|passwd|pwd)\s*[=:]\s*['\"][^'\"]{6,}['\"]",
    "Generic Password":     r"(?i)(password|passwd|secret[_\-]?key)\s*=\s*['\"][^'\"\\]{8,}['\"]",
    "Slack Token":          r"xox[baprs]-[0-9A-Za-z\-]{10,}",
    "Stripe Key":           r"sk_live_[0-9A-Za-z]{24,}",
    "SendGrid Key":         r"SG\.[A-Za-z0-9\-_]{22}\.[A-Za-z0-9\-_]{43}",
    "Twilio Account":       r"AC[a-f0-9]{32}",
    "Heroku API Key":       r"(?i)heroku[_\-]?api[_\-]?key\s*[=:]\s*['\"]?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    "Connection String":    r"(?i)(jdbc|mongodb(\+srv)?|postgresql|mysql|redis)://[^:]+:[^@/\s]+@",
}

# Extensions de fichiers à IGNORER (binaires, images, etc.)
IGNORED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".pdf",
    ".zip", ".tar", ".gz", ".exe", ".dll", ".so", ".dylib",
    ".lock", ".sum", ".woff", ".woff2", ".ttf", ".eot",
}

# Fichiers à ignorer explicitement (tests, exemples)
IGNORED_FILE_PATTERNS = [
    r"test[s]?[/\\]",
    r"spec[s]?[/\\]",
    r"__tests__[/\\]",
    r"\.test\.",
    r"\.spec\.",
    r"example[s]?[/\\]",
    r"sample[s]?[/\\]",
    r"mock[s]?[/\\]",
    r"fixture[s]?[/\\]",
    r"\.example$",
    r"\.sample$",
    r"README",
]

# Valeurs factices courantes — ignorer (pas des vraies credentials)
FAKE_VALUE_PATTERNS = [
    r"^(your[_\-]?|my[_\-]?|example[_\-]?|test[_\-]?|fake[_\-]?|dummy[_\-]?|placeholder)",
    r"^\*+$",
    r"^x+$",
    r"^<.*>$",
    r"^\$\{.*\}$",
    r"^\{\{.*\}\}$",
    r"^process\.env\.",
    r"^os\.environ",
    r"^config\.",
    r"^settings\.",
    r"^ENV\[",
]


# ── Structures de données ─────────────────────────────────────────────────────

@dataclass
class SecretFinding:
    """Un secret détecté dans un fichier stagé."""
    file_path:   str
    line_number: int
    secret_type: str
    matched_text: str       # extrait tronqué (jamais le secret complet)
    severity:    str = "CRITICAL"
    context:     str = ""   # ligne originale (masquée)

    @property
    def masked_text(self) -> str:
        """Masque le secret — affiche seulement les 4 premiers chars + ***."""
        if len(self.matched_text) > 8:
            return self.matched_text[:4] + "***" + self.matched_text[-2:]
        return "***"


@dataclass
class SecretScanReport:
    """Rapport complet du scan de secrets."""
    findings:       List[SecretFinding] = field(default_factory=list)
    scanned_files:  int = 0
    blocked:        bool = False
    error:          Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def has_secrets(self) -> bool:
        return len(self.findings) > 0

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "CRITICAL")


# ── Scanner ───────────────────────────────────────────────────────────────────

class GitSecretScanner:
    """
    Scanne les fichiers stagés pour détecter les secrets.

    Stratégie :
      1. Récupère le diff staged (git diff --cached)
      2. Pour chaque fichier, scanne le contenu ajouté (lignes "+")
      3. Applique les regex SECRET_PATTERNS
      4. Filtre les faux positifs (valeurs factices, fichiers de test)
    """

    def scan_staged(self, project_path: Path) -> SecretScanReport:
        """Point d'entrée principal — scan des fichiers stagés."""
        report = SecretScanReport()

        try:
            # Récupérer le diff staged complet
            result = subprocess.run(
                ["git", "diff", "--cached", "--unified=0"],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", cwd=str(project_path),
            )
            if result.returncode != 0:
                report.error = f"git diff --cached failed: {result.stderr.strip()}"
                return report

            diff_content = result.stdout
            if not diff_content.strip():
                return report  # Rien de stagé

            # Parser le diff par fichier
            file_sections = self._parse_diff_by_file(diff_content)
            report.scanned_files = len(file_sections)

            for file_path, added_lines in file_sections.items():
                if self._should_ignore_file(file_path):
                    continue
                findings = self._scan_file_lines(file_path, added_lines)
                report.findings.extend(findings)

            report.blocked = report.has_secrets

        except Exception as e:
            report.error = str(e)

        return report

    def scan_file_content(self, file_path: str, content: str) -> List[SecretFinding]:
        """Scan direct du contenu d'un fichier (utilisé par le mode GitHub)."""
        if self._should_ignore_file(file_path):
            return []
        lines = [(i + 1, line) for i, line in enumerate(content.splitlines())]
        return self._scan_file_lines(file_path, lines)

    # ── Parsing du diff ───────────────────────────────────────────────────────

    def _parse_diff_by_file(self, diff: str) -> Dict[str, List[Tuple[int, str]]]:
        """
        Parse le diff git et extrait les lignes ajoutées par fichier.
        Retourne : { file_path: [(line_number, content), ...] }
        """
        files: Dict[str, List[Tuple[int, str]]] = {}
        current_file = None
        current_line = 0

        for line in diff.splitlines():
            # En-tête de fichier : +++ b/path/to/file.py
            if line.startswith("+++ b/"):
                current_file = line[6:].strip()
                files.setdefault(current_file, [])
                current_line = 0

            # Hunk header : @@ -a,b +c,d @@
            elif line.startswith("@@") and current_file:
                match = re.search(r"\+(\d+)", line)
                if match:
                    current_line = int(match.group(1)) - 1

            # Ligne ajoutée
            elif line.startswith("+") and not line.startswith("+++") and current_file:
                current_line += 1
                files[current_file].append((current_line, line[1:]))

            # Ligne de contexte (inchangée)
            elif not line.startswith("-") and current_file:
                current_line += 1

        return files

    # ── Scan d'un fichier ─────────────────────────────────────────────────────

    def _scan_file_lines(
        self, file_path: str, lines: List[Tuple[int, str]]
    ) -> List[SecretFinding]:
        """Applique tous les patterns à chaque ligne ajoutée."""
        findings = []

        for line_num, line_content in lines:
            # Ignorer les commentaires de code évidents
            stripped = line_content.strip()
            if stripped.startswith(("#", "//", "*", "/*", "<!--", "--")):
                # Sauf si c'est du code commenté contenant des vrais secrets
                if not any(kw in stripped.lower() for kw in ("todo", "fixme", "hack")):
                    # Scanner quand même pour les cas comme: // api_key = "real_key"
                    pass

            for secret_type, pattern in SECRET_PATTERNS.items():
                matches = re.finditer(pattern, line_content)
                for match in matches:
                    matched = match.group(0)

                    # Filtrer les faux positifs
                    if self._is_fake_value(matched, line_content):
                        continue

                    findings.append(SecretFinding(
                        file_path    = file_path,
                        line_number  = line_num,
                        secret_type  = secret_type,
                        matched_text = matched,
                        context      = self._mask_line(line_content),
                    ))

        return findings

    # ── Filtres de faux positifs ──────────────────────────────────────────────

    def _should_ignore_file(self, file_path: str) -> bool:
        """Retourne True si le fichier doit être ignoré."""
        path = Path(file_path)
        if path.suffix.lower() in IGNORED_EXTENSIONS:
            return True
        normalized = file_path.replace("\\", "/")
        return any(re.search(p, normalized, re.IGNORECASE) for p in IGNORED_FILE_PATTERNS)

    def _is_fake_value(self, matched: str, context: str) -> bool:
        """Retourne True si la valeur ressemble à un placeholder/fake."""
        # Extraire juste la valeur (après = ou :)
        value_match = re.search(r"[=:]\s*['\"]?(.+?)['\"]?\s*$", matched)
        value = value_match.group(1) if value_match else matched

        value_lower = value.strip().lower()
        return any(re.search(p, value_lower) for p in FAKE_VALUE_PATTERNS)

    def _mask_line(self, line: str) -> str:
        """Masque les valeurs sensibles dans la ligne pour l'affichage."""
        for pattern in SECRET_PATTERNS.values():
            line = re.sub(
                pattern,
                lambda m: m.group(0)[:4] + "***" if len(m.group(0)) > 6 else "***",
                line,
            )
        return line.strip()


# ── Affichage terminal ────────────────────────────────────────────────────────

def render_secret_scan_report(report: SecretScanReport) -> None:
    """Affiche le rapport dans le terminal (utilisé par git_hook.py)."""
    if not report.has_secrets:
        print(f"  {_GR}✓  Aucun secret détecté dans les fichiers stagés.{_R}\n")
        return

    print(f"\n  {_RD}{_B}🔐 SECRETS DÉTECTÉS — COMMIT BLOQUÉ{_R}")
    print(f"  {_RD}{len(report.findings)} secret(s) dans {len(set(f.file_path for f in report.findings))} fichier(s){_R}\n")

    by_file: Dict[str, List[SecretFinding]] = {}
    for finding in report.findings:
        by_file.setdefault(finding.file_path, []).append(finding)

    for file_path, findings in by_file.items():
        print(f"  {_YL}📄 {file_path}{_R}")
        for f in findings:
            print(f"    {_RD}✗  Ligne {f.line_number} : [{f.secret_type}]{_R}")
            print(f"       {_CY}{f.masked_text}{_R}")
            print(f"       {f.context[:80]}")
        print()

    print(f"  {_RD}Actions recommandées :{_R}")
    print(f"  {_RD}  1. Supprimer le(s) secret(s) du code source{_R}")
    print(f"  {_RD}  2. Utiliser des variables d'environnement (.env){_R}")
    print(f"  {_RD}  3. Si déjà pushé : révoquer immédiatement la clé{_R}")
    print(f"  {_DM}  Pour forcer (déconseillé) : git commit --no-verify{_R}\n")


_scanner = GitSecretScanner()


def scan_staged_secrets(project_path: Path) -> SecretScanReport:
    """Point d'entrée public — utilisé par git_hook.py et tool_secret_scan (git_tools.py)."""
    return _scanner.scan_staged(project_path)
