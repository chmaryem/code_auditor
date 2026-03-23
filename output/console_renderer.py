"""
console_renderer.py — Affichage console de toutes les analyses.

Contient TOUT ce qui touche à l'affichage (migré depuis incremental_analyzer.py) :
  - Constantes ANSI
  - parse_fix_blocks()   : parse le texte LLM → blocs structurés
  - make_diff()          : diff coloré current vs fixed
  - print_block()        : affiche un seul bloc de correction
  - print_results()      : résumé complet d'une analyse
  - print_minor_change() : affiche RAS pour changement mineur
"""
from __future__ import annotations
import re
from datetime import datetime
from pathlib import Path


def enable_windows_ansi():
    try:
        import ctypes, sys
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetConsoleMode(
                ctypes.windll.kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

enable_windows_ansi()

_R  = "\033[0m"
_BD = "\033[1m"
_DM = "\033[2m"
_RD = "\033[91m"
_GR = "\033[92m"
_YL = "\033[93m"
_CY = "\033[96m"
_OR = "\033[38;5;208m"
_W    = 72
_SEP  = "\u2500" * _W
_SEP2 = "\u2550" * _W

_SEV = {
    "CRITICAL": (_RD, "\U0001f534", "CRITIQUE"),
    "HIGH":     (_OR, "\U0001f7e0", "HAUTE"),
    "MEDIUM":   (_YL, "\U0001f7e1", "MOYENNE"),
    "LOW":      ("\033[94m", "\U0001f535", "FAIBLE"),
}


def parse_fix_blocks(text: str) -> list:
    """Parse le texte LLM → liste de blocs {problem, severity, location, ...}"""
    blocks = []
    parts  = re.split(r'-{3,}\s*FIX START\s*-{3,}', text, flags=re.IGNORECASE)
    for raw in parts[1:]:
        end = re.search(r'-{3,}\s*FIX END\s*-{3,}', raw, re.IGNORECASE)
        if end:
            raw = raw[:end.start()]

        def _f(name):
            m = re.search(r'\*\*' + re.escape(name) + r'\*\*\s*:?\s*(.+?)(?=\n\s*\*\*|\Z)',
                          raw, re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ""

        def _code(section):
            m = re.search(r'\*\*' + re.escape(section) + r'\*\*.*?```\w*\n(.*?)```',
                          raw, re.DOTALL | re.IGNORECASE)
            return m.group(1).rstrip() if m else ""

        sev_raw  = _f("SEVERITY").upper().split()[0] if _f("SEVERITY") else "MEDIUM"
        severity = sev_raw if sev_raw in _SEV else "MEDIUM"
        location = _f("LOCATION")
        line_m   = re.search(r'[:\s](\d{1,5})\b', location)
        problem  = _f("PROBLEM")
        if not problem:
            continue
        blocks.append({
            "problem":      problem,
            "severity":     severity,
            "location":     location,
            "line_number":  int(line_m.group(1)) if line_m else None,
            "current_code": _code("CURRENT CODE"),
            "fixed_code":   _code("FIXED CODE"),
            "why":          _f("WHY"),
        })
    return blocks

def _compute_delta(new_blocks: list, previous_text: str) -> tuple:
    """
    Compare les blocs actuels avec l'analyse précédente.
    Retourne (new_blocks, known_blocks) :
      new_blocks   : problèmes qui n'existaient pas au run précédent
      known_blocks : problèmes déjà signalés au run précédent (résumé compact)

    Matching par similarité du texte du problème (sans case, sans ponctuation).
    On évite le matching exact car Gemini reformule légèrement d'un run à l'autre.
    """
    import re as _re

    def _normalize(text: str) -> str:
        return _re.sub(r"[^a-z0-9]", "", text.lower())[:80]

    if not previous_text:
        return new_blocks, []

    old_blocks   = parse_fix_blocks(previous_text)
    old_norms    = {_normalize(b["problem"]) for b in old_blocks}

    new_only  = []
    known     = []
    for b in new_blocks:
        norm = _normalize(b["problem"])
        # Matching partiel : on cherche si la norm actuelle contient 60%+ d'une norm ancienne
        matched = False
        for old_n in old_norms:
            if old_n and norm and (old_n[:40] in norm or norm[:40] in old_n):
                matched = True
                break
        if matched:
            known.append(b)
        else:
            new_only.append(b)

    return new_only, known


def make_diff(current: str, fixed: str) -> str:
    if not current and not fixed:
        return ""
    cur_set = {l.strip() for l in current.splitlines() if l.strip()}
    fix_set = {l.strip() for l in fixed.splitlines()   if l.strip()}
    out = []
    for l in current.splitlines():
        if l.strip() in cur_set - fix_set:
            out.append(f"  {_RD}- {l}{_R}")
    for l in fixed.splitlines():
        if l.strip() in fix_set - cur_set:
            out.append(f"  {_GR}+ {l}{_R}")
    return "\n".join(out[:12])


def print_block(block: dict, file_name: str) -> None:
    color, icon, label = _SEV.get(block["severity"], (_YL, "🟡", "MOYENNE"))
    loc      = block.get("location", "")
    line_num = block.get("line_number")
    print(f"\n{icon} [{_BD}{color}{label}{_R}] {_BD}{block['problem']}{_R}")
    if line_num:
        print(f"   \U0001f4cd {_CY}{file_name}:{line_num}{_R}  {_DM}({loc}){_R}")
    elif loc:
        print(f"   \U0001f4cd {_CY}{file_name}{_R}  {_DM}\u2192 {loc}{_R}")
    diff = make_diff(block.get("current_code", ""), block.get("fixed_code", ""))
    if diff:
        print(); print(diff)
    if block.get("why"):
        why = block["why"].replace("\n", " ").strip()
        if len(why) > 140: why = why[:137] + "\u2026"
        print(f"\n   \U0001f4a1 {why}")


def print_results(text: str, file_name: str, context: dict,
                  elapsed: float, analyzed_count: int, score: int, impacted: list,
                  previous_analysis: str = "") -> None:
    all_blocks = parse_fix_blocks(text)
    new_blocks, known_blocks = _compute_delta(all_blocks, previous_analysis)

    # Le header reflète les NOUVEAUX problèmes uniquement
    blocks = new_blocks
    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for b in new_blocks:
        counts[b["severity"]] = counts.get(b["severity"], 0) + 1

    now = datetime.now().strftime("%H:%M:%S")
    hc  = _RD if counts["CRITICAL"] else _OR if counts["HIGH"] else _YL if counts["MEDIUM"] else _GR

    # Label du score selon sa valeur — évite la confusion "100/100 avec 5 critiques"
    # Ce score mesure l'IMPORTANCE DU CHANGEMENT (ChangeAnalyzer), pas la qualité du code
    if score >= 70:
        score_label = f"Changement majeur"
    elif score >= 30:
        score_label = f"Changement modéré"
    elif score > 0:
        score_label = f"Changement mineur"
    else:
        score_label = f"Aucun changement"

    parts = [
        f"{_DM}[{now}]{_R}",
        f"{_BD}{file_name}{_R}",
        f"{_DM}{score_label} ({score}/100){_R}",
    ]
    if counts["CRITICAL"]: parts.append(f"\U0001f534 {_RD}{_BD}{counts['CRITICAL']} Critique(s){_R}")
    if counts["HIGH"]:     parts.append(f"\U0001f7e0 {_OR}{counts['HIGH']} Haute(s){_R}")
    if counts["MEDIUM"]:   parts.append(f"\U0001f7e1 {_YL}{counts['MEDIUM']} Moyenne(s){_R}")
    if not new_blocks and not known_blocks: parts.append(f"\U0001f7e2 {_GR}OK{_R}")
    elif not new_blocks and known_blocks:   parts.append(f"{_DM}\u21a9 {len(known_blocks)} déjà connu(s){_R}")
    parts.append(f"{_DM}{elapsed:.1f}s{_R}")

    print(f"\n{_DM}{_SEP}{_R}")
    print("  " + f"  {_DM}\u2502{_R}  ".join(parts))
    print(f"{_DM}{_SEP}{_R}")

    if not all_blocks:
        clean = text.strip()
        if any(k in clean for k in ("\u2705", "no major issues", "code quality is good", "RAS")):
            print(f"\n  {_GR}\u2705  Aucun problème majeur détecté.{_R}\n")
        else:
            print(f"\n{clean}\n")
        print(f"{_DM}{_SEP2}{_R}")
        print(f"  {_DM}{elapsed:.1f}s  \u2502  Analysés : {analyzed_count}{_R}\n")
        return

    for block in blocks:
        print(_SEP)
        print_block(block, file_name)

    print(f"\n{_DM}{_SEP}{_R}")

    # Résumé compact des problèmes déjà connus (run précédent)
    if known_blocks:
        known_crits  = sum(1 for b in known_blocks if b["severity"] == "CRITICAL")
        known_highs  = sum(1 for b in known_blocks if b["severity"] == "HIGH")
        known_others = len(known_blocks) - known_crits - known_highs
        parts_known  = []
        if known_crits:  parts_known.append(f"{_RD}{known_crits} critique(s){_R}")
        if known_highs:  parts_known.append(f"{_OR}{known_highs} haute(s){_R}")
        if known_others: parts_known.append(f"{_DM}{known_others} autre(s){_R}")
        summary = " · ".join(parts_known)
        print(f"  {_DM}↩  {len(known_blocks)} problème(s) déjà signalé(s) au run précédent"
              f" ({summary}{_DM}){_R}")

    if impacted:
        names = ", ".join(Path(p).name for p in impacted[:4])
        extra = f" +{len(impacted)-4}" if len(impacted) > 4 else ""
        print(f"\u26a0\ufe0f  {_YL}Impact sur {len(impacted)} dépendant(s) : {_BD}{names}{extra}{_R}")
    print(f"{_DM}{_SEP2}{_R}")

    # Si 0 nouveaux problèmes mais des connus → message clair
    if not new_blocks and known_blocks:
        print(f"  {_GR}\u2705  Aucun nouveau problème — {len(known_blocks)} problème(s) précédent(s) "
              f"toujours présent(s).{_R}")

    print(f"  {_DM}{elapsed:.1f}s  \u2502  Analysés : {_BD}{analyzed_count}{_R}\n")


def print_minor_change(reason: str) -> None:
    """Affiche RAS pour un changement mineur — zéro token LLM consommé."""
    print(f"  {_GR}✅ RAS — {reason}{_R}\n")

def print_solution(solution_text: str, file_name: str, changes: list,
                   language: str, elapsed: float = 0.0, analyzed_count: int = 0,
                   score: int = 0, impacted: list = None) -> None:
    """
    Affiche la classe complète réécrite directement dans le terminal.
    Aucune sauvegarde fichier — tout dans la console.
    """
    import re as _re

    W = 72
    now = datetime.now().strftime("%H:%M:%S")

    # Regex robuste : accepte espaces variables et lignes vides autour des marqueurs
    match = _re.search(
        r"---+\s*SOLUTION\s+START\s*---+[^`]*```\w*\n(.*?)```[^-]*---+\s*SOLUTION\s+END",
        solution_text, _re.DOTALL | _re.IGNORECASE
    )
    code_block = match.group(1).rstrip() if match else ""

    # Dernier recours : premier ```java ... ``` de la réponse
    if not code_block:
        fb = _re.search(r"```(?:java|\w+)\n(package\s+\S+.*?)```",
                        solution_text, _re.DOTALL | _re.IGNORECASE)
        code_block = fb.group(1).rstrip() if fb else ""

    # ── En-tête ───────────────────────────────────────────────────────────────
    print(f"\n{_DM}{'═'*W}{_R}")
    print(f"  {_GR}{_BD}Solution complète — {file_name}{_R}"
          f"  {_DM}[{now}]  {elapsed:.1f}s{_R}")
    print(f"{_DM}{'═'*W}{_R}")

    if not code_block:
        print(f"  {_YL}⚠  Aucun code extrait — réponse brute :{_R}\n")
        print(solution_text[:800])
        print(f"\n{_DM}{'═'*W}{_R}\n")
        return

    lines     = code_block.splitlines()
    n_lines   = len(lines)
    n_methods = sum(1 for l in lines
                    if any(kw in l for kw in ("public ", "private ", "protected "))
                    and "(" in l and not l.strip().startswith("//"))

    print(f"  {_GR}✓{_R}  {n_lines} lignes · ~{n_methods} méthode(s)\n")

    # ── Code affiché ligne par ligne ──────────────────────────────────────────
    print(f"{_DM}{'─'*W}{_R}")
    for line in lines:
        # Colorer les mots-clés Java critiques pour la lisibilité
        display = line
        if "try (" in line or "try{" in line:
            display = f"{_GR}{line}{_R}"
        elif any(x in line for x in ("conn.rollback", "ROLLBACK", "setAutoCommit")):
            display = f"{_GR}{line}{_R}"
        elif any(x in line for x in ("SQL injection", "PROBLEM", "CRITICAL", "SECURITY")):
            display = f"{_DM}{line}{_R}"
        print(f"  {display}")
    print(f"{_DM}{'─'*W}{_R}")

    # ── Résumé des corrections (nettoie le Markdown Gemini) ───────────────────
    if changes:
        # Filtrer les entrées vides et les lignes purement Markdown
        clean = []
        for c in changes:
            c = _re.sub(r"\*+", "", c).strip()   # enlever ** et *
            c = _re.sub(r"`([^`]+)`", r"\1", c)  # enlever les backticks
            if c and len(c) > 4 and not c.startswith("General"):
                clean.append(c)

        if clean:
            print(f"\n  {_BD}Corrections appliquées :{_R}")
            for ch in clean[:20]:
                print(f"    {_GR}✓{_R}  {ch}")
            if len(clean) > 20:
                print(f"    {_DM}... et {len(clean)-20} autre(s){_R}")

    # ── Pied de page ──────────────────────────────────────────────────────────
    if impacted:
        names = ", ".join(Path(p).name for p in (impacted or [])[:4])
        extra = f" +{len(impacted)-4}" if len(impacted) > 4 else ""
        print(f"\n  {_YL}⚠  Impact sur {len(impacted)} dépendant(s) : {_BD}{names}{extra}{_R}")

    print(f"\n{_DM}{'═'*W}{_R}\n")


def print_targeted_methods(methods: list, file_name: str, remaining: list,
                            elapsed: float, analyzed_count: int, score: int,
                            impacted: list, previous_analysis: str = "") -> None:
    """
    Affiche les méthodes réécrites ciblées par le LLM.
    Chaque méthode est affichée avec son code complet prêt à copier.
    """
    now = datetime.now().strftime("%H:%M:%S")

    print(f"\n{_DM}{_SEP}{_R}")
    parts = [
        f"{_DM}[{now}]{_R}",
        f"{_BD}{file_name}{_R}",
        f"{_CY}{len(methods)} méthode(s) réécrite(s){_R}",
        f"{_DM}{elapsed:.1f}s{_R}",
    ]
    print("  " + f"  {_DM}\u2502{_R}  ".join(parts))
    print(f"{_DM}{_SEP}{_R}")

    for i, m in enumerate(methods, 1):
        name = m.get("name", "?")
        code = m.get("code", "")
        why  = m.get("why",  "")
        n_lines = len(code.splitlines())

        print(f"\n{_GR}\u25b6{_R}  {_BD}{name}(){_R}  {_DM}({n_lines} lignes){_R}")
        if why:
            short = why.replace("\n", " ").strip()
            if len(short) > 120: short = short[:117] + "\u2026"
            print(f"   {_DM}{short}{_R}")

        if code:
            print(f"\n{_DM}```{_R}")
            # Afficher les 20 premières lignes max dans le terminal
            lines = code.splitlines()
            for l in lines[:20]:
                print(f"  {l}")
            if len(lines) > 20:
                print(f"  {_DM}... ({len(lines)-20} lignes supplémentaires dans le fichier){_R}")
            print(f"{_DM}```{_R}")

    # Problèmes restants (autres méthodes, diffs simples)
    if remaining:
        _, known = _compute_delta(remaining, previous_analysis)
        new_remaining = [b for b in remaining if b not in known]
        if new_remaining:
            print(f"\n{_DM}Autres problèmes détectés :{_R}")
            for block in new_remaining[:5]:
                color, icon, label = _SEV.get(block["severity"], (_YL, "🟡", "MOYENNE"))
                print(f"  {icon} [{color}{label}{_R}] {block['problem']}")

    print(f"\n{_DM}{_SEP2}{_R}")
    if impacted:
        names = ", ".join(Path(p).name for p in impacted[:4])
        print(f"\u26a0\ufe0f  {_YL}Impact : {_BD}{names}{_R}")
    print(f"  {_DM}{elapsed:.1f}s  \u2502  Analysés : {_BD}{analyzed_count}{_R}\n")