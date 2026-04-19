import os
import json
import time
import sys
import re
from pathlib import Path
from typing import List, Dict, Any

# Ajouter le répertoire racine au path pour importer les services
sys.path.append(os.getcwd())

from agents.analysis_agent import analysis_agent
from output.console_renderer import parse_fix_blocks
from agents.code_agent import code_agent
from services.llm_service import assistant_agent

# Couleurs ANSI
_R = "\033[0m"
_B = "\033[1m"
_GR = "\033[92m"
_YL = "\033[93m"
_RD = "\033[91m"
_CY = "\033[96m"

class BenchmarkRunner:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.files_dir = base_dir / "files"
        self.annotations_dir = base_dir / "annotations"
        
        # Statistiques
        self.results = []
        self.metrics = {
            "TP": 0,  # True Positives: Bug attendu et trouvé
            "FP": 0,  # False Positives: Bug trouvé mais non attendu
            "FN": 0,  # False Negatives: Bug attendu mais non trouvé
        }

    def run(self):
        print(f"\n{_B}🚀 DÉMARRAGE DU BENCHMARK QUALITÉ LLM{_R}\n")
        
        # Initialisation minimale des services
        print(f"[{_CY}INFO{_R}] Initialisation des services...")
        assistant_agent.__init__() # Re-init pour être propre
        analysis_agent.set_llm_service(assistant_agent)
        
        benchmark_files = list(self.files_dir.glob("*"))
        
        for file_path in benchmark_files:
            self._process_file(file_path)
            
        self._print_final_report()

    def _process_file(self, file_path: Path):
        print(f"\n{_B}📄 Analyse de {file_path.name}...{_R}")
        
        # 1. Charger l'annotation
        annotation_path = self.annotations_dir / f"{file_path.stem}.json"
        if not annotation_path.exists():
            print(f"  {_RD}✘ Annotation manquante pour {file_path.name}{_R}")
            return
            
        with open(annotation_path, "r") as f:
            annotation = json.load(f)
            
        expected_bugs = annotation.get("expected_bugs", [])
        
        # 2. Lire le code
        code = file_path.read_text(encoding="utf-8")
        
        # 3. Lancer l'analyse (Context pour forcer block_fix dans le benchmark)
        context = {
            "file_path": str(file_path),
            "language": code_agent.detect_language(file_path),
            "neighborhood": {"predecessors": [], "successors": [], "criticality": 0},
            "post_solution_mode": True,  # FORCE LE MODE BLOCK_FIX
            "post_solution_hint": "CHOOSE block_fix strategy with individual ---FIX START--- blocks for EACH issue found. full_class is FORBIDDEN. If the code is good, use block_fix with 0 blocks."
        }
        
        start_time = time.time()
        analysis_result = analysis_agent.analyze(code, context)
        elapsed = time.time() - start_time
        
        raw_text = analysis_result.get("analysis", "")
        # print(f"DEBUG RAW TEXT:\n{raw_text[:200]}...\n{'='*20}")
        found_blocks = parse_fix_blocks(raw_text)
        
        print(f"  Temps : {elapsed:.1f}s | Blocs trouvés : {len(found_blocks)}")
        
        # 4. Évaluation
        self._evaluate(file_path.name, expected_bugs, found_blocks)

    def _evaluate(self, filename: str, expected: List[Dict], found: List[Dict]):
        # On essaie de matcher chaque bug attendu avec un bug trouvé
        matched_found_indices = set()
        
        for exp in expected:
            match_found = False
            for i, f in enumerate(found):
                if i in matched_found_indices:
                    continue
                
                # Check pattern (plus ou moins flexible)
                found_pattern = self._normalize_pattern(f.get("problem", ""))
                if exp["pattern"].lower() in found_pattern.lower():
                    match_found = True
                    matched_found_indices.add(i)
                    break
            
            if match_found:
                self.metrics["TP"] += 1
                print(f"  {_GR}✓ MATCH:{_R} Found expected bug '{exp['pattern']}'")
            else:
                self.metrics["FN"] += 1
                print(f"  {_RD}✘ MISSING:{_R} Expected bug '{exp['pattern']}' not found")
                
        # Les blocs trouvés qui n'ont pas été matchés sont des FP
        fps = len(found) - len(matched_found_indices)
        if fps > 0:
            self.metrics["FP"] += fps
            print(f"  {_YL}⚠ FALSE POSITIVES:{_R} {fps} non-expected bug(s) reported")
        elif not expected and not found:
             print(f"  {_GR}✓ CLEAN:{_R} Correctly identified as bug-free")

    def _normalize_pattern(self, text: str) -> str:
        # Simplification pour le matching
        return text.replace("_", "").replace(" ", "").lower()

    def _print_final_report(self):
        tp = self.metrics["TP"]
        fp = self.metrics["FP"]
        fn = self.metrics["FN"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{_B}{'='*60}{_R}")
        print(f"{_B}📊 RAPPORT FINAL DE QUALITÉ LLM{_R}")
        print(f"{'='*60}")
        print(f"  Vrais Positifs (TP) : {tp}")
        print(f"  Faux Positifs (FP)  : {fp}")
        print(f"  Faux Négatifs (FN)  : {fn}")
        print(f"{'-'*60}")
        print(f"  {_B}PRECISION :{_R} {precision:.2%}")
        print(f"  {_B}RECALL    :{_R} {recall:.2%}")
        print(f"  {_B}F1-SCORE  :{_R} {f1:.2f}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    runner = BenchmarkRunner(Path("benchmarks"))
    runner.run()
