"""
Analyseur de projet complet avec cohérence
"""
from pathlib import Path
from typing import List, Dict, Any, Set
from services.code_parser import parser
from services.llm_service import assistant_agent
from services.graph_service import dependency_builder
import re


class ProjectAnalyzer:
    """Analyse complète d'un projet avec maintien de cohérence"""
    
    def __init__(self):
        self.dependency_graph = None
        self.all_analyses = {}
        self.proposed_changes = {} 
        
    def analyze_full_project(self, project_path: Path, max_files: int = 10) -> Dict[str, Any]:
        """
        Analyse complète avec contexte global
        
        Returns:
            {
                'structure_analysis': analyse de l'architecture,
                'file_analyses': analyses détaillées par fichier,
                'refactoring_plan': plan de refactoring global,
                'conflicts': conflits détectés,
                'dependency_graph': graphe de dépendances
            }
        """
        
        print(" Construction du graphe de dépendances...")
        # 1. Construire le graphe de dépendances
        self.dependency_graph = dependency_builder.build_from_project(project_path)
        
        # 2. Analyser la structure globale
        print(" Analyse de la structure globale...")
        structure_analysis = dependency_builder.analyze_flows()
        
        # 3. Identifier les fichiers critiques (les plus connectés)
        print(" Identification des fichiers critiques...")
        critical_files = self._identify_critical_files(structure_analysis, max_files)
        
        print(f" {len(critical_files)} fichiers critiques identifiés\n")
        
        # 4. Analyser chaque fichier AVEC son contexte de dépendances
        for i, file_path in enumerate(critical_files, 1):
            print(f"\n Analyse {i}/{len(critical_files)}: {file_path.name}")
            
            # Construire le contexte du fichier
            context = self._build_file_context(file_path)
            
            # Lire le code
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
            except Exception as e:
                print(f" Erreur de lecture: {e}")
                continue
            
            # UTILISER assistant_agent.analyze_code_with_rag() !
            # C'est la méthode qui utilise le RAG et les best practices
            analysis = assistant_agent.analyze_code_with_rag(
                code=code,
                context={
                    'file_path': str(file_path),
                    'language': parser._detect_language(file_path),
                    'dependencies': context['dependencies'],
                    'dependents': context['dependents'],
                    'criticality_score': context['criticality'],
                    'is_entry_point': context['is_entry_point']
                }
            )
            
            self.all_analyses[str(file_path)] = analysis
            
            # Extraire les changements proposés
            self._extract_proposed_changes(str(file_path), analysis['analysis'])
        
        # 5. Générer un plan de refactoring cohérent
        print("\n Génération du plan de refactoring global...")
        refactoring_plan = assistant_agent.generate_refactoring_plan(
            list(self.all_analyses.values())
        )
        
        # 6. Vérifier la cohérence des corrections
        print(" Vérification de la cohérence...")
        conflicts = self._detect_conflicts()
        
        return {
            'structure_analysis': structure_analysis,
            'file_analyses': self.all_analyses,
            'refactoring_plan': refactoring_plan,
            'conflicts': conflicts,
            'dependency_graph': self.dependency_graph,
            'critical_files': [str(f) for f in critical_files]
        }
    
    def _identify_critical_files(self, structure: Dict, max_files: int) -> List[Path]:
        """
        Identifie les fichiers critiques à analyser en priorité
        Basé sur le couplage (nombre de dépendances + dépendants)
        """
        coupling = structure['coupling_metrics']
        
        # Trier par score de couplage (afferent + efferent)
        critical = sorted(
            coupling.items(),
            key=lambda x: x[1]['afferent'] + x[1]['efferent'],
            reverse=True
        )
        
        # Extraire les chemins de fichiers (format: "file:path")
        file_paths = []
        for node_id, metrics in critical[:max_files * 2]: 
            if node_id.startswith('file:'):
                path = Path(node_id.replace('file:', ''))
                if path.exists():
                    file_paths.append(path)
                    if len(file_paths) >= max_files:
                        break
        
        return file_paths
    
    def _build_file_context(self, file_path: Path) -> Dict[str, Any]:
        """Construit le contexte de dépendances d'un fichier"""
        node_id = f"file:{file_path}"
        
        # Dépendances (ce que ce fichier utilise)
        dependencies = []
        if self.dependency_graph.has_node(node_id):
            dependencies = [
                n.replace('file:', '') 
                for n in self.dependency_graph.successors(node_id)
                if n.startswith('file:')
            ]
        
        # Dépendants (ce qui utilise ce fichier)
        dependents = []
        if self.dependency_graph.has_node(node_id):
            dependents = [
                n.replace('file:', '')
                for n in self.dependency_graph.predecessors(node_id)
                if n.startswith('file:')
            ]
        
        # Criticité basée sur le nombre de dépendants
        criticality = len(dependents)
        
        # Est-ce un point d'entrée? (pas de dépendants)
        is_entry_point = len(dependents) == 0
        
        return {
            'dependencies': dependencies,
            'dependents': dependents,
            'criticality': criticality,
            'is_entry_point': is_entry_point
        }
    
    def _extract_proposed_changes(self, file_path: str, analysis_text: str):
        """Extrait les changements proposés de l'analyse"""
        # Patterns pour détecter les renommages de méthodes/classes
        patterns = [
            r'rename\s+(\w+)\s+(?:to|→)\s+(\w+)',
            r'renommer\s+(\w+)\s+(?:en|→)\s+(\w+)',
            r'(\w+)\s*→\s*(\w+)',
        ]
        
        changes = []
        for pattern in patterns:
            matches = re.finditer(pattern, analysis_text, re.IGNORECASE)
            for match in matches:
                changes.append({
                    'type': 'rename',
                    'old_name': match.group(1),
                    'new_name': match.group(2),
                    'file': file_path
                })
        
        self.proposed_changes[file_path] = changes
    
    def _detect_conflicts(self) -> List[Dict]:
        """
        Détecte les conflits entre corrections proposées
        Ex: Si FileA renomme une méthode utilisée par FileB
        """
        conflicts = []
        
        # Pour chaque changement proposé
        for file_path, changes in self.proposed_changes.items():
            for change in changes:
                if change['type'] == 'rename':
                    # Vérifier si d'autres fichiers utilisent l'ancien nom
                    conflict = self._check_rename_conflict(
                        file_path,
                        change['old_name'],
                        change['new_name']
                    )
                    if conflict:
                        conflicts.append(conflict)
        
        return conflicts
    
    def _check_rename_conflict(self, source_file: str, old_name: str, new_name: str) -> Dict:
        """Vérifie si un renommage casse d'autres fichiers"""
        node_id = f"file:{source_file}"
        
        if not self.dependency_graph.has_node(node_id):
            return None
        
        # Fichiers qui dépendent de source_file
        dependents = [
            n.replace('file:', '')
            for n in self.dependency_graph.predecessors(node_id)
            if n.startswith('file:')
        ]
        
        if not dependents:
            return None
        
        # Vérifier si ces fichiers utilisent old_name
        affected_files = []
        for dep_file in dependents:
            try:
                with open(dep_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                if old_name in content:
                    affected_files.append(dep_file)
            except:
                pass
        
        if affected_files:
            return {
                'type': 'rename_conflict',
                'source_file': source_file,
                'old_name': old_name,
                'new_name': new_name,
                'affected_files': affected_files,
                'severity': 'HIGH',
                'message': f"Renommer '{old_name}' → '{new_name}' dans {source_file} "
                          f"affectera {len(affected_files)} autre(s) fichier(s)"
            }
        
        return None


# Instance globale
project_analyzer = ProjectAnalyzer()