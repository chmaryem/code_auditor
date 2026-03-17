"""
File Watcher - Surveillance en temps réel du système de fichiers
"""
import time
from pathlib import Path
from typing import Callable, Set, Dict
from threading import Timer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from config import config


class CodeChangeHandler(FileSystemEventHandler):
    """
    Gestionnaire d'événements du système de fichiers
    Filtre les événements pertinents et applique le debouncing
    """
    
    def __init__(self, callback: Callable[[Path], None], debounce_seconds: float = 2.0):
        """
        Args:
            callback: Fonction appelée quand un fichier change (après debouncing)
            debounce_seconds: Temps d'attente après le dernier événement
        """
        super().__init__()
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        
        # Timers pour le debouncing (un par fichier)
        self.debounce_timers: Dict[str, Timer] = {}
        
        # Timestamps des derniers événements (filtrage dupliqués)
        self.last_event_times: Dict[str, float] = {}
        
        # Extensions surveillées
        self.watched_extensions = set(config.analysis.supported_languages)
        self._build_extension_set()
        
        # Dossiers exclus
        self.excluded_dirs = set(config.analysis.exclude_patterns)
    
    def _build_extension_set(self):
        """Construit l'ensemble des extensions à surveiller"""
        # Mapping language → extensions
        lang_to_ext = {
            'python': {'.py'},
            'javascript': {'.js', '.jsx'},
            'typescript': {'.ts', '.tsx'},
            'java': {'.java'}
        }
        
        extensions = set()
        for lang in self.watched_extensions:
            extensions.update(lang_to_ext.get(lang, set()))
        
        self.watched_extensions = extensions
    
    def _should_process_file(self, file_path: Path) -> bool:
        """
        Détermine si un fichier doit être traité
        
        Args:
            file_path: Chemin du fichier
            
        Returns:
            True si le fichier doit être analysé
        """
        # Vérifier l'extension
        if file_path.suffix not in self.watched_extensions:
            return False
        
        # Vérifier les dossiers exclus
        for excluded in self.excluded_dirs:
            # Convertir le pattern en vérification simple
            excluded_clean = excluded.replace('**/', '').replace('/**', '')
            if excluded_clean in file_path.parts:
                return False
        
        return True
    
    def _schedule_analysis(self, file_path: str):
        """
        Programme une analyse avec debouncing
        
        Args:
            file_path: Chemin du fichier modifié
        """
        # Annuler le timer précédent s'il existe
        if file_path in self.debounce_timers:
            self.debounce_timers[file_path].cancel()
        
        # Créer un nouveau timer
        timer = Timer(
            self.debounce_seconds,
            self._execute_callback,
            args=[file_path]
        )
        self.debounce_timers[file_path] = timer
        timer.start()
    
    def _execute_callback(self, file_path: str):
        """
        Exécute le callback après le délai de debouncing
        
        Args:
            file_path: Chemin du fichier à analyser
        """
        # Nettoyer le timer
        if file_path in self.debounce_timers:
            del self.debounce_timers[file_path]
        
        # Appeler le callback
        try:
            self.callback(Path(file_path))
        except Exception as e:
            print(f" Erreur lors de l'analyse de {file_path}: {e}")
    
    def on_modified(self, event: FileSystemEvent):
        """Appelé quand un fichier est modifié"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        if not self._should_process_file(file_path):
            return
        
        # Filtrer événements dupliqués (< 0.5s)
        file_key = str(file_path)
        now = time.time()
        
        if file_key in self.last_event_times:
            if now - self.last_event_times[file_key] < 0.5:
                return  # Ignorer doublon
        
        self.last_event_times[file_key] = now
        self._schedule_analysis(file_key)
    
    def on_created(self, event: FileSystemEvent):
        """Appelé quand un fichier est créé"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        if self._should_process_file(file_path):
            print(f" Nouveau fichier : {file_path.name}")
            self._schedule_analysis(str(file_path))
    
    def on_deleted(self, event: FileSystemEvent):
        """Appelé quand un fichier est supprimé"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        if self._should_process_file(file_path):
            print(f"  Fichier supprimé : {file_path.name}")
            # Pour les suppressions, pas de debouncing
            try:
                self.callback(file_path, deleted=True)
            except Exception as e:
                print(f" Erreur lors de la gestion de suppression : {e}")


class FileWatcher:
    """
    Surveillant de fichiers principal
    Orchestre la surveillance et la gestion des événements
    """
    
    def __init__(self, project_path: Path, callback: Callable[[Path], None]):
        """
        Args:
            project_path: Chemin racine du projet à surveiller
            callback: Fonction appelée pour chaque changement de fichier
        """
        self.project_path = project_path
        self.callback = callback
        
        # Observer Watchdog
        self.observer = Observer()
        
        # Handler des événements
        self.handler = CodeChangeHandler(
            callback=self._on_file_changed,
            debounce_seconds=config.watcher.debounce_seconds  # 4.0s — corrige la double analyse
        )
        
        # État
        self.is_running = False
        self.files_processed = 0
    
    def _on_file_changed(self, file_path: Path, deleted: bool = False):
        """
        Callback interne appelé par le handler
        
        Args:
            file_path: Fichier modifié
            deleted: True si le fichier a été supprimé
        """
        self.files_processed += 1
        
        try:
            if deleted:
                self.callback(file_path, deleted=True)
            else:
                # Vérifier que le fichier existe toujours
                if not file_path.exists():
                    print(f"  Fichier n'existe plus : {file_path.name}")
                    return
                
                self.callback(file_path, deleted=False)
        
        except Exception as e:
            print(f" Erreur dans le callback : {e}")
            import traceback
            traceback.print_exc()
    
    def start(self):
        """Démarre la surveillance"""
        if self.is_running:
            print(" Le watcher est déjà actif")
            return
        
        print(f"\n Initialisation du file watcher...")
        print(f" Répertoire surveillé : {self.project_path}")
        print(f" Extensions : {', '.join(self.handler.watched_extensions)}")
        print(f" Dossiers exclus : {', '.join(list(self.handler.excluded_dirs)[:3])}...\n")
        
        # Programmer la surveillance récursive
        self.observer.schedule(
            self.handler,
            str(self.project_path),
            recursive=True
        )
        
        # Démarrer l'observer
        self.observer.start()
        self.is_running = True
        
        print(" Surveillance active")
        print(" En attente de modifications...\n")
    
    def stop(self):
        """Arrête la surveillance"""
        if not self.is_running:
            return
        
        print(f"\n Arrêt de la surveillance...")
        
        # Arrêter l'observer
        self.observer.stop()
        self.observer.join(timeout=5)
        
        self.is_running = False
        
        print(f" Surveillance arrêtée")
        print(f" Fichiers traités : {self.files_processed}\n")
    
    def watch(self):
        """
        Démarre la surveillance et attend (bloquant)
        Utiliser Ctrl+C pour arrêter
        """
        self.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n  Interruption utilisateur (Ctrl+C)")
            self.stop()
    
    def get_stats(self) -> Dict[str, any]:
        """Retourne les statistiques de surveillance"""
        return {
            'is_running': self.is_running,
            'files_processed': self.files_processed,
            'project_path': str(self.project_path)
        }


# Exemple d'utilisation
if __name__ == "__main__":
    def on_file_change(file_path: Path, deleted: bool = False):
        """Callback de test"""
        if deleted:
            print(f"  → Fichier supprimé : {file_path}")
        else:
            print(f"  → Analyse de : {file_path}")
            print(f"  → Taille : {file_path.stat().st_size} bytes")
    
    # Créer le watcher
    watcher = FileWatcher(
        project_path=Path.cwd(),
        callback=on_file_change
    )
    
    # Démarrer la surveillance (bloquant)
    watcher.watch()