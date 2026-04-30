# Résumé : Intégration du Test Generator dans Code Auditor AI

## Contexte et Objectif

**Problème** : Le système "System Watch" analysait le code en temps réel mais ne proposait pas de générer des tests unitaires lorsque du code nouveau/modifié n'était pas couvert.

**Solution** : Intégrer un pipeline "Test-Aware" à 3 niveaux qui détecte les manques de tests et propose au développeur de les générer via IA.

---

## Architecture du Pipeline Test-Aware (3 Niveaux)

### Niveau 1 : Test Gap Detection (0 token)
- **Quand** : À chaque sauvegarde de fichier (Ctrl+S)
- **Comment** : Analyse heuristique sans appel LLM
- **Résultat** : Score d'impact (0-100) si des entités publiques n'ont pas de tests

### Niveau 2 : Test Proposal Notifier
- **Quand** : Si le score d'impact ≥ 50 ou couverture < 50%
- **Comment** : Notification console discrète (style GitSessionTracker)
- **Résultat** : Message non-bloquant avec commande CLI suggérée

### Niveau 3 : Test Generation (on-demand)
- **Quand** : Sur demande explicite du développeur
- **Comment** : LLM + RAG (exemples de tests du projet)
- **Résultat** : Fichier de test généré selon les conventions du projet

---

## Fichiers Créés

| Fichier | Rôle |
|---------|------|
| `services/test_discovery.py` | Détecte les conventions de test du projet (pytest, JUnit, Jest) et mappe source ↔ test |
| `agents/test_gap_agent.py` | Détection 0-token des manques de tests + calcul du score d'impact |
| `agents/test_proposal_notifier.py` | Affiche notifications console (INFO/WARN/URGENT) avec commande CLI |
| `agents/test_generator_agent.py` | Génère les tests via LLM + RAG quand le dev le demande |

## Fichiers Modifiés

| Fichier | Modification |
|---------|--------------|
| `core/orchestrator.py` | Étape 4.7 ajoutée : appelle TestGapAgent après parsing AST |
| `smart_git/git_session_tracker.py` | Intègre les test_gaps dans le SessionSnapshot + notification groupée |
| `main.py` | Nouvelle commande CLI `generate-tests` |

---

## Workflow Complet

### 1. Détection Temps Réel (Watch Mode)
```
Développeur modifie UserService.java → Ctrl+S
         ↓
    Orchestrator._analyze_file()
         ↓
    Étape 4.7 : TestGapAgent.analyze()
         ↓
    [INFO] UserService.java : 3 méthodes publiques sans test
           └─ python main.py generate-tests .../UserService.java --project ...
```

### 2. Notification de Session (GitSessionTracker)
```
Toutes les 3 minutes : scan des fichiers uncommitted
         ↓
    TestDiscoveryService.has_test_file() ?
         ↓
    Si non : ajouté à snapshot.test_gaps
         ↓
    Notification batch : "3 fichiers sans tests"
```

### 3. Génération On-Demand
```
Développeur exécute : generate-tests --write
         ↓
    TestGeneratorAgent :
      1. Détecte convention (JUnit via pom.xml)
      2. Extrait signatures publiques
      3. Récupère exemples de tests existants (RAG)
      4. Prompt LLM avec contexte
      5. Écrit src/test/java/.../UserServiceTest.java
```

---

## Commandes pour Tester sur `sample-projet`

### Étape 1 : Lancer la surveillance
```bash
cd C:\Users\Asus\Desktop\code_auditor
python main.py watch C:\Users\Asus\Desktop\sample-projet
```

### Étape 2 : Modifier un fichier Java
Ouvrir `sample-projet/src/main/java/tn/esprit/sampleprojet/UserService.java`
Ajouter une méthode publique, sauvegarder (Ctrl+S).

**Attendu dans la console** :
```
[TEST GAP] UserService.java → tests manquants
           └─ python main.py generate-tests .../UserService.java --project ...
```

### Étape 3 : Générer le test manquant
```bash
# Mode preview (affiche sans écrire)
python main.py generate-tests C:\Users\Asus\Desktop\sample-projet\src\main\java\tn\esprit\sampleprojet\UserService.java --project C:\Users\Asus\Desktop\sample-projet

# Mode écriture (crée le fichier)
python main.py generate-tests C:\Users\Asus\Desktop\sample-projet\src\main\java\tn\esprit\sampleprojet\UserService.java --project C:\Users\Asus\Desktop\sample-projet --write
```

### Étape 4 : Vérifier le résultat
```bash
dir C:\Users\Asus\Desktop\sample-projet\src\test\java\tn\esprit\sampleprojet\
```

Doit afficher : `UserServiceTest.java` créé

---

## Points Clés Techniques

### Avantage "0 Token" pour la détection
- Le TestGapAgent n'utilise **pas** de LLM pour détecter
- Il parse le code source (regex/AST simple) et vérifie l'existence du fichier de test
- Coût en tokens uniquement lors de la génération on-demand

### Notification Non-Bloquante
- Style GitSessionTracker : messages colorés mais pas de popup bloquant
- Respecte le workflow du développeur
- Propose toujours une action concrète (commande CLI)

### RAG pour la génération
- Le TestGeneratorAgent récupère 3 fichiers de test similaires dans le projet
- Le prompt LLM inclut ces exemples pour respecter le style/conventions
- Génération adaptée au framework détecté (JUnit, pytest, Jest)

---

## Résultat Attendu

| Avant | Après |
|-------|-------|
| Code modifié sans tests détecté manuellement | Notification automatique à chaque save |
| Tests écrits à la main ou oubliés | Génération IA avec contexte projet |
| Aucune visibilité sur la couverture | Score d'impact + suivi session Git |
| Conventions de test incohérentes | Détection auto + respect des patterns existants |

---

## Prochaines Étapes (Améliorations Futures)

1. **Coverage tracking** : Intégrer `pytest-cov` / JaCoCo pour mesurer la couverture réelle
2. **Test auto-run** : Exécuter les tests générés pour valider qu'ils passent
3. **Integration avec CI** : Bloquer la PR si des fichiers critiques n'ont pas de tests

---

## Fichier de Démonstration

Projet de test : `C:\Users\Asus\Desktop\sample-projet`
- 8 fichiers Java source
- 1 fichier de test existant (SampleProjetApplicationTests.java)
- 7 fichiers sans tests → cibles pour la démonstration

**Fichier recommandé pour la démo** : `UserService.java` (logique métier simple, pas de dépendances complexes)
