# Code Auditor — CI/CD Workflow Documentation

## Vue d'ensemble

Le système CI/CD de Code Auditor permet d'analyser automatiquement les Pull Requests sur GitHub en utilisant l'IA (Gemini/OpenRouter) pour reviewer le code.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Développeur   │────▶│  Push sur PR    │────▶│ GitHub Actions  │
│                 │     │                 │     │   se déclenche  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Commentaires   │◀────│  Analyse IA     │◀────│  Code Auditor   │
│   sur la PR     │     │  (RAG + KG)     │     │   s'exécute     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Architecture du Système CI/CD

### 1. Déploiement (`ci_deploy_agent.py`)

C'est l'agent qui déploie le workflow sur un repository GitHub.

#### Flux de déploiement

```
Développeur
    │
    │ python main.py ci-deploy --repo owner/repo
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    ci_deploy_agent.py                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [0/6] Validation Token                                     │
│        ├── Vérifie GITHUB_TOKEN                              │
│        ├── Test API /user                                    │
│        └── ✓ ou ✗                                           │
│                                                             │
│  [0b/6] Validation Secrets                                  │
│        ├── Vérifie GOOGLE_API_KEY                           │
│        └── Vérifie OPENROUTER_API_KEY                       │
│                                                             │
│  [1/6] Check Workflow existant                              │
│        ├── MCP get_file_content(".github/workflows/ci.yml") │
│        └── Si existe et pas --force → arrêt                 │
│                                                             │
│  [2/6] Détection Profil                                     │
│        ├── Check pom.xml → Java/Maven                       │
│        ├── Check package.json → Node.js                     │
│        ├── Check requirements.txt → Python                  │
│        └── Sinon → unknown/unknown                          │
│                                                             │
│  [3/6] Génération YAML                                      │
│        └── workflow_generator.py                            │
│                                                             │
│  [3b/6] Validation YAML                                     │
│        ├── Parse YAML                                        │
│        ├── Check jobs build-test + code-review              │
│        └── ✓ ou ✗                                           │
│                                                             │
│  [4/6] Préparation requirements-ci.txt                      │
│        ├── Lit fichier local                                 │
│        └── Sinon → génère version minimal                   │
│                                                             │
│  [5/6] Push des fichiers                                    │
│        ├── MCP push_file(requirements-ci.txt)               │
│        └── MCP push_file(ci.yml)                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Repository GitHub (workflow actif)
```

#### Commande de déploiement

```bash
# Déploiement standard
python main.py ci-deploy --repo chmaryem/mon-projet

# Forcer le remplacement d'un workflow existant
python main.py ci-deploy --repo chmaryem/mon-projet --force

# Spécifier une branche
python main.py ci-deploy --repo chmaryem/mon-projet --branch develop
```

---

### 2. Générateur de Workflow (`workflow_generator.py`)

Génère le fichier YAML GitHub Actions adapté au langage du projet.

#### Structure du YAML généré

```yaml
name: "CI/CD — Build + Code Auditor"

# Déclencheurs
trigger:
  - pull_request (opened, synchronize, reopened)
  - push (branche main)

# Permissions nécessaires
permissions:
  - statuses: write      # Pour poster les status checks
  - pull-requests: write # Pour commenter la PR
  - contents: read       # Pour lire le code

# Deux jobs en parallèle
jobs:
  1. build-test:    # Compile et teste le projet
  2. code-review:   # Analyse IA du code
```

#### Détection automatique du langage

| Fichier détecté | Langage | Build System |
|-----------------|---------|--------------|
| `pom.xml` | Java | Maven |
| `build.gradle` ou `gradlew` | Java | Gradle |
| `package.json` | JavaScript/TypeScript | npm/yarn |
| `requirements.txt` ou `setup.py` | Python | pip |

---

### 3. Job "code-review" (exécuté par GitHub Actions)

```yaml
# Job 2 : Code Auditor IA Review
code-review:
  name: "Code Auditor Review"
  runs-on: ubuntu-latest
  timeout-minutes: 20
  if: github.event_name == 'pull_request'  # Uniquement sur PR
  
  steps:
    1. Checkout du projet cible
    2. Checkout de Code Auditor (chmaryem/code_auditor)
    3. Setup Python 3.11
    4. Setup Node.js 20 (pour MCP)
    5. Cache pip (optimisation)
    6. Install Code Auditor (pip install)
    7. Pre-install MCP GitHub server
    8. Analyse PR (python -m ci_cd.ci_runner)
```

#### Détail du step "Analyse PR"

```yaml
- name: "Analyse PR"
  working-directory: code_auditor_tool    # ← Exécute dans ce dossier
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    GITHUB_PERSONAL_ACCESS_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
    GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
    PYTHONPATH: "${{ github.workspace }}/code_auditor_tool"  # ← Important !
  run: |
    python -m ci_cd.ci_runner \
      --repo "${{ github.repository }}" \
      --pr "${{ github.event.pull_request.number }}"
```

**Pourquoi `PYTHONPATH` est crucial :**
- Sans lui, Python ne trouve pas le module `ci_cd`
- Il indique à Python où chercher les modules

---

### 4. Runner CI (`ci_runner.py`)

Point d'entrée exécuté dans GitHub Actions.

#### Séquence d'exécution

```
┌─────────────────────────────────────────────────────────┐
│                    ci_runner.py                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [0/5] Test connexion MCP                             │
│        ├── subprocess.run(["npx", "--yes", ...])        │
│        └── Vérifie que Node.js/MCP sont disponibles     │
│                                                         │
│  [1/5] Récupération SHA HEAD                           │
│        ├── GitHub API: GET /repos/{owner}/{repo}/     │
│        └── Récupère le SHA du dernier commit de la PR   │
│                                                         │
│  [2/5] Post Status "Pending"                           │
│        ├── POST /repos/{owner}/{repo}/statuses/{sha}   │
│        └── state: "pending", description: "Analyse..." │
│                                                         │
│  [3/5] Analyse PR (review_pr)                          │
│        ├── Import smart_git.pr_review_agent            │
│        ├── asyncio.run(review_pr(owner, repo, pr))       │
│        └── Retourne: verdict, score, issues...          │
│                                                         │
│  [4/5] Post Status Final                                │
│        ├── Si APPROVE → state: "success"                │
│        ├── Si COMMENT → state: "success"                │
│        └── Si REQUEST_CHANGES → state: "failure"      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### Codes de sortie

| Verdict | Exit Code | Bloque le merge ? |
|---------|-----------|-------------------|
| APPROVE | 0 | Non |
| COMMENT | 0 | Non |
| REQUEST_CHANGES | 1 | **Oui** |

---

### 5. Status Reporter (`ci_status_reporter.py`)

Poste les status checks sur GitHub.

#### API GitHub utilisée

```python
# POST /repos/{owner}/{repo}/statuses/{sha}
{
  "state": "pending" | "success" | "failure",
  "target_url": "https://github.com/.../actions/runs/...",
  "description": "Code Auditor analyse en cours...",
  "context": "Code Auditor / PR Review"
}
```

#### États possibles

| État | Description | Icône PR |
|------|-------------|----------|
| `pending` | Analyse en cours | 🟡 Jaune |
| `success` | Analyse OK (APPROVE/COMMENT) | 🟢 Vert |
| `failure` | Bugs trouvés (REQUEST_CHANGES) | 🔴 Rouge |

---

## Flux Complet : De la PR au Commentaire

```
1. Développeur crée une PR
   │
   ▼
2. GitHub Actions détecte la PR
   └── trigger: pull_request
   │
   ▼
3. Workflow YAML s'exécute
   ├── Job build-test: compile/test (optionnel)
   └── Job code-review: analyse IA
   │
   ▼
4. ci_runner.py s'exécute
   ├── [0] Test MCP
   ├── [1] Récupère SHA
   ├── [2] Post "pending"
   ├── [3] Analyse (review_pr)
   └── [4] Post résultat
   │
   ▼
5. Résultat visible sur la PR
   ├── Status check (vert/jaune/rouge)
   └── Commentaire avec détails
```

---

## Fichiers importants

### côté Repository cible (déployés)

| Fichier | Description |
|---------|-------------|
| `.github/workflows/ci.yml` | Workflow GitHub Actions |
| `requirements-ci.txt` | Dépendances Python pour le runner |

### côté Code Auditor (source)

| Fichier | Rôle |
|---------|------|
| `ci_cd/ci_deploy_agent.py` | Déploie le workflow |
| `ci_cd/workflow_generator.py` | Génère le YAML |
| `ci_cd/ci_runner.py` | Exécute l'analyse dans Actions |
| `ci_cd/ci_status_reporter.py` | Poste les status |

---

## Secrets requis sur GitHub

Configure dans : `Settings > Secrets > Actions`

| Secret | Obligatoire | Description |
|--------|-------------|-------------|
| `GITHUB_TOKEN` | ✅ Oui | Fourni automatiquement par GitHub Actions |
| `GOOGLE_API_KEY` | ✅ Oui | Clé API Gemini pour l'analyse IA |
| `OPENROUTER_API_KEY` | ❌ Non | Alternative à Gemini (fallback) |

---

## Dépannage courant

### Erreur : `No module named 'ci_cd'`

**Cause** : PYTHONPATH mal configuré

**Solution** : Vérifier que le workflow YAML contient :
```yaml
env:
  PYTHONPATH: "${{ github.workspace }}/code_auditor_tool"
```

### Erreur : `MCP server timeout`

**Cause** : Node.js/npm non installé ou problème réseau

**Solution** : Vérifier que le step "Setup Node.js 20" est présent

### Erreur : `401 Unauthorized`

**Cause** : Token GitHub invalide ou permissions insuffisantes

**Solution** : Vérifier `GITHUB_TOKEN` et permissions du repo

---

## Résumé des validations ajoutées

| Validation | Fichier | Quand exécuté ? |
|------------|---------|-----------------|
| Token GitHub valide | `ci_deploy_agent.py` | Avant déploiement |
| Secrets présents | `ci_deploy_agent.py` | Avant déploiement (warning) |
| YAML valide | `workflow_generator.py` | Avant push |
| requirements-ci.txt | `ci_deploy_agent.py` | Génération auto si manquant |
| Connexion MCP | `ci_runner.py` | Au démarrage dans Actions |
