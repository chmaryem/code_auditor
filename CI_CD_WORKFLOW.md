# Code Auditor — CI/CD Workflow Documentation

## Vue d'ensemble

Le système CI/CD de Code Auditor déploie un pipeline classique **Build + Test + SonarQube** sur les repositories GitHub cibles.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Développeur   │────▶│  Push sur PR    │────▶│ GitHub Actions  │
│                 │     │                 │     │   se déclenche  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
                                    ┌──────────────────────────────────┐
                                    │         Workflow GitHub Actions    │
                                    │  ┌─────────────┐  ┌─────────────┐  │
                                    │  │ build-test  │──▶│ sonar-scan  │  │
                                    │  │ Build + Test│  │   SonarQube │  │
                                    │  └─────────────┘  └─────────────┘  │
                                    └──────────────────────────────────┘
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
│  [0/5] Validation Token                                     │
│        ├── Vérifie GITHUB_TOKEN                              │
│        ├── Test API /user                                    │
│        └── ✓ ou ✗                                           │
│                                                             │
│  [0b/5] Validation Secrets                                  │
│        ├── Vérifie SONAR_TOKEN                               │
│        └── Vérifie SONAR_HOST_URL                            │
│                                                             │
│  [1/5] Check Workflow existant                              │
│        ├── MCP get_file_content(".github/workflows/ci.yml") │
│        └── Si existe et pas --force → arrêt                 │
│                                                             │
│  [2/5] Détection Profil                                     │
│        ├── Check pom.xml → Java/Maven                       │
│        ├── Check package.json → Node.js                     │
│        ├── Check requirements.txt → Python                  │
│        └── Sinon → unknown/unknown                          │
│                                                             │
│  [3/5] Génération YAML                                      │
│        └── workflow_generator.py                            │
│                                                             │
│  [3b/5] Validation YAML                                     │
│        ├── Parse YAML                                        │
│        ├── Check jobs build-test + sonar-scan               │
│        └── ✓ ou ✗                                           │
│                                                             │
│  [4/5] Push du workflow YAML                                │
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
name: "CI/CD — Build + SonarQube"

# Déclencheurs
trigger:
  - pull_request (opened, synchronize, reopened)
  - push (branches main, develop)

# Permissions nécessaires
permissions:
  - contents: read          # Pour lire le code
  - statuses: write         # Pour poster les status checks
  - pull-requests: read     # Pour lire la PR
  - security-events: write  # Pour SonarQube

# Deux jobs séquentiels
jobs:
  1. build-test:  # Compile et teste le projet
  2. sonar-scan:  # Analyse qualité SonarQube (dépend de build-test)
```

#### Détection automatique du langage

| Fichier détecté | Langage | Build System |
|-----------------|---------|--------------|
| `pom.xml` | Java | Maven |
| `build.gradle` ou `gradlew` | Java | Gradle |
| `package.json` | JavaScript/TypeScript | npm/yarn |
| `requirements.txt` ou `setup.py` | Python | pip |
| `pyproject.toml` | Python | poetry |
| `Cargo.toml` | Rust | cargo |
| `go.mod` | Go | go |

---

### 3. Job "build-test" (exécuté par GitHub Actions)

Compile et teste le projet selon le langage détecté.

**Java / Maven :**
- Setup Java (version détectée dans `pom.xml`)
- `mvn compile`
- `mvn test` (continue-on-error : les tests n'empêchent pas SonarQube)
- `mvn package`

**JavaScript / npm :**
- Setup Node.js 20
- `npm ci`
- `npm run build`
- `npm test`

**Python / pip :**
- Setup Python 3.11
- `pip install -r requirements.txt`
- `pytest`

---

### 4. Job "sonar-scan" (exécuté par GitHub Actions)

Analyse qualité du code via SonarQube.

**Dépendance :**
```yaml
needs: build-test
if: always() && needs.build-test.result == 'success'
```

**Java / Maven :**
- Plugin SonarQube officiel via Maven

**Autres langages :**
- Téléchargement du `sonar-scanner-cli`
- Analyse des sources avec le token configuré

**Quality Gate :**
- Check optionnel du Quality Gate SonarQube

---

## Flux Complet : Du push à l'analyse SonarQube

```
1. Développeur push sur PR
   │
   ▼
2. GitHub Actions détecte la PR
   └── trigger: pull_request
   │
   ▼
3. Workflow YAML s'exécute
   ├── Job build-test : compile + tests
   └── Job sonar-scan : analyse qualité (si build-test OK)
   │
   ▼
4. Résultat visible sur la PR
   ├── Status checks GitHub (vert/rouge)
   └── Rapport SonarQube (si configuré)
```

---

## Fichiers importants

### côté Repository cible (déployés)

| Fichier | Description |
|---------|-------------|
| `.github/workflows/ci.yml` | Workflow GitHub Actions (build + SonarQube) |

### côté Code Auditor (source)

| Fichier | Rôle |
|---------|------|
| `ci_cd/ci_deploy_agent.py` | Déploie le workflow sur le repo cible |
| `ci_cd/workflow_generator.py` | Génère le YAML adapté au langage |

---

## Secrets requis sur GitHub

Configure dans : `Settings > Secrets > Actions`

| Secret | Obligatoire | Description |
|--------|-------------|-------------|
| `GITHUB_TOKEN` | ✅ Oui | Fourni automatiquement par GitHub Actions |
| `SONAR_TOKEN` | ✅ Oui | Token d'analyse SonarQube |
| `SONAR_HOST_URL` | ✅ Oui | URL SonarQube (ex: `https://sonarcloud.io`) |

---

## Dépannage courant

### Erreur : `401 Unauthorized`

**Cause** : Token GitHub invalide ou permissions insuffisantes

**Solution** : Vérifier `GITHUB_TOKEN` et permissions du repo

### Erreur : SonarQube analysis failed

**Cause** : `SONAR_TOKEN` ou `SONAR_HOST_URL` manquant

**Solution** : Ajouter les secrets dans `Settings > Secrets > Actions`

---

## Résumé des validations

| Validation | Fichier | Quand exécuté ? |
|------------|---------|-----------------|
| Token GitHub valide | `ci_deploy_agent.py` | Avant déploiement |
| Secrets présents | `ci_deploy_agent.py` | Avant déploiement (warning) |
| YAML valide | `workflow_generator.py` | Avant push |
