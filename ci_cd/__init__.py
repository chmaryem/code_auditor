# ci_cd/ — Module CI/CD de Code Auditor
#
# Contient :
#   ci_deploy_agent.py    — Déploie le workflow GitHub Actions via MCP
#   workflow_generator.py — Génère le YAML adapté au langage détecté
#   ci_runner.py          — Pont GitHub Actions → review_pr()
#   ci_status_reporter.py — POST status check sur GitHub Statuses API
