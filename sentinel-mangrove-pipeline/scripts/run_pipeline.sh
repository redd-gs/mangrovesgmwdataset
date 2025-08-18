#!/usr/bin/env bash
set -euo pipefail

# Localiser le dossier racine du projet à partir de ce script
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

# Choisir le venv prioritaire
if [ -d "$ROOT_DIR/.venv" ]; then
  VENV_DIR="$ROOT_DIR/.venv"
elif [ -d "$ROOT_DIR/venv" ]; then
  VENV_DIR="$ROOT_DIR/venv"
else
  echo "[ERREUR] Aucun environnement (.venv ou venv). Crée-le :"
  echo "  python -m venv .venv && . .venv/Scripts/activate && pip install -r requirements.txt"
  exit 1
fi

# Activer (Windows Git Bash => Scripts; Linux => bin)
if [ -f "$VENV_DIR/Scripts/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/Scripts/activate"
elif [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  echo "[ERREUR] Script d'activation introuvable."
  exit 1
fi

# Charger .env si présent
if [ -f "$ROOT_DIR/.env" ]; then
  echo "[INFO] Chargement .env"
  set -a
  while IFS='=' read -r k v; do
    [ -z "$k" ] && continue
    case "$k" in \#*) continue;; esac
    v="${v%$'\r'}"
    export "$k"="$v"
  done < "$ROOT_DIR/.env"
  set +a
fi

echo "[INFO] Répertoire courant: $(pwd)"
echo "[INFO] Exécutable Python: $(which python)"
echo "[INFO] Lancement pipeline..."
python "$ROOT_DIR/src/main.py"

deactivate
echo "[INFO] Terminé."