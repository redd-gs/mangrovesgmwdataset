$ErrorActionPreference = "Stop"

Write-Host "[INFO] Suppression venv..."
Remove-Item -Recurse -Force .venv, venv -ErrorAction SilentlyContinue

Write-Host "[INFO] Suppression caches..."
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Remove-Item -Recurse -Force .pytest_cache, .mypy_cache, .ruff_cache, .coverage -ErrorAction SilentlyContinue

Write-Host "[INFO] Terminé."