#!/bin/bash

set -e

# Colors
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Directory structure
DIRS=(
  "data/raw"
  "data/processed"
  "docker"
  "docs"
  "infra/terraform"
  "logs"
  "models"
  "notebooks"
  "scripts/preprocessing"
  "scripts/ocr"
  "scripts/phi_detection"
  "scripts/masking"
  "src/cli"
  "src/core"
  "src/services"
  "src/ui/gradio_app"
  "tests"
)

# Create directories and .gitkeep files
for dir in "${DIRS[@]}"; do
  mkdir -p "$dir"
  touch "$dir/.gitkeep"
done

echo -e "${GREEN}Project directories created with .gitkeep placeholders.${NC}"

# Initialize git repo
if [ ! -d ".git" ]; then
  git init -b main
  git remote add origin "git@github.com:your-org/my-masker.git"
  echo -e "${GREEN}Initialized new git repository.${NC}"
fi

# .gitignore template
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
*.so
*.egg
*.egg-info/
dist/
build/
*.pyo
*.pyd
*.db
*.sqlite3
*.log

# VS Code
.vscode/
.history/

# macOS
.DS_Store
.AppleDouble
.LSOverride

# Docker
*.tar
docker-compose.override.yml

# Data files
data/raw/*
data/processed/*
logs/
models/
*.csv
*.tsv
*.xlsx
*.h5
*.pth
*.ckpt

# Env
.env
.env.*
.venv/
EOF

echo -e "${GREEN}.gitignore created.${NC}"

# Enable Git LFS and track data folders
if ! git lfs &>/dev/null; then
  echo "Git LFS not installed. Please install it first."
  exit 1
fi

git lfs install
git lfs track "data/raw/*"
git lfs track "data/processed/*"
echo -e "${GREEN}Git LFS enabled and tracking data folders.${NC}"

# Add and commit
git add .
git commit -m "feat: project scaffold"
echo -e "${GREEN}Initial commit done: feat: project scaffold${NC}"
