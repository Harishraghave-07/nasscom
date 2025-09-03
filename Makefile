.PHONY: env

env:
	@echo "Creating Python 3.11 virtual environment in .venv..."
	python3.11 -m venv .venv
	@echo "Activating and installing dependencies..."
	. .venv/bin/activate && \
		pip install --upgrade pip && \
		pip install easyocr==1.7.1 opencv-python-headless==4.9.0.80 pillow==10.3.0 spacy~=3.6 gradio~=4.0 python-dotenv==1.0.1 loguru==0.7.2 pytest==8.2.0 black==24.4.2 isort==5.13.2 flake8==7.0.0
	@echo "Generating requirements.txt..."
	. .venv/bin/activate && pip freeze > requirements.txt
	@echo "Creating pre-commit config..."
	@cat > .pre-commit-config.yaml <<'EOL'
repos:
  - repo: https://github.com/psf/black
    rev: 24.4.2
    hooks:
      - id: black
  - repo: https://github.com/PyCQA/isort
    rev: 5.13.2
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
EOL
	@echo "Creating .env.example..."
	@echo "LOG_LEVEL=INFO" > .env.example
	@echo "DATA_PATH=./data/raw" >> .env.example
	@echo "✅ Environment setup complete."
