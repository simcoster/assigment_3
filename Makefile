.PHONY: cli streamlit help

help:
	@echo Usage:
	@echo   make cli         - terminal agent
	@echo   make streamlit   - Streamlit chat UI

cli:
	uv run python main.py

streamlit:
	uv run streamlit run streamlit_app.py
