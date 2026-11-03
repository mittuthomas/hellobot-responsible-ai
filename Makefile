.PHONY: install api ui test evals up down
install: ; pip install -r requirements.txt
api:     ; uvicorn app.main:app --reload --port 8000
ui:      ; API_URL=http://localhost:8000 streamlit run ui/dashboard.py
test:    ; pytest -q
evals:   ; python -m evals.runner
up:      ; docker compose up --build
down:    ; docker compose down
