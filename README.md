# Fake News Detection

A fake news detection pipeline with a Bi-LSTM model, optional NewsAPI verification, and a React + Vite frontend.

## Project structure

- backend/ - data exploration, preprocessing, training, evaluation, and API
- frontend/ - React UI (Vite)
- processed/ - generated artifacts (ignored in git)
- WELFake_Dataset.csv - source dataset (ignored in git)

## Data

1. Download the WELFake dataset and place `WELFake_Dataset.csv` in the project root.
2. The dataset is ignored by git by default.

## Backend setup

Create a virtual environment and install dependencies:

```bash
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1

pip install pandas numpy matplotlib seaborn scikit-learn nltk joblib torch fastapi uvicorn httpx
```

Optional: download NLTK data once:

```bash
python -m nltk.downloader stopwords punkt
```

## Pipeline steps

Run these from the project root:

```bash
python backend/1_explore_data.py
python backend/2_preprocess.py
python backend/3_train_lstm.py
python backend/4_evaluate.py
```

Artifacts created:

- data_exploration.png (exploration plots)
- processed/train.csv, processed/val.csv, processed/test.csv
- processed/vocab.json
- processed/best_model.pt
- evaluation_plots.png (evaluation plots)

## API

The FastAPI app loads `processed/vocab.json` and `processed/best_model.pt`.

1. Set a NewsAPI key in `backend/5_api.py` (replace the hardcoded `NEWS_API_KEY`).
2. Run the API:

```bash
# Note: Python module names cannot start with a digit.
# Rename backend/5_api.py to backend/api.py, then run:
uvicorn backend.api:app --reload
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Example news article text...\"}"
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend expects the API to be available at `http://127.0.0.1:8000`.

## Notes

- Generated data and models are ignored by git (see .gitignore).
- If you change the API port, update the frontend API base URL accordingly.
