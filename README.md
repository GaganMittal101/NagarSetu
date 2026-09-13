# NagarSetu — Clean Project

## One project root
There is no nested NagarSetu folder.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run .\app\app.py
```

## Data pipeline

```powershell
python .\src\data\validate_schema.py
python .\src\data\clean_complaints.py
python .\src\data\build_dataset.py
```

## Reliability rule

The UI does not present invented severity/confidence values as real AI. Ground truth must come from verified labels or measured outcomes.

