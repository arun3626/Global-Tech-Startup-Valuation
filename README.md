# Global Tech Startup Valuation Prediction & Analytics 🚀

A portfolio-ready Data Analytics + Machine Learning project using the supplied **Global Tech Startups 2026** CSV.

## Objective
Analyze startup funding, revenue, operations and AI adoption, then predict **startup valuation (USD millions)** with regression models.

## Dataset
- Rows: 25,000
- Columns: 17
- Target: `Valuation_USD_Millions`
- Cleaned rows: 25,000

## Presentation Structure
1. Project Title
2. Problem Statement
3. Dataset
4. Data Cleaning
5. Exploratory Data Analysis
6. Machine Learning Model
7. Dashboard
8. Findings
9. Conclusion

## Project Structure
```text
global-tech-startup-valuation/
├── data_raw/
├── data_processed/
├── src/
│   ├── data_cleaning.py
│   ├── eda.py
│   └── train_model.py
├── dashboard/
│   └── app.py
├── notebooks/
├── outputs/
├── requirements.txt
├── PRESENTATION_CONTENT.md
└── README.md
```

## Run
```bash
pip install -r requirements.txt
python src/data_cleaning.py
python src/eda.py
python src/train_model.py
streamlit run dashboard/app.py
```

## ML
Target: `Valuation_USD_Millions`

Models:
- Linear Regression baseline
- Random Forest Regressor

Metrics:
- MAE
- RMSE
- R² Score

The pipeline handles categorical encoding and numerical scaling automatically.

## Dashboard
The Streamlit dashboard provides KPI cards, filters, valuation charts, startup comparisons and a valuation prediction form.

## GitHub
Recommended repository name:
`global-tech-startup-valuation-prediction`

Upload the contents of this project folder to GitHub.
