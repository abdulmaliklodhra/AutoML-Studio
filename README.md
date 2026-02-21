# 🤖 AutoML Studio — PyCaret Powered Machine Learning App

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://abdulmaliklodhra-automl-studio.streamlit.app)

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=for-the-badge&logo=streamlit)
![PyCaret](https://img.shields.io/badge/PyCaret-3.3+-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![GitHub](https://img.shields.io/badge/GitHub-abdulmaliklodhra-181717?style=for-the-badge&logo=github)

**Upload any dataset → Select a target variable → Let AI run the full ML pipeline automatically!**

🌐 **Live App:** https://abdulmaliklodhra-automl-studio.streamlit.app

</div>

---

## 🚀 Features

| Feature                      | Description                                           |
| ---------------------------- | ----------------------------------------------------- |
| 📤 **Upload Any Dataset**    | Supports CSV and Excel (`.xlsx`, `.xls`) files        |
| 🧠 **Auto Task Detection**   | Automatically decides: Classification or Regression   |
| ⚙️ **Full PyCaret Pipeline** | `setup → compare_models → finalize_model`             |
| 📊 **Model Leaderboard**     | All models ranked with cross-validated metrics        |
| 📈 **Interactive Charts**    | Plotly bar charts, line charts, correlation heatmap   |
| 🌟 **Feature Importance**    | Visual feature importance for tree-based models       |
| ⬇️ **Download Results**      | Export HTML report, model `.pkl` file, CSV comparison |
| 🎨 **Premium UI**            | Dark glassmorphism theme with gradient animations     |

---

## 🗂️ Project Structure

```
app_pycaret/
│
├── app.py              ← Main Streamlit application (UI + logic)
├── ml_pipeline.py      ← PyCaret backend (pipeline + helpers)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
│
└── .venv/              ← Python 3.10 virtual environment (created by setup)
```

---

## ⚙️ Setup Instructions

### Prerequisites

- **Python 3.10** installed on your system
- `pip` package manager
- Windows 10/11 (PowerShell or Command Prompt)

---

### Step 1: Navigate to Project Folder

```powershell
cd "d:\AI_KA_CHILLA_2026\15_machine_learning\01_Pycaret\app_pycaret"
```

### Step 2: Create Virtual Environment

```powershell
python -m venv .venv
```

> ⚠️ Make sure `python` refers to Python 3.10. You can verify with:
>
> ```powershell
> python --version
> ```

### Step 3: Activate Virtual Environment

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt
.\.venv\Scripts\activate.bat
```

You should see `(.venv)` in your terminal prompt.

### Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

> ⏳ This may take 5–10 minutes as PyCaret installs many ML libraries.

### Step 5: Launch the App

```powershell
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

---

## 📖 How to Use

1. **Upload your dataset** using the sidebar uploader (CSV or Excel)
2. **Select the target column** from the dropdown
3. Watch the app **auto-detect** the task type (Classification or Regression)
4. Optionally adjust **cross-validation folds** and **session ID**
5. Click **🚀 Run ML Pipeline** to start training
6. View the **Model Leaderboard**, **Charts**, and **Feature Importance**
7. **Download** your best model or HTML report

---

## 🧠 Auto-Detection Logic

The app uses the following heuristic to decide task type:

| Condition                                                | Task               |
| -------------------------------------------------------- | ------------------ |
| Target dtype is `object` / `category` / `bool`           | **Classification** |
| Target is `integer` with ≤ 20 unique values              | **Classification** |
| Target is continuous `float` or high-cardinality integer | **Regression**     |

---

## 📊 PyCaret Pipeline Steps

### Classification

```python
from pycaret.classification import setup, compare_models, finalize_model
setup(data=df, target='target_col', session_id=42)
best_model = compare_models()           # Compares 15+ classifiers
final_model = finalize_model(best_model) # Retrain on full dataset
```

### Regression

```python
from pycaret.regression import setup, compare_models, finalize_model
setup(data=df, target='target_col', session_id=42)
best_model = compare_models()           # Compares 18+ regressors
final_model = finalize_model(best_model) # Retrain on full dataset
```

---

## 📦 Dependencies

| Package         | Version | Purpose                   |
| --------------- | ------- | ------------------------- |
| `streamlit`     | ≥ 1.32  | Web application framework |
| `pycaret[full]` | ≥ 3.3   | AutoML pipeline engine    |
| `pandas`        | ≥ 2.0   | Data manipulation         |
| `numpy`         | ≥ 1.24  | Numerical computing       |
| `plotly`        | ≥ 5.18  | Interactive charts        |
| `matplotlib`    | ≥ 3.7   | Static plots              |
| `seaborn`       | ≥ 0.12  | Heatmaps + styling        |
| `openpyxl`      | ≥ 3.1   | Excel file support        |

---

## 🛠️ Troubleshooting

**Q: App shows an error when running the pipeline?**  
A: Make sure your dataset has no columns with all NaN values. Drop empty columns before uploading.

**Q: Feature importance is not showing?**  
A: Linear models (SVM, KNN, Ridge) don't always expose feature coefficients. The chart will fall back to descriptive statistics.

**Q: PyCaret setup is taking very long?**  
A: Large datasets (100k+ rows) or many features may take longer. Reduce the fold count to 3 for faster runs.

**Q: PowerShell says "execution of scripts is disabled"?**  
A: Run this once in PowerShell as Administrator:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📄 License

MIT License — Free to use, modify, and distribute.

---

<div align="center">
Made with ❤️ using <strong>PyCaret</strong> + <strong>Streamlit</strong>
</div>
