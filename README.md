<div align="center">

# 🎯 ScoreIQ — Student Exam Performance Predictor

**A production-ready Machine Learning web application that predicts a student's Mathematics exam score based on demographic and academic inputs.**

[🚀 Live Demo](#) · [📦 Installation](#-installation) · [🧠 How It Works](#-how-it-works) · [📁 Project Structure](#-project-structure)


## 📌 Overview

**ScoreIQ** is an end-to-end ML web application built with Flask that takes a student's academic and socioeconomic profile as input and predicts their **Mathematics score** using a trained regression model.

The project demonstrates a complete ML pipeline — from data ingestion and preprocessing to model training, serialization, and real-time inference via a polished web UI.

> **Key insight:** Features like parental education level, lunch type (a socioeconomic proxy), test preparation, and reading/writing scores are strong predictors of math performance.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🤖 **ML Prediction** | Trained regression model predicts math score in < 1 second |
| 🌐 **Flask Web App** | Clean GET/POST routing with Jinja2 templating |
| 🎨 **Premium UI** | Editorial-style design with animated result panel |
| 📊 **Live Insights** | Grade band, percentile estimate, gap-to-100 on result |
| 🔄 **Full Pipeline** | End-to-end: ingest → transform → train → predict |
| 📦 **Modular Codebase** | Separated concerns: pipeline, components, utils |

---

## 🧠 How It Works

```
User Input (Web Form)
        │
        ▼
┌───────────────────┐
│   Flask Route     │  POST /predictdata
│  predict_datapoint│
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   CustomData      │  Collects & structures form fields
│   (Data Class)    │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  PredictPipeline  │  Loads model + preprocessor from artifacts/
│                   │  Applies StandardScaler → Model inference
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   home.html       │  Renders result panel with score,
│   (Jinja2)        │  grade, percentile, insight message
└───────────────────┘
```

### Input Features

| Feature | Type | Values |
|---------|------|--------|
| `gender` | Categorical | male, female |
| `race_ethnicity` | Categorical | group A–E |
| `parental_level_of_education` | Categorical | some high school → master's degree |
| `lunch` | Categorical | standard, free/reduced |
| `test_preparation_course` | Categorical | none, completed |
| `reading_score` | Numerical | 0–100 |
| `writing_score` | Numerical | 0–100 |

**Target:** `math_score` (continuous, 0–100)

---

## 📁 Project Structure

```
mlproject-main/
│
├── app.py                          # Flask application entry point
│
├── src/
│   ├── __init__.py
│   ├── exception.py                # Custom exception handler
│   ├── logger.py                   # Logging configuration
│   ├── utils.py                    # Shared utility functions
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py       # Load & split raw dataset
│   │   ├── data_transformation.py  # Feature encoding & scaling
│   │   └── model_trainer.py        # Train & evaluate models
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── predict_pipeline.py     # CustomData + PredictPipeline classes
│       └── train_pipeline.py       # Orchestrates training components
│
├── artifacts/                      # Saved model & preprocessor (.pkl)
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── train.csv
│   └── test.csv
│
├── templates/
│   ├── index.html                  # Landing page  →  route: /
│   └── home.html                   # Form + Result  →  route: /predictdata
│
├── notebook/
│   └── EDA_and_Model_Training.ipynb
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip

### 1. Clone the Repository

```bash
git clone ---repo
cd mlproject-main
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model *(first run only)*

```bash
python src/pipeline/train_pipeline.py
```

> This generates `artifacts/model.pkl` and `artifacts/preprocessor.pkl`.

### 5. Run the App

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🌐 Routes

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Landing page (`index.html`) |
| `GET` | `/predictdata` | Prediction form (`home.html`) |
| `POST` | `/predictdata` | Submit form → run prediction → render result |

---

## 🔬 Model & Performance

The training pipeline evaluates multiple regression algorithms and selects the best performer:

| Model | R² Score |
|-------|----------|
| Linear Regression | ~0.87 |
| Ridge Regression | ~0.88 |
| **CatBoost Regressor** | **~0.92** ✅ |
| XGBoost Regressor | ~0.91 |
| Random Forest | ~0.90 |

> Best model and preprocessor are serialized to `artifacts/` and loaded at inference time.

**Feature Importance (top predictors):**
1. `writing_score` — strongest correlated feature
2. `reading_score`
3. `test_preparation_course` — completed vs none
4. `lunch` — standard vs free/reduced
5. `parental_level_of_education`

---

## 🖥️ UI Preview

### Landing Page (`/`)
- Split-screen hero with animated score card
- "How it works" 3-step breakdown
- Links directly to the prediction form

### Prediction Form (`/predictdata`)
- Real-time progress bar as fields are filled
- Grouped form sections: Personal → Academic Context → Scores
- Live `/100` preview on score inputs

### Result Panel (after POST)
- Animated score meter
- Grade badge (A+ → F), Percentile estimate, Gap-to-100
- Contextual performance message
- Two-column layout: form stays visible alongside result

---

## 🧩 Key Classes

### `CustomData` — `src/pipeline/predict_pipeline.py`
Collects raw form inputs and converts them into a Pandas DataFrame ready for the preprocessing pipeline.

```python
data = CustomData(
    gender="female",
    race_ethnicity="group B",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="completed",
    reading_score=78.0,
    writing_score=82.0
)
df = data.get_data_as_data_frame()
```

### `PredictPipeline` — `src/pipeline/predict_pipeline.py`
Loads the saved preprocessor and model from `artifacts/`, transforms input, and returns a prediction.

```python
pipeline = PredictPipeline()
result = pipeline.predict(df)
# result → array([84.3])
```

---

## 📦 Dependencies

```txt
Flask
numpy
pandas
scikit-learn
catboost
xgboost
dill
```

Install all via:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'Add your feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please follow PEP8 style guidelines and include docstrings for new functions.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---


<div align="center">

Made with ❤️ and a lot of ☕

⭐ **Star this repo if you found it useful!** ⭐

</div>