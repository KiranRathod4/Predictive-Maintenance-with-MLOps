
# Predictive Maintenance with MLOps

## 1. Problem Statement
Industrial companies lose millions from unexpected machine breakdowns.  
Traditional preventive maintenance (fixed schedules) is inefficient.  

**Goal:** Build an ML system to predict equipment failures in advance using sensor data (temperature, vibration, pressure, etc.), enabling timely intervention.

---

## 2. Data Sources
- [NASA Turbofan Engine Degradation Dataset (CMAPSS)](https://data.nasa.gov/)  


The project is built with:

*  **Machine Learning (Random Forest Regressor)**
*  **FastAPI backend** for serving model predictions
*  **Interactive HTML + JavaScript frontend** for real-time prediction and dashboard
*  **MLflow** for model tracking, experiment management, and versioning
*  **Modular MLOps-ready architecture** for scalability and CI/CD integration

---

##  Why I Built This Project

Predictive maintenance is one of the most impactful applications of AI in the industry.
Through this project, I wanted to:

* Apply **data science + MLOps concepts** in a real-world use case.
* Learn how to **serve ML models in production** using FastAPI.
* Build a **user-friendly frontend** to visualize predictions.
* Integrate **MLflow** to track and compare model performance.
* Understand the **end-to-end lifecycle** — from data to deployment.

---

## 🧰 Tech Stack

| Layer                    | Tools / Technologies Used               |
| ------------------------ | --------------------------------------- |
| **Language**             | Python, JavaScript, HTML, CSS           |
| **Backend Framework**    | FastAPI                                 |
| **Frontend**             | TailwindCSS, Vanilla JS                 |
| **ML & Data Science**    | Pandas, NumPy, Scikit-learn, Matplotlib |
| **Experiment Tracking**  | MLflow                                  |
| **Version Control**      | Git, GitHub                             |
| **Deployment Ready For** | Render / Railway / AWS EC2              |
| **Database (Optional)**  | SQLite (for MLflow tracking)            |

---

##  Project Architecture

```
predictive-maintenance-with-mlops/
│
├── backend/
│   ├── main.py                  # FastAPI backend with ML endpoints
│   ├── model.pkl                # Trained ML model (Random Forest)
│   ├── utils.py                 # Helper functions for preprocessing
│   └── requirements.txt
│
├── frontend/
│   ├── index.html               # Dashboard page
│   ├── predict.html             # RUL prediction form
│   ├── model-metrics.html       # Model performance metrics
│   ├── api-health.html          # API health page
│   ├── services/
│   │   └── api.js               # API service for frontend
│   └── script.js                # UI logic and event handling
│
├── mlflow.db                    # Local MLflow tracking database
├── notebook/
│   ├── eda_and_model.ipynb      # Data analysis and model training
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

##  Features

✅ **Machine Learning**

* Predicts Remaining Useful Life (RUL) of engines using sensor data.
* Model trained on NASA Turbofan Engine dataset (C-MAPSS).
* Feature engineering and preprocessing handled in pipeline.

✅ **FastAPI Backend**

* RESTful endpoints for prediction, health check, and model info.
* Batch prediction and reload model functionality.
* CORS enabled for frontend integration.

✅ **Frontend Dashboard**

* Simple and responsive UI built with TailwindCSS.
* Real-time prediction form and dashboard with live API connection.
* Displays RUL value, model info, and timestamp.

✅ **MLflow Integration**

* Tracks model runs, hyperparameters, metrics (RMSE, R²).
* Version control for each trained model.
* UI for experiment comparison and performance analysis.

✅ **Deployment-Ready**

* Works locally via FastAPI server (`localhost:8000`).
* Ready for cloud deployment on Render or Railway.
* Modular folder structure for CI/CD pipeline setup.

---

##  API Endpoints

| Endpoint         | Method | Description                             |
| ---------------- | ------ | --------------------------------------- |
| `/health`        | GET    | Check if API is running                 |
| `/model/info`    | GET    | Get model metadata and training metrics |
| `/predict`       | POST   | Predict RUL for a single data input     |
| `/predict/batch` | POST   | Upload CSV for batch predictions        |
| `/model/reload`  | POST   | Reload or refresh model in memory       |

---

##  Example Prediction Request

**Request:**

```json
POST /predict
{
  "engine_id": 1,
  "cycle": 120,
  "max_cycle": 250,
  "sensor_2": -0.0007,
  "sensor_3": -0.0004,
  "sensor_4": 100.0,
  "sensor_5": 518.67,
  ...
}
```

**Response:**

```json
{
  "engine_id": 1,
  "rul_prediction": 89.3,
  "model_version": "v1.0",
  "prediction_timestamp": "2025-10-10 12:45:32",
  "status": "Success"
}
```

---

## 💻 Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/KiranRathod4/Predictive-Maintenance-with-MLOps.git
cd Predictive-Maintenance-with-MLOps
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Linux/Mac
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Start MLflow (optional)

```bash
mlflow ui
```

Access MLflow at: [http://localhost:5000](http://localhost:5000)

### 5️⃣ Run FastAPI Server

```bash
uvicorn backend.main:app --reload
```

Server runs at: [http://localhost:8000](http://localhost:8000)

### 6️⃣ Open Frontend

Open `frontend/predict.html` in your browser.

---

## 🌐 Deployment Guide


---

## 📊 Results

| Metric              | Value                             |
| ------------------- | --------------------------------- |
| **Validation RMSE** | 3.75                              |
| **Validation R²**   | 0.997                             |
| **Model Type**      | RandomForestRegressor             |
| **Dataset**         | NASA C-MAPSS Turbofan Engine Data |

---

##  Learning Outcomes

Through this project, I learned:

* How to **design a scalable ML system** using MLOps best practices.
* Serving ML models via **FastAPI REST API**.
* Building a **real-time dashboard** for predictions.
* Tracking and managing ML experiments using **MLflow**.
* Setting up **GitHub version control and deployment pipelines**.

---

## 🧑‍💻 Author

**Kiran Rathod**
🎓 Data Science & Machine Learning Enthusiast
💼 Exploring MLOps, AI Systems, and End-to-End Deployment
🌐 [LinkedIn](https://www.linkedin.com/in/kiranrathod05/) | [GitHub](https://github.com/KiranRathod4)

---

## 🪪 License

This project is open-source under the **MIT License**.
Feel free to use, modify, and distribute with attribution.

---


Pipeline:

1.raw data (data/raw.dvc)
↓

2.preprocess → outputs data/processed/
↓

3.features → outputs data/features/
↓

4.train → saves model in models/
↓

5.evaluate → saves metrics in metrics.json + plots


### why containarization in docker 
our setup only works on your machine with your virtualenv.
If someone else runs it, they might hit:

Different Python version

Missing dependencies

Wrong paths (Windows vs Linux)

Docker solves this → packages code + dependencies + environment into a container.
Result → works everywhere, the same way.
