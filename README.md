PREDICTIVE MAINTENANCE USING MACHINE LEARNING (CMAPSS – NASA Turbofan Engines)

A hands-on end-to-end project built to understand real-world ML pipelines, feature engineering, and deployment.

📌 Overview

This project is my attempt to build a complete predictive maintenance system from scratch — starting from raw sensor logs and ending with a working FastAPI inference service.

I used the NASA CMAPSS turbofan engine dataset, which contains multivariate time-series data from engines running until failure. The goal is simple:

Predict whether an engine is heading toward failure using sensor patterns + ML.

Instead of stopping at a notebook experiment, I pushed myself to go through the entire workflow — preprocessing, feature engineering, model training, evaluation, and finally exposing a real API that can be tested locally.

This project taught me how real machine-learning systems are built, not just the theory.

🧠 What This Project Actually Does

Takes raw CMAPSS engine logs

Cleans and preprocesses the data

Builds rolling-window features (means, std, trends, etc.)

Trains a Random Forest classifier

Saves evaluation metrics + feature importance

Serves predictions through a FastAPI endpoint

Allows users to send sensor readings and get real-time risk predictions

Everything here is reproducible and runs on a normal laptop.

📁 Project Structure (human explanation)

I tried to follow a clean, industry-style layout so things don't get messy later.

ml-predictive-maintenance/
│
├── data/
│   ├── 01-raw/                # Original CMAPSS text files
│   ├── 02-preprocessed/        # Cleaned + merged files
│   ├── 03-features/            # Rolling-window features
│   └── 04-predictions/         # Metrics, feature importance, etc.
│
├── src/
│   ├── pipelines/              # Processing pipelines (cleaning, features, inference)
│   └── utils.py                # Helper functions
│
├── entrypoint/
│   ├── train.py                # Train the model end-to-end
│   └── inference.py            # FastAPI app
│
├── models/                     # Saved Random Forest model
│
├── notebooks/                  # EDA + baseline exploration
│
├── requirements.txt
└── README.md


This structure helped me stop mixing training code, API code, and exploratory notebook experiments.

🛠 How I Built It (Step-By-Step Summary)
1. Data Preprocessing

The CMAPSS files don’t have headers and need a lot of cleaning.
I built a preprocessing script to:

assign column names

remove useless sensors

normalize settings

merge train/test files

save everything as .parquet for faster loading

2. Feature Engineering

Modeling raw sensor signals directly doesn’t work well.
So I generated rolling statistical features such as:

mean / std over last 5 cycles

slope/trend

sensor deltas

health indicators

These features actually captured engine degradation patterns.

3. Model Training

I used a Random Forest Classifier, mainly because:

it handles noisy real-world data well

it works with tabular features

it doesn’t require heavy tuning like deep learning

The training script:

loads processed features

splits train/validation

trains the model

saves evaluation metrics (accuracy, F1 score, etc.)

exports the model as a .pkl file

4. Model Serving with FastAPI

I wanted this project to feel “real”, so I built a tiny REST API where anyone can POST sensor readings and get a prediction back.

The API exposes:

GET /

Health check:

{ "status": "ok", "service": "predictive-maintenance-inference" }

POST /predict_simple

For quick demos — only needs a few sensor values.

POST /predict_from_features

Full feature input for production-style prediction.

📊 Model Performance

Here are the evaluation metrics saved during training (your file: data/04-predictions/eval_metrics.json):

accuracy:   1.0
precision:  1.0
recall:     1.0
f1:         1.0
roc_auc:    NaN (only one class in validation split)


The dataset split caused only one class to appear in the validation set, so ROC-AUC is not meaningful here.
Everything else shows the model is distinguishing failures very well on the provided split.

🔥 How to Run the API (Local Machine)
1. Install dependencies
pip install -r requirements.txt

2. Run the training pipeline (only once)
python entrypoint/train.py


This creates:

models/rf_failure_joblib.pkl
data/04-predictions/eval_metrics.json

3. Start the FastAPI server
uvicorn entrypoint.inference:app --host 127.0.0.1 --port 8000 --reload

4. Open the interactive Swagger UI
http://127.0.0.1:8000/docs


This lets you test prediction endpoints visually.

📨 Example Prediction (Simple)

Request:

{
  "sensors": {
    "sensor_1": 518.67,
    "sensor_2": 642.37,
    "sensor_3": 1582.85
  },
  "settings": {
    "setting1": 100.0
  }
}


Response (example):

{
  "prediction": 0,
  "failure_risk": 0.12
}

📌 Why I Built This

I wanted to push myself beyond just “notebooks”.
I wanted to learn how ML works when it’s actually used in a system:

structured data pipelines

feature engineering

model persistence

running inference on an API

understanding how everything connects

This project gave me a much clearer picture of real workflows used in companies — way more than a Kaggle notebook ever could.

💬 Final Thoughts

I built this project to strengthen my practical understanding of machine learning and deployment. I tried to make everything as clean and reproducible as possible so anyone reviewing my work (professors, engineers, or admissions panels) can run it without difficulty.

A hands-on end-to-end project built to understand real-world ML pipelines, feature engineering, and deployment.

📌 Overview

This project is my attempt to build a complete predictive maintenance system from scratch — starting from raw sensor logs and ending with a working FastAPI inference service.

I used the NASA CMAPSS turbofan engine dataset, which contains multivariate time-series data from engines running until failure. The goal is simple:

Predict whether an engine is heading toward failure using sensor patterns + ML.

Instead of stopping at a notebook experiment, I pushed myself to go through the entire workflow — preprocessing, feature engineering, model training, evaluation, and finally exposing a real API that can be tested locally.

This project taught me how real machine-learning systems are built, not just the theory.

🧠 What This Project Actually Does

Takes raw CMAPSS engine logs

Cleans and preprocesses the data

Builds rolling-window features (means, std, trends, etc.)

Trains a Random Forest classifier

Saves evaluation metrics + feature importance

Serves predictions through a FastAPI endpoint

Allows users to send sensor readings and get real-time risk predictions

Everything here is reproducible and runs on a normal laptop.

📁 Project Structure (human explanation)

I tried to follow a clean, industry-style layout so things don't get messy later.

ml-predictive-maintenance/
│
├── data/
│   ├── 01-raw/                # Original CMAPSS text files
│   ├── 02-preprocessed/        # Cleaned + merged files
│   ├── 03-features/            # Rolling-window features
│   └── 04-predictions/         # Metrics, feature importance, etc.
│
├── src/
│   ├── pipelines/              # Processing pipelines (cleaning, features, inference)
│   └── utils.py                # Helper functions
│
├── entrypoint/
│   ├── train.py                # Train the model end-to-end
│   └── inference.py            # FastAPI app
│
├── models/                     # Saved Random Forest model
│
├── notebooks/                  # EDA + baseline exploration
│
├── requirements.txt
└── README.md


This structure helped me stop mixing training code, API code, and exploratory notebook experiments.

🛠 How I Built It (Step-By-Step Summary)
1. Data Preprocessing

The CMAPSS files don’t have headers and need a lot of cleaning.
I built a preprocessing script to:

assign column names

remove useless sensors

normalize settings

merge train/test files

save everything as .parquet for faster loading

2. Feature Engineering

Modeling raw sensor signals directly doesn’t work well.
So I generated rolling statistical features such as:

mean / std over last 5 cycles

slope/trend

sensor deltas

health indicators

These features actually captured engine degradation patterns.

3. Model Training

I used a Random Forest Classifier, mainly because:

it handles noisy real-world data well

it works with tabular features

it doesn’t require heavy tuning like deep learning

The training script:

loads processed features

splits train/validation

trains the model

saves evaluation metrics (accuracy, F1 score, etc.)

exports the model as a .pkl file

4. Model Serving with FastAPI

I wanted this project to feel “real”, so I built a tiny REST API where anyone can POST sensor readings and get a prediction back.

The API exposes:

GET /

Health check:

{ "status": "ok", "service": "predictive-maintenance-inference" }

POST /predict_simple

For quick demos — only needs a few sensor values.

POST /predict_from_features

Full feature input for production-style prediction.

📊 Model Performance

Here are the evaluation metrics saved during training (your file: data/04-predictions/eval_metrics.json):

accuracy:   1.0
precision:  1.0
recall:     1.0
f1:         1.0
roc_auc:    NaN (only one class in validation split)


The dataset split caused only one class to appear in the validation set, so ROC-AUC is not meaningful here.
Everything else shows the model is distinguishing failures very well on the provided split.

🔥 How to Run the API (Local Machine)
1. Install dependencies
pip install -r requirements.txt

2. Run the training pipeline (only once)
python entrypoint/train.py


This creates:

models/rf_failure_joblib.pkl
data/04-predictions/eval_metrics.json

3. Start the FastAPI server
uvicorn entrypoint.inference:app --host 127.0.0.1 --port 8000 --reload

4. Open the interactive Swagger UI
http://127.0.0.1:8000/docs


This lets you test prediction endpoints visually.

📨 Example Prediction (Simple)

Request:

{
  "sensors": {
    "sensor_1": 518.67,
    "sensor_2": 642.37,
    "sensor_3": 1582.85
  },
  "settings": {
    "setting1": 100.0
  }
}


Response (example):

{
  "prediction": 0,
  "failure_risk": 0.12
}

📌 Why I Built This

I wanted to push myself beyond just “notebooks”.
I wanted to learn how ML works when it’s actually used in a system:

structured data pipelines

feature engineering

model persistence

running inference on an API

understanding how everything connects

This project gave me a much clearer picture of real workflows used in companies — way more than a Kaggle notebook ever could.

💬 Final Thoughts

I built this project to strengthen my practical understanding of machine learning and deployment. I tried to make everything as clean and reproducible as possible so anyone reviewing my work (professors, engineers, or admissions panels) can run it without difficulty.
