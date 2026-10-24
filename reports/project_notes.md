Customer-Churn-Prediction/
│
├── app/
├── artifacts/
├── configs/
├── data/
│   ├── raw/
│   │     Telco_customer_churn.xlsx
│   └── processed/
│
├── logs/
├── models/
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── reports/
│   ├── images/
│   └── project_notes.md
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── inference.py
│   └── utils.py
│
├── tests/
│
├── .gitignore
├── README.md
├── requirements.txt
└── create_project.py
============================================================================================================
business problems:
What exactly does "churn" mean for this business?
Which customer attributes are available?
How is success measured? (Accuracy, Recall, etc.)
What happens if the model predicts incorrectly?
Can the business act on the predictions?

============================================================================================================

dataset: 
IBM Telco Customer Churn Dataset
(https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset?resource=download)

Where does the data come from?
Who created it?
Is it reliable?
What problem does it solve?
What are its limitations?
Why not another dataset?
Why did you choose this dataset?

============================================================================================================
Commit 1
Project structure

Commit 2
Business understanding

Commit 3
Dataset understanding

Commit 4
EDA

Commit 5
Preprocessing

Commit 6
Feature engineering

Commit 7
Baseline model

Commit 8
Hyperparameter tuning

Commit 9
Deployment

Commit 10
Documentation

src/data_loader.py
This will be the only file responsible for loading data.