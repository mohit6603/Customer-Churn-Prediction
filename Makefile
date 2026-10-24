# Common project commands. Usage: make <target>

.PHONY: install train api app test lint drift mlflow docker

install:
	pip install -r requirements.txt

train:          ## Retrain, version, register and log the model
	python -m src.train_pipeline

api:            ## Serve the model at http://127.0.0.1:8000/docs
	uvicorn api.main:app --reload

app:            ## Streamlit UI
	streamlit run app/streamlit_app.py

test:
	python -m pytest -q

lint:
	ruff check .

drift:          ## make drift FILE=path/to/new_customers.csv
	python -m src.monitoring $(FILE)

mlflow:         ## Browse experiment runs at http://127.0.0.1:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db

docker:
	docker build -t churn-api . && docker run -p 8000:8000 -v "$$(pwd)/logs:/app/logs" churn-api
