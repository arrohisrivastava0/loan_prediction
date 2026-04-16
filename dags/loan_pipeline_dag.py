from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys

import sys
import os

project_path = "/opt/airflow/project"

if os.path.exists(project_path):
    sys.path.append(project_path)
else:
    sys.path.append(os.getcwd())

from scripts.run_pipeline import run_pipeline

with DAG(
    dag_id="loan_training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
) as dag:

    train_task = PythonOperator(
        task_id="train_and_register_model",
        python_callable=run_pipeline
    )