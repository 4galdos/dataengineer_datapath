
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="etl_salaries",
    default_args=default_args,
    description="ETL de salarios con PySpark y dataset en Google Drive",
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    # Descargar el CSV desde Google Drive
    t_download = BashOperator(
        task_id="download_dataset",
        bash_command=(
            "gdown --id 1UZmU8RSJryGVtj6GZi7L9jNRB2kp8dV_ "
            "-O /opt/airflow/staging/TemporalFile/salaries.csv"
        ),
    )

    # Ejecutar el script de PySpark
    t_run_spark = BashOperator(
        task_id="run_pyspark",
        bash_command=(
            "spark-submit /opt/airflow/scripts/job_pyspark.py "
            "--input /opt/airflow/staging/TemporalFile/salaries.csv "
            "--output /opt/airflow/output"
        ),
    )

    # Enviar correo con el archivo del top 10 desde el script de PySpark
    t_send_email = EmailOperator(
        task_id="send_email",
        to="andree.galdos.correa@gmail.com",
        subject="Top 10 Análisis Salaries",        
        files=["/opt/airflow/output/top10_highest_salaries.csv"],  # archivo generado por PySpark
    )

    t_download >> t_run_spark >> t_send_email
