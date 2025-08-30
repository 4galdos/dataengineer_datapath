# ETL Local con Airflow + PySpark (Google Drive → Staging → CSVs)

## Pasos
1) Exporta tu CSV de Google Drive con un **enlace público** y copia la URL (o el file id).
2) Arranca Airflow:
   ```bash
   docker-compose up -d
   ```
3) Define `GDRIVE_URL`:
   - Opción A (rápida): al levantar, pasa la variable de entorno:
     ```bash
     GDRIVE_URL='https://drive.google.com/uc?id=FILE_ID' docker-compose up -d
     ```
   - Opción B (UI): en Airflow (Admin → Variables) crea `GDRIVE_URL` con la URL.
4) En la UI (http://localhost:8080), activa y ejecuta el DAG **etl_salaries_local_gdrive**.
5) Resultados en `output/`:
   - `clean_data/`
   - `avg_salary_by_experience/`
   - `avg_salary_by_job/`
   - `salary_max_min_by_job/`
   - `top10_highest_salaries/`

> Nota: la carpeta de staging es `staging/TemporalFile/` (equivale al "almacenamiento temporal").
