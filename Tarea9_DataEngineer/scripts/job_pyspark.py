import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, lower, avg, max as sf_max, min as sf_min, desc
import shutil
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

spark = SparkSession.builder.appName("ETL-Salaries-Local").getOrCreate()

df = (spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(args.input))

# Normalización de texto
for c in ["job_title","experience_level","company_location","salary_currency","employment_type","company_size","employee_residence"]:
    if c in df.columns:
        df = df.withColumn(c, lower(trim(col(c))))

# Asegurar salary_in_usd
if "salary_in_usd" not in df.columns and "salary" in df.columns:
    df = df.withColumnRenamed("salary","salary_in_usd")

# Eliminar nulos en columnas clave
required = [c for c in ["job_title","experience_level","company_location","salary_in_usd"] if c in df.columns]
for c in required:
    df = df.filter(col(c).isNotNull())

# Guardar datos limpios (como referencia/warehouse simulado)
df.coalesce(1).write.mode("overwrite").option("header",True).csv(f"{args.output}/clean_data")

# 1) Promedio por experience_level
avg_by_exp = df.groupBy("experience_level").agg(avg("salary_in_usd").alias("avg_salary"))
avg_by_exp.orderBy(desc("avg_salary")).coalesce(1).write.mode("overwrite").option("header",True).csv(f"{args.output}/avg_salary_by_experience")

# 2) Promedio por job_title
avg_by_job = df.groupBy("job_title").agg(avg("salary_in_usd").alias("avg_salary"))
avg_by_job.orderBy(desc("avg_salary")).coalesce(1).write.mode("overwrite").option("header",True).csv(f"{args.output}/avg_salary_by_job")

# 3) Máximo y mínimo por cargo
max_min_by_job = df.groupBy("job_title").agg(
    sf_max(col("salary_in_usd")).alias("max_salary"),
    sf_min(col("salary_in_usd")).alias("min_salary")
)
max_min_by_job.orderBy(desc("max_salary")).coalesce(1).write.mode("overwrite").option("header",True).csv(f"{args.output}/salary_max_min_by_job")

# 4) Top 10 salarios más altos
cols = [c for c in ["job_title","experience_level","company_location","salary_in_usd"] if c in df.columns]
top10 = df.select(*cols).orderBy(desc("salary_in_usd")).limit(10)
top10.coalesce(1).write.mode("overwrite").option("header",True).csv(f"{args.output}/top10_highest_salaries")

# Buscar el archivo generado part-*.csv
part_file = glob.glob(f"{args.output}/part-*.csv")[0]

# Definir ruta final con nombre fijo
final_file = f"{args.output}/top10_highest_salaries.csv"

# Renombrar/mover
shutil.move(part_file, final_file)

print("✅ Proceso finalizado. Revisa la carpeta de salida:", args.output)
spark.stop()
