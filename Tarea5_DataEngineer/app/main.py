from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import uuid
import json
import os

app = FastAPI()

# Cargar modelo y preprocesador
ruta_model = os.path.join("..", "model.pkl")

model = joblib.load(ruta_model)
preprocessor = joblib.load(ruta_model)

HISTORY_FILE = "history.json"

# Crear archivo si no existe
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

class Proceso(BaseModel):
    Uso_CPU: float
    Uso_Memoria: float
    Numero_Hilos: int
    Tiempo_Ejecucion: float
    Numero_Errores: int
    Tipo_Proceso: str

@app.post("/predict")
def predict(data: Proceso):
    df = pd.DataFrame([data.dict()])
    X = preprocessor.transform(df)
    pred = model.predict(X)[0]
    pred_id = str(uuid.uuid4())

    resultado = {
        "id": pred_id,
        "entrada": data.dict(),
        "prediccion": int(pred)
    }

    # Guardar historial
    with open(HISTORY_FILE, "r+") as f:
        history = json.load(f)
        history.append(resultado)
        f.seek(0)
        json.dump(history, f, indent=4)

    return resultado

@app.get("/predictions")
def get_history():
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)

@app.get("/predictions/{pred_id}")
def get_prediction_by_id(pred_id: str):
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
        result = next((item for item in history if item["id"] == pred_id), None)
        if result:
            return result
        raise HTTPException(status_code=404, detail="No encontrado")
    
@app.post("/predict")
def predict(data: Proceso):
    df = pd.DataFrame([data.dict()])                         # ✅ Convertir datos a DataFrame
    X = preprocessor.transform(df)                           # ✅ Preprocesar datos
    pred = model.predict(X)[0]                               # ✅ Hacer predicción
    pred_id = str(uuid.uuid4())                              # ✅ Crear ID único

    resultado = {
        "id": pred_id,
        "entrada": data.dict(),                              # ✅ Guardar entrada original
        "prediccion": int(pred)                              # ✅ Guardar resultado
    }

    # ✅ Guardar en historial
    with open(HISTORY_FILE, "r+") as f:
        history = json.load(f)
        history.append(resultado)
        f.seek(0)
        json.dump(history, f, indent=4)

    return resultado                                         # ✅ Devolver resultado

@app.put("/predict/{pred_id}")
def update_prediction(pred_id: str, new_data: Proceso):
    with open(HISTORY_FILE, "r+") as f:
        history = json.load(f)
        for item in history:
            if item["id"] == pred_id:
                item["entrada"] = new_data.dict()
                df = pd.DataFrame([new_data.dict()])
                X = preprocessor.transform(df)
                item["prediccion"] = int(model.predict(X)[0])
                break
        else:
            raise HTTPException(status_code=404, detail="No encontrado")

        f.seek(0)
        f.truncate()
        json.dump(history, f, indent=4)
        return {"msg": "Actualizado"}

@app.delete("/delete/{pred_id}")
def delete_prediction(pred_id: str):
    with open(HISTORY_FILE, "r+") as f:
        history = json.load(f)
        new_history = [item for item in history if item["id"] != pred_id]
        if len(new_history) == len(history):
            raise HTTPException(status_code=404, detail="No encontrado")
        f.seek(0)
        f.truncate()
        json.dump(new_history, f, indent=4)
        return {"msg": "Eliminado"}
