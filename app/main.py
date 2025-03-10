from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor

# Inicializar FastAPI
app = FastAPI()

# Configurar plantillas y archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Cargar la nueva tabla
df_restaurantes = pd.read_csv("ML_DATA.csv")

# Filtrar las ciudades más populares
top_estados = df_restaurantes["Estado"].value_counts().nlargest(5).index
df_filtered = df_restaurantes[df_restaurantes["Estado"].isin(top_estados)]

# Codificación OneHot para el estado
encoder = OneHotEncoder(sparse_output=False)
estado_encoded = encoder.fit_transform(df_filtered[["Estado"]])
estado_encoded_df = pd.DataFrame(estado_encoded, columns=encoder.categories_[0])
df_filtered = pd.concat([df_filtered, estado_encoded_df], axis=1)

# Seleccionar características (añadir Takeout y Delivery)
features = ["Estado", "Valoraciones", "Estacionamiento Discapacitados", "Almuerzo", "Cena", 
            "Comer Solo", "Grupos", "Turistas", "Debito", "NFC", "Credito", "Servicio Rapido", 
            "Takeout", "Delivery"]
X = df_filtered[features]
y = df_filtered["Promedio"]

# Preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('estado', OneHotEncoder(), ['Estado']),
        ('num', 'passthrough', ["Valoraciones", "Estacionamiento Discapacitados", "Almuerzo", "Cena", 
                                 "Comer Solo", "Grupos", "Turistas", "Debito", "NFC", "Credito", "Servicio Rapido", "Takeout", "Delivery"])
    ])


# Modelo
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Entrenar el modelo
X = X.dropna()
y = y[X.index]
model.fit(X, y)

# Ruta para el formulario
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Ruta de predicción
@app.post("/predict/", response_class=HTMLResponse)
async def predict(
    request: Request,
    estado: str = Form(...),
    valoraciones: int = Form(...),
    estacionamiento: str = Form(...),
    almuerzo: str = Form(...),
    cena: str = Form(...),
    comer_solo: str = Form(...),
    grupos: str = Form(...),
    turistas: str = Form(...),
    debito: str = Form(...),
    nfc: str = Form(...),
    credito: str = Form(...),
    servicio_rapido: str = Form(...),
    takeout: str = Form(...),  # Nuevo campo
    delivery: str = Form(...), # Nuevo campo
):
    try:
        # Convertir "SI"/"NO" a 1/0
        def convertir_si_no(valor):
            return 1 if valor.upper() == "SI" else 0
        
        # Crear DataFrame con datos ingresados
        input_data = pd.DataFrame([{ 
            "Estado": estado,
            "Valoraciones": valoraciones,
            "Estacionamiento Discapacitados": convertir_si_no(estacionamiento),
            "Almuerzo": convertir_si_no(almuerzo),
            "Cena": convertir_si_no(cena),
            "Comer Solo": convertir_si_no(comer_solo),
            "Grupos": convertir_si_no(grupos),
            "Turistas": convertir_si_no(turistas),
            "Debito": convertir_si_no(debito),
            "NFC": convertir_si_no(nfc),
            "Credito": convertir_si_no(credito),
            "Servicio Rapido": convertir_si_no(servicio_rapido),
            "Takeout": convertir_si_no(takeout),  # Nuevo parámetro
            "Delivery": convertir_si_no(delivery) # Nuevo parámetro
        }])
                

        # Transformar y calcular similitud
        X_transformed = model.named_steps['preprocessor'].transform(input_data)
        X_full_transformed = model.named_steps['preprocessor'].transform(X)
        cos_similarities = cosine_similarity(X_full_transformed, X_transformed)
        idx_mas_similar = np.argmax(cos_similarities)
        
        # Obtener restaurante más similar
        restaurante_similar = df_filtered.iloc[idx_mas_similar]
        prediction = {
            "Nombre": restaurante_similar["Nombre"],
            "Promedio": restaurante_similar["Promedio"],
            "Estado": restaurante_similar["Estado"],
            "Valoraciones": restaurante_similar["Valoraciones"]
        }
        return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))




