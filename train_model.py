import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar los datos
df = pd.read_csv("ML_DATA.csv")

# Eliminar columnas innecesarias
data = df.drop(columns=['gmap_id', 'Estado', 'Unnamed: 0', 'Categoria1', 'Categoria2', 'Categoria3', 'Categoria4', 'Categoria5'])

# Separar características (X) y etiquetas (y)
X = data.drop(columns=['Nombre'])
y = data['Nombre']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Guardar el modelo entrenado
joblib.dump(model, "model.pkl")
