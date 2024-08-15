from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel

# Charger le modèle et le CountVectorizer
model = load('logreg.joblib')
vectorizer = load('logreg_vectorizer.joblib')  # Assurez-vous que c'est le bon chemin

app = FastAPI()

class RequestBody(BaseModel):
    message: str

@app.post("/predict")
def predict(data: RequestBody):
    # Convertir le message en une représentation numérique
    new_data = vectorizer.transform([data.message])

    # Faire la prédiction
    prediction = model.predict(new_data)

    # Convertir la prédiction en un entier Python standard
    prediction_int = int(prediction[0])

    return {"class": prediction_int}
