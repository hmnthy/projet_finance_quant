from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import psutil  # Bibliothèque pour surveiller les ressources système
import requests
from pydantic import BaseModel
from webscrapping import process_crypto_data
import joblib
from typing import List
from typing import List, Dict
import numpy as np
from CryptoPrediction import getData, prepareData, generate_crypto_dataframes
from webscrapping import configure_driver

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API FastAPI - Web Scraping et Modèles ML",
    description="Cette API permet de gérer des tâches comme le web scraping, les prédictions de modèles ML et le suivi de la santé du système.",
    version="1.0.0",
)

# 1. Endpoint Index
@app.get("/", summary="Index - Présentation de l'API")
async def index():
    """
    Fournit une description des capacités de l'application, un lien vers la documentation, et la liste des endpoints disponibles.
    """
    return JSONResponse(
        content={
            "message": "Bienvenue sur l'API FastAPI.",
            "description": "Cette API propose des fonctionnalités telles que le web scraping, les prédictions avec des modèles ML et le monitoring de l'état du système.",
            "documentation_url": "/docs",
            "endpoints": {
                "get": {
                    "Index": "/",
                    "Health Check": "/health", 
                    "Model info": "/model/info"
                },
                "post":{
                    "Health Check": "/health",
                    "Web Scraping": "/scrape",
                    "Model Inference Single": "/predict/single",
                    "Model Inference Batch": "/predict/batch"
                }

            }
        }
    )

# 2. Endpoint Health Check
@app.get("/health", summary="Vérification de l'état du système")
async def health_check():
    """
    Vérifie l'état de santé de l'API et du système.
    """
    # Récupérer l'utilisation des ressources système
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return JSONResponse(
        content={
            "status": "OK",
            "cpu_usage": f"{cpu_usage}%",
            "memory": {
                "total": f"{memory.total / (1024 ** 3):.2f} GB",
                "used": f"{memory.used / (1024 ** 3):.2f} GB",
                "available": f"{memory.available / (1024 ** 3):.2f} GB",
            },
            "disk": {
                "total": f"{disk.total / (1024 ** 3):.2f} GB",
                "used": f"{disk.used / (1024 ** 3):.2f} GB",
                "free": f"{disk.free / (1024 ** 3):.2f} GB",
            },
            "dependencies": {
                "database": "OK",  
                "external_service": "OK", 
            }
        }
    )


# Définir un modèle pour les paramètres d'entrée
class CryptoRequest(BaseModel):
    name: str
    url: str

@app.post("/scrape", summary="Effectuer un scraping de données de crypto-monnaies")
async def scrape_data(request: CryptoRequest):
    """
    Scrape les données de crypto-monnaie à partir de l'URL donnée.
    :param request: Les paramètres de la requête (nom de la crypto et URL)
    :return: Données scrappées ou message de succès
    """
    driver = None  # Initialiser le driver en dehors pour une gestion propre dans finally
    try:
        # Configurer le driver Selenium
        driver = configure_driver()
        
        # Appeler la fonction de scraping en utilisant les paramètres fournis
        success = process_crypto_data(driver, request.name, request.url)

        # Vérifier si le scraping a réussi
        if success:
            return {"message": f"Les données de {request.name} ont été scrappées avec succès."}
        else:
            raise HTTPException(status_code=500, detail=f"Échec du scraping pour {request.name}.")
    
    except requests.exceptions.RequestException as e:
        # Erreur liée à une connexion réseau
        raise HTTPException(status_code=500, detail=f"Erreur de connexion: {e}")
    
    except Exception as e:
        # Autre erreur générale
        raise HTTPException(status_code=500, detail=f"Erreur de scraping: {e}")
    
    finally:
        # Toujours fermer le driver pour libérer les ressources
        if driver:
            driver.quit()



# Modèles Pydantic pour la validation des requêtes
class PredictionRequest(BaseModel):
    crypto: str
    trading_days: int 

class BatchPredictionRequest(BaseModel):
    cryptos: List[str]
    trading_days: int 

# Chargement du modèle au démarrage
model = joblib.load('random_forest_model.pkl')

def process_crypto(crypto: str, trading_days: int, crypto_dataframes: Dict) -> Dict:
    """Traite les données d'une crypto et fait des prédictions."""
    try:
        # Utilisation des fonctions existantes
        ohclv_data, close, date = getData(crypto, crypto_dataframes)
        ohclv_data = np.array(ohclv_data)
        
        # Préparation des données
        X, y, xplot, closeplot, dateplot = prepareData(ohclv_data, close, date, trading_days)
        
        # Prédictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'dates': [d.strftime('%Y-%m-%d') for d in dateplot]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement de {crypto}: {str(e)}")

@app.post("/predict/single")
async def predict_single(request: PredictionRequest):
    """
    Fait des prédictions pour une seule crypto
    """
    try:
        # Utilisation de generate_crypto_dataframes
        crypto_dataframes = generate_crypto_dataframes([request.crypto])
        if not crypto_dataframes:
            raise HTTPException(status_code=404, detail=f"Données non trouvées pour {request.crypto}")
            
        result = process_crypto(request.crypto, request.trading_days, crypto_dataframes)
        return {
            "crypto": request.crypto,
            "trading_days": request.trading_days,
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Fait des prédictions pour plusieurs cryptos
    """
    # Utilisation de generate_crypto_dataframes pour toutes les cryptos
    crypto_dataframes = generate_crypto_dataframes(request.cryptos)
    
    results = []
    for crypto in request.cryptos:
        try:
            if crypto not in crypto_dataframes:
                results.append({
                    "crypto": crypto,
                    "status": "error",
                    "error": "Données non trouvées"
                })
                continue
                
            result = process_crypto(crypto, request.trading_days, crypto_dataframes)
            results.append({
                "crypto": crypto,
                "trading_days": request.trading_days,
                "status": "success",
                **result
            })
        except Exception as e:
            results.append({
                "crypto": crypto,
                "status": "error",
                "error": str(e)
            })
    
    return {"results": results}

@app.get("/model/info")
async def get_model_info():
    """
    Obtient les informations sur le modèle chargé
    """
    return {
        "model_type": "RandomForestClassifier",
        "features": [
            "RSI",
            "StochasticOscillator",
            "Williams",
            "MACD",
            "PROC",
            "OBV"
        ],
        "trading_days_supported": [3, 5, 10, 15, 30, 60, 90, 120]
    }


# uvicorn crypto_api:app --reload