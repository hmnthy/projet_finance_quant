import os
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pandas import read_csv
from datetime import datetime
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
from TechnicalIndicators import *
from TechnicalIndicators import *
from DataPreprocessor import DataPreprocessor


def generate_crypto_dataframes(cryptos):
    """
    Cette méthode lit les fichiers CSV correspondant à une liste de cryptos donnée
    et renvoie un dictionnaire contenant les DataFrames pour chaque crypto.

    Arguments :
        cryptos (list): Une liste de noms de cryptomonnaies (exemple : ["BTC", "ETH", "XRP"])

    Retourne :
        dict : Un dictionnaire contenant les DataFrames pour chaque crypto,
               où les clés sont les noms des cryptos et les valeurs sont les DataFrames.
    """
    # Obtenir le chemin absolu du fichier courant
    dir_name = os.path.dirname(os.path.abspath(__file__))
    
    # Initialiser le dictionnaire pour stocker les DataFrames
    crypto_dataframes = {}

    # Parcourir chaque crypto dans la liste
    for crypto in cryptos:
        try:
            # Construire le chemin du fichier CSV correspondant
            filename = crypto + "_data.csv"
            csv_file = os.path.join(dir_name, "Data", filename)

            # Lire le fichier CSV en tant que DataFrame
            # Si le fichier n'existe pas, une exception sera levée
            if os.path.exists(csv_file):
                crypto_df = pd.read_csv(csv_file)

                # Convertir la colonne 'Date' en type datetime
                crypto_df['Date'] = pd.to_datetime(crypto_df['Date'], format="%b %d, %Y")

                # Trier par date croissante
                crypto_df = crypto_df.sort_values(by='Date')

                # Nettoyer les colonnes numériques (supprimer les '$' et ',' et convertir en float)
                for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']:
                    if col in crypto_df.columns:
                        crypto_df[col] = crypto_df[col].replace({'\$': '', ',': ''}, regex=True).astype(float)

                # Supprimer les lignes complètement vides
                crypto_df = crypto_df.dropna(how='all')

                # Définir 'Date' comme index
                crypto_df.set_index('Date', inplace=True)

                # Ajouter le DataFrame au dictionnaire
                crypto_dataframes[crypto] = crypto_df
            else:
                print(f"Fichier CSV non trouvé pour {crypto}: {csv_file}")
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier pour {crypto}: {e}")

    # Retourner le dictionnaire contenant les DataFrames
    print("Les DataFrames générés sont :")
    print(crypto_dataframes.keys())

    return crypto_dataframes

###############################################################################

def getData(crypto_name, crypto_dataframes):
    """
    Charge les données financières à partir des DataFrames fournis et applique un lissage.

    Paramètres :
    crypto_name : str
        Le nom de la crypto-monnaie ('BTC', 'ETH', 'XRP', 'BNB', 'SOL', 'LINK').
    crypto_dataframes : dict
        Dictionnaire contenant les DataFrames des différentes cryptos.

    Retour :
    tuple :
        - smoothened_ohclv_data : np.ndarray (Données OHLCV lissées)
        - close_prices : np.ndarray (Prix de clôture)
        - dates : list (Dates des transactions)
    """

    print(f"\n--- getData pour {crypto_name} ---")

    # Vérification si la crypto-monnaie est valide
    if crypto_name not in crypto_dataframes:
        raise ValueError(f"La crypto {crypto_name} n'est pas reconnue. Choisissez parmi {list(crypto_dataframes.keys())}.")

    # Récupérer le DataFrame correspondant
    data = crypto_dataframes[crypto_name]

    print(f"Forme du DataFrame initial: {data.shape}")
    print(f"Colonnes disponibles : {list(data.columns)}")

    # Vérification si 'Date' est dans l'index et le réinitialiser
    if data.index.name == 'Date':
        data = data.reset_index()

    print(f"Head du DataFrame initial: {data.head}")

    # Vérification des colonnes nécessaires
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']
    for col in required_columns:
        if col not in data.columns:
            raise KeyError(f"La colonne {col} est absente des données fournies pour {crypto_name}.")

    # Vérification du format de la colonne 'Date'
    if data['Date'].isnull().any():
        raise ValueError("La colonne Date contient des valeurs manquantes ou mal formatées.")

    # S'assurer que la colonne 'Date' est de type datetime
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # Vérification après conversion des dates
    if data['Date'].isnull().any():
        raise ValueError("Des valeurs non valides ont été trouvées dans la colonne Date après conversion.")

    # Trier les données par ordre chronologique (croissant)
    data = data.sort_values(by='Date', ascending=True).reset_index(drop=True)

    print(f"Forme du DataFrame après tri chronologique : {data.shape}")
    print(f"Période des données: {data['Date'].min()} à {data['Date'].max()}")

    # Extraction des colonnes OHLCV sous forme de tableau NumPy
    ohclv_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].values

    print("\nAperçu des 5 premières lignes des données OHLCV brutes :")
    print(ohclv_data[:5])
    print(f"Forme des données OHLCV : {ohclv_data.shape}")

    # Vérification du format des données OHLCV
    if len(ohclv_data.shape) != 2 or ohclv_data.shape[1] != 5:
        raise ValueError("Les données OHLCV ne sont pas au format attendu (n, 5).")

    # Application du lissage des données
    smoother = DataPreprocessor()
    smoothened_ohclv_data = smoother.PandaSmoother(ohclv_data)

    print("\nAperçu des 5 premières lignes des données OHLCV lissées :")
    print(smoothened_ohclv_data[:5])
    print(f"Forme des données OHLCV lissées : {smoothened_ohclv_data.shape}")

    # Extraction des prix de clôture et des dates
    close_prices = np.array(data["Close"])
    dates = list(data["Date"])

    print("\n--- Résumé des sorties ---")
    print(f"Forme des prix de clôture : {close_prices.shape}")
    print(f"Prix de clôture (5 premiers) : {close_prices[:5]}")
    print(f"Nombre total de dates : {len(dates)}")
    print(f"5 premières dates : {dates[:5]}")

    return smoothened_ohclv_data, close_prices, dates

###############################################################################

def getTechnicalIndicators(X, d):
    """
    Calcule plusieurs indicateurs techniques à partir des données OHLCV.

    Paramètres :
    X : np.ndarray
        Tableau des données du marché (Open, High, Low, Close, Volume).
    d : int
        Nombre de jours pour le calcul du taux de variation du prix (PROC).

    Retour :
    np.ndarray :
        Une matrice contenant les valeurs des indicateurs techniques.
    """

    print("\n--- Début du calcul des indicateurs techniques ---")
    print(f"Forme des données d'entrée X: {X.shape}")

    # Vérification de la validité de l'entrée
    if X.shape[1] < 5:
        raise ValueError("L'entrée doit contenir au moins 5 colonnes : Open, High, Low, Close, Volume.")

    # Calcul des indicateurs techniques
    RSI = getRSI(X[:,3])
    print(f"Forme de RSI : {RSI.shape}, 5 premières valeurs :\n{RSI[:5]}")

    StochasticOscillator = getStochasticOscillator(X)
    print(f"Forme de StochasticOscillator : {StochasticOscillator.shape}, 5 premières valeurs :\n{StochasticOscillator[:5]}")

    Williams = getWilliams(X)
    print(f"Forme de Williams : {Williams.shape}, 5 premières valeurs :\n{Williams[:5]}")

    MACD = getMACD(X[:,3])
    print(f"Forme de MACD : {MACD.shape}, 5 premières valeurs :\n{MACD[:5]}")

    PROC = getPriceRateOfChange(X[:,3], d)
    print(f"Forme de PROC : {PROC.shape}, 5 premières valeurs :\n{PROC[:5]}")

    OBV = getOnBalanceVolume(X)
    print(f"Forme de OBV : {OBV.shape}, 5 premières valeurs :\n{OBV[:5]}")

    # Vérification de la longueur minimale des indicateurs
    min_len = min(len(RSI), len(StochasticOscillator), len(Williams), len(MACD), len(PROC), len(OBV))

    if min_len == 0:
        raise ValueError("Un des indicateurs techniques a une longueur de zéro.")

    # Uniformisation des longueurs des indicateurs
    RSI = RSI[-min_len:]
    StochasticOscillator = StochasticOscillator[-min_len:]
    Williams = Williams[-min_len:]
    MACD = MACD[-min_len:]
    PROC = PROC[-min_len:]
    OBV = OBV[-min_len:]

    print(f"Forme après normalisation RSI : {RSI.shape}")
    print(f"Forme après normalisation StochasticOscillator : {StochasticOscillator.shape}")
    print(f"Forme après normalisation Williams : {Williams.shape}")
    print(f"Forme après normalisation MACD : {MACD.shape}")
    print(f"Forme après normalisation PROC : {PROC.shape}")
    print(f"Forme après normalisation OBV : {OBV.shape}")

    # Construction de la matrice des caractéristiques
    feature_matrix = np.c_[RSI[:,0],
                           StochasticOscillator[:,0],
                           Williams[:,0],
                           MACD[:,0],
                           PROC[:,0],
                           OBV[:,0]]
    print(f"Forme de la matrice des caractéristiques : {feature_matrix.shape}")
    print(f"5 premières lignes de la matrice des caractéristiques :\n{feature_matrix[:5]}")

    # Vérification des valeurs NaN
    if np.isnan(feature_matrix).any():
        raise ValueError("La matrice des caractéristiques contient des valeurs NaN.")

    print("\n--- Fin du calcul des indicateurs techniques ---")

    return feature_matrix

###############################################################################

def prepareData(X, close, date, d):
    """
    Prépare les données pour la modélisation en calculant des indicateurs techniques
    et en créant les étiquettes de sortie.

    Paramètres :
    X : np.ndarray
        Données du marché contenant Open, High, Low, Close, Volume.
    close : np.ndarray
        Série des prix de clôture.
    date : list
        Liste des dates correspondantes aux données.
    d : int
        Nombre de jours pour le décalage.

    Retour :
    tuple :
        - feature_matrix_1 : np.ndarray (Données d'entraînement - Train)
        - y : np.ndarray (Étiquettes de la classification)
        - feature_matrix_2 : np.ndarray (Données de test - Test)
        - closeplot : np.ndarray (Prix de clôture pour le traçage)
        - date : list (Dates correspondantes)
    """

    print("\n--- Début de la préparation des données ---")

    # Vérification des dimensions des entrées
    print(f"Forme des données X : {X.shape}")
    print(f"Forme de la série de clôture : {close.shape}")
    print(f"Nombre total de dates : {len(date)}")
    print(f"Nombre de jours de décalage (d) : {d}")

    # Calcul des indicateurs techniques
    feature_matrix = getTechnicalIndicators(X, d)
    print(f"Forme de la matrice des caractéristiques calculée : {feature_matrix.shape}")

    # Ajustement de la taille des données
    number_of_samples = feature_matrix.shape[0]
    date = date[-number_of_samples:]
    close = close[-number_of_samples:]

    print(f"Forme ajustée de la série de clôture : {close.shape}")
    print(f"Nombre total de dates après ajustement : {len(date)}")

    # Création des étiquettes pour la prévision
    y0 = close[:-d]
    y1 = close[d:]
    y = np.sign(y1 - y0)

    print(f"Forme de y0 : {y0.shape}, valeurs (5 premières): {y0[:5]}")
    print(f"Forme de y1 : {y1.shape}, valeurs (5 premières): {y1[:5]}")
    print(f"Forme de y après sign : {y.shape}, valeurs (5 premières): {y[:5]}")

    # Sélection des caractéristiques
    feature_matrix_1 = feature_matrix[:-d, :6]
    feature_matrix_2 = feature_matrix[-1000:, :6]
    closeplot = close[-1000:]
    date = date[-1000:]

    print(f"feature_matrix_1: {feature_matrix_1.shape}")
    print(f"len(y) : {len(y)}")
    print(f"feature_matrix_2: {feature_matrix_2.shape}")
    print(f"closeplot: {closeplot.shape}")
    print(f"len(date): {len(date)}")

    print("\n--- Fin de la préparation des données ---")

    return feature_matrix_1, y, feature_matrix_2, closeplot, date

###############################################################################

def plotClassificationResult(y_test, y_pred):
    """ 
    Affiche la matrice de confusion sous forme d'un graphique à barres.
    
    Arguments :
        y_test (array-like): Valeurs réelles des étiquettes.
        y_pred (array-like): Prédictions du modèle.
    """
    # Calculer la matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Vérifier que la matrice de confusion est valide
    if conf_matrix.shape != (2, 2):
        print("Erreur : La matrice de confusion doit être de dimension (2,2).")
        return

    # Définir les labels et les positions des barres
    x = [i + 3.0 for i in range(4)]
    xlabel = ["TP", "FN", "FP", "TN"]

    # Création du graphique
    plt.figure()
    plt.grid(True)
    plt.bar(x, conf_matrix.reshape(-1), color=np.random.random((4,3)))

    # Ajouter les labels aux axes
    plt.xticks([i + 3.0 for i in range(4)], xlabel)
    plt.xlabel("Catégorie")
    plt.ylabel("Valeur")
    plt.title("Matrice de confusion")

    # Ajouter les valeurs sur les barres
    for i, v in enumerate(conf_matrix.reshape(-1)):
        plt.text(x[i], v + 0.05, str(v), ha='center', fontsize=12)

    # Afficher le graphique
    plt.show()

###############################################################################

def pipeline_crypto_rf(crypto, crypto_dataframes, trading_day):
    """
    Pipeline complet pour le traitement des données et la modélisation d'une crypto-monnaie.

    Paramètres :
    crypto : str
        Nom de la crypto-monnaie.
    crypto_dataframes : dict
        Dictionnaire des DataFrames de crypto-monnaies.
    trading_day : int
        Nombre de jours de trading utilisés pour la préparation des données.

    Retour :
    None
    """

    print(f"Traitement des données pour {crypto}")

    # Étape 1: Chargement des données
    ohclv_data, close, date = getData(crypto, crypto_dataframes)
    ohclv_data = np.array(ohclv_data)

    # Étape 2: Préparation des données
    X, y, xplot, closeplot, dateplot = prepareData(ohclv_data, close, date, trading_day)

    # Vérification des classes uniques dans y
    print(f"Classes uniques dans y avant traitement: {np.unique(y)}")

    # Correction des valeurs 0 dans y pour éviter des erreurs
    y[y == 0] = 1

    print(f"Données préparées pour {crypto}, X shape: {X.shape}, y shape: {y.shape}")

    # Affichage des premières lignes des données préparées
    print(f"\nAperçu des 10 premières lignes de X pour {crypto}:")
    print(X[:10])
    print(f"\nAperçu des 10 premières valeurs de y pour {crypto}:")
    print(y[:10])

    # Étape 3: Division des données en ensembles train/test
    indices = np.arange(len(X))  # Création des indices pour suivre les échantillons
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices, shuffle=False, random_state=42
    )

    # Affichage des tailles des ensembles
    print(f"\nTaille des données après split pour {crypto}:")
    print(f"- Taille de X_train : {X_train.shape}")
    print(f"- Taille de X_test  : {X_test.shape}")
    print(f"- Taille de y_train : {y_train.shape}")
    print(f"- Taille de y_test  : {y_test.shape}")

    # Affichage des indices des ensembles d'entraînement et de test
    print(f"\nIndices des données de 'train' pour {crypto}:")
    print(train_indices)
    print(f"\nIndices des données de 'test' pour {crypto}:")
    print(test_indices)

    # Comptage des classes dans les ensembles
    train_counts = Counter(y_train)
    test_counts = Counter(y_test)

    print(f"\nDistribution des étiquettes dans y_train pour {crypto}:")
    print(f"Nombre de +1: {train_counts[1]}, Nombre de -1: {train_counts[-1]}")
    print(f"\nDistribution des étiquettes dans y_test pour {crypto}:")
    print(f"Nombre de +1: {test_counts[1]}, Nombre de -1: {test_counts[-1]}")

    # Étape 4: Modélisation avec RandomForest
    model = RandomForestClassifier(n_estimators=100, criterion="gini", random_state=42)

    # Évaluation par validation croisée
    scores = cross_val_score(model, X_train, y_train, cv=5)

    print("\nCross Validation scores:")
    for i, score in enumerate(scores):
        print(f"Validation Set {i} score: {score:.4f}")

    # Entraînement du modèle
    model.fit(X_train, y_train)

    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Étape 5: Évaluation des performances
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    specificity = recall_score(y_test, y_pred, pos_label=-1)

    print("\nRésultats sur 'test':")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")

    # Étape 6: Tracé de la courbe ROC
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
    auc_score = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {crypto}')
    plt.legend()
    plt.show()

    plotClassificationResult(y_test, y_pred)

###############################################################################

def pipeline_crypto_xgb(crypto, crypto_dataframes, trading_day):
    """
    Pipeline complet pour le traitement des données et la modélisation d'une crypto-monnaie.

    Paramètres :
    crypto : str
        Nom de la crypto-monnaie.
    crypto_dataframes : dict
        Dictionnaire des DataFrames de crypto-monnaies.
    trading_day : int
        Nombre de jours de trading utilisés pour la préparation des données.

    Retour :
    None
    """

    print(f"Traitement des données pour {crypto}")

    # Étape 1: Chargement des données
    ohclv_data, close, date = getData(crypto, crypto_dataframes)
    ohclv_data = np.array(ohclv_data)

    # Étape 2: Préparation des données
    X, y, xplot, closeplot, dateplot = prepareData(ohclv_data, close, date, trading_day)

    # Vérification des classes uniques dans y
    print(f"Classes uniques dans y avant traitement: {np.unique(y)}")

    # # Correction des valeurs 0 dans y pour éviter des erreurs
    # y[y == 0] = 1

    # Correction des valeurs de y : convertir -1 en 0
    y = np.where(y == -1, 0, y)

    print(f"Données préparées pour {crypto}, X shape: {X.shape}, y shape: {y.shape}")

    # Affichage des premières lignes des données préparées
    print(f"\nAperçu des 10 premières lignes de X pour {crypto}:")
    print(X[:10])
    print(f"\nAperçu des 10 premières valeurs de y pour {crypto}:")
    print(y[:10])

    # Étape 3: Division des données en ensembles train/test
    indices = np.arange(len(X))  # Création des indices pour suivre les échantillons
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices, shuffle=False, random_state=42
    )

    # Affichage des tailles des ensembles
    print(f"\nTaille des données après split pour {crypto}:")
    print(f"- Taille de X_train : {X_train.shape}")
    print(f"- Taille de X_test  : {X_test.shape}")
    print(f"- Taille de y_train : {y_train.shape}")
    print(f"- Taille de y_test  : {y_test.shape}")

    ####################################################################

    # Affichage des indices des ensembles d'entraînement et de test
    print(f"\nIndices des données de 'train' pour {crypto}:")
    print(train_indices)
    print(f"\nIndices des données de 'test' pour {crypto}:")
    print(test_indices)

    # Comptage des classes dans les ensembles
    train_counts = Counter(y_train)
    test_counts = Counter(y_test)

    print(f"\nDistribution des étiquettes dans y_train pour {crypto}:")
    print(f"Nombre de +1: {train_counts[1]}, Nombre de -1: {train_counts[-1]}")
    print(f"\nDistribution des étiquettes dans y_test pour {crypto}:")
    print(f"Nombre de +1: {test_counts[1]}, Nombre de -1: {test_counts[-1]}")

    ####################################################################

    # Étape 4: Modélisation avec XGB
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        eval_metric="logloss"
    )

    # Évaluation par validation croisée
    scores = cross_val_score(model, X_train, y_train, cv=5)

    print("\nCross Validation scores:")
    for i, score in enumerate(scores):
        print(f"Validation Set {i} score: {score:.4f}")

    # Entraînement du modèle
    model.fit(X_train, y_train)

    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)

    ####################################################################

    # Étape 5: Évaluation des performances
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    specificity = recall_score(y_test, y_pred, pos_label=0)

    print("\nRésultats sur 'test':")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")

    # Étape 6: Tracé de la courbe ROC + plotClassificationResult
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
    # auc_score = auc(fpr, tpr)

    if len(np.unique(y_test)) < 2:
        print(f"Attention: y_test ne contient qu'une seule classe, impossible de calculer AUC.")
        auc_score = np.nan
    else:
        if np.unique(y_prob).size == 1:
            print("Attention: y_prob contient une seule valeur, courbe ROC invalide.")
            auc_score = np.nan
        else:
            fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
            auc_score = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {crypto} and trading_day = {trading_day} (XGB)')
    plt.legend()
    plt.show()


###############################################################################

def main() :

    # Définition des données de crypto-monnaie
    cryptos = ["BTC"] # "BTC", "ETH", "XRP", "BNB", "SOL", "LINK"
    Trading_Days = [50] # [5, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

    # Appeler la méthode pour générer les DataFrames
    crypto_dataframes = generate_crypto_dataframes(cryptos)

    for crypto in cryptos: 
        for Trading_Day in Trading_Days:
            pipeline_crypto_xgb(crypto, crypto_dataframes, Trading_Day)
	
###############################################################################

# Pour exécuter main()
if __name__ == "__main__":
    main()