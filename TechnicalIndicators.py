# Récupérer les méthodes crées
import numpy as np

def ema(x, p):
    """
    Calcul de la moyenne mobile exponentielle (EMA).

    Paramètres :
    x : np.ndarray
        Série de données sous forme d'un tableau NumPy.
    p : int
        Période de lissage pour l'EMA.

    Retour :
    np.ndarray :
        Série de données lissées avec EMA.
    """

    # Vérification des types d'entrée
    if not isinstance(x, np.ndarray):
        raise TypeError("L'entrée doit être un tableau NumPy.")

    # Vérification de la taille de la série de données
    if p > len(x):
        raise ValueError("La période p ne peut pas être supérieure à la taille de la série temporelle.")

    # Suppression des dimensions inutiles pour garantir un tableau unidimensionnel
    x = x.squeeze()

    # Affichage des informations sur les données d'entrée
    print("\n--- EMA - Informations sur les données d'entrée ---")
    print(f"Forme de x : {x.shape}")
    print(f"Valeurs (5 premières) : {x[:5]}")
    print(f"Période de lissage : {p}")

    # Calcul de la moyenne initiale pour les p premières valeurs
    prev_ema = x[:p].mean()

    # Initialisation de la liste pour stocker les valeurs EMA
    ema_values = [prev_ema]

    # Calcul du multiplicateur pour le lissage exponentiel
    multiplier = 2 / float(p + 1)

    # Calcul de l'EMA pour chaque valeur restante de la série temporelle
    for i in range(p, len(x)):
        cur_ema = (x[i] - prev_ema) * multiplier + prev_ema
        prev_ema = cur_ema
        ema_values.append(cur_ema)

    # Conversion de la liste en tableau NumPy
    ema_array = np.array(ema_values)

    # Affichage des résultats
    print("\n--- Résultat du calcul de l'EMA ---")
    print(f"Forme de la sortie : {ema_array.shape}")
    print(f"Valeurs (5 premières) : {ema_array[:5]}")
    print(f"Valeurs (5 dernières) : {ema_array[-5:]}")

    return ema_array

###############################################################################

def getRSI(x):
    """
    Calcule l'indice de force relative (RSI) sur une série de données.

    Paramètres :
    x : np.ndarray
        Série de données de prix.

    Retour :
    np.ndarray :
        Un tableau contenant les valeurs du RSI et les prix correspondants.
    """

    # Suppression des dimensions inutiles
    x = x.squeeze()

    # Vérification de la taille minimale des données
    if len(x) < 14:
        raise ValueError("La taille des données doit être d'au moins 14 pour calculer le RSI.")

    # Affichage des informations sur les données d'entrée
    print("\n--- RSI - Informations sur les données d'entrée ---")
    print(f"Forme de x : {x.shape}")
    print(f"Valeurs (5 premières) : {x[:5]}")
    print(f"Valeurs (5 dernières) : {x[-5:]}")

    # Définition des variables initiales
    n = len(x)
    x0 = x[:n-1]
    x1 = x[1:]
    change = x1 - x0

    # Calcul des gains et pertes initiaux
    gain = np.sum(change[change > 0][:14])
    loss = np.sum(np.abs(change[change < 0][:14]))

    avgGain = [gain / 14.0]
    avgLoss = [loss / 14.0]

    # Calcul progressif de l'EMA des gains/pertes
    for i in range(14, n-1):
        if change[i] >= 0:
            avgGain.append((avgGain[-1] * 13 + change[i]) / 14.0)
            avgLoss.append((avgLoss[-1] * 13) / 14.0)
        else:
            avgGain.append((avgGain[-1] * 13) / 14.0)
            avgLoss.append((avgLoss[-1] * 13 + abs(change[i])) / 14.0)

    # Calcul du RSI
    avgGain = np.array(avgGain)
    avgLoss = np.array(avgLoss)
    RS = avgGain / avgLoss
    RSI = 100 - (100 / (1 + RS))

    result = np.c_[RSI, x1[13:]]

    # Affichage des résultats
    print("\n--- Résultat du calcul du RSI ---")
    print(f"Forme de la sortie : {result.shape}")
    print(f"Valeurs RSI (5 premières): {result[:5, 0]}")
    print(f"Prix correspondants (5 premiers): {result[:5, 1]}")
    print(f"Valeurs RSI (5 dernières): {result[-5:, 0]}")
    print(f"Prix correspondants (5 derniers): {result[-5:, 1]}")

    return result

###############################################################################

def getStochasticOscillator(x):
    """
    Calcule l'oscillateur stochastique basé sur les données de marché.

    Paramètres :
    x : np.ndarray
        Données de marché sous forme d'un tableau NumPy avec les colonnes Open, High, Low, Close.

    Retour :
    np.ndarray :
        Un tableau contenant les valeurs de %K et les prix de clôture correspondants.
    """

    # Vérification de la taille minimale des colonnes
    if x.shape[1] < 4:
        raise ValueError("L'entrée doit contenir au moins 4 colonnes : Open, High, Low, Close.")

    # Extraction des colonnes High, Low et Close
    high = x[:,1].squeeze()
    low = x[:,2].squeeze()
    close = x[:,3].squeeze()

    # Affichage des informations d'entrée
    print("\n--- StochasticOscillator - Informations sur les données d'entrée ---")
    print(f"Forme des données : {x.shape}")
    print(f"Valeurs High (5 premières) : {high[:5]}")
    print(f"Valeurs Low (5 premières) : {low[:5]}")
    print(f"Valeurs Close (5 premières) : {close[:5]}")

    # Vérification du nombre minimal de données
    n = len(high)
    if n < 14:
        raise ValueError("Les données doivent contenir au moins 14 périodes pour calculer l'oscillateur stochastique.")

    # Calcul des plus hauts et plus bas sur une fenêtre de 14 jours
    highestHigh = [high[i:i+14].max() for i in range(n-13)]
    lowestLow = [low[i:i+14].min() for i in range(n-13)]

    # Affichage des données calculées
    print("\n--- Valeurs calculées ---")
    print(f"Taille de highestHigh : {len(highestHigh)}")
    print(f"Taille de lowestLow : {len(lowestLow)}")
    print(f"Valeurs highestHigh (5 premières) : {highestHigh[:5]}")
    print(f"Valeurs lowestLow (5 premières) : {lowestLow[:5]}")

    # Conversion des listes en tableaux NumPy
    highestHigh = np.array(highestHigh)
    lowestLow = np.array(lowestLow)

    # Vérification de la division par zéro
    if np.any(highestHigh == lowestLow):
        raise ValueError("Le prix le plus haut et le plus bas sont égaux, division par zéro détectée.")

    # Calcul de l'oscillateur stochastique %K
    k = 100 * ((close[13:] - lowestLow) / (highestHigh - lowestLow))

    # Affichage des résultats
    print("\n--- Résultats finaux ---")
    print(f"Forme de k : {k.shape}")
    print(f"Valeurs %K (5 premières) : {k[:5]}")
    print(f"Valeurs Close (5 dernières) : {close[-5:]}")

    # Retourner le résultat avec les valeurs %K et les prix de clôture correspondants
    result = np.c_[k, close[13:]]

    print("\n--- Résumé de la sortie du StochasticOscillator ---")
    print(f"Forme de la sortie : {result.shape}")
    print(f"Valeurs de sortie (5 premières lignes) :\n{result[:5]}")

    return result

###############################################################################

def getWilliams(x):
    """
    Calcule l'indicateur Williams %R basé sur les données de marché.

    Paramètres :
    x : np.ndarray
        Données de marché sous forme d'un tableau NumPy avec les colonnes Open, High, Low, Close.

    Retour :
    np.ndarray :
        Un tableau contenant les valeurs de Williams %R et les prix de clôture correspondants.
    """

    # Vérification de la taille minimale des colonnes
    if x.shape[1] < 4:
        raise ValueError("L'entrée doit contenir au moins 4 colonnes : Open, High, Low, Close.")

    # Extraction des colonnes High, Low et Close
    high = x[:,1].squeeze()
    low = x[:,2].squeeze()
    close = x[:,3].squeeze()

    # Affichage des informations sur les données d'entrée
    print("\n--- Williams - Informations sur les données d'entrée ---")
    print(f"Forme des données : {x.shape}")
    print(f"Valeurs High (5 premières) : {high[:5]}")
    print(f"Valeurs Low (5 premières) : {low[:5]}")
    print(f"Valeurs Close (5 premières) : {close[:5]}")

    # Vérification du nombre minimal de données
    n = len(high)
    if n < 14:
        raise ValueError("Les données doivent contenir au moins 14 périodes pour calculer l'indicateur Williams %R.")

    # Calcul des plus hauts et plus bas sur une fenêtre de 14 jours
    highestHigh = [high[i:i+14].max() for i in range(n-13)]
    lowestLow = [low[i:i+14].min() for i in range(n-13)]

    # Conversion des listes en tableaux NumPy
    highestHigh = np.array(highestHigh)
    lowestLow = np.array(lowestLow)

    # Affichage des plus hauts et plus bas calculés
    print("\n--- Valeurs calculées ---")
    print(f"Forme de highestHigh : {highestHigh.shape}")
    print(f"Forme de lowestLow : {lowestLow.shape}")
    print(f"Valeurs highestHigh (5 premières) : {highestHigh[:5]}")
    print(f"Valeurs lowestLow (5 premières) : {lowestLow[:5]}")

    # Vérification de la division par zéro
    if np.any(highestHigh == lowestLow):
        raise ValueError("Le prix le plus haut et le plus bas sont égaux, division par zéro détectée.")

    # Calcul de l'indicateur Williams %R
    w = -100 * ((highestHigh - close[13:]) / (highestHigh - lowestLow))

    # Affichage des résultats de l'indicateur Williams %R
    print("\n--- Résultats finaux ---")
    print(f"Forme de w : {w.shape}")
    print(f"Valeurs %R (5 premières) : {w[:5]}")
    print(f"Valeurs Close (5 dernières) : {close[-5:]}")

    # Retourner le résultat avec les valeurs %R et les prix de clôture correspondants
    result = np.c_[w, close[13:]]

    print("\n--- Résumé de la sortie du Williams ---")
    print(f"Forme de la sortie : {result.shape}")
    print(f"Valeurs de sortie (5 premières lignes) :\n{result[:5]}")

    return result

###############################################################################

def getMACD(close):
    """
    Calcule l'indicateur MACD basé sur la moyenne mobile exponentielle.

    Paramètres :
    close : np.ndarray
        Série de prix de clôture.

    Retour :
    np.ndarray :
        Tableau contenant les valeurs MACD et les prix de clôture correspondants.
    """

    # Vérification du type de données
    if not isinstance(close, np.ndarray):
        raise TypeError("L'entrée doit être un tableau NumPy.")

    # Suppression des dimensions inutiles
    close = close.squeeze()

    print("\n--- MACD - Informations sur les données d'entrée ---")
    print(f"Forme des prix de clôture : {close.shape}")
    print(f"Prix de clôture (5 premiers) : {close[:5]}")

    # Calcul des EMA de périodes 12 et 26
    ma1 = ema(close, 12)
    ma2 = ema(close, 26)

    print("\n--- Calcul des EMA ---")
    print(f"Forme de ma1 (EMA 12) : {ma1.shape}")
    print(f"Forme de ma2 (EMA 26) : {ma2.shape}")
    print(f"EMA 12 (5 premiers) : {ma1[:5]}")
    print(f"EMA 26 (5 premiers) : {ma2[:5]}")

    # Vérification de la longueur suffisante des EMA
    if len(ma1) < 14 or len(ma2) < 14:
        raise ValueError("Les moyennes mobiles doivent contenir au moins 14 éléments.")

    # Calcul de l'indicateur MACD
    macd = ma1[14:] - ma2

    print("\n--- Résultats du MACD ---")
    print(f"Forme de MACD : {macd.shape}")
    print(f"Valeurs MACD (5 premières) : {macd[:5]}")

    # Retourner les résultats sous forme de tableau avec les prix correspondants
    result = np.c_[macd, close[len(close) - len(macd):]]

    print("\n--- Résumé de la sortie du MACD---")
    print(f"Forme de la sortie : {result.shape}")
    print(f"Valeurs de sortie (5 premières lignes) :\n{result[:5]}")

    return result

###############################################################################

def getPriceRateOfChange(close, n_days):
    """
    Calcule le taux de changement du prix sur une période donnée.

    Paramètres :
    close : np.ndarray
        Série de prix de clôture.
    n_days : int
        Nombre de jours pour le calcul du taux de changement.

    Retour :
    np.ndarray :
        Tableau contenant le taux de changement et les prix correspondants.
    """

    # Vérification du type de données
    if not isinstance(close, np.ndarray):
        raise TypeError("L'entrée doit être un tableau NumPy.")

    # Vérification de la période
    if n_days >= len(close):
        raise ValueError("Le nombre de jours ne peut pas être supérieur à la taille de la série de prix.")

    # Suppression des dimensions inutiles
    close = close.squeeze()

    print("\n--- PriceRateOfChange - Informations sur les données d'entrée ---")
    print(f"Forme des prix de clôture : {close.shape}")
    print(f"Prix de clôture (5 premiers) : {close[:5]}")
    print(f"Période utilisée pour le calcul : {n_days}")

    # Séparation des données en deux séries
    x0 = close[:len(close) - n_days]
    x1 = close[n_days:]

    print("\n--- Données après séparation ---")
    print(f"Forme de x0 : {x0.shape}, valeurs (5 premières) : {x0[:5]}")
    print(f"Forme de x1 : {x1.shape}, valeurs (5 premières) : {x1[:5]}")

    # Vérification de la division par zéro
    if np.any(x0 == 0):
        raise ValueError("La division par zéro a été détectée dans les données de prix.")

    # Calcul du taux de changement
    PriceRateOfChange = (x1 - x0) / x0

    print("\n--- Calcul du taux de changement ---")
    print(f"Forme du taux de changement : {PriceRateOfChange.shape}")
    print(f"Taux de changement (5 premiers) : {PriceRateOfChange[:5]}")

    # Retourner les résultats sous forme de tableau avec les prix correspondants
    result = np.c_[PriceRateOfChange, x1]

    print("\n--- Résumé de la sortie du PriceRateOfChange ---")
    print(f"Forme de la sortie : {result.shape}")
    print(f"Valeurs de sortie (5 premières lignes) :\n{result[:5]}")

    return result

###############################################################################

def getOnBalanceVolume(X):
    """
    Calcule l'indicateur On-Balance Volume (OBV) basé sur les données de marché.

    Paramètres :
    X : np.ndarray
        Données de marché sous forme d'un tableau NumPy avec les colonnes Open, High, Low, Close, Volume.

    Retour :
    np.ndarray :
        Tableau contenant les valeurs OBV et les prix de clôture correspondants.
    """

    # Vérification de la taille minimale des colonnes
    if X.shape[1] < 5:
        raise ValueError("L'entrée doit contenir au moins 5 colonnes : Open, High, Low, Close, Volume.")

    # Extraction des colonnes Close et Volume
    close = X[:, 3].squeeze()
    volume = X[:, 4].squeeze()[1:]

    # Affichage des informations de base
    print("\n--- OnBalanceVolume - Informations sur les données d'entrée ---")
    print(f"Forme des données d'entrée X: {X.shape}")
    print(f"Forme de la colonne Close : {close.shape}")
    print(f"Forme de la colonne Volume : {volume.shape}")
    print(f"Prix de clôture (5 premiers) : {close[:5]}")
    print(f"Volume (5 premiers) : {volume[:5]}")

    # Vérification de la validité du volume
    if np.any(volume < 0):
        raise ValueError("Le volume ne peut pas être négatif.")

    # Calcul du nombre total de données
    n = len(close)

    # Séparation des prix pour le calcul des changements
    x0 = close[:n-1]
    x1 = close[1:]
    change = x1 - x0

    print("\n--- Détails du calcul des variations de prix ---")
    print(f"Forme de x0 : {x0.shape}, valeurs (5 premières) : {x0[:5]}")
    print(f"Forme de x1 : {x1.shape}, valeurs (5 premières) : {x1[:5]}")
    print(f"Variations de prix (5 premières) : {change[:5]}")

    # Initialisation des valeurs OBV
    OBV = []
    prev_OBV = 0

    # Calcul des valeurs OBV en fonction des variations de prix
    for i in range(n-1):
        if change[i] > 0:
            current_OBV = prev_OBV + volume[i]
        elif change[i] < 0:
            current_OBV = prev_OBV - volume[i]
        else:
            current_OBV = prev_OBV
        OBV.append(current_OBV)
        prev_OBV = current_OBV

    # Conversion de la liste OBV en tableau NumPy
    OBV = np.array(OBV)

    print("\n--- Résultats du calcul OBV ---")
    print(f"Forme de OBV : {OBV.shape}")
    print(f"Valeurs OBV (5 premières) : {OBV[:5]}")

    # Retourner le tableau avec les valeurs OBV et les prix correspondants
    result = np.c_[OBV, x1]

    print("\n--- Résumé de la sortie du OBV ---")
    print(f"Forme de la sortie : {result.shape}")
    print(f"Valeurs de sortie (5 premières lignes) :\n{result[:5]}")

    return result