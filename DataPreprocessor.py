
import numpy as np
import pandas as pd

class DataPreprocessor:

    def __init__(self):
        pass

    def CustomSmoother(self, x, alpha):
        if not isinstance(x, np.ndarray):
            raise ValueError("Input x should be a NumPy array.")

        s0 = x[0]
        smoothed_statistic = [s0]
        n = x.shape[0]
        for i in range(1, n):
            s1 = alpha * x[i] + (1 - alpha) * s0
            smoothed_statistic.append(s1)
            s0 = s1

        result = np.array(smoothed_statistic)

        # Affichage des résultats
        print("\n--- Résultat de CustomSmoother ---")
        print(f"Forme d'entrée : {x.shape}")
        print(f"Forme de sortie : {result.shape}")
        print(f"Valeurs (5 premières): {result[:5]}")

        return result

    def PandaSmoother(self, x):
      if not isinstance(x, (np.ndarray, list)):
          raise ValueError("Input x should be a NumPy array or list.")

      x = np.array(x)

      if x.ndim == 2:
          smoothed_data = np.apply_along_axis(
              lambda col: pd.Series(col).ewm(span=20, adjust=False).mean().fillna(col[0]).values,
              axis=0,
              arr=x
          )
      else:
          smoothed_data = pd.Series(x).ewm(span=20, adjust=False).mean().fillna(x[0]).values

      if smoothed_data.shape != x.shape:
          raise ValueError("Le format des données de sortie ne correspond pas à celui de l'entrée.")

      # Affichage des résultats
      print("\n--- Résultat de PandaSmoother ---")
      print(f"Forme d'entrée : {x.shape}")
      print(f"Forme de sortie : {smoothed_data.shape}")
      print(f"Valeurs (5 premières):\n {smoothed_data[:5]}")

      return smoothed_data