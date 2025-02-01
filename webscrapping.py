from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import random

def configure_driver():
    chrome_options = Options()
    # Options pour éviter la détection
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-browser-side-navigation")
    chrome_options.add_argument("--disable-gpu")
    
    # Ajout d'un user agent réaliste
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    # Configuration des préférences
    chrome_prefs = {}
    chrome_options.experimental_options["prefs"] = chrome_prefs
    chrome_prefs["profile.default_content_settings"] = {"images": 2}
    chrome_prefs["profile.managed_default_content_settings"] = {"images": 2}

    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

def random_sleep():
    """Ajoute un délai aléatoire pour simuler un comportement humain"""
    time.sleep(random.uniform(2, 5))

def process_crypto_data(driver, name, url):
    try:
        print(f"[{name}] Début du traitement...")
        driver.get(url)
        random_sleep()

        # Attente explicite pour que la page soit complètement chargée
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Gestion des cookies avec retry
        for _ in range(3):
            try:
                cookie_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
                )
                cookie_button.click()
                print(f"[{name}] Cookies acceptés")
                break
            except Exception:
                print(f"[{name}] Pas de bannière de cookies visible")
                pass

        random_sleep()

        # Extraction directe des données du tableau
        try:
            table = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            
            # Extraction des en-têtes
            headers = []
            for header in table.find_elements(By.TAG_NAME, "th"):
                headers.append(header.text.strip())
            
            # Extraction des données
            rows_data = []
            rows = table.find_elements(By.TAG_NAME, "tr")[1:]  # Skip header row
            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                row_data = [cell.text.strip() for cell in cells]
                if row_data:  # Vérifie que la ligne n'est pas vide
                    rows_data.append(row_data)
            
            # Création du DataFrame
            df = pd.DataFrame(rows_data, columns=headers)
            df['Cryptocurrency'] = name
            
            # Sauvegarde en CSV
            filename = f"{name}_data_{time.strftime('%Y%m%d')}.csv"
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"[{name}] Données sauvegardées dans {filename}")
            
            return True
            
        except Exception as e:
            print(f"[{name}] Erreur lors de l'extraction des données: {str(e)}")
            return False

    except Exception as e:
        print(f"[{name}] Erreur générale: {str(e)}")
        return False

def main():
    urls = {
        "BTC": "https://coinmarketcap.com/currencies/bitcoin/historical-data/",
        "ETH": "https://coinmarketcap.com/currencies/ethereum/historical-data/",
        "XRP": "https://coinmarketcap.com/currencies/xrp/historical-data/",
        "BNB": "https://coinmarketcap.com/currencies/bnb/historical-data/",
        "SOL": "https://coinmarketcap.com/currencies/solana/historical-data/",
        "LINK": "https://coinmarketcap.com/currencies/chainlink/historical-data/"
    }

    retries = 3
    for attempt in range(retries):
        try:
            driver = configure_driver()
            for name, url in urls.items():
                success = process_crypto_data(driver, name, url)
                if not success:
                    print(f"[{name}] Échec du traitement, passage à la crypto suivante")
                random_sleep()
            break
        except Exception as e:
            print(f"Tentative {attempt + 1}/{retries} échouée: {str(e)}")
            if driver:
                driver.quit()
            if attempt < retries - 1:
                print("Nouvelle tentative dans 10 secondes...")
                time.sleep(10)
        finally:
            if driver:
                driver.quit()

if __name__ == "__main__":
    main()