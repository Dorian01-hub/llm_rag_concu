import os
import requests
import zipfile
import gdown


folder = "chromadb"
file_id = "1_X2ZnuLuPsqSO2JGLbxQOY44T9JN85rB"
url = f"https://drive.google.com/uc?id={file_id}"
output_zip = "chromadb.zip"
destination = "chromadb"

def download_and_extract_chromadb():
    if  os.path.exists(destination):
        print("📦 Téléchargement des données ChromaDB...")

        try:
            # Utiliser gdown à la place de requests
            gdown.download(url, output=output_zip, quiet=False)

            # Décompression
            with zipfile.ZipFile(output_zip, "r") as zip_ref:
                zip_ref.extractall("test/")

            os.remove(output_zip)
            print(f"✅ Dossier extrait dans `{destination}`")
        except Exception as e:
            print(f"❌ Erreur pendant le téléchargement ou la décompression : {e}")
    else:
        print(f"📁 Le dossier `{destination}` existe déjà.")

    return destination

# Lancer la fonction
if __name__ == "__main__":
    chemin = download_and_extract_chromadb()
    print(f"📂 Chemin utilisé : {chemin}")
    
