import os
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
DATASET_PATH = "./dataset"
TARGET_SIZE = (28, 28) 
THRESHOLD = 127         

def process_image(file_path):
    """
    Transforme n'importe quelle image (PNG, JPG) en matrice binaire 28x28.
    """
    try:
        # 1. Acquisition et Grayscale (L = Luminance)
        img = Image.open(file_path).convert('L')
        
        # 2. Resize automatique à 28x28
        img = img.resize(TARGET_SIZE)
        
        # 3. Conversion en matrice NumPy
        matrix = np.array(img)
        
        # 4. Binarisation : pixel = 1 (noir/chiffre), pixel = 0 (blanc/fond)
        binary_matrix = (matrix < THRESHOLD).astype(int)
        
        return binary_matrix
    except Exception as e:
        print(f"❌ Erreur lors du traitement de {file_path}: {e}")
        return None

def load_dataset(folder_path):
    """
    Charge toutes les images PNG du dossier dataset pour servir de modèles.
    """
    dataset = {}
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"⚠️ Dossier '{folder_path}' créé. Placez vos PNG (0.png, 1.png...) dedans.")
        return dataset

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".png"):
            label = os.path.splitext(filename)[0] # Récupère le nom sans .png
            matrix = process_image(os.path.join(folder_path, filename))
            if matrix is not None:
                dataset[label] = matrix
                
    print(f"✅ Dataset chargé : {len(dataset)} modèles trouvés.")
    return dataset

def predict(input_matrix, dataset):
    """
    Compare la matrice de test avec chaque modèle et retourne le meilleur score.
    """
    best_digit = "Inconnu"
    max_score = -1
    
    for label, template in dataset.items():
        # np.sum(input_matrix == template) compte les pixels identiques
        score = np.sum(input_matrix == template)
        
        if score > max_score:
            max_score = score
            best_digit = label
            
    # Calcul du pourcentage de confiance (sur 784 pixels au total)
    confidence = (max_score / (TARGET_SIZE[0] * TARGET_SIZE[1])) * 100
    return best_digit, confidence

# --- EXECUTION PRINCIPALE ---
if __name__ == "__main__":
    print("=== IA Digit Recognizer (Pillow + NumPy) ===")
    
    # Chargement initial des modèles
    reference_data = load_dataset(DATASET_PATH)
    
    if not reference_data:
        print("Fin du programme : Aucun modèle de référence trouvé.")
    else:
        while True:
            print("-" * 30)
            filename = input("Entrez le nom de l'image à analyser (ou 'quit') : ")
            
            if filename.lower() == 'quit':
                break
            
            # Traitement de l'image de test
            test_matrix = process_image(filename)
            
            if test_matrix is not None:
                # Prédiction
                digit, score = predict(test_matrix, reference_data)
                
                print(f"\n🎯 RÉSULTAT : Chiffre {digit}")
                print(f"📊 CONFIANCE : {score:.2f}%")
                
                # On remplace 1 par '#' et 0 par '.' pour voir le chiffre
                print("\nAperçu de ce que l'IA a vu :")
                for row in test_matrix:
                    print("".join(['#' if p == 1 else '.' for p in row]))