#!/usr/bin/env python3
"""
Script di debug per verificare il dataset e il training
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ===== VERIFICA DATASET =====
def check_dataset():
    """Verifica che il dataset contenga skin valide"""
    dataset_path = "/skin_dataset"
    
    print("=== VERIFICA DATASET ===")
    
    # Conta file
    png_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.png'):
                png_files.append(os.path.join(root, file))
    
    print(f"Trovati {len(png_files)} file PNG")
    
    if len(png_files) == 0:
        print("ERRORE: Nessun file PNG trovato!")
        return False
    
    # Verifica dimensioni e formato
    valid_skins = 0
    invalid_skins = 0
    
    for i, file_path in enumerate(png_files[:10]):  # Controlla primi 10
        try:
            img = Image.open(file_path)
            width, height = img.size
            mode = img.mode
            
            print(f"\nFile {i+1}: {os.path.basename(file_path)}")
            print(f"  Dimensioni: {width}x{height}")
            print(f"  Formato: {mode}")
            
            # Verifica se è una skin valida
            if width == 64 and height == 64:
                valid_skins += 1
                print("  ✓ Skin valida")
            else:
                invalid_skins += 1
                print("  ✗ Dimensioni non corrette (deve essere 64x64)")
                
        except Exception as e:
            print(f"  ✗ Errore apertura: {e}")
            invalid_skins += 1
    
    print(f"\n=== RISULTATO ===")
    print(f"Skin valide: {valid_skins}")
    print(f"Skin non valide: {invalid_skins}")
    
    # Salva alcune skin originali per confronto
    os.makedirs("debug_output", exist_ok=True)
    
    print("\nSalvataggio prime 5 skin originali...")
    for i in range(min(5, len(png_files))):
        try:
            img = Image.open(png_files[i]).convert('RGBA')
            img = img.resize((64, 64), Image.NEAREST)
            img.save(f"debug_output/original_skin_{i+1}.png")
        except:
            pass
    
    return valid_skins > 0

# ===== VERIFICA NORMALIZZAZIONE =====
def test_normalization():
    """Testa il processo di normalizzazione/denormalizzazione"""
    print("\n=== TEST NORMALIZZAZIONE ===")
    
    # Carica una skin
    dataset_path = "./skin_dataset"
    png_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.png'):
                png_files.append(os.path.join(root, file))
                break
        if png_files:
            break
    
    if not png_files:
        print("Nessuna skin trovata per il test!")
        return
    
    # Carica e processa
    original = Image.open(png_files[0]).convert('RGBA')
    original = original.resize((64, 64), Image.NEAREST)
    original.save("debug_output/test_original.png")
    
    # Applica trasformazioni
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
    ])
    
    tensor = transform(original)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor min: {tensor.min():.2f}, max: {tensor.max():.2f}")
    
    # Denormalizza
    denorm = tensor * 0.5 + 0.5
    print(f"Denorm min: {denorm.min():.2f}, max: {denorm.max():.2f}")
    
    # Salva risultato
    save_image(denorm, "debug_output/test_denormalized.png")
    
    # Verifica che siano simili
    denorm_array = (denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    reconstructed = Image.fromarray(denorm_array, 'RGBA')
    reconstructed.save("debug_output/test_reconstructed.png")

# ===== TRAINING SEMPLIFICATO =====
def simple_training_test():
    """Test di training super semplificato per verificare il learning"""
    print("\n=== TEST TRAINING SEMPLIFICATO ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset minimale
    class SimpleDataset(Dataset):
        def __init__(self):
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5]*4, [0.5]*4)
            ])
            
            # Trova tutte le skin
            self.images = []
            for root, dirs, files in os.walk("./skin_dataset"):
                for file in files:
                    if file.endswith('.png'):
                        self.images.append(os.path.join(root, file))
            
            print(f"Dataset: {len(self.images)} immagini")
        
        def __len__(self):
            return min(len(self.images), 100)  # Usa solo prime 100
        
        def __getitem__(self, idx):
            img = Image.open(self.images[idx]).convert('RGBA')
            img = img.resize((64, 64), Image.NEAREST)
            return self.transform(img)
    
    # Generatore MOLTO semplice
    class TinyGenerator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(100, 256, 4, 1, 0),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(True),
                
                torch.nn.ConvTranspose2d(256, 128, 4, 2, 1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(True),
                
                torch.nn.ConvTranspose2d(128, 64, 4, 2, 1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(True),
                
                torch.nn.ConvTranspose2d(64, 32, 4, 2, 1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(True),
                
                torch.nn.ConvTranspose2d(32, 4, 4, 2, 1),
                torch.nn.Tanh()
            )
        
        def forward(self, x):
            return self.net(x)
    
    # Test rapido
    dataset = SimpleDataset()
    if len(dataset) == 0:
        print("Dataset vuoto!")
        return
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Salva batch reale
    real_batch = next(iter(loader))
    save_image(real_batch * 0.5 + 0.5, "debug_output/real_batch.png", nrow=2)
    
    # Test generatore
    gen = TinyGenerator().to(device)
    noise = torch.randn(4, 100, 1, 1, device=device)
    
    with torch.no_grad():
        fake = gen(noise)
        save_image(fake * 0.5 + 0.5, "debug_output/generator_test.png", nrow=2)
    
    print("Test completato. Controlla debug_output/")

# ===== ANALISI CHECKPOINT =====
def analyze_checkpoint():
    """Analizza i checkpoint salvati"""
    print("\n=== ANALISI CHECKPOINT ===")
    
    if not os.path.exists("models/final_model.pth"):
        print("Nessun modello finale trovato!")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("models/final_model.pth", map_location=device)
    
    print("Checkpoint contiene:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    # Verifica dimensioni layer
    if 'generator' in checkpoint:
        print("\nDimensioni layer generatore:")
        for name, param in checkpoint['generator'].items():
            if 'weight' in name:
                print(f"  {name}: {param.shape}")

# ===== MAIN =====
if __name__ == "__main__":
    # Esegui tutti i test
    print("ESECUZIONE DIAGNOSTICA COMPLETA\n")
    
    # 1. Verifica dataset
    dataset_ok = check_dataset()
    
    if not dataset_ok:
        print("\n❌ PROBLEMA CRITICO: Dataset non valido!")
        print("Assicurati che ./skin_dataset contenga file PNG 64x64")
        exit(1)
    
    # 2. Test normalizzazione
    test_normalization()
    
    # 3. Test training semplificato
    simple_training_test()
    
    # 4. Analisi checkpoint
    analyze_checkpoint()
    
    print("\n✅ Diagnostica completata!")
    print("Controlla la cartella debug_output/ per i risultati")
