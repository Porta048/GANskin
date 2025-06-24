#!/usr/bin/env python3
"""
Sistema di Miglioramento Modello AI per Skin Minecraft
Strumento avanzato per ottimizzare le performance del generatore.

Espande il dataset tramite data augmentation intelligente e riallena
il modello con parametri ottimizzati per migliorare la qualità delle skin generate.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
import random
from datetime import datetime
import time

# Import dei nostri moduli
from models import SkinGenerator, SkinDiscriminator
from dataset import SkinDataset
import config

class ModelImprover:
    """
    Sistema completo per migliorare le performance del modello GAN.
    
    Include funzionalità di:
    - Espansione dataset con augmentation intelligente
    - Retraining ottimizzato con parametri anti-mode-collapse
    - Validazione automatica dei miglioramenti
    """
    
    def __init__(self):
        print("SISTEMA DI MIGLIORAMENTO MODELLO AI")
        print("=" * 50)
        
        # Configurazione hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device utilizzato: {self.device}")
        
        # Inizializza modelli
        self.generator = SkinGenerator().to(self.device)
        self.discriminator = SkinDiscriminator().to(self.device)
        
        # Optimizer con parametri ottimizzati per stabilità
        # Learning rate più basso per training più stabile
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        
        # Loss function con label smoothing per ridurre overfitting
        self.criterion = nn.BCELoss()
        
        # Carica modello esistente come base
        self.load_existing_model()
        
        print("Sistema inizializzato!")

    def load_existing_model(self):
        """
        Carica il modello esistente come punto di partenza.
        Il miglioramento parte sempre dal modello già addestrato.
        """
        if os.path.exists(config.MODEL_PATH):
            try:
                checkpoint = torch.load(config.MODEL_PATH, map_location=self.device)
                self.generator.load_state_dict(checkpoint['generator_state_dict'])
                self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                print("Modello base caricato per miglioramento!")
                return True
            except Exception as e:
                print(f"⚠️ Errore caricamento: {e}")
        
        print("⚠️ Nessun modello base trovato - si parte da zero")
        return False

    def create_augmented_skin(self, original_skin):
        """
        Crea variazioni intelligenti di una skin esistente.
        
        Applica trasformazioni realistiche che mantengono la coerenza
        della skin ma aumentano la diversità del dataset.
        
        Args:
            original_skin: Array numpy della skin originale
            
        Returns:
            List di skin augmentate
        """
        augmented_skins = []
        
        try:
            # Converte a PIL per manipolazione
            base_img = Image.fromarray(original_skin, 'RGBA')
            
            # Variazione 1: Leggero cambio colori (tono pelle diverso)
            enhancer = ImageEnhance.Color(base_img)
            color_variant = enhancer.enhance(random.uniform(0.8, 1.3))
            augmented_skins.append(np.array(color_variant))
            
            # Variazione 2: Contrasto modificato (abbronzatura/pallore)
            enhancer = ImageEnhance.Contrast(base_img)
            contrast_variant = enhancer.enhance(random.uniform(0.9, 1.2))
            augmented_skins.append(np.array(contrast_variant))
            
            # Variazione 3: Luminosità (diversa illuminazione)
            enhancer = ImageEnhance.Brightness(base_img)
            brightness_variant = enhancer.enhance(random.uniform(0.9, 1.1))
            augmented_skins.append(np.array(brightness_variant))
            
            # Variazione 4: Versione leggermente sfocata (stile diverso)
            blurred = base_img.filter(ImageFilter.BLUR)
            # Mix tra originale e blur per effetto sottile
            blended = Image.blend(base_img, blurred, 0.3)
            augmented_skins.append(np.array(blended))
            
            # Variazione 5: Hue shift per colori capelli/vestiti diversi
            # Converte in HSV, modifica tonalità, riconverte
            hsv_img = base_img.convert('HSV')
            hsv_array = np.array(hsv_img)
            
            # Shift random della tonalità (solo su pixel non trasparenti)
            mask = original_skin[:,:,3] > 0
            if np.any(mask):
                hue_shift = random.randint(-30, 30)
                hsv_array[mask, 0] = (hsv_array[mask, 0] + hue_shift) % 256
                
                shifted_img = Image.fromarray(hsv_array, 'HSV').convert('RGBA')
                augmented_skins.append(np.array(shifted_img))
            
        except Exception as e:
            print(f"Errore augmentation: {e}")
            # In caso di errore, ritorna almeno l'originale
            augmented_skins = [original_skin]
        
        return augmented_skins

    def expand_dataset(self, target_size=1000):
        """
        Espande il dataset da 121 skin a target_size tramite augmentation.
        
        Crea variazioni intelligenti delle skin esistenti mantenendo
        la qualità e diversità necessarie per un training efficace.
        """
        print(f"\nESPANSIONE DATASET: 121 → {target_size} skin")
        print("=" * 40)
        
        # Carica dataset esistente
        original_dataset = SkinDataset()
        original_count = len(original_dataset)
        
        print(f"Dataset originale: {original_count} skin")
        print("Generazione variazioni intelligenti...")
        
        augmented_count = 0
        skins_to_generate = target_size - original_count
        
        # Genera skin augmentate
        for i in range(skins_to_generate):
            try:
                # Scegli skin casuale dal dataset
                idx = random.randint(0, original_count - 1)
                original_tensor = original_dataset[idx]
                
                # Converte tensor in numpy array
                original_skin = original_tensor.permute(1, 2, 0).numpy()
                original_skin = (original_skin * 255).astype(np.uint8)
                
                # Crea versioni augmentate
                augmented_variants = self.create_augmented_skin(original_skin)
                
                # Salva una delle varianti
                if augmented_variants:
                    selected_variant = random.choice(augmented_variants)
                    
                    # Salva nel dataset
                    timestamp = int(time.time() * 1000) % 1000000  # Timestamp unico
                    filename = f"aug_{i:04d}_{timestamp}.png"
                    filepath = os.path.join(config.DATASET_PATH, filename)
                    
                    img = Image.fromarray(selected_variant, 'RGBA')
                    img.save(filepath)
                    
                    augmented_count += 1
                    
                    # Progress ogni 100 skin
                    if (i + 1) % 100 == 0:
                        print(f"   Progresso: {i+1}/{skins_to_generate} skin generate")
                        
            except Exception as e:
                print(f"Errore generazione skin {i}: {e}")
                continue
        
        print(f"Dataset espanso: +{augmented_count} skin augmentate")
        print(f"Totale finale: {original_count + augmented_count} skin")
        
        return augmented_count

    def improved_training(self, epochs=150):
        """
        Esegue training migliorato con parametri ottimizzati.
        
        Utilizza tecniche avanzate:
        - Learning rate più basso per stabilità
        - Gradient clipping per evitare esplosioni
        - Label smoothing per ridurre overfitting
        - Validazione periodica per monitorare progress
        """
        print(f"\nTRAINING MIGLIORATO - {epochs} EPOCHE")
        print("=" * 45)
        
        # Ricarica dataset espanso
        expanded_dataset = SkinDataset()
        print(f"Dataset per training: {len(expanded_dataset)} skin")
        
        # DataLoader ottimizzato
        dataloader = DataLoader(
            expanded_dataset,
            batch_size=16,  # Batch size fisso per stabilità
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        
        print("Inizio training ottimizzato...")
        start_time = time.time()
        
        # Variabili per monitoraggio
        best_g_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_d_loss = 0
            epoch_g_loss = 0
            num_batches = 0
            
            for i, real_imgs in enumerate(dataloader):
                try:
                    real_imgs = real_imgs.to(self.device)
                    b_size = real_imgs.size(0)
                    
                    # Label smoothing più aggressivo per stabilità
                    real_labels = torch.full((b_size,), 0.85, device=self.device, dtype=torch.float)
                    fake_labels = torch.full((b_size,), 0.15, device=self.device, dtype=torch.float)

                    # === TRAINING DISCRIMINATOR ===
                    self.optimizer_D.zero_grad()
                    
                    # Valutazione real images
                    output_real = self.discriminator(real_imgs)
                    loss_d_real = self.criterion(output_real, real_labels)
                    
                    # Generazione fake images
                    noise = torch.randn(b_size, config.LATENT_DIM, 1, 1, device=self.device)
                    fake_imgs = self.generator(noise)
                    
                    # Valutazione fake images
                    output_fake = self.discriminator(fake_imgs.detach())
                    loss_d_fake = self.criterion(output_fake, fake_labels)
                    
                    # Backpropagation discriminator
                    loss_d = loss_d_real + loss_d_fake
                    loss_d.backward()
                    
                    # Gradient clipping per stabilità
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                    
                    self.optimizer_D.step()

                    # === TRAINING GENERATOR ===
                    self.optimizer_G.zero_grad()
                    
                    # Generator prova a ingannare discriminator
                    output = self.discriminator(fake_imgs)
                    loss_g = self.criterion(output, real_labels)
                    loss_g.backward()
                    
                    # Gradient clipping per generator
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                    
                    self.optimizer_G.step()
                    
                    # Accumula statistiche
                    epoch_d_loss += loss_d.item()
                    epoch_g_loss += loss_g.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Errore batch {i}: {e}")
                    continue

            # Statistiche epoca
            avg_d_loss = epoch_d_loss / max(num_batches, 1)
            avg_g_loss = epoch_g_loss / max(num_batches, 1)
            elapsed = time.time() - start_time
            
            print(f"Epoca [{epoch+1:3d}/{epochs}] - "
                  f"D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f} - "
                  f"Tempo: {elapsed:.1f}s")
            
            # Early stopping basato su loss generator
            if avg_g_loss < best_g_loss:
                best_g_loss = avg_g_loss
                patience_counter = 0
                # Salva il miglior modello
                self.save_improved_model()
            else:
                patience_counter += 1
            
            # Preview ogni 10 epoche
            if (epoch + 1) % 10 == 0:
                self.generate_test_preview(epoch + 1)
            
            # Early stopping se non migliora per 30 epoche
            if patience_counter >= 30:
                print(f"Early stopping attivato all'epoca {epoch+1}")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining migliorato completato in {total_time:.1f} secondi!")
        
        return True

    def generate_test_preview(self, epoch):
        """Genera preview per monitorare i progressi."""
        try:
            self.generator.eval()
            with torch.no_grad():
                noise = torch.randn(1, config.LATENT_DIM, 1, 1, device=self.device)
                generated = self.generator(noise)
                
                skin_array = generated.squeeze().permute(1, 2, 0).cpu().numpy()
                skin_array = np.clip(skin_array * 255, 0, 255).astype(np.uint8)
                
                img = Image.fromarray(skin_array, 'RGBA')
                preview_path = f"improved_preview_epoch_{epoch:03d}.png"
                img.save(preview_path)
                
            self.generator.train()
        except Exception as e:
            print(f"Errore preview: {e}")

    def save_improved_model(self):
        """Salva il modello migliorato."""
        try:
            checkpoint = {
                'generator_state_dict': self.generator.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            }
            torch.save(checkpoint, config.MODEL_PATH)
        except Exception as e:
            print(f"Errore salvataggio: {e}")

    def test_improvements(self):
        """
        Testa il modello migliorato generando skin di esempio.
        Permette di valutare visivamente se i miglioramenti sono efficaci.
        """
        print("\nTEST MODELLO MIGLIORATO")
        print("=" * 30)
        
        self.generator.eval()
        test_results = []
        
        with torch.no_grad():
            for i in range(3):
                # Genera skin test
                noise = torch.randn(1, config.LATENT_DIM, 1, 1, device=self.device)
                generated = self.generator(noise)
                
                skin_array = generated.squeeze().permute(1, 2, 0).cpu().numpy()
                skin_array = np.clip(skin_array * 255, 0, 255).astype(np.uint8)
                
                # Valutazione qualitativa semplice
                quality = np.var(skin_array)  # Varietà colori come metrica
                
                # Salva skin test
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"IMPROVED_test_{i+1}_{timestamp}_Q{quality:.1f}.png"
                
                img = Image.fromarray(skin_array, 'RGBA')
                img.save(filename)
                
                test_results.append((filename, quality))
                print(f"Test {i+1}: {filename} (Q: {quality:.1f})")
        
        print(f"\nMedia qualità: {np.mean([q for _, q in test_results]):.1f}")
        print("Confronta le skin generate con quelle precedenti!")
        
        return test_results

def main():
    """Interfaccia principale per il miglioramento del modello."""
    print("MIGLIORAMENTO MODELLO AI - MINECRAFT SKIN")
    print("=" * 50)
    
    improver = ModelImprover()
    
    print("\nCosa vuoi fare?")
    print("1. Miglioramento completo (espansione + retraining)")
    print("2. Solo espansione dataset")  
    print("3. Solo retraining migliorato")
    print("4. Test modello attuale")
    
    choice = input("\nScegli (1-4): ").strip()
    
    if choice == "1":
        # Processo completo di miglioramento
        print("\nAVVIO MIGLIORAMENTO COMPLETO")
        print("Questo processo può richiedere diverso tempo...")
        
        # Step 1: Espansione dataset
        augmented = improver.expand_dataset(1000)
        
        if augmented > 0:
            # Step 2: Training migliorato
            improver.improved_training(150)
            
            # Step 3: Test finale
            improver.test_improvements()
            
            print("\nMIGLIORAMENTO COMPLETATO!")
            print("Il modello dovrebbe ora generare skin di qualità superiore!")
        
    elif choice == "2":
        # Solo espansione dataset
        improver.expand_dataset(1000)
        
    elif choice == "3":
        # Solo retraining
        improver.improved_training(150)
        
    elif choice == "4":
        # Solo test
        improver.test_improvements()
        
    else:
        print("Scelta non valida")

if __name__ == "__main__":
    main() 