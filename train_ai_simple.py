#!/usr/bin/env python3
"""
Sistema di Training AI per Generazione Skin Minecraft
Sviluppato per addestrare una rete GAN (Generative Adversarial Network) 
specializzata nella creazione di skin personalizzate per Minecraft.

L'approccio utilizzato è semplificato ma efficace, focalizzato sui risultati pratici.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from PIL import Image
import os
from datetime import datetime
import time

# Importiamo i nostri moduli personalizzati
from models import SkinGenerator, SkinDiscriminator
from dataset import SkinDataset
import config

class SimpleAITrainer:
    def __init__(self, max_samples=None):
        print("Inizializzazione Sistema di Training AI")
        print("=" * 50)
        
        # Configurazione hardware - utilizza GPU se disponibile per velocità
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device utilizzato: {self.device}")
        
        # Gestione del dataset - possiamo limitarlo per test più rapidi
        full_dataset = SkinDataset()
        
        if max_samples and max_samples < len(full_dataset):
            # Per test rapidi, prendiamo solo un subset casuale
            indices = torch.randperm(len(full_dataset))[:max_samples]
            self.dataset = Subset(full_dataset, indices)
            print(f"Dataset limitato: {len(self.dataset)} skin (su {len(full_dataset)} totali)")
        else:
            self.dataset = full_dataset
            print(f"Dataset completo: {len(self.dataset)} skin")
        
        # Inizializzazione dei modelli neurali
        # Generator: crea skin da rumore casuale
        # Discriminator: distingue skin reali da quelle generate
        self.generator = SkinGenerator().to(self.device)
        self.discriminator = SkinDiscriminator().to(self.device)
        
        # Optimizer Adam - configurazione standard per GAN
        # Parametri beta ottimizzati per la stabilità del training
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Binary Cross Entropy Loss - standard per problemi di classificazione binaria
        self.criterion = nn.BCELoss()
        
        # Tentativo di caricare un modello pre-esistente
        self.load_model()
        
        print("Inizializzazione completata!")

    def train(self, epochs=100, batch_size=None):
        """
        Funzione principale di training della GAN.
        
        Il processo alterna tra:
        1. Training del Discriminator (impara a riconoscere skin reali)
        2. Training del Generator (impara a creare skin realistiche)
        """
        
        # Calcolo automatico del batch size ottimale in base al dataset
        if batch_size is None:
            if len(self.dataset) < 1000:
                batch_size = min(8, len(self.dataset))
            elif len(self.dataset) < 5000:
                batch_size = 16
            else:
                batch_size = 32
        
        # Verifica che abbiamo abbastanza dati
        if len(self.dataset) < batch_size:
            batch_size = len(self.dataset)
            print(f"⚠️ Batch size ridotto a {batch_size}")
        
        print(f"\nAvvio training per {epochs} epoche")
        print(f"Batch size: {batch_size}")
        print(f"Dataset size: {len(self.dataset)}")
        print("=" * 50)
        
        # DataLoader per il caricamento efficiente dei batch
        dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True,  # Mescoliamo i dati per evitare pattern
            num_workers=0,  # Su Windows è più stabile
            drop_last=True  # Evitiamo batch di dimensioni diverse
        )
        
        start_time = time.time()
        
        # Loop principale di training
        for epoch in range(epochs):
            epoch_d_loss = 0
            epoch_g_loss = 0
            num_batches = 0
            
            for i, real_imgs in enumerate(dataloader):
                try:
                    real_imgs = real_imgs.to(self.device)
                    b_size = real_imgs.size(0)
                    
                    # Label smoothing - trucco per migliorare la stabilità
                    # Invece di 1.0 e 0.0, usiamo 0.9 e 0.1
                    real_labels = torch.full((b_size,), 0.9, device=self.device, dtype=torch.float)
                    fake_labels = torch.full((b_size,), 0.1, device=self.device, dtype=torch.float)

                    # === TRAINING DEL DISCRIMINATOR ===
                    self.optimizer_D.zero_grad()
                    
                    # Valutazione su immagini reali
                    output_real = self.discriminator(real_imgs)
                    loss_d_real = self.criterion(output_real, real_labels)
                    
                    # Generazione di immagini fake per il test
                    noise = torch.randn(b_size, config.LATENT_DIM, 1, 1, device=self.device)
                    fake_imgs = self.generator(noise)
                    
                    # Valutazione su immagini fake (senza aggiornare il generator)
                    output_fake = self.discriminator(fake_imgs.detach())
                    loss_d_fake = self.criterion(output_fake, fake_labels)
                    
                    # Backpropagation del discriminator
                    loss_d = loss_d_real + loss_d_fake
                    loss_d.backward()
                    self.optimizer_D.step()

                    # === TRAINING DEL GENERATOR ===
                    self.optimizer_G.zero_grad()
                    
                    # Il generator cerca di "ingannare" il discriminator
                    # Vogliamo che le fake images vengano classificate come reali
                    output = self.discriminator(fake_imgs)
                    loss_g = self.criterion(output, real_labels)
                    loss_g.backward()
                    self.optimizer_G.step()
                    
                    # Statistiche per monitoraggio
                    epoch_d_loss += loss_d.item()
                    epoch_g_loss += loss_g.item()
                    num_batches += 1
                    
                    # Progress feedback durante il training
                    if (i + 1) % 50 == 0:
                        print(f"  Batch [{i+1:3d}/{len(dataloader)}] - "
                              f"Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")
                    
                except Exception as e:
                    print(f"⚠️ Errore nel batch {i}: {e}")
                    continue

            # Statistiche dell'epoca completata
            avg_d_loss = epoch_d_loss / max(num_batches, 1)
            avg_g_loss = epoch_g_loss / max(num_batches, 1)
            elapsed = time.time() - start_time
            
            print(f"Epoca [{epoch+1:3d}/{epochs}] COMPLETATA - "
                  f"Loss D: {avg_d_loss:.4f}, Loss G: {avg_g_loss:.4f} - "
                  f"Tempo totale: {elapsed:.1f}s")
            
            # Generiamo anteprime per monitorare i progressi
            if (epoch + 1) % 5 == 0:
                self.generate_preview(epoch + 1)
        
        print(f"\nTraining completato in {time.time() - start_time:.1f} secondi!")
        self.save_model()

    def generate_preview(self, epoch):
        """
        Genera una skin di anteprima per vedere i progressi del training.
        Utile per monitorare visivamente se il modello sta migliorando.
        """
        try:
            self.generator.eval()  # Modalità valutazione
            with torch.no_grad():  # Disabilita il calcolo del gradiente
                # Genera rumore casuale
                noise = torch.randn(1, config.LATENT_DIM, 1, 1, device=self.device)
                generated = self.generator(noise)
                
                # Conversione da tensor PyTorch a immagine PIL
                skin_array = generated.squeeze().permute(1, 2, 0).cpu().numpy()
                skin_array = np.clip(skin_array * 255, 0, 255).astype(np.uint8)
                
                # Salvataggio dell'anteprima
                img = Image.fromarray(skin_array, 'RGBA')
                preview_path = f"preview_epoch_{epoch:03d}.png"
                img.save(preview_path)
                print(f"  Anteprima salvata: {preview_path}")
                
            self.generator.train()  # Torna in modalità training
            
        except Exception as e:
            print(f"  ⚠️ Errore generazione anteprima: {e}")

    def generate_skin(self):
        """
        Genera una singola skin usando il modello addestrato.
        Restituisce un array numpy pronto per essere salvato come immagine.
        """
        self.generator.eval()
        with torch.no_grad():
            # Input casuale nel latent space
            noise = torch.randn(1, config.LATENT_DIM, 1, 1, device=self.device)
            generated = self.generator(noise)
            
            # Conversione e normalizzazione
            skin_array = generated.squeeze().permute(1, 2, 0).cpu().numpy()
            skin_array = np.clip(skin_array * 255, 0, 255).astype(np.uint8)
            
            return skin_array

    def save_model(self):
        """Salva lo stato completo del modello per utilizzi futuri."""
        os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
        
        # Salviamo tutti i componenti necessari per riprendere il training
        checkpoint = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
        }
        torch.save(checkpoint, config.MODEL_PATH)
        print(f"Modello salvato: {config.MODEL_PATH}")

    def load_model(self):
        """Carica un modello precedentemente salvato, se esistente."""
        if os.path.exists(config.MODEL_PATH):
            try:
                checkpoint = torch.load(config.MODEL_PATH, map_location=self.device)
                self.generator.load_state_dict(checkpoint['generator_state_dict'])
                self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
                self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
                print("Modello caricato con successo!")
                return True
            except Exception as e:
                print(f"⚠️ Errore caricamento modello: {e}")
        return False

def main():
    """Interfaccia utente principale per l'utilizzo del sistema."""
    print("\nOpzioni di Training:")
    print("1. Training completo (100 epoche, tutto il dataset)")
    print("2. Training veloce (20 epoche, dataset limitato)")
    print("3. Training test (5 epoche, 500 skin)")
    print("4. Genera skin singola")
    
    choice = input("\nScegli (1-4): ").strip()
    
    if choice == "1":
        # Training completo per risultati ottimali
        trainer = SimpleAITrainer()
        trainer.train(epochs=100)
    elif choice == "2":
        # Training rapido per test e sviluppo
        trainer = SimpleAITrainer(max_samples=2000)
        trainer.train(epochs=20)
    elif choice == "3":
        # Training di test molto veloce
        trainer = SimpleAITrainer(max_samples=500)
        trainer.train(epochs=5)
    elif choice == "4":
        # Generazione singola usando modello esistente
        trainer = SimpleAITrainer()
        skin = trainer.generate_skin()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_skin_{timestamp}.png"
        Image.fromarray(skin, 'RGBA').save(filename)
        print(f"Skin generata: {filename}")
    else:
        print("Scelta non valida")

if __name__ == "__main__":
    main() 