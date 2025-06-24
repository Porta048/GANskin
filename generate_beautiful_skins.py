#!/usr/bin/env python3
"""
Generatore Avanzato di Skin Belle per Minecraft
Sistema specializzato nella creazione di skin di alta qualità estetica.

Utilizza algoritmi di valutazione automatica della qualità visiva
e tecniche di post-processing per migliorare l'aspetto finale.
"""

import torch
import numpy as np
from PIL import Image, ImageEnhance
import os
from datetime import datetime
import cv2

# Import dei nostri moduli
from models import SkinGenerator
import config

class BeautifulSkinGenerator:
    """
    Generatore specializzato per skin esteticamente superiori.
    
    Combina generazione AI con valutazione qualitativa automatica
    e post-processing per ottenere risultati visivamente accattivanti.
    """
    
    def __init__(self):
        # Configurazione hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device utilizzato: {self.device}")
        
        # Caricamento modello AI pre-addestrato
        self.generator = SkinGenerator().to(self.device)
        self.load_model()
        print("Sistema pronto per generare skin di qualità!")

    def load_model(self):
        """Carica il modello GAN salvato."""
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError("Modello non trovato! Esegui prima il training.")
        
        try:
            checkpoint = torch.load(config.MODEL_PATH, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.generator.eval()  # Modalità inferenza
            print("MODELLO MASTER (100 epoche) caricato!")
        except Exception as e:
            raise Exception(f"Errore caricamento modello: {e}")

    def evaluate_quality(self, skin_array):
        """
        Valuta la qualità estetica di una skin usando metriche computazionali.
        
        Combina diversi fattori:
        - Varietà cromatica (diversità di colori)
        - Dettaglio visivo (contrasto e nitidezza)
        - Bilanciamento strutturale
        - Nitidezza generale (nuovo)
        
        Returns:
            float: Punteggio qualità 0-10
        """
        try:
            # Converte in formato OpenCV per analisi
            bgr_skin = cv2.cvtColor(skin_array[:,:,:3], cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr_skin, cv2.COLOR_BGR2GRAY)
            
            # Metrica 1: Varietà cromatica (entropy dei colori)
            color_hist = cv2.calcHist([bgr_skin], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
            color_variety = cv2.compareHist(color_hist, np.ones_like(color_hist), cv2.HISTCMP_BHATTACHARYYA)
            
            # Metrica 2: Nitidezza (Laplacian variance) - MIGLIORATA
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 200, 3.0)  # Max 3 punti per nitidezza
            
            # Metrica 3: Contrasto locale (standard deviation)
            contrast = np.std(gray)
            contrast_score = min(contrast / 30, 2.0)  # Max 2 punti per contrasto
            
            # Metrica 4: Bilanciamento dell'immagine
            non_transparent = skin_array[:,:,3] > 0
            coverage = np.mean(non_transparent)
            balance_penalty = abs(coverage - 0.5) * 2
            balance_score = max(0, 2.0 - balance_penalty)  # Max 2 punti per bilanciamento
            
            # Metrica 5: Definizione bordi (edge detection)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (64 * 64)
            edge_score = min(edge_density * 10, 1.0)  # Max 1 punto per bordi definiti
            
            # Calcolo punteggio finale (0-10) con peso maggiore su nitidezza
            quality_score = (
                min(color_variety * 6, 2.0) +  # Max 2 punti per varietà
                sharpness_score +               # Max 3 punti per nitidezza (PESO MAGGIORE)
                contrast_score +                # Max 2 punti per contrasto  
                balance_score +                 # Max 2 punti per bilanciamento
                edge_score                      # Max 1 punto per bordi
            )
            
            return max(0, min(10, quality_score))
            
        except Exception as e:
            print(f"Errore valutazione qualità: {e}")
            return 2.0

    def enhance_skin(self, skin_array):
        """
        Applica post-processing aggressivo per migliorare nitidezza e dettagli.
        
        Include:
        - Sharpening più aggressivo
        - Riduzione blur
        - Miglioramento contrasto
        - Ottimizzazione colori
        """
        try:
            # Conversione a PIL per manipolazione
            img = Image.fromarray(skin_array, 'RGBA')
            
            # 1. Sharpening MOLTO aggressivo per ridurre sfocatura
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(2.5)  # Aumentato da 1.3 a 2.5
            
            # 2. Contrasto aumentato per definire meglio i dettagli
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.4)  # Aumentato da 1.2 a 1.4
            
            # 3. Conversione a numpy per filtri OpenCV
            img_array = np.array(img)
            bgr_part = cv2.cvtColor(img_array[:,:,:3], cv2.COLOR_RGB2BGR)
            
            # 4. Kernel di sharpening personalizzato
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            sharpened_bgr = cv2.filter2D(bgr_part, -1, kernel)
            
            # 5. Riconversione mantenendo canale alpha
            sharpened_rgb = cv2.cvtColor(sharpened_bgr, cv2.COLOR_BGR2RGB)
            final_array = img_array.copy()
            final_array[:,:,:3] = sharpened_rgb
            
            # 6. Clip per evitare overflow
            final_array = np.clip(final_array, 0, 255)
            
            return final_array.astype(np.uint8)
            
        except Exception as e:
            print(f"Errore enhancement: {e}")
            return skin_array

    def generate_beautiful_skin(self, max_attempts=25):  # Aumentati i tentativi
        """
        Genera una skin bella con criteri più severi per la nitidezza.
        
        Args:
            max_attempts: Numero massimo di tentativi per trovare una skin nitida
            
        Returns:
            tuple: (skin_array, quality_score)
        """
        best_skin = None
        best_quality = 0
        
        print("Generazione skin nitida in corso...")
        
        with torch.no_grad():
            for attempt in range(max_attempts):
                # Genera skin candidata
                noise = torch.randn(1, config.LATENT_DIM, 1, 1, device=self.device)
                generated = self.generator(noise)
                
                # Conversione a numpy
                skin_array = generated.squeeze().permute(1, 2, 0).cpu().numpy()
                skin_array = np.clip(skin_array * 255, 0, 255).astype(np.uint8)
                
                # Applica enhancement PRIMA della valutazione
                enhanced_skin = self.enhance_skin(skin_array)
                
                # Valuta qualità sulla skin enhanced
                quality = self.evaluate_quality(enhanced_skin)
                
                print(f"   Tentativo {attempt+1}/{max_attempts}: Qualità {quality:.1f}/10")
                
                # Criteri più severi: accetta solo se > 6.0 per nitidezza
                if quality > best_quality and quality >= 6.0:
                    best_quality = quality
                    best_skin = enhanced_skin.copy()
                    
                # Se troviamo qualità > 7.5, fermiamoci (già ottima)
                if quality >= 7.5:
                    break
        
        return best_skin, best_quality

    def save_beautiful_skin(self, skin_array, quality):
        """
        Salva la skin con nome descrittivo basato sulla qualità.
        
        Include timestamp e classificazione qualitativa nel filename.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Classificazione qualitativa
        if quality >= 8:
            quality_label = "ECCELLENTE"
        elif quality >= 6:
            quality_label = "MOLTO_BELLA"
        elif quality >= 4:
            quality_label = "BELLA"
        else:
            quality_label = "NORMALE"
        
        # Genera filename descrittivo
        filename = f"BEAUTIFUL_skin_{timestamp}_Q{quality:.1f}_{quality_label}.png"
        
        # Salva immagine
        img = Image.fromarray(skin_array, 'RGBA')
        img.save(filename)
        
        return filename

def main():
    """Interfaccia utente per la generazione di skin belle."""
    print("GENERATORE SKIN BELLE ED ORDINATE")
    print("Device: cpu")
    
    try:
        # Inizializza il generatore
        generator = BeautifulSkinGenerator()
        
        print("\nOpzioni disponibili:")
        print("1. Genera 1 skin bella singola")
        print("2. Genera 5 skin belle (collezione)")
        print("3. Genera 10 skin belle (set completo)")
        
        choice = input("\nScegli (1-3): ").strip()
        
        if choice == "1":
            count = 1
        elif choice == "2":
            count = 5
        elif choice == "3":
            count = 10
        else:
            print("Scelta non valida")
            return
        
        print(f"\nGENERAZIONE {count} SKIN BELLE ED ORDINATE")
        print("=" * 50)
        
        total_quality = 0
        generated_files = []
        
        # Loop generazione
        for i in range(count):
            print(f"\nSKIN {i+1}/{count}:")
            
            # Genera skin bella
            skin_array, quality = generator.generate_beautiful_skin()
            
            if skin_array is not None:
                # Salva con nome descrittivo
                filename = generator.save_beautiful_skin(skin_array, quality)
                
                print(f"   Salvata: {filename}")
                print(f"   Qualità finale: {quality:.1f}/10")
                
                # Classificazione qualitativa
                if quality >= 8:
                    print("   ECCELLENTE!")
                elif quality >= 6:
                    print("   MOLTO BELLA!")
                elif quality >= 4:
                    print("   BELLA!")
                else:
                    print("   NORMALE")
                
                total_quality += quality
                generated_files.append(filename)
            else:
                print("   Errore generazione")
        
        # Statistiche finali
        if generated_files:
            avg_quality = total_quality / len(generated_files)
            best_file = max(generated_files, key=lambda f: float(f.split('_Q')[1].split('_')[0]))
            
            print(f"\nRISULTATI FINALI:")
            print("=" * 30)
            print(f"Skin generate: {len(generated_files)}/{count}")
            print(f"Qualità media: {avg_quality:.1f}/10")
            print(f"Miglior skin: {best_file}")
            print("Il modello può migliorare ancora!")
        
    except Exception as e:
        print(f"Errore: {e}")

if __name__ == "__main__":
    main() 