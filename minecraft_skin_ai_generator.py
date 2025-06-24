import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import requests
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
import io
import base64
from datetime import datetime
import threading

app = Flask(__name__)
CORS(app)

class SkinDataset(Dataset):
    """Dataset personalizzato per le skin di Minecraft"""
    
    def __init__(self, data_dir="./skin_dataset", transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.skin_files = []
        self.metadata = {}
        
        # Crea directory se non esiste
        os.makedirs(data_dir, exist_ok=True)
        
        # Carica le skin esistenti
        self.load_existing_skins()
    
    def load_existing_skins(self):
        """Carica le skin esistenti nel dataset"""
        if os.path.exists(self.data_dir):
            for file in os.listdir(self.data_dir):
                if file.endswith('.png'):
                    self.skin_files.append(os.path.join(self.data_dir, file))
        
        # Carica metadata se esiste
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
    
    def __len__(self):
        return len(self.skin_files)
    
    def __getitem__(self, idx):
        img_path = self.skin_files[idx]
        image = Image.open(img_path).convert('RGBA')
        
        # Ridimensiona a 64x64 se necessario
        if image.size != (64, 64):
            image = image.resize((64, 64), Image.NEAREST)
        
        # Converti in array numpy e normalizza
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Converti in tensore PyTorch
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # HWC -> CHW
        
        return image_tensor

class ConditionalSkinGenerator(nn.Module):
    """Generatore condizionale per skin con temi specifici"""
    
    def __init__(self, latent_dim=128, num_classes=5):
        super(ConditionalSkinGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Embedding per i temi
        self.label_embedding = nn.Embedding(num_classes, 50)
        
        # Generatore principale
        self.main = nn.Sequential(
            # Input: latent_dim + 50
            nn.Linear(latent_dim + 50, 256 * 8 * 8),
            nn.ReLU(True),
            
            # Reshape to (256, 8, 8)
            nn.Unflatten(1, (256, 8, 8)),
            
            # Upsampling con convoluzione trasposta
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # Output layer per RGBA
            nn.ConvTranspose2d(32, 4, 3, 1, 1),     # 64x64, 4 canali
            nn.Sigmoid()  # Output tra 0 e 1
        )
    
    def forward(self, z, labels):
        # Combina noise e label embedding
        label_embed = self.label_embedding(labels)
        input_tensor = torch.cat([z, label_embed], dim=1)
        
        return self.main(input_tensor)

class SkinDiscriminator(nn.Module):
    """Discriminatore per distinguere skin reali da generate"""
    
    def __init__(self):
        super(SkinDiscriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: (4, 64, 64)
            nn.Conv2d(4, 32, 4, 2, 1),              # 32x32
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, 4, 2, 1),             # 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1),            # 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1),           # 4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Classificazione finale
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.main(img)

class MinecraftSkinAI:
    """Sistema AI completo per generare skin Minecraft"""
    
    def __init__(self, data_dir="./skin_dataset"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Inizializzando Minecraft Skin AI Generator")
        print(f"Device: {self.device}")
        
        self.data_dir = data_dir
        self.latent_dim = 128
        
        # Temi/stili disponibili per le skin
        self.themes = {
            'warrior': 0,    # Guerriero/Cavaliere
            'mage': 1,       # Mago/Stregone
            'nature': 2,     # Natura/Animali
            'tech': 3,       # Tecnologico/Futuristico
            'shadow': 4      # Oscuro/Assassino
        }
        
        # Inizializza modelli neurali
        self.generator = ConditionalSkinGenerator(
            self.latent_dim, 
            len(self.themes)
        ).to(self.device)
        
        self.discriminator = SkinDiscriminator().to(self.device)
        
        # Ottimizzatori con parametri specifici per GAN
        self.optimizer_G = optim.Adam(
            self.generator.parameters(), 
            lr=0.0002, 
            betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), 
            lr=0.0002, 
            betas=(0.5, 0.999)
        )
        
        # Funzione di loss
        self.criterion = nn.BCELoss()
        
        # Dataset e stato del training
        self.dataset = SkinDataset(data_dir)
        self.is_trained = False
        self.training_losses = {'generator': [], 'discriminator': []}
        
        # Carica modelli pre-addestrati se esistono
        self.load_models()
        
        print(f"Dataset attuale: {len(self.dataset)} skin")
        print(f"Temi disponibili: {list(self.themes.keys())}")
        
        # Auto-addestramento se necessario
        if not self.is_trained and len(self.dataset) >= 50:
            print(f"Avvio training automatico su {len(self.dataset)} skin...")
            import threading
            def auto_train():
                try:
                    self.train(epochs=80, batch_size=8)
                    print("✅ Training automatico completato!")
                except Exception as e:
                    print(f"❌ Errore training automatico: {e}")
            
            thread = threading.Thread(target=auto_train)
            thread.daemon = True
            thread.start()
    
    def train(self, epochs=100, batch_size=8):
        """Addestra il modello GAN"""
        if len(self.dataset) < 10:
            print("⚠️ Dataset troppo piccolo!")
            return
        
        dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True
        )
        
        print(f"🚀 Iniziando training con {len(self.dataset)} skin per {epochs} epochs...")
        
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(epochs):
            epoch_loss_G = 0
            epoch_loss_D = 0
            num_batches = 0
            
            for batch_idx, real_skins in enumerate(dataloader):
                batch_size_actual = real_skins.size(0)
                real_skins = real_skins.to(self.device)
                
                # Label per training
                real_labels = torch.ones(batch_size_actual, 1).to(self.device)
                fake_labels = torch.zeros(batch_size_actual, 1).to(self.device)
                
                # === ADDESTRA DISCRIMINATORE ===
                self.optimizer_D.zero_grad()
                
                # Loss su skin reali
                output_real = self.discriminator(real_skins)
                loss_real = self.criterion(output_real, real_labels)
                
                # Genera skin fake
                z = torch.randn(batch_size_actual, self.latent_dim).to(self.device)
                random_themes = torch.randint(0, len(self.themes), (batch_size_actual,)).to(self.device)
                fake_skins = self.generator(z, random_themes)
                
                # Loss su skin fake
                output_fake = self.discriminator(fake_skins.detach())
                loss_fake = self.criterion(output_fake, fake_labels)
                
                # Loss totale discriminatore
                loss_D = (loss_real + loss_fake) / 2
                loss_D.backward()
                self.optimizer_D.step()
                
                # === ADDESTRA GENERATORE ===
                self.optimizer_G.zero_grad()
                
                # Il generatore vuole ingannare il discriminatore
                output_fake_G = self.discriminator(fake_skins)
                loss_G = self.criterion(output_fake_G, real_labels)
                
                loss_G.backward()
                self.optimizer_G.step()
                
                # Accumula loss per statistiche
                epoch_loss_G += loss_G.item()
                epoch_loss_D += loss_D.item()
                num_batches += 1
                
                if batch_idx % 5 == 0:
                    print(f"📈 Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                          f"Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")
            
            # Salva loss medie dell'epoca
            avg_loss_G = epoch_loss_G / num_batches
            avg_loss_D = epoch_loss_D / num_batches
            
            self.training_losses['generator'].append(avg_loss_G)
            self.training_losses['discriminator'].append(avg_loss_D)
            
            print(f"🎯 Epoch {epoch+1} completata - Avg Loss_D: {avg_loss_D:.4f}, Avg Loss_G: {avg_loss_G:.4f}")
            
            # Salva modelli periodicamente
            if (epoch + 1) % 20 == 0:
                self.save_models()
                print(f"💾 Modelli salvati all'epoch {epoch + 1}")
        
        self.is_trained = True
        self.save_models()
        print("🎉 Training completato!")
    
    def generate_skin(self, theme='random', seed=None):
        """Genera una nuova skin utilizzando l'AI"""
        if not self.is_trained and len(self.dataset) >= 50:
            print("Avvio training immediato...")
            self.train(epochs=80, batch_size=8)
        
        self.generator.eval()
        
        if seed is not None:
            torch.manual_seed(seed)
        
        with torch.no_grad():
            # Genera rumore latente
            z = torch.randn(1, self.latent_dim).to(self.device)
            
            # Seleziona tema
            if theme == 'random':
                theme_idx = torch.randint(0, len(self.themes), (1,)).to(self.device)
                used_theme = list(self.themes.keys())[theme_idx.item()]
            else:
                if theme not in self.themes:
                    theme = 'warrior'  # fallback
                theme_idx = torch.tensor([self.themes[theme]]).to(self.device)
                used_theme = theme
            
            # Genera skin
            generated_skin = self.generator(z, theme_idx)
            
            # Converti in array numpy
            skin_array = generated_skin.squeeze().cpu().numpy()
            skin_array = (skin_array * 255).astype(np.uint8)
            skin_array = skin_array.transpose(1, 2, 0)  # CHW -> HWC
            
            return skin_array, used_theme
    
    def generate_fusion_skin(self, fusion_method='intelligent', num_samples=20):
        """Genera una skin che combina tutti i dati del dataset"""
        print(f"Generando FUSION SKIN da {len(self.dataset)} skin totali...")
        
        if len(self.dataset) == 0:
            print("ERRORE: Dataset vuoto!")
            return None, "empty"
        
        # Campiona diverse skin dal dataset in modo sicuro
        max_samples = min(num_samples, len(self.dataset))
        all_indices = list(range(len(self.dataset)))
        np.random.shuffle(all_indices)
        sample_indices = all_indices[:max_samples]
        
        fusion_skins = []
        
        for idx in sample_indices:
            skin_tensor = self.dataset[idx]  # Tensore 4D (RGBA, 64, 64)
            skin_array = (skin_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            fusion_skins.append(skin_array)
        
        if fusion_method == 'intelligent':
            # Metodo intelligente: combina caratteristiche diverse
            final_skin = self._intelligent_fusion(fusion_skins)
        elif fusion_method == 'average':
            # Media semplice
            final_skin = self._average_fusion(fusion_skins)
        elif fusion_method == 'mosaic':
            # Mosaico di parti diverse
            final_skin = self._mosaic_fusion(fusion_skins)
        else:
            # Blend avanzato
            final_skin = self._advanced_blend(fusion_skins)
        
        return final_skin, f"fusion_{fusion_method}_{len(fusion_skins)}skins"
    
    def _intelligent_fusion(self, skins):
        """Fusione intelligente che preserva le caratteristiche migliori"""
        print("Applicando fusione intelligente...")
        
        # Inizia con una skin di base
        base_skin = skins[0].copy()
        
        # Definisci le regioni della skin Minecraft
        regions = {
            'head_front': (8, 8, 16, 16),      # Faccia
            'head_top': (8, 0, 16, 8),        # Testa sopra
            'body_front': (20, 20, 28, 32),   # Corpo davanti
            'arm_right': (44, 20, 48, 32),    # Braccio destro
            'arm_left': (36, 52, 40, 64),     # Braccio sinistro  
            'leg_right': (4, 20, 8, 32),      # Gamba destra
            'leg_left': (20, 52, 24, 64)      # Gamba sinistra
        }
        
        # Per ogni regione, scegli la versione migliore da una skin casuale
        for region_name, (x1, y1, x2, y2) in regions.items():
            random_idx = np.random.randint(0, len(skins))
            random_skin = skins[random_idx]
            
            # Combina la regione con quella esistente
            alpha = 0.7  # 70% nuova skin, 30% base
            base_skin[y1:y2, x1:x2] = (
                alpha * random_skin[y1:y2, x1:x2] + 
                (1 - alpha) * base_skin[y1:y2, x1:x2]
            ).astype(np.uint8)
        
        # Aggiungi dettagli da altre skin
        for i, skin in enumerate(skins[1:6]):  # Usa prime 5 skin aggiuntive
            mask = np.random.rand(64, 64, 4) > 0.8  # 20% di possibilità per pixel
            base_skin = np.where(mask, skin, base_skin)
        
        return base_skin
    
    def _average_fusion(self, skins):
        """Media pesata di tutte le skin"""
        print("Applicando media pesata...")
        
        # Converti in float per calcoli
        skin_arrays = [skin.astype(np.float32) for skin in skins]
        
        # Media pesata (più peso alle prime skin)
        weights = np.exp(-np.linspace(0, 2, len(skin_arrays)))
        weights = weights / weights.sum()
        
        result = np.zeros_like(skin_arrays[0])
        for skin, weight in zip(skin_arrays, weights):
            result += skin * weight
        
        return result.astype(np.uint8)
    
    def _mosaic_fusion(self, skins):
        """Crea un mosaico usando parti di skin diverse"""
        print("Creando mosaico fusion...")
        
        result = np.zeros((64, 64, 4), dtype=np.uint8)
        
        # Dividi in blocchi 8x8 
        for y in range(0, 64, 8):
            for x in range(0, 64, 8):
                # Scegli una skin casuale per questo blocco
                random_idx = np.random.randint(0, len(skins))
                random_skin = skins[random_idx]
                result[y:y+8, x:x+8] = random_skin[y:y+8, x:x+8]
        
        # Applica un filtro di smoothing
        for i in range(3):  # 3 passaggi di smoothing
            for y in range(1, 63):
                for x in range(1, 63):
                    # Media con i pixel adiacenti
                    neighbors = result[y-1:y+2, x-1:x+2]
                    result[y, x] = neighbors.mean(axis=(0,1)).astype(np.uint8)
        
        return result
    
    def _advanced_blend(self, skins):
        """Blend avanzato con preservazione delle caratteristiche"""
        print("Applicando blend avanzato...")
        
        # Inizia con la prima skin
        result = skins[0].astype(np.float32)
        
        # Applica ogni skin successiva con blend mode diversi
        blend_modes = ['multiply', 'overlay', 'soft_light', 'color_dodge']
        
        for i, skin in enumerate(skins[1:min(8, len(skins))]):
            mode = blend_modes[i % len(blend_modes)]
            alpha = 0.3 - (i * 0.03)  # Alpha decrescente
            
            skin_float = skin.astype(np.float32)
            
            if mode == 'multiply':
                blended = (result * skin_float) / 255.0
            elif mode == 'overlay':
                blended = np.where(result < 128, 
                                 2 * result * skin_float / 255.0,
                                 255 - 2 * (255 - result) * (255 - skin_float) / 255.0)
            elif mode == 'soft_light':
                blended = (1 - 2 * skin_float / 255.0) * result**2 / 255.0 + \
                         2 * skin_float * result / 255.0
            else:  # color_dodge
                blended = np.where(skin_float == 255, 255,
                                 np.minimum(255, result * 255 / (255 - skin_float + 1e-10)))
            
            result = alpha * blended + (1 - alpha) * result
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def save_models(self):
        """Salva i modelli addestrati"""
        os.makedirs("./models", exist_ok=True)
        
        checkpoint = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'training_losses': self.training_losses,
            'is_trained': self.is_trained,
            'themes': self.themes,
            'latent_dim': self.latent_dim
        }
        
        torch.save(checkpoint, './models/minecraft_skin_gan.pth')
        print("Modelli salvati in ./models/minecraft_skin_gan.pth")
    
    def load_models(self):
        """Carica modelli pre-addestrati"""
        model_path = './models/minecraft_skin_gan.pth'
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                self.generator.load_state_dict(checkpoint['generator_state_dict'])
                self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
                self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
                self.training_losses = checkpoint.get('training_losses', {'generator': [], 'discriminator': []})
                self.is_trained = checkpoint.get('is_trained', False)
                
                print("SUCCESSO: Modelli pre-addestrati caricati con successo!")
                
            except Exception as e:
                print(f"ERRORE caricando modelli: {e}")
    
    def skin_to_base64(self, skin_array):
        """Converte array numpy in base64"""
        img = Image.fromarray(skin_array, 'RGBA')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode()

    def generate_intelligent_hybrid_skin(self):
        """Genera una skin combinando AI e FUSION in modo intelligente"""
        print("Avvio algoritmo di generazione ibrida intelligente...")
        
        # Seleziona strategia di generazione basata su casualità pesata
        strategy_weights = {
            'ai_pure': 0.3,                    # 30% AI pura
            'ai_enhanced_fusion': 0.25,        # 25% AI + miglioramenti FUSION  
            'multi_fusion_blend': 0.25,        # 25% Multi-FUSION blend
            'evolutionary_fusion': 0.2         # 20% FUSION evolutiva
        }
        
        # Scelta strategia
        strategies = list(strategy_weights.keys())
        weights = list(strategy_weights.values())
        chosen_strategy = np.random.choice(strategies, p=weights)
        
        print(f"Strategia selezionata: {chosen_strategy}")
        
        # Genera seed unico basato su timestamp per garantire variabilità
        unique_seed = int(datetime.now().timestamp() * 1000000) % 2147483647
        np.random.seed(unique_seed)
        torch.manual_seed(unique_seed)
        
        if chosen_strategy == 'ai_pure' and self.is_trained:
            return self._generate_ai_pure_variable()
        elif chosen_strategy == 'ai_enhanced_fusion':
            return self._generate_ai_enhanced_fusion()
        elif chosen_strategy == 'multi_fusion_blend':
            return self._generate_multi_fusion_blend()
        else:  # evolutionary_fusion
            return self._generate_evolutionary_fusion()
    
    def _generate_ai_pure_variable(self):
        """Genera skin AI pura con alta variabilità"""
        print("Generazione AI pura con alta variabilità...")
        
        # Usa parametri casuali per massima variabilità
        theme_variations = ['random', 'warrior', 'mage', 'nature', 'tech', 'shadow']
        selected_theme = np.random.choice(theme_variations)
        
        # Genera con seed variabile
        skin_array, theme = self.generate_skin(selected_theme)
        return skin_array, f"AI_VARIABLE_{theme}"
    
    def _generate_ai_enhanced_fusion(self):
        """Combina AI con elementi FUSION per risultati unici"""
        print("Generazione AI potenziata con elementi FUSION...")
        
        try:
            # Genera base AI se possibile
            if self.is_trained:
                base_skin, ai_theme = self.generate_skin('random')
            else:
                # Fallback a FUSION se AI non disponibile
                base_skin, _ = self.generate_fusion_skin('intelligent', 15)
                ai_theme = 'fusion_base'
            
            # Genera elementi FUSION da aggiungere
            fusion_skin, _ = self.generate_fusion_skin('mosaic', 
                                                      np.random.randint(8, 25))
            
            # Combina AI base con elementi FUSION usando blend intelligente
            final_skin = self._intelligent_blend(base_skin, fusion_skin)
            
            return final_skin, f"AI_ENHANCED_{ai_theme}_with_fusion"
            
        except Exception as e:
            print(f"Errore in AI enhanced fusion: {e}")
            # Fallback sicuro
            return self.generate_fusion_skin('intelligent', 20)
    
    def _generate_multi_fusion_blend(self):
        """Combina multiple strategie FUSION per risultati complessi"""
        print("Generazione multi-FUSION blend...")
        
        # Genera multiple skin FUSION con metodi diversi
        fusion_methods = ['intelligent', 'average', 'mosaic', 'advanced']
        fusion_results = []
        
        for method in fusion_methods:
            skin_data, _ = self.generate_fusion_skin(
                method, 
                np.random.randint(5, 15)
            )
            if skin_data is not None:
                fusion_results.append(skin_data)
        
        if len(fusion_results) < 2:
            # Fallback se non abbastanza risultati
            return self.generate_fusion_skin('intelligent', 25)
        
        # Combina i risultati con algoritmo avanzato
        final_skin = self._advanced_multi_blend(fusion_results)
        return final_skin, f"MULTI_FUSION_BLEND_{len(fusion_results)}methods"
    
    def _generate_evolutionary_fusion(self):
        """Simula evoluzione genetica per skin uniche"""
        print("Generazione FUSION evolutiva...")
        
        # Genera popolazione iniziale
        population_size = np.random.randint(6, 12)
        population = []
        
        for _ in range(population_size):
            method = np.random.choice(['intelligent', 'average', 'mosaic', 'advanced'])
            samples = np.random.randint(3, 20)
            skin_data, _ = self.generate_fusion_skin(method, samples)
            if skin_data is not None:
                population.append(skin_data)
        
        if len(population) < 3:
            return self.generate_fusion_skin('advanced', 30)
        
        # Simula "evoluzione" combinando le migliori caratteristiche
        evolved_skin = self._evolutionary_combination(population)
        return evolved_skin, f"EVOLUTIONARY_FUSION_gen{len(population)}"
    
    def _intelligent_blend(self, skin1, skin2):
        """Blend intelligente tra due skin"""
        # Blend diversificato per regioni
        result = skin1.copy().astype(np.float32)
        skin2_float = skin2.astype(np.float32)
        
        # Regioni con blend diversi
        regions = [
            ((0, 0, 32, 16), 0.4),      # Testa - blend leggero
            ((0, 16, 56, 32), 0.6),     # Corpo - blend medio
            ((0, 32, 64, 64), 0.3),     # Gambe - blend leggero
        ]
        
        for (x1, y1, x2, y2), alpha in regions:
            region_blend = alpha * skin2_float[y1:y2, x1:x2] + (1-alpha) * result[y1:y2, x1:x2]
            result[y1:y2, x1:x2] = region_blend
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _advanced_multi_blend(self, skins):
        """Blend avanzato di multiple skin"""
        if len(skins) == 0:
            return np.zeros((64, 64, 4), dtype=np.uint8)
        
        result = skins[0].astype(np.float32)
        
        for i, skin in enumerate(skins[1:]):
            # Peso decrescente per ogni skin aggiuntiva
            weight = 0.5 / (i + 1)
            result = weight * skin.astype(np.float32) + (1-weight) * result
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _evolutionary_combination(self, population):
        """Combina popolazione usando principi evolutivi"""
        # Seleziona "genitori" migliori (casuali per ora)
        np.random.shuffle(population)
        parents = population[:3]
        
        # Crossover: combina caratteristiche dei genitori
        result = parents[0].astype(np.float32)
        
        # Eredita caratteristiche da altri genitori
        for y in range(0, 64, 8):
            for x in range(0, 64, 8):
                parent_idx = np.random.randint(0, len(parents))
                parent_region = parents[parent_idx][y:y+8, x:x+8]
                
                # Mutation: aggiungi variabilità
                mutation_factor = 0.9 + np.random.random() * 0.2  # 0.9-1.1
                mutated_region = parent_region.astype(np.float32) * mutation_factor
                
                result[y:y+8, x:x+8] = mutated_region
        
        return np.clip(result, 0, 255).astype(np.uint8)



# Istanza globale del sistema AI
print("Inizializzando Minecraft Skin AI...")
ai_generator = MinecraftSkinAI()

# === GENERAZIONE AUTOMATICA ALL'AVVIO ===
def generate_startup_skin():
    """Genera automaticamente una skin intelligente all'avvio del programma"""
    try:
        print("\n=== GENERAZIONE AUTOMATICA INTELLIGENTE ALL'AVVIO ===")
        
        # Controlla se abbiamo skin nel dataset
        if len(ai_generator.dataset) == 0:
            print("ATTENZIONE: Nessuna skin trovata nel dataset! Impossibile generare skin.")
            return
            
        print(f"Dataset caricato: {len(ai_generator.dataset)} skin")
        print(f"Modello addestrato: {'Si' if ai_generator.is_trained else 'No'}")
        
        # Crea directory per le skin generate se non esiste
        os.makedirs('downloaded_skins', exist_ok=True)
        
        # Usa sempre il generatore intelligente ibrido
        print("Avvio generazione intelligente ibrida...")
        skin_array, method_info = ai_generator.generate_intelligent_hybrid_skin()
        
        if skin_array is not None:
            # Crea nome file con timestamp
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            skin_id = np.random.randint(1000, 9999)
            filename = f"minecraft_skin_INTELLIGENT-{timestamp}-{skin_id}_{method_info}.png"
            
            # Salva la skin
            img = Image.fromarray(skin_array, 'RGBA')
            filepath = os.path.join('downloaded_skins', filename)
            img.save(filepath)
            
            print(f"SUCCESSO: Skin intelligente generata: {filename}")
            print(f"Percorso: {filepath}")
            print(f"Metodo: {method_info}")
            print("=" * 60)
        else:
            print("ERRORE: Impossibile generare skin intelligente")
            
    except Exception as e:
        print(f"ERRORE durante generazione automatica: {e}")

# Esegui generazione automatica all'avvio
generate_startup_skin()

# === FLASK ENDPOINTS ===

@app.route('/generate_and_download', methods=['GET'])
def generate_and_download_skin():
    """Genera skin casuale e la scarica automaticamente come file PNG"""
    try:
        # Genera skin casuale
        skin_array, theme = ai_generator.generate_skin('random')
        
        # Crea file temporaneo
        skin_id = f"AI-{datetime.now().strftime('%Y%m%d%H%M%S')}-{np.random.randint(1000, 9999)}"
        filename = f"minecraft_skin_{skin_id}_{theme}.png"
        
        # Converti array in immagine PIL
        img = Image.fromarray(skin_array, 'RGBA')
        
        # Salva in buffer
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Restituisci file per download
        return send_file(
            buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Endpoint per generare FUSION SKIN che combina tutto il dataset
@app.route('/generate_fusion', methods=['GET'])
def generate_fusion():
    """Genera una FUSION SKIN che combina tutti i dati del dataset"""
    try:
        fusion_method = request.args.get('method', 'intelligent')  # intelligent, average, mosaic, advanced
        num_samples = int(request.args.get('samples', 20))  # Numero di skin da campionare
        
        print(f"Richiesta FUSION SKIN - Metodo: {fusion_method}, Campioni: {num_samples}")
        
        # Genera la FUSION skin
        skin_array, fusion_info = ai_generator.generate_fusion_skin(
            fusion_method=fusion_method, 
            num_samples=num_samples
        )
        
        if skin_array is None:
            return jsonify({'error': 'Errore nella generazione fusion'}), 500
        
        # Crea file temporaneo
        skin_id = f"FUSION-{datetime.now().strftime('%Y%m%d%H%M%S')}-{np.random.randint(1000, 9999)}"
        filename = f"minecraft_skin_{skin_id}_{fusion_info}.png"
        
        # Converti array in immagine PIL
        img = Image.fromarray(skin_array, 'RGBA')
        
        # Salva in buffer
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Salva anche localmente per riferimento
        os.makedirs('downloaded_skins', exist_ok=True)
        local_path = os.path.join('downloaded_skins', filename)
        img.save(local_path)
        print(f"FUSION SKIN salvata: {filename}")
        
        # Restituisci file per download
        return send_file(
            buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Ottieni stato del sistema AI"""
    return jsonify({
        'success': True,
        'dataset_size': len(ai_generator.dataset),
        'is_trained': ai_generator.is_trained,
        'available_themes': list(ai_generator.themes.keys()),
        'device': str(ai_generator.device),
        'model_info': {
            'latent_dim': ai_generator.latent_dim,
            'training_epochs': len(ai_generator.training_losses['generator'])
        },
        'training_losses': {
            'generator_last': ai_generator.training_losses['generator'][-1] if ai_generator.training_losses['generator'] else None,
            'discriminator_last': ai_generator.training_losses['discriminator'][-1] if ai_generator.training_losses['discriminator'] else None
        },
        'fusion_methods': ['intelligent', 'average', 'mosaic', 'advanced']
    })

# Endpoint per informazioni sui metodi fusion disponibili  
@app.route('/fusion_methods', methods=['GET'])
def fusion_methods():
    """Ottieni informazioni sui metodi fusion disponibili"""
    return jsonify({
        'available_methods': {
            'intelligent': {
                'name': 'Fusione Intelligente',
                'description': 'Combina regioni specifiche (testa, corpo, braccia) da skin diverse in modo anatomicamente corretto',
                'best_for': 'Skin bilanciate e realistiche',
                'parameters': 'Preserva struttura anatomica'
            },
            'average': {
                'name': 'Media Pesata',
                'description': 'Media pesata di tutte le skin campionate con priorità decrescente',
                'best_for': 'Colori e texture omogenei',
                'parameters': 'Risultato uniforme e bilanciato'
            },
            'mosaic': {
                'name': 'Mosaico',
                'description': 'Crea blocchi 8x8 da skin diverse con algoritmo di smoothing avanzato',
                'best_for': 'Effetti artistici e pattern unici',
                'parameters': 'Texture frammentata creativa'
            },
            'advanced': {
                'name': 'Blend Avanzato',
                'description': 'Usa blend modes professionali (multiply, overlay, soft light, color dodge)',
                'best_for': 'Effetti fotografici e artistici',
                'parameters': 'Blend modes cinematografici'
            }
        },
        'usage': {
            'endpoint': '/generate_fusion',
            'parameters': {
                'method': 'Metodo di fusione (intelligent, average, mosaic, advanced)',
                'samples': 'Numero di skin da campionare dal dataset (5-50, default: 20)'
            },
            'examples': [
                '/generate_fusion?method=intelligent&samples=30 - Fusione anatomica con 30 skin',
                '/generate_fusion?method=mosaic&samples=15 - Mosaico artistico con 15 skin',  
                '/generate_fusion?method=advanced&samples=25 - Blend cinematografico con 25 skin',
                '/generate_fusion?method=average&samples=40 - Media pesata con 40 skin'
            ]
        },
        'dataset_info': {
            'total_skins': len(ai_generator.dataset),
            'recommended_samples': min(25, len(ai_generator.dataset)),
            'max_samples': min(50, len(ai_generator.dataset))
        }
    })

# === NUOVO METODO DI GENERAZIONE INTELLIGENTE ===
def add_intelligent_hybrid_generator():
    """Aggiunge il metodo di generazione ibrida intelligente alla classe MinecraftSkinAI"""
    
    def generate_intelligent_hybrid_skin(self):
        """Genera una skin combinando AI e FUSION in modo intelligente"""
        print("Avvio algoritmo di generazione ibrida intelligente...")
        
        # Seleziona strategia di generazione basata su casualità pesata
        strategy_weights = {
            'ai_pure': 0.3,                    # 30% AI pura
            'ai_enhanced_fusion': 0.25,        # 25% AI + miglioramenti FUSION  
            'multi_fusion_blend': 0.25,        # 25% Multi-FUSION blend
            'evolutionary_fusion': 0.2         # 20% FUSION evolutiva
        }
        
        # Scelta strategia
        strategies = list(strategy_weights.keys())
        weights = list(strategy_weights.values())
        chosen_strategy = np.random.choice(strategies, p=weights)
        
        print(f"Strategia selezionata: {chosen_strategy}")
        
        # Genera seed unico basato su timestamp per garantire variabilità
        unique_seed = int(datetime.now().timestamp() * 1000000) % 2147483647
        np.random.seed(unique_seed)
        torch.manual_seed(unique_seed)
        
        if chosen_strategy == 'ai_pure' and self.is_trained:
            return self._generate_ai_pure_variable()
        elif chosen_strategy == 'ai_enhanced_fusion':
            return self._generate_ai_enhanced_fusion()
        elif chosen_strategy == 'multi_fusion_blend':
            return self._generate_multi_fusion_blend()
        else:  # evolutionary_fusion
            return self._generate_evolutionary_fusion()
    
    def _generate_ai_pure_variable(self):
        """Genera skin AI pura con alta variabilità"""
        print("Generazione AI pura con alta variabilità...")
        
        # Usa parametri casuali per massima variabilità
        theme_variations = ['random', 'warrior', 'mage', 'nature', 'tech', 'shadow']
        selected_theme = np.random.choice(theme_variations)
        
        # Genera con seed variabile
        skin_array, theme = self.generate_skin(selected_theme)
        return skin_array, f"AI_VARIABLE_{theme}"
    
    def _generate_ai_enhanced_fusion(self):
        """Combina AI con elementi FUSION per risultati unici"""
        print("Generazione AI potenziata con elementi FUSION...")
        
        try:
            # Genera base AI se possibile
            if self.is_trained:
                base_skin, ai_theme = self.generate_skin('random')
            else:
                # Fallback a FUSION se AI non disponibile
                base_skin, _ = self.generate_fusion_skin('intelligent', 15)
                ai_theme = 'fusion_base'
            
            # Genera elementi FUSION da aggiungere
            fusion_skin, _ = self.generate_fusion_skin('mosaic', 
                                                      np.random.randint(8, 25))
            
            # Combina AI base con elementi FUSION usando blend intelligente
            final_skin = self._intelligent_blend(base_skin, fusion_skin)
            
            return final_skin, f"AI_ENHANCED_{ai_theme}_with_fusion"
            
        except Exception as e:
            print(f"Errore in AI enhanced fusion: {e}")
            # Fallback sicuro
            return self.generate_fusion_skin('intelligent', 20)
    
    def _generate_multi_fusion_blend(self):
        """Combina multiple strategie FUSION per risultati complessi"""
        print("Generazione multi-FUSION blend...")
        
        # Genera multiple skin FUSION con metodi diversi
        fusion_methods = ['intelligent', 'average', 'mosaic', 'advanced']
        fusion_results = []
        
        for method in fusion_methods:
            skin_data, _ = self.generate_fusion_skin(
                method, 
                np.random.randint(5, 15)
            )
            if skin_data is not None:
                fusion_results.append(skin_data)
        
        if len(fusion_results) < 2:
            # Fallback se non abbastanza risultati
            return self.generate_fusion_skin('intelligent', 25)
        
        # Combina i risultati con algoritmo avanzato
        final_skin = self._advanced_multi_blend(fusion_results)
        return final_skin, f"MULTI_FUSION_BLEND_{len(fusion_results)}methods"
    
    def _generate_evolutionary_fusion(self):
        """Simula evoluzione genetica per skin uniche"""
        print("Generazione FUSION evolutiva...")
        
        # Genera popolazione iniziale
        population_size = np.random.randint(6, 12)
        population = []
        
        for _ in range(population_size):
            method = np.random.choice(['intelligent', 'average', 'mosaic', 'advanced'])
            samples = np.random.randint(3, 20)
            skin_data, _ = self.generate_fusion_skin(method, samples)
            if skin_data is not None:
                population.append(skin_data)
        
        if len(population) < 3:
            return self.generate_fusion_skin('advanced', 30)
        
        # Simula "evoluzione" combinando le migliori caratteristiche
        evolved_skin = self._evolutionary_combination(population)
        return evolved_skin, f"EVOLUTIONARY_FUSION_gen{len(population)}"
    
    def _intelligent_blend(self, skin1, skin2):
        """Blend intelligente tra due skin"""
        # Blend diversificato per regioni
        result = skin1.copy().astype(np.float32)
        skin2_float = skin2.astype(np.float32)
        
        # Regioni con blend diversi
        regions = [
            ((0, 0, 32, 16), 0.4),      # Testa - blend leggero
            ((0, 16, 56, 32), 0.6),     # Corpo - blend medio
            ((0, 32, 64, 64), 0.3),     # Gambe - blend leggero
        ]
        
        for (x1, y1, x2, y2), alpha in regions:
            region_blend = alpha * skin2_float[y1:y2, x1:x2] + (1-alpha) * result[y1:y2, x1:x2]
            result[y1:y2, x1:x2] = region_blend
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _advanced_multi_blend(self, skins):
        """Blend avanzato di multiple skin"""
        if len(skins) == 0:
            return np.zeros((64, 64, 4), dtype=np.uint8)
        
        result = skins[0].astype(np.float32)
        
        for i, skin in enumerate(skins[1:]):
            # Peso decrescente per ogni skin aggiuntiva
            weight = 0.5 / (i + 1)
            result = weight * skin.astype(np.float32) + (1-weight) * result
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _evolutionary_combination(self, population):
        """Combina popolazione usando principi evolutivi"""
        # Seleziona "genitori" migliori (casuali per ora)
        np.random.shuffle(population)
        parents = population[:3]
        
        # Crossover: combina caratteristiche dei genitori
        result = parents[0].astype(np.float32)
        
        # Eredita caratteristiche da altri genitori
        for y in range(0, 64, 8):
            for x in range(0, 64, 8):
                parent_idx = np.random.randint(0, len(parents))
                parent_region = parents[parent_idx][y:y+8, x:x+8]
                
                # Mutation: aggiungi variabilità
                mutation_factor = 0.9 + np.random.random() * 0.2  # 0.9-1.1
                mutated_region = parent_region.astype(np.float32) * mutation_factor
                
                result[y:y+8, x:x+8] = mutated_region
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    # Aggiungi i metodi alla classe
    MinecraftSkinAI.generate_intelligent_hybrid_skin = generate_intelligent_hybrid_skin
    MinecraftSkinAI._generate_ai_pure_variable = _generate_ai_pure_variable
    MinecraftSkinAI._generate_ai_enhanced_fusion = _generate_ai_enhanced_fusion
    MinecraftSkinAI._generate_multi_fusion_blend = _generate_multi_fusion_blend
    MinecraftSkinAI._generate_evolutionary_fusion = _generate_evolutionary_fusion
    MinecraftSkinAI._intelligent_blend = _intelligent_blend
    MinecraftSkinAI._advanced_multi_blend = _advanced_multi_blend
    MinecraftSkinAI._evolutionary_combination = _evolutionary_combination

if __name__ == '__main__':
    app.run(debug=True, port=5000) 