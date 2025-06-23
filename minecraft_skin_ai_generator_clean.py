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
        print(f"🎮 Inizializzando Minecraft Skin AI Generator")
        print(f"📱 Device: {self.device}")
        
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
        
        print(f"📊 Dataset attuale: {len(self.dataset)} skin")
        print(f"🎨 Temi disponibili: {list(self.themes.keys())}")
        
        # Auto-addestramento se necessario
        if not self.is_trained and len(self.dataset) >= 50:
            print(f"🤖 Avvio training automatico su {len(self.dataset)} skin...")
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
            print("🤖 Avvio training immediato...")
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
        print("💾 Modelli salvati in ./models/minecraft_skin_gan.pth")
    
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
                
                print("✅ Modelli pre-addestrati caricati con successo!")
                
            except Exception as e:
                print(f"⚠️ Errore caricando modelli: {e}")
    
    def skin_to_base64(self, skin_array):
        """Converte array numpy in base64"""
        img = Image.fromarray(skin_array, 'RGBA')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode()

# Istanza globale del sistema AI
print("🎮 Inizializzando Minecraft Skin AI...")
skin_ai = MinecraftSkinAI()

# === FLASK ENDPOINTS ===

@app.route('/generate_and_download', methods=['GET'])
def generate_and_download_skin():
    """Genera skin casuale e la scarica automaticamente come file PNG"""
    try:
        # Genera skin casuale
        skin_array, theme = skin_ai.generate_skin('random')
        
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

@app.route('/status', methods=['GET'])
def get_status():
    """Ottieni stato del sistema AI"""
    return jsonify({
        'success': True,
        'dataset_size': len(skin_ai.dataset),
        'is_trained': skin_ai.is_trained,
        'available_themes': list(skin_ai.themes.keys()),
        'device': str(skin_ai.device),
        'model_info': {
            'latent_dim': skin_ai.latent_dim,
            'training_epochs': len(skin_ai.training_losses['generator'])
        },
        'training_losses': {
            'generator_last': skin_ai.training_losses['generator'][-1] if skin_ai.training_losses['generator'] else None,
            'discriminator_last': skin_ai.training_losses['discriminator'][-1] if skin_ai.training_losses['discriminator'] else None
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000) 