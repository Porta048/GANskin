#!/usr/bin/env python3
"""
Sistema GAN 

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# ===== CONFIGURAZIONE =====
DATASET_PATH = "./skin_dataset"
BATCH_SIZE = 64
LEARNING_RATE_G = 0.0001
LEARNING_RATE_D = 0.0004
BETA1 = 0.5
BETA2 = 0.999
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1 # Valore per lo smoothing (0.9 per reali, 0.1 per fake)
LATENT_DIM = 100
EPOCHS = 300
SAVE_INTERVAL = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== DATASET =====
class MinecraftSkinDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
        ])
        
        # Carica tutti i file PNG
        self.images = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.png'):
                    self.images.append(os.path.join(root, file))
        
        print(f"Trovate {len(self.images)} skin nel dataset")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            img_path = self.images[idx]
            img = Image.open(img_path).convert('RGBA')
            
            # Controllo dimensioni
            if img.size != (64, 64):
                # Se l'immagine non è 64x64, prova con un'altra
                return self.__getitem__(np.random.randint(0, len(self)))

            return self.transform(img)
        except Exception as e:
            # Se fallisce, logga l'errore e ritorna un'immagine random
            return self.__getitem__(np.random.randint(0, len(self)))

# ===== GENERATORE SEMPLICE =====
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: latent vector 100
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512 x 4 x 4
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 x 8 x 8
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 16 x 16
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64 x 32 x 32
            
            nn.ConvTranspose2d(64, 4, 4, 2, 1, bias=False),
            nn.Tanh()
            # 4 x 64 x 64 (RGBA)
        )
    
    def forward(self, input):
        return self.main(input)

# ===== DISCRIMINATORE SEMPLICE =====
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: 4 x 64 x 64
            nn.Conv2d(4, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 32 x 32
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 16 x 16
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 8 x 8
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 4 x 4
            
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # 1 x 1 x 1
        )
    
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

# ===== INIZIALIZZAZIONE PESI =====
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ===== TRAINING =====
def train():
    # Crea cartelle necessarie
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Dataset e DataLoader
    dataset = MinecraftSkinDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # Crea modelli
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    
    # Inizializza pesi
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Loss e ottimizzatori
    criterion = nn.BCELoss()
    optimizerD = optim.AdamW(netD.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
    optimizerG = optim.AdamW(netG.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
    
    # Labels con smoothing
    real_label = 1.0 - LABEL_SMOOTHING
    fake_label = LABEL_SMOOTHING
    
    # Fixed noise per visualizzazione progressi
    fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=device)
    
    print(f"\nInizio training su {device}")
    print(f"Dataset: {len(dataset)} skin")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epoche: {EPOCHS}\n")
    
    # Training loop
    for epoch in range(EPOCHS):
        for i, data in enumerate(tqdm(dataloader, desc=f"Epoca {epoch+1}/{EPOCHS}")):
            
            ############################
            # (1) Update D network
            ###########################
            netD.zero_grad()
            real_data = data.to(device)
            batch_size = real_data.size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            
            output = netD(real_data)
            errD_real = criterion(output, label)
            errD_real.backward()
            
            # Train with fake
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimizerD.step()
            
            ############################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()
        
        # Salva immagini di esempio
        if (epoch + 1) % SAVE_INTERVAL == 0:
            with torch.no_grad():
                fake = netG(fixed_noise)
                save_image(fake, f"outputs/epoch_{epoch+1}.png", normalize=True)
            print(f"\nSalvate immagini epoca {epoch+1}")
            
            # Salva modello
            torch.save({
                'generator': netG.state_dict(),
                'discriminator': netD.state_dict(),
            }, f"models/checkpoint_epoch_{epoch+1}.pth")
    
    # Salva modello finale
    torch.save({
        'generator': netG.state_dict(),
        'discriminator': netD.state_dict(),
    }, "models/final_model.pth")
    
    print("\nTraining completato!")

# ===== GENERAZIONE =====
def generate_skins(num_skins=10):
    # Carica modello
    netG = Generator().to(device)
    checkpoint = torch.load("models/final_model.pth", map_location=device)
    netG.load_state_dict(checkpoint['generator'])
    netG.eval()
    
    os.makedirs("generated_skins", exist_ok=True)
    
    print(f"\nGenerazione di {num_skins} skin...")
    
    with torch.no_grad():
        for i in range(num_skins):
            noise = torch.randn(1, LATENT_DIM, 1, 1, device=device)
            fake = netG(noise)
            
            # Denormalizza e converti in immagine
            fake = fake * 0.5 + 0.5  # da [-1,1] a [0,1]
            fake = fake.squeeze().cpu()
            
            # Converti in PIL e salva
            img_array = (fake.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_array, 'RGBA')
            img.save(f"generated_skins/skin_{i+1}.png")
    
    print(f"Generate {num_skins} skin in generated_skins/")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        generate_skins(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    else:
        train()
