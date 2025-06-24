import torch
import torch.nn as nn
import config

class SkinGenerator(nn.Module):
    """
    Generatore per skin Minecraft basato su architettura DCGAN.
    
    Converte un vettore di rumore casuale (latent space) in una skin 64x64 RGBA.
    La rete utilizza trasposte convoluzioni per upsampling progressivo.
    """
    def __init__(self):
        super(SkinGenerator, self).__init__()
        self.latent_dim = config.LATENT_DIM
        
        # Architettura DCGAN standard con modifiche per output RGBA
        # Progressione: 1x1 -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.main = nn.Sequential(
            # Layer 1: Da vettore latente a feature map 4x4
            nn.ConvTranspose2d(self.latent_dim, 512, 4, 1, 0, bias=False), 
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=False),  # ReLU senza inplace per stabilità
            
            # Layer 2: 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=False),
            
            # Layer 3: 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=False),
            
            # Layer 4: 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=False),
            
            # Layer finale: 32x32 -> 64x64 con 4 canali RGBA
            nn.ConvTranspose2d(64, 4, 4, 2, 1, bias=False), 
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, z):
        """
        Forward pass del generatore.
        
        Args:
            z: Tensor di rumore casuale shape (batch_size, latent_dim)
            
        Returns:
            Tensor shape (batch_size, 4, 64, 64) rappresentante le skin generate
        """
        # Reshape del vettore latente per convoluzione trasposata
        return self.main(z.view(z.size(0), self.latent_dim, 1, 1))

class SkinDiscriminator(nn.Module):
    """
    Discriminatore per skin Minecraft - classifica real vs fake.
    
    Prende in input una skin 64x64 RGBA e restituisce una probabilità
    che l'immagine sia reale (da dataset) piuttosto che generata.
    """
    def __init__(self):
        super(SkinDiscriminator, self).__init__()
        
        # Architettura convoluzionale standard per discriminatori
        # Progressione: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4 -> 1x1
        self.main = nn.Sequential(
            # Layer 1: 64x64 -> 32x32
            # Primo layer senza BatchNorm (pratica standard)
            nn.Conv2d(4, 64, 4, 2, 1, bias=False), 
            nn.LeakyReLU(0.2, inplace=False),  # LeakyReLU per evitare sparse gradients
            
            # Layer 2: 32x32 -> 16x16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2, inplace=False),
            
            # Layer 3: 16x16 -> 8x8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2, inplace=False),
            
            # Layer 4: 8x8 -> 4x4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(0.2, inplace=False),
            
            # Layer finale: 4x4 -> 1x1 (classificazione binaria)
            nn.Conv2d(512, 1, 4, 1, 0, bias=False), 
            nn.Sigmoid()  # Output probabilità [0, 1]
        )

    def forward(self, img):
        """
        Forward pass del discriminatore.
        
        Args:
            img: Tensor shape (batch_size, 4, 64, 64) rappresentante le skin
            
        Returns:
            Tensor shape (batch_size,) con probabilità che l'input sia reale
        """
        # Flat output per compatibilità con BCE Loss
        return self.main(img).view(-1, 1).squeeze(1) 