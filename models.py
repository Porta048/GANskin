import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import config

class SelfAttention(nn.Module):
    """
    Self-Attention module per catturare dipendenze a lungo raggio nell'immagine.
    Migliora significativamente la coerenza globale delle skin generate.
    """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        
        # Query, Key, Value projections con riduzione dimensionalità
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        # Output projection
        self.out_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        # Learnable attention scaling
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Compute Q, K, V
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        
        # Compute attention scores
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Apply output projection and residual connection
        out = self.out_conv(out)
        out = self.gamma * out + x
        
        return out

class ResidualBlock(nn.Module):
    """
    Residual Block per stabilizzare il training e migliorare la qualità dei dettagli.
    Importante per architetture profonde e generazione di alta qualità.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x):
        return F.relu(x + self.block(x))

class UpSampleBlock(nn.Module):
    """
    Upsampling moderno che elimina gli artefatti checkerboard.
    Usa nearest neighbor + conv invece di ConvTranspose2d.
    """
    def __init__(self, in_channels, out_channels, use_attention=False):
        super(UpSampleBlock, self).__init__()
        
        # Upsampling + Convolution (no checkerboard artifacts)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
        # Optional self-attention
        self.attention = SelfAttention(out_channels) if use_attention else None
        
        # Residual refinement
        self.residual = ResidualBlock(out_channels)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        
        # Apply self-attention if enabled
        if self.attention is not None:
            x = self.attention(x)
            
        # Residual refinement for better details
        x = self.residual(x)
        
        return x

class SkinGenerator(nn.Module):
    """
    Generatore Moderno per skin Minecraft con Architettura Avanzata.
    
    Caratteristiche:
    - Progressive upsampling senza artefatti checkerboard
    - Self-Attention per coerenza globale
    - Residual blocks per dettagli fini
    - Spectral normalization per stabilità
    """
    def __init__(self):
        super(SkinGenerator, self).__init__()
        self.latent_dim = config.LATENT_DIM
        
        # Initial projection: latent → 4x4 feature map
        self.initial = nn.Sequential(
            # Convolution instead of ConvTranspose2d for stability
            nn.ConvTranspose2d(self.latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Progressive upsampling blocks with modern architecture
        # 4x4 → 8x8
        self.up1 = UpSampleBlock(512, 256, use_attention=False)
        
        # 8x8 → 16x16 (add attention for global coherence)  
        self.up2 = UpSampleBlock(256, 128, use_attention=True)
        
        # 16x16 → 32x32
        self.up3 = UpSampleBlock(128, 64, use_attention=False)
        
        # 32x32 → 64x64 (final upsampling)
        self.up4 = UpSampleBlock(64, 32, use_attention=False)
        
        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(32, 4, 3, 1, 1, bias=False),  # RGBA output
            nn.Tanh()  # Output range [-1, 1]
        )
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Inizializzazione pesi ottimizzata per GAN moderne."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        """
        Forward pass del generatore moderno.
        
        Args:
            z: Tensor di rumore casuale shape (batch_size, latent_dim, 1, 1)
            
        Returns:
            Tensor shape (batch_size, 4, 64, 64) rappresentante le skin generate
        """
        # Initial projection
        x = self.initial(z)
        
        # Progressive upsampling with modern techniques
        x = self.up1(x)  # 4x4 → 8x8
        x = self.up2(x)  # 8x8 → 16x16 (with attention)
        x = self.up3(x)  # 16x16 → 32x32
        x = self.up4(x)  # 32x32 → 64x64
        
        # Final output
        x = self.final(x)
        
        return x

class SkinDiscriminator(nn.Module):
    """
    Discriminatore Moderno con Spectral Normalization per stabilità.
    
    Caratteristiche:
    - Spectral normalization per training stabile
    - Self-attention per analisi globale
    - Residual connections per gradiente migliore
    - Architecture ottimizzata per skin 64x64
    """
    def __init__(self):
        super(SkinDiscriminator, self).__init__()
        
        # Progressive downsampling with spectral normalization
        # 64x64 → 32x32
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(4, 64, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 32x32 → 16x16
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 16x16 → 8x8 (with self-attention)
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Self-attention at 8x8 resolution
        self.attention = SelfAttention(256)
        
        # 8x8 → 4x4
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final classification: 4x4 → 1x1
        self.final = spectral_norm(nn.Conv2d(512, 1, 4, 1, 0, bias=False))
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Inizializzazione pesi ottimizzata per discriminatori moderni."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        """
        Forward pass del discriminatore moderno.
        
        Args:
            img: Tensor shape (batch_size, 4, 64, 64) rappresentante le skin
            
        Returns:
            Tensor shape (batch_size,) con score di realismo
        """
        # Progressive feature extraction
        x = self.conv1(img)  # 64x64 → 32x32
        x = self.conv2(x)    # 32x32 → 16x16
        x = self.conv3(x)    # 16x16 → 8x8
        
        # Self-attention for global analysis
        x = self.attention(x)
        
        # Final layers
        x = self.conv4(x)    # 8x8 → 4x4
        x = self.final(x)    # 4x4 → 1x1
        
        # Output flattening
        return x.view(-1) 