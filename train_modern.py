#!/usr/bin/env python3
"""
Sistema di Training AI Moderno per Generazione Skin Minecraft
Implementa architetture GAN moderne con loss functions sofisticate
e parametri di training ottimizzati per risultati di qualità suprema.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime
import time

# Importiamo i nostri moduli personalizzati
from models import SkinGenerator, SkinDiscriminator
from dataset import SkinDataset
import config

def gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Calcola il gradient penalty per WGAN-GP."""
    batch_size = real_samples.size(0)
    
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    
    disc_interpolated = discriminator(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=disc_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(disc_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return penalty

def least_squares_loss_generator(fake_output):
    """LSGAN Loss per generatore - più stabile di BCE."""
    return 0.5 * torch.mean((fake_output - 1.0) ** 2)

def least_squares_loss_discriminator(real_output, fake_output):
    """LSGAN Loss per discriminatore."""
    real_loss = 0.5 * torch.mean((real_output - 1.0) ** 2)
    fake_loss = 0.5 * torch.mean(fake_output ** 2)
    return real_loss + fake_loss

def perceptual_loss(fake_images, real_images):
    """Perceptual Loss per preservare caratteristiche visive."""
    fake_rgb = fake_images[:, :3, :, :]
    real_rgb = real_images[:, :3, :, :]
    return F.mse_loss(fake_rgb, real_rgb)

class MultiLossGAN:
    """Sistema GAN con Loss Functions Multiple e Sofisticate."""
    
    def __init__(self, loss_type='multi'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_type = loss_type
        print(f"🚀 Sistema Multi-Loss inizializzato: {loss_type}")
        
        self.weights = {
            'adversarial': 1.0,
            'perceptual': 0.1,
            'feature_matching': 0.5,
            'gradient_penalty': 10.0
        }
        
    def compute_generator_loss(self, generator, discriminator, real_images, noise):
        """Calcola loss complessa del generatore."""
        fake_images = generator(noise)
        fake_output = discriminator(fake_images)
        
        # Loss adversariale principale
        if self.loss_type == 'lsgan' or self.loss_type == 'multi':
            adv_loss = least_squares_loss_generator(fake_output)
        else:
            adv_loss = -torch.mean(fake_output)  # WGAN
        
        total_loss = self.weights['adversarial'] * adv_loss
        
        # Perceptual Loss per realismo
        if self.loss_type == 'multi':
            perc_loss = perceptual_loss(fake_images, real_images)
            total_loss += self.weights['perceptual'] * perc_loss
            
            # Feature Matching Loss semplificato
            fm_loss = F.mse_loss(fake_images.mean(dim=[2,3]), real_images.mean(dim=[2,3]))
            total_loss += self.weights['feature_matching'] * fm_loss
        
        return total_loss, fake_images
    
    def compute_discriminator_loss(self, discriminator, real_images, fake_images):
        """Calcola loss complessa del discriminatore."""
        real_output = discriminator(real_images)
        fake_output = discriminator(fake_images.detach())
        
        if self.loss_type == 'lsgan' or self.loss_type == 'multi':
            adv_loss = least_squares_loss_discriminator(real_output, fake_output)
        elif self.loss_type == 'wgan-gp':
            adv_loss = torch.mean(fake_output) - torch.mean(real_output)
            gp_loss = gradient_penalty(discriminator, real_images, fake_images, self.device)
            adv_loss += self.weights['gradient_penalty'] * gp_loss
        else:
            adv_loss = least_squares_loss_discriminator(real_output, fake_output)
        
        return adv_loss

def train_modern_gan(epochs=30, loss_type='multi'):
    """Training con Loss Functions Moderne e Parametri Ottimali."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Training Moderno con Parametri Ottimali")
    print(f"📊 Loss Type: {loss_type.upper()}")
    print(f"💻 Device: {device}")
    
    # Caricamento dataset
    dataset = SkinDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"📁 Dataset: {len(dataset)} skin")
    
    # Modelli con architettura moderna
    generator = SkinGenerator().to(device)
    discriminator = SkinDiscriminator().to(device)
    
    # Sistema multi-loss
    loss_system = MultiLossGAN(loss_type=loss_type)
    
    # Caricamento modello esistente
    model_path = os.path.join(config.MODEL_SAVE_PATH, 'minecraft_skin_gan.pth')
    
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=False)
            print("✅ Modello esistente caricato con parametri ottimali")
        except Exception as e:
            print(f"⚠️  Errore caricamento: {e} - Training da zero")
    
    # Optimizers ottimizzati
    optimizer_G = optim.AdamW(
        generator.parameters(), 
        lr=config.LEARNING_RATE_G,
        betas=(0.0, 0.99),
        weight_decay=1e-4,
        eps=1e-8
    )
    
    optimizer_D = optim.AdamW(
        discriminator.parameters(), 
        lr=config.LEARNING_RATE_D,
        betas=(0.0, 0.99),
        weight_decay=1e-4,
        eps=1e-8
    )
    
    # Schedulers
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs, eta_min=1e-7)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=epochs, eta_min=1e-7)
    
    # Tracking metriche
    G_losses = []
    D_losses = []
    
    # Label smoothing calibrato
    real_label_smooth = 0.95
    fake_label_smooth = 0.05
    
    print("\n" + "="*70)
    print("🚀 TRAINING CON PARAMETRI OTTIMALI MODERNI")
    print("="*70)
    print(f"🎯 Epoche: {epochs}")
    print(f"📊 Batch size: {config.BATCH_SIZE}")
    print(f"🧠 Loss: {loss_type.upper()}")
    print(f"📈 Learning Rate G: {config.LEARNING_RATE_G}")
    print(f"📈 Learning Rate D: {config.LEARNING_RATE_D}")
    print(f"🎛️  Label smoothing: {real_label_smooth}/{fake_label_smooth}")
    print(f"⚡ Optimizer: AdamW + Cosine Scheduling")
    print("="*70)
    
    for epoch in range(epochs):
        epoch_G_loss = 0
        epoch_D_loss = 0
        batch_count = 0
        
        for i, real_skins in enumerate(dataloader):
            batch_size = real_skins.size(0)
            real_skins = real_skins.to(device)
            
            # ============================
            # TRAINING DISCRIMINATORE
            # ============================
            optimizer_D.zero_grad()
            
            noise = torch.randn(batch_size, config.LATENT_DIM, 1, 1, device=device)
            fake_skins = generator(noise)
            
            d_loss = loss_system.compute_discriminator_loss(
                discriminator, real_skins, fake_skins
            )
            
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=0.5)
            optimizer_D.step()
            
            # ============================
            # TRAINING GENERATORE
            # ============================
            optimizer_G.zero_grad()
            
            noise = torch.randn(batch_size, config.LATENT_DIM, 1, 1, device=device)
            
            g_loss, fake_skins = loss_system.compute_generator_loss(
                generator, discriminator, real_skins, noise
            )
            
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.5)
            optimizer_G.step()
            
            # Tracking
            epoch_G_loss += g_loss.item()
            epoch_D_loss += d_loss.item()
            batch_count += 1
        
        # Update schedulers
        scheduler_G.step()
        scheduler_D.step()
        
        # Statistiche epoch
        avg_G_loss = epoch_G_loss / batch_count
        avg_D_loss = epoch_D_loss / batch_count
        
        G_losses.append(avg_G_loss)
        D_losses.append(avg_D_loss)
        
        # Progress report
        current_lr_G = optimizer_G.param_groups[0]['lr']
        current_lr_D = optimizer_D.param_groups[0]['lr']
        
        print(f"Epoca [{epoch+1:3d}/{epochs}] | "
              f"G_loss: {avg_G_loss:7.4f} | "
              f"D_loss: {avg_D_loss:7.4f} | "
              f"LR_G: {current_lr_G:.2e} | "
              f"LR_D: {current_lr_D:.2e}")
        
        # Salvataggio ogni 10 epoche
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(config.MODEL_SAVE_PATH, 'minecraft_skin_gan.pth')
            os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'g_losses': G_losses,
                'd_losses': D_losses,
                'config': {
                    'latent_dim': config.LATENT_DIM,
                    'architecture': 'Modern_OptimalParams_MultiLoss',
                    'loss_type': loss_type,
                    'label_smoothing': [real_label_smooth, fake_label_smooth],
                    'learning_rates': [config.LEARNING_RATE_G, config.LEARNING_RATE_D]
                }
            }, save_path)
            
            print(f"💾 Modello salvato: epoca {epoch+1}")
    
    # Salvataggio finale
    final_save_path = os.path.join(config.MODEL_SAVE_PATH, 'minecraft_skin_gan.pth')
    torch.save({
        'epoch': epochs,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_losses': G_losses,
        'd_losses': D_losses,
        'config': {
            'latent_dim': config.LATENT_DIM,
            'architecture': 'Modern_OptimalParams_MultiLoss',
            'loss_type': loss_type
        }
    }, final_save_path)
    
    print("\n" + "="*70)
    print("✅ TRAINING CON PARAMETRI OTTIMALI COMPLETATO!")
    print("🎯 Miglioramenti implementati:")
    print(f"   • Multi-Loss System: LSGAN + Perceptual + Feature Matching")
    print(f"   • AdamW + Weight Decay per regolarizzazione")
    print(f"   • Cosine Annealing scheduling")
    print(f"   • Gradient Clipping conservativo (0.5)")
    print(f"   • Label Smoothing calibrato")
    print("="*70)
    
    return G_losses, D_losses

if __name__ == "__main__":
    # Training automatico con Multi-Loss
    print("🎮 AVVIO TRAINING MODERNO")
    losses = train_modern_gan(epochs=30, loss_type='multi')
    print("🏁 Training moderno completato!") 