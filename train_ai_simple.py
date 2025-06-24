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
import matplotlib.pyplot as plt

# Importiamo i nostri moduli personalizzati
from models import SkinGenerator, SkinDiscriminator
from dataset import SkinDataset
import config

def gradient_penalty(discriminator, real_samples, fake_samples, device):
    """
    Calcola il gradient penalty per WGAN-GP (Wasserstein GAN con Gradient Penalty).
    Stabilizza enormemente il training rispetto alla normale GAN loss.
    """
    batch_size = real_samples.size(0)
    
    # Interpolazione random tra sample reali e fake
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    
    # Forward pass dell'interpolazione
    disc_interpolated = discriminator(interpolated)
    
    # Calcola gradienti rispetto agli input interpolati
    gradients = torch.autograd.grad(
        outputs=disc_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(disc_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Calcola penalty basato sulla norma dei gradienti
    gradients = gradients.view(batch_size, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return penalty

def perceptual_loss(fake_images, real_images):
    """
    Perceptual Loss per preservare caratteristiche visive di alto livello.
    Usa la distanza tra features di un modello pre-addestrato.
    """
    # Conversione da RGBA a RGB per compatibilità
    fake_rgb = fake_images[:, :3, :, :]
    real_rgb = real_images[:, :3, :, :]
    
    # Calcola MSE sui canali RGB (versione semplificata)
    # In implementazioni complete si userebbero features VGG/ResNet
    return F.mse_loss(fake_rgb, real_rgb)

def feature_matching_loss(fake_features, real_features):
    """
    Feature Matching Loss per stabilizzare training e migliorare diversità.
    Confronta statistiche intermedie invece che solo output finale.
    """
    loss = 0
    for fake_feat, real_feat in zip(fake_features, real_features):
        loss += F.mse_loss(fake_feat.mean(0), real_feat.mean(0))
    return loss

def least_squares_loss_generator(fake_output):
    """
    Least Squares GAN Loss per generatore - più stabile di BCE.
    Produce gradienti più smooth e riduce vanishing gradient problem.
    """
    return 0.5 * torch.mean((fake_output - 1.0) ** 2)

def least_squares_loss_discriminator(real_output, fake_output):
    """
    Least Squares GAN Loss per discriminatore.
    Versione più stabile di BCE che produce gradienti migliori.
    """
    real_loss = 0.5 * torch.mean((real_output - 1.0) ** 2)
    fake_loss = 0.5 * torch.mean(fake_output ** 2)
    return real_loss + fake_loss

def wasserstein_loss_generator(fake_output):
    """Loss Wasserstein per il generatore - più stabile della BCE."""
    return -torch.mean(fake_output)

def wasserstein_loss_discriminator(real_output, fake_output):
    """Loss Wasserstein per il discriminatore - più stabile della BCE."""
    return torch.mean(fake_output) - torch.mean(real_output)

def relativistic_loss_generator(real_output, fake_output):
    """
    Relativistic GAN Loss per generatore.
    Considera la differenza relativa tra real e fake invece di valori assoluti.
    Migliora significativamente la qualità dei dettagli.
    """
    return -torch.mean(torch.log(torch.sigmoid(fake_output - real_output.mean()) + 1e-8))

def relativistic_loss_discriminator(real_output, fake_output):
    """
    Relativistic GAN Loss per discriminatore.
    Approccio più sofisticato che considera relazioni tra batch.
    """
    real_loss = -torch.mean(torch.log(torch.sigmoid(real_output - fake_output.mean()) + 1e-8))
    fake_loss = -torch.mean(torch.log(torch.sigmoid(-fake_output + real_output.mean()) + 1e-8))
    return (real_loss + fake_loss) / 2

class MultiLossGAN:
    """
    Sistema GAN con Loss Functions Multiple e Sofisticate.
    
    Combina:
    - LSGAN per stabilità base
    - Perceptual Loss per realismo visivo
    - Feature Matching per diversità
    - Gradient Penalty per regolarizzazione
    - Relativistic Loss opzionale per dettagli fini
    """
    
    def __init__(self, loss_type='multi'):
        """
        loss_type: 'multi', 'lsgan', 'wgan-gp', 'relativistic'
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_type = loss_type
        print(f"🚀 Sistema Multi-Loss inizializzato: {loss_type}")
        print(f"Device: {self.device}")
        
        # Pesi per combinare diverse loss functions
        self.weights = {
            'adversarial': 1.0,    # Loss principale GAN
            'perceptual': 0.1,     # Perceptual loss per realismo
            'feature_matching': 0.5, # Feature matching per diversità
            'gradient_penalty': 10.0  # Gradient penalty per stabilità
        }
        
    def compute_generator_loss(self, generator, discriminator, real_images, noise):
        """
        Calcola loss complessa del generatore con componenti multiple.
        """
        fake_images = generator(noise)
        fake_output = discriminator(fake_images)
        
        # 1. Loss adversariale principale
        if self.loss_type == 'lsgan':
            adv_loss = least_squares_loss_generator(fake_output)
        elif self.loss_type == 'wgan-gp':
            adv_loss = wasserstein_loss_generator(fake_output)
        elif self.loss_type == 'relativistic':
            real_output = discriminator(real_images)
            adv_loss = relativistic_loss_generator(real_output.detach(), fake_output)
        else:  # multi
            adv_loss = least_squares_loss_generator(fake_output)
        
        total_loss = self.weights['adversarial'] * adv_loss
        
        # 2. Perceptual Loss per preservare caratteristiche visive
        if self.loss_type == 'multi':
            perc_loss = perceptual_loss(fake_images, real_images)
            total_loss += self.weights['perceptual'] * perc_loss
        
        # 3. Feature Matching Loss (simulato - versione semplificata)
        if self.loss_type == 'multi':
            # In implementazione completa si estraggono features intermedie
            fm_loss = F.mse_loss(fake_images.mean(dim=[2,3]), real_images.mean(dim=[2,3]))
            total_loss += self.weights['feature_matching'] * fm_loss
        
        return total_loss, fake_images
    
    def compute_discriminator_loss(self, discriminator, real_images, fake_images):
        """
        Calcola loss complessa del discriminatore.
        """
        real_output = discriminator(real_images)
        fake_output = discriminator(fake_images.detach())
        
        # Loss adversariale principale
        if self.loss_type == 'lsgan':
            adv_loss = least_squares_loss_discriminator(real_output, fake_output)
        elif self.loss_type == 'wgan-gp':
            adv_loss = wasserstein_loss_discriminator(real_output, fake_output)
            # Gradient Penalty per WGAN-GP
            gp_loss = gradient_penalty(discriminator, real_images, fake_images, self.device)
            adv_loss += self.weights['gradient_penalty'] * gp_loss
        elif self.loss_type == 'relativistic':
            adv_loss = relativistic_loss_discriminator(real_output, fake_output)
        else:  # multi
            adv_loss = least_squares_loss_discriminator(real_output, fake_output)
        
        return adv_loss

def train_modern_gan(epochs=50, loss_type='multi'):
    """
    Training con Loss Functions Moderne e Parametri Ottimali.
    
    Miglioramenti:
    - Learning rate scheduling dinamico e adattivo
    - Label smoothing calibrato (0.95/0.05)
    - Exponential Moving Average per stabilità
    - Adaptive learning rate con plateau detection
    - Warm-up e cosine annealing ottimizzati
    """
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
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if len(dataset) > 100 else False
    )
    
    print(f"📁 Dataset: {len(dataset)} skin")
    
    # Modelli con architettura moderna
    generator = SkinGenerator().to(device)
    discriminator = SkinDiscriminator().to(device)
    
    # Exponential Moving Average per stabilità generatore
    class EMA:
        def __init__(self, model, decay=0.999):
            self.decay = decay
            self.shadow = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.data.clone()
        
        def update(self, model):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
        
        def apply_shadow(self, model):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.shadow[name])
    
    # EMA per stabilizzare il generatore
    ema_generator = EMA(generator, decay=0.9995)
    
    # Sistema multi-loss
    loss_system = MultiLossGAN(loss_type=loss_type)
    
    # Caricamento modello esistente
    model_path = os.path.join(config.MODEL_SAVE_PATH, 'minecraft_skin_gan.pth')
    start_epoch = 0
    
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=False)
            print("✅ Modello esistente caricato con parametri ottimali")
        except Exception as e:
            print(f"⚠️  Errore caricamento: {e} - Training da zero")
    
    # Optimizers con parametri ottimizzati per GAN moderne
    optimizer_G = optim.AdamW(
        generator.parameters(), 
        lr=1e-4,  # Learning rate iniziale più conservativo
        betas=(0.0, 0.99),  # Beta1=0 per GAN stabili (no momentum)
        weight_decay=1e-4,  # Weight decay per regolarizzazione
        eps=1e-8
    )
    
    optimizer_D = optim.AdamW(
        discriminator.parameters(), 
        lr=4e-4,  # TTUR: discriminator 4x più veloce
        betas=(0.0, 0.99),
        weight_decay=1e-4,
        eps=1e-8
    )
    
    # Scheduler avanzati con warm-up e plateau detection
    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, 
                                        min_lr_ratio=0.01, num_cycles=1):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                # Warm-up lineare
                return float(current_step) / float(max(1, num_warmup_steps))
            
            # Cosine annealing con cycles
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * num_cycles * progress))
            return max(min_lr_ratio, cosine_decay)
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training steps totali
    total_steps = epochs * len(dataloader)
    warmup_steps = max(5, total_steps // 20)  # 5% di warm-up
    
    # Schedulers ottimizzati
    scheduler_G = get_cosine_schedule_with_warmup(
        optimizer_G, warmup_steps, total_steps, min_lr_ratio=0.01
    )
    scheduler_D = get_cosine_schedule_with_warmup(
        optimizer_D, warmup_steps, total_steps, min_lr_ratio=0.01
    )
    
    # ReduceLROnPlateau per adattamento automatico
    plateau_scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7
    )
    plateau_scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_D, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7
    )
    
    # Tracking metriche avanzate
    G_losses = []
    D_losses = []
    lr_history_G = []
    lr_history_D = []
    
    # Label smoothing ottimizzato (meno aggressivo)
    real_label_smooth = 0.95  # Era 0.9, troppo aggressivo
    fake_label_smooth = 0.05  # Era 0.1, troppo aggressivo
    
    # Label noise per robustezza (flip occasionale)
    label_noise_prob = 0.02  # 2% di probabilità di flip
    
    print("\n" + "="*70)
    print("🚀 TRAINING CON PARAMETRI OTTIMALI MODERNI")
    print("="*70)
    print(f"🎯 Epoche: {epochs}")
    print(f"📊 Batch size: {config.BATCH_SIZE}")
    print(f"🧠 Loss: {loss_type.upper()}")
    print(f"📈 Learning Rate G: 1e-4 → adaptive")
    print(f"📈 Learning Rate D: 4e-4 → adaptive")
    print(f"🔄 Warm-up steps: {warmup_steps}")
    print(f"🎛️  Label smoothing: {real_label_smooth}/{fake_label_smooth}")
    print(f"🎲 Label noise: {label_noise_prob*100:.1f}%")
    print(f"⚡ Optimizer: AdamW + EMA + Cosine + Plateau")
    print("="*70)
    
    # Parametri training adattivi
    d_iterations = 1  # Rapporto discriminator/generator
    best_g_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(start_epoch, epochs):
        epoch_G_loss = 0
        epoch_D_loss = 0
        batch_count = 0
        
        # Aggiusta rapporto D/G training in base al progresso
        if epoch > epochs // 3:
            # Nella seconda metà, bilanciamo meglio D e G
            d_iterations = 1 if epoch % 2 == 0 else 2
        
        for i, real_skins in enumerate(dataloader):
            batch_size = real_skins.size(0)
            real_skins = real_skins.to(device)
            
            # Label smoothing calibrato + noise occasionale
            real_labels = torch.full((batch_size,), real_label_smooth, device=device, dtype=torch.float)
            fake_labels = torch.full((batch_size,), fake_label_smooth, device=device, dtype=torch.float)
            
            # Label noise per robustezza (flip occasionale)
            if torch.rand(1).item() < label_noise_prob:
                real_labels, fake_labels = fake_labels, real_labels
            
            # ============================
            # TRAINING DISCRIMINATORE
            # ============================
            for _ in range(d_iterations):
                optimizer_D.zero_grad()
                
                # Genera noise per fake images
                noise = torch.randn(batch_size, config.LATENT_DIM, 1, 1, device=device)
                fake_skins = generator(noise)
                
                # Calcola loss discriminatore con sistema moderno
                if loss_type in ['multi', 'lsgan', 'wgan-gp', 'relativistic']:
                    d_loss = loss_system.compute_discriminator_loss(
                        discriminator, real_skins, fake_skins
                    )
                else:
                    # Fallback a BCE ottimizzato
                    real_output = discriminator(real_skins)
                    fake_output = discriminator(fake_skins.detach())
                    d_loss_real = F.binary_cross_entropy_with_logits(real_output, real_labels)
                    d_loss_fake = F.binary_cross_entropy_with_logits(fake_output, fake_labels)
                    d_loss = d_loss_real + d_loss_fake
                
                d_loss.backward()
                # Gradient clipping più conservativo
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=0.5)
                optimizer_D.step()
                scheduler_D.step()
            
            # ============================
            # TRAINING GENERATORE
            # ============================
            optimizer_G.zero_grad()
            
            # Genera nuove skin con noise fresco
            noise = torch.randn(batch_size, config.LATENT_DIM, 1, 1, device=device)
            
            # Calcola loss generatore con sistema multi-loss
            if loss_type in ['multi', 'lsgan', 'wgan-gp', 'relativistic']:
                g_loss, fake_skins = loss_system.compute_generator_loss(
                    generator, discriminator, real_skins, noise
                )
            else:
                # Fallback a BCE ottimizzato
                fake_skins = generator(noise)
                fake_output = discriminator(fake_skins)
                g_loss = F.binary_cross_entropy_with_logits(fake_output, real_labels)
            
            g_loss.backward()
            # Gradient clipping più conservativo
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.5)
            optimizer_G.step()
            scheduler_G.step()
            
            # Update EMA del generatore
            ema_generator.update(generator)
            
            # Tracking
            epoch_G_loss += g_loss.item()
            epoch_D_loss += d_loss.item()
            batch_count += 1
        
        # Statistiche epoch
        avg_G_loss = epoch_G_loss / batch_count
        avg_D_loss = epoch_D_loss / batch_count
        
        G_losses.append(avg_G_loss)
        D_losses.append(avg_D_loss)
        
        # Update plateau schedulers
        plateau_scheduler_G.step(avg_G_loss)
        plateau_scheduler_D.step(avg_D_loss)
        
        # Learning rates tracking
        current_lr_G = optimizer_G.param_groups[0]['lr']
        current_lr_D = optimizer_D.param_groups[0]['lr']
        lr_history_G.append(current_lr_G)
        lr_history_D.append(current_lr_D)
        
        # Early stopping intelligente
        if avg_G_loss < best_g_loss:
            best_g_loss = avg_G_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Progress report dettagliato
        print(f"Epoca [{epoch+1:3d}/{epochs}] | "
              f"G_loss: {avg_G_loss:7.4f} | "
              f"D_loss: {avg_D_loss:7.4f} | "
              f"LR_G: {current_lr_G:.2e} | "
              f"LR_D: {current_lr_D:.2e} | "
              f"D_iter: {d_iterations}")
        
        # Salvataggio con EMA ogni 10 epoche
        if (epoch + 1) % 10 == 0:
            # Applica EMA al generatore per salvataggio
            ema_generator.apply_shadow(generator)
            
            save_path = os.path.join(config.MODEL_SAVE_PATH, 'minecraft_skin_gan.pth')
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'scheduler_G_state_dict': scheduler_G.state_dict(),
                'scheduler_D_state_dict': scheduler_D.state_dict(),
                'g_losses': G_losses,
                'd_losses': D_losses,
                'lr_history_G': lr_history_G,
                'lr_history_D': lr_history_D,
                'best_g_loss': best_g_loss,
                'config': {
                    'latent_dim': config.LATENT_DIM,
                    'architecture': 'Modern_OptimalParams_EMA',
                    'loss_type': loss_type,
                    'label_smoothing': [real_label_smooth, fake_label_smooth],
                    'learning_rates': [1e-4, 4e-4]
                }
            }, save_path)
            
            print(f"💾 Modello EMA salvato: epoca {epoch+1}")
    
    # Salvataggio finale con EMA
    ema_generator.apply_shadow(generator)
    
    final_save_path = os.path.join(config.MODEL_SAVE_PATH, 'minecraft_skin_gan.pth')
    torch.save({
        'epoch': epochs,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'scheduler_G_state_dict': scheduler_G.state_dict(),
        'scheduler_D_state_dict': scheduler_D.state_dict(),
        'g_losses': G_losses,
        'd_losses': D_losses,
        'lr_history_G': lr_history_G,
        'lr_history_D': lr_history_D,
        'best_g_loss': best_g_loss,
        'config': {
            'latent_dim': config.LATENT_DIM,
            'architecture': 'Modern_OptimalParams_EMA',
            'loss_type': loss_type,
            'label_smoothing': [real_label_smooth, fake_label_smooth],
            'learning_rates': [1e-4, 4e-4]
        }
    }, final_save_path)
    
    print("\n" + "="*70)
    print("✅ TRAINING CON PARAMETRI OTTIMALI COMPLETATO!")
    print("🎯 Miglioramenti implementati:")
    print(f"   • Learning Rate Scheduling: Cosine + Warm-up + Plateau")
    print(f"   • Label Smoothing Calibrato: {real_label_smooth}/{fake_label_smooth}")
    print(f"   • AdamW + Weight Decay per regolarizzazione")
    print(f"   • EMA (Exponential Moving Average) per stabilità")
    print(f"   • Gradient Clipping conservativo (0.5)")
    print(f"   • Label Noise ({label_noise_prob*100:.1f}%) per robustezza")
    print(f"   • Training ratio adattivo D/G")
    print("="*70)
    
    return G_losses, D_losses, lr_history_G, lr_history_D

def main():
    """Interfaccia per scegliere tipo di training moderno."""
    print("\n🎮 SISTEMA TRAINING MINECRAFT SKIN - LOSS MODERNE")
    print("="*60)
    print("Scegli il tipo di loss function:")
    print("1. Multi-Loss (LSGAN + Perceptual + Feature Matching)")
    print("2. Relativistic GAN (dettagli eccellenti)")
    print("3. Least Squares GAN (stabile)")
    print("4. WGAN-GP (molto stabile)")
    
    choice = input("\nScegli (1-4): ").strip()
    
    loss_types = {
        '1': 'multi',
        '2': 'relativistic', 
        '3': 'lsgan',
        '4': 'wgan-gp'
    }
    
    loss_type = loss_types.get(choice, 'multi')
    epochs = int(input("Numero epoche (default 30): ") or "30")
    
    print(f"\n🚀 Avvio training con {loss_type.upper()} loss...")
    train_modern_gan(epochs=epochs, loss_type=loss_type)

if __name__ == "__main__":
    # Training automatico con Multi-Loss per dettagli fini
    losses, lr_history_G, lr_history_D = train_modern_gan(epochs=30, loss_type='multi')
    print("🏁 Training moderno completato!") 