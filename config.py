# Configurazione principale del sistema AI per skin Minecraft
# Modifica questi parametri per ottimizzare performance e risultati

# Percorsi filesystem
DATASET_PATH = "./skin_dataset"  # Directory con le skin di training
MODEL_PATH = "./models/minecraft_skin_gan.pth"  # File del modello salvato

# Iperparametri della rete neurale
LATENT_DIM = 128  # Dimensione vettore latente per il generatore (matching modello esistente)