# Configurazione principale del sistema AI per skin Minecraft
# Modifica questi parametri per ottimizzare performance e risultati

# Percorsi filesystem
DATASET_PATH = "./skin_dataset"  # Directory con le skin di training
MODEL_PATH = "./models/minecraft_skin_gan.pth"  # File del modello salvato
MODEL_SAVE_PATH = "./models"  # Directory per salvare i modelli

# Iperparametri della rete neurale moderna
LATENT_DIM = 128  # Dimensione vettore latente per il generatore
BATCH_SIZE = 32   # Batch size per training
LEARNING_RATE_G = 1e-4  # Learning rate conservativo generatore
LEARNING_RATE_D = 4e-4  # Learning rate discriminatore (TTUR)
EPOCHS = 30       # Numero epoche di default