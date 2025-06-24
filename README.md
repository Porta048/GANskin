# Sistema AI per Generazione Skin Minecraft

Un sistema completo di intelligenza artificiale per generare skin personalizzate per Minecraft utilizzando reti neurali GAN (Generative Adversarial Networks).

## Caratteristiche Principali

- **Generazione AI**: Crea skin uniche utilizzando un modello GAN addestrato
- **Valutazione Qualità**: Sistema automatico di scoring delle skin generate
- **Post-Processing**: Miglioramento automatico di contrasto, nitidezza e colori
- **Server Web**: Interfaccia web per generazione tramite browser
- **Dataset Intelligente**: Gestione automatica con pulizia e augmentation
- **Training Ottimizzato**: Sistema di addestramento con early stopping e gradient clipping

## Struttura del Progetto

```
skin-minecraft-AI/
├── config.py              # Configurazione principale del sistema
├── models.py               # Architetture neurali (Generator + Discriminator)
├── dataset.py              # Gestione dataset con pulizia automatica
├── train_ai_simple.py      # Sistema di training principale
├── generate_beautiful_skins.py  # Generatore skin di alta qualità
├── improve_model.py        # Sistema miglioramento modello avanzato
├── app.py                  # Server web Flask
├── requirements.txt        # Dipendenze Python
├── models/                 # Modelli addestrati
│   └── minecraft_skin_gan.pth
└── skin_dataset/           # Dataset di training (121+ skin)
```

## Installazione

1. **Clona il repository**:
```bash
git clone <repository-url>
cd skin-minecraft-AI
```

2. **Installa le dipendenze**:
```bash
pip install -r requirements.txt
```

3. **Verifica la struttura**:
- Assicurati che `skin_dataset/` contenga le skin di training
- Il modello addestrato sarà in `models/minecraft_skin_gan.pth`

## Utilizzo

### Training del Modello

Per addestrare il sistema AI:

```bash
python train_ai_simple.py
```

**Opzioni disponibili**:
1. Training completo (100 epoche, tutto il dataset)
2. Training veloce (20 epoche, dataset limitato)
3. Training test (5 epoche, 500 skin)
4. Genera skin singola

### Generazione Skin Belle

Per generare skin di alta qualità:

```bash
python generate_beautiful_skins.py
```

**Funzionalità**:
- Genera da 1 a 10 skin con valutazione qualitativa
- Sistema di scoring automatico (0-10)
- Post-processing per migliorare l'aspetto
- Classificazione automatica (ECCELLENTE, MOLTO BELLA, BELLA, NORMALE)

### Server Web

Per utilizzare l'interfaccia web:

```bash
python app.py
```

Accedi a `http://localhost:5000` per l'interfaccia grafica.

### Miglioramento Modello

Per ottimizzare le performance:

```bash
python improve_model.py
```

**Opzioni**:
1. Miglioramento completo (espansione dataset + retraining)
2. Solo espansione dataset (121 → 1000 skin)
3. Solo retraining ottimizzato
4. Test modello attuale

## Architettura Tecnica

### Modelli Neurali

**Generator (SkinGenerator)**:
- Architettura DCGAN standard
- Input: Vettore latente 100D
- Output: Skin 64x64 RGBA
- Progressione: 1x1 → 4x4 → 8x8 → 16x16 → 32x32 → 64x64

**Discriminator (SkinDiscriminator)**:
- Classificatore binario real/fake
- Input: Skin 64x64 RGBA
- Output: Probabilità [0, 1]
- Progressione: 64x64 → 32x32 → 16x16 → 8x8 → 4x4 → 1x1

### Sistema di Qualità

Il sistema valuta le skin su diversi parametri:
- **Varietà cromatica**: Diversità dei colori utilizzati
- **Contrasto e dettagli**: Definizione visiva degli elementi
- **Bilanciamento**: Proporzione tra aree piene e trasparenti
- **Coerenza**: Struttura logica della skin

### Tecniche di Ottimizzazione

- **Label Smoothing**: Riduce overfitting (0.85/0.15 invece di 1.0/0.0)
- **Gradient Clipping**: Previene esplosioni del gradiente
- **Early Stopping**: Ferma training quando non migliora
- **Data Augmentation**: Espande dataset con variazioni intelligenti
- **Learning Rate Adattivo**: 0.0002 → 0.0001 per stabilità

## Performance

**Dataset Base**: 121 skin originali pulite
**Training Standard**: 100 epoche, ~3 ore
**Qualità Media**: 4-6/10 (BELLA - MOLTO BELLA)
**Miglioramento Possibile**: Fino a 7-8/10 con dataset espanso

## Configurazione

Modifica `config.py` per personalizzare:

```python
# Percorsi
DATASET_PATH = "./skin_dataset"
MODEL_PATH = "./models/minecraft_skin_gan.pth"

# Parametri modello
LATENT_DIM = 100  # Dimensione vettore latente
```

## Risoluzione Problemi

**Errore "Modello non trovato"**:
- Esegui prima `python train_ai_simple.py` per addestrare

**Skin di bassa qualità**:
- Utilizza `python improve_model.py` per migliorare il modello
- Considera espansione dataset

**Errori di memoria**:
- Riduci batch_size in `train_ai_simple.py`
- Utilizza CPU invece di GPU se necessario

## Miglioramenti Futuri

- Dataset più ampio (2000+ skin) per maggiore varietà
- Modelli condizionali per stili specifici
- Interfaccia web più avanzata con editor
- Esportazione diretta per Minecraft
- Sistema di rating collaborativo

## Contributi

Il progetto è aperto a miglioramenti! Aree di interesse:
- Ottimizzazione architetture neurali
- Nuove tecniche di augmentation
- Miglioramento interfaccia utente
- Espansione dataset di qualità

## Note Tecniche

Sviluppato utilizzando:
- **PyTorch** per le reti neurali
- **Pillow** per elaborazione immagini
- **Flask** per il server web
- **OpenCV** per analisi qualitativa
- **NumPy** per computazione numerica

Il sistema è stato progettato per essere modulare, estendibile e facile da utilizzare sia per sviluppatori che per utenti finali.