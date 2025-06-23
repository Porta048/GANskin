# 🎮 Minecraft Skin AI Generator

Un sistema avanzato bassu su **reti neurali GAN (Generative Adversarial Networks)** per generare skin di Minecraft uniche e belle.

## 🚀 Caratteristiche Principali

- **🧠 AI Avanzata**: Utilizza GAN condizionali per generare skin realistiche
- **🎨 Temi Personalizzabili**: 5 temi diversi (Warrior, Mage, Nature, Tech, Shadow)
- **📊 Dataset Intelligente**: Sistema automatico di gestione del dataset
- **⚡ Training Automatico**: Il modello si addestra automaticamente quando necessario
- **🌐 API REST**: Server Flask con endpoint per tutte le funzionalità
- **💾 Persistenza**: Salvataggio automatico dei modelli addestrati

## 🏗️ Architettura del Sistema

### Modello GAN Condizionale
- **Generatore**: ConditionalSkinGenerator con embedding per temi
- **Discriminatore**: SkinDiscriminator per distinguere skin reali da generate
- **Latent Space**: 128 dimensioni per massima varietà
- **Condizionamento**: Embedding di 50 dimensioni per ogni tema

### Struttura della Rete
```
Generatore: Rumore (128D) + Tema (50D) → Linear → ConvTranspose2D → 64x64x4 RGBA
Discriminatore: 64x64x4 RGBA → Conv2D → Linear → Probabilità Real/Fake
```

## 🛠️ Installazione

1. **Clona il repository**:
```bash
git clone <repository-url>
cd skin-minecraft-AI
```

2. **Installa le dipendenze**:
```bash
pip install -r requirements.txt
```

3. **Avvia il sistema**:
```bash
python minecraft_skin_ai_generator.py
```

## 📊 Dataset e Training

### Dataset Automatico
Il sistema crea automaticamente un dataset di esempio con:
- 100 skin di esempio (20 per tema)
- Pattern procedurali diversificati
- Colori tematici appropriati
- Metadata per ogni skin

### Training del Modello
- **Epochs**: Default 100 (configurabile)
- **Batch Size**: Default 8 (ottimizzato per GPU limitate)
- **Loss Function**: Binary Cross Entropy
- **Ottimizzatori**: Adam con learning rate 0.0002

## 🌐 API Endpoints

### Generazione Skin

#### `GET /generate`
Genera una skin casuale
```json
{
  "success": true,
  "skin_id": "AI-20231201120000-1234",
  "theme": "warrior",
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "timestamp": "2023-12-01T12:00:00",
  "generated_by": "Neural_Network_GAN"
}
```

#### `GET /generate/<theme>`
Genera skin con tema specifico
- Temi disponibili: `warrior`, `mage`, `nature`, `tech`, `shadow`, `random`

### Training e Dataset

#### `POST /train`
Avvia il training del modello
```json
{
  "epochs": 100,
  "batch_size": 8
}
```

#### `POST /upload_skin`
Carica una skin nel dataset
- File: `skin` (immagine PNG 64x64)
- Form data: `theme` (opzionale)

#### `POST /create_dataset`
Crea dataset di esempio proceduralmente

### Stato Sistema

#### `GET /status`
Ottieni informazioni complete sul sistema
```json
{
  "success": true,
  "dataset_size": 100,
  "is_trained": true,
  "available_themes": ["warrior", "mage", "nature", "tech", "shadow"],
  "device": "cuda",
  "model_info": {
    "latent_dim": 128,
    "training_epochs": 100
  }
}
```

## 🎨 Temi Disponibili

### 🗡️ Warrior (Guerriero)
- Colori: Marroni, rossi, oro
- Stile: Armature, spade, scudi

### 🪄 Mage (Mago)
- Colori: Viola, blu, oro, bianco
- Stile: Robes mistiche, cappelli a punta

### 🌿 Nature (Natura)
- Colori: Verdi, marroni, gialli
- Stile: Foglie, corteccia, elementi naturali

### 🤖 Tech (Tecnologico)
- Colori: Azzurri, grigi, arancioni
- Stile: Circuiti, metalli, luci LED

### 🥷 Shadow (Ombra)
- Colori: Neri, grigi scuri, rossi
- Stile: Ninja, assassini, cappucci

## 🔧 Configurazione Avanzata

### Parametri del Modello
```python
# Dimensioni dello spazio latente
latent_dim = 128

# Numero di temi
num_classes = 5

# Architettura generatore
- Linear: 128+50 → 256*8*8
- ConvTranspose2d: 256→128→64→32→4
- Attivazione finale: Sigmoid

# Architettura discriminatore  
- Conv2d: 4→32→64→128→256
- Linear finale: 256*4*4 → 1
- Attivazione finale: Sigmoid
```

### Training Ottimizzato
```python
# Ottimizzatori Adam
learning_rate = 0.0002
betas = (0.5, 0.999)

# Batch normalization per stabilità
# Dropout per regolarizzazione
# Label smoothing per training robusto
```

## 📁 Struttura File

```
skin-minecraft-AI/
├── minecraft_skin_ai_generator.py  # Sistema principale
├── requirements.txt                # Dipendenze
├── README.md                      # Documentazione
├── models/                        # Modelli addestrati
│   └── minecraft_skin_gan.pth    # Checkpoint GAN
├── skin_dataset/                  # Dataset skin
│   ├── metadata.json             # Metadata temi
│   └── *.png                     # Immagini skin
```

## 🚀 Utilizzo Avanzato

### Training Personalizzato
```python
# Avvia training con parametri custom
skin_ai.train(epochs=200, batch_size=16)

# Aggiungi skin personalizzate
skin_ai.add_skin_to_dataset("my_skin.png", "warrior")
```

### Generazione Programmatica
```python
# Genera skin con seed per riproducibilità
skin_array, theme = skin_ai.generate_skin(theme='mage', seed=12345)

# Converti in base64 per web
base64_str = skin_ai.skin_to_base64(skin_array)
```

## 🐛 Troubleshooting

### Problemi Comuni

1. **CUDA non disponibile**: Il sistema usa automaticamente CPU
2. **Dataset vuoto**: Chiama `/create_dataset` per crearne uno di esempio
3. **Modello non addestrato**: Il training parte automaticamente al primo utilizzo
4. **Memoria insufficiente**: Riduci `batch_size` a 4 o 2

### Ottimizzazioni Performance
- **GPU**: Installa PyTorch con CUDA per accelerazione GPU
- **Memoria**: Usa batch_size più piccoli se hai poca RAM
- **Dataset**: Aggiungi più skin per risultati migliori

## 🤝 Contributi

Per migliorare il sistema:
1. Aggiungi più skin di qualità al dataset
2. Sperimenta con nuovi temi
3. Ottimizza l'architettura della rete
4. Implementa nuove funzionalità

## 📜 Licenza

Progetto open source - sentiti libero di usarlo e modificarlo!

---

**🎮 Buona generazione di skin con l'AI! 🎨** 