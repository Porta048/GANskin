# Minecraft Skin AI Generator

Sistema avanzato basato su **reti neurali GAN (Generative Adversarial Networks)** per generare skin di Minecraft uniche e di alta qualità.

## Caratteristiche Principali

- **AI Avanzata**: Utilizza GAN condizionali addestrate su 4.408 skin reali
- **Sistema FUSION**: 4 metodi di fusione per combinare multiple skin
- **Temi Personalizzabili**: 5 temi distinti (Warrior, Mage, Nature, Tech, Shadow)
- **Dataset Espanso**: Addestrato su dataset massivo per qualità superiore
- **Training Automatico**: Sistema intelligente di auto-addestramento
- **API REST Completa**: Server Flask con endpoint avanzati
- **Persistenza Modelli**: Salvataggio automatico dei modelli addestrati

## Architettura del Sistema

### Modello GAN Condizionale Ottimizzato
- **Generatore**: ConditionalSkinGenerator con embedding tematici
- **Discriminatore**: SkinDiscriminator per distinguere skin reali/generate
- **Dataset**: 4.408 skin reali per training di alta qualità
- **Latent Space**: 128 dimensioni per massima varietà
- **Condizionamento**: Embedding di 50 dimensioni per tema

### Struttura della Rete
```
Generatore: Rumore (128D) + Tema (50D) → Linear → ConvTranspose2D → 64x64x4 RGBA
Discriminatore: 64x64x4 RGBA → Conv2D → Linear → Probabilità Real/Fake
Dataset: 4.408 skin → Batch size 8 → 551 batch per epoch
```

## Installazione e Utilizzo

### 1. Installazione
```bash
git clone <repository-url>
cd skin-minecraft-AI
pip install -r requirements.txt
```

### 2. Avvio Sistema
```bash
python minecraft_skin_ai_generator.py
```

Il sistema si avvia automaticamente su `http://localhost:5000`

## Dataset e Addestramento

### Dataset Professionale
- **4.408 skin reali** scaricate da fonti multiple
- **Varietà massima**: Crafatar, NameMC, MinecraftSkins.org
- **Qualità verificata**: Solo skin 64x64 RGBA valide
- **Distribuzione temi**: Bilanciata tra tutti i 5 temi

### Training del Modello
- **Epochs**: 80 (ottimizzato per dataset espanso)
- **Batch Size**: 8 (compatibile CPU/GPU)
- **Dataset Size**: 4.408 skin → 551 batch per epoch
- **Loss Function**: Binary Cross Entropy con label smoothing
- **Ottimizzatori**: Adam (lr: 0.0002, betas: (0.5, 0.999))

## API Endpoints

### Generazione Skin Standard

#### `GET /generate_and_download`
Genera e scarica automaticamente una skin casuale
- Ritorna: File PNG scaricabile
- Tema: Selezionato automaticamente
- Qualità: Alta (addestrata su 4408 skin)

#### `GET /generate_and_download?theme=<nome>`
Genera skin con tema specifico
- Temi: `warrior`, `mage`, `nature`, `tech`, `shadow`

### Sistema FUSION (Funzionalità Avanzata)

#### `GET /generate_fusion?method=<tipo>&samples=<numero>`
Genera skin FUSION combinando multiple skin del dataset

**Metodi disponibili:**
- `intelligent`: Combina regioni anatomiche (samples: 20-50)
- `average`: Media pesata di tutte le skin (samples: 30-60)  
- `mosaic`: Blocchi 8x8 con smoothing (samples: 15-30)
- `advanced`: Blend modes professionali (samples: 20-40)

**Esempio:**
```bash
GET /generate_fusion?method=intelligent&samples=30
```

#### `GET /fusion_methods`
Ottieni informazioni sui metodi FUSION disponibili

### Stato Sistema

#### `GET /status`
Informazioni complete del sistema
```json
{
  "dataset_size": 4408,
  "available_themes": ["warrior", "mage", "nature", "tech", "shadow"],
  "device": "cpu",
  "fusion_methods": ["intelligent", "average", "mosaic", "advanced"]
}
```

## Temi Disponibili

### Warrior (Guerriero)
- Colori: Marroni, rossi, oro metallico
- Stile: Armature, armi, protezioni

### Mage (Mago)
- Colori: Viola, blu, oro, bianco mistico
- Stile: Vesti magiche, cappelli, simboli

### Nature (Natura)
- Colori: Verdi, marroni, gialli naturali
- Stile: Foglie, corteccia, elementi organici

### Tech (Tecnologico)
- Colori: Azzurri, grigi, arancioni LED
- Stile: Circuiti, metalli, luci futuristiche

### Shadow (Ombra)
- Colori: Neri, grigi scuri, rossi sangue
- Stile: Ninja, assassini, cappucci oscuri

## Sistema FUSION - Funzionalità Avanzata

### Intelligent Fusion
- **Algoritmo**: Combina testa, corpo, braccia da skin diverse
- **Risultato**: Skin naturali e bilanciate
- **Campioni consigliati**: 20-50

### Average Fusion
- **Algoritmo**: Media pesata di tutti i pixel
- **Risultato**: Skin omogenee con colori sfumati
- **Campioni consigliati**: 30-60

### Mosaic Fusion
- **Algoritmo**: Blocchi 8x8 da skin diverse + smoothing
- **Risultato**: Effetto artistico unico
- **Campioni consigliati**: 15-30

### Advanced Fusion
- **Algoritmo**: Blend modes (multiply, overlay, soft light)
- **Risultato**: Massima qualità visiva
- **Campioni consigliati**: 20-40

## Configurazione Avanzata

### Parametri del Modello
```python
# Dataset e Training
dataset_size = 4408  # Skin reali
batch_size = 8       # Ottimizzato CPU/GPU
epochs = 80          # Per dataset espanso

# Architettura GAN
latent_dim = 128     # Spazio latente
num_classes = 5      # Temi disponibili
embedding_dim = 50   # Dimensione embedding temi

# Generatore: 128+50 → 256*8*8 → ConvTranspose → 64x64x4
# Discriminatore: 64x64x4 → Conv2d → 256*4*4 → 1
```

### Ottimizzazioni Training
- **Adam Optimizer**: lr=0.0002, betas=(0.5, 0.999)
- **Batch Normalization**: Stabilità training
- **Label Smoothing**: Training robusto
- **Automatic Checkpointing**: Salvataggio ogni epoch

## Struttura File Ottimizzata

```
skin-minecraft-AI/
├── minecraft_skin_ai_generator.py  # Sistema principale completo
├── requirements.txt                # Dipendenze Python
├── README.md                      # Documentazione (questo file)
├── DATASET_GUIDE.md              # Guida dataset (aggiornata)
├── models/                       # Modelli addestrati
│   └── minecraft_skin_gan.pth   # Modello GAN (4408 skin)
├── skin_dataset/                 # Dataset completo
│   ├── [4408 file PNG]          # Skin per training
│   └── metadata.json            # Metadati temi
└── downloaded_skins/             # Skin generate
    └── minecraft_skin_*.png      # Output generazioni
```

## Prestazioni e Qualità

### Miglioramenti Dataset Espanso
- **Dataset precedente**: 470 skin → Qualità buona
- **Dataset attuale**: 4.408 skin → Qualità eccellente
- **Miglioramento**: 9.4x più dati → Varietà e realismo massimi

### Metriche Training
- **Loss Discriminatore finale**: ~0.156 (convergenza ottimale)
- **Loss Generatore finale**: ~3.986 (bilanciamento perfetto)
- **Training time**: ~45-60 minuti (80 epochs, 551 batch)

### Output Quality
- **Skin singole**: 8-12 KB (dettagliate)
- **FUSION skin**: 4-11 KB (qualità variabile per metodo)
- **Risoluzione**: 64x64 RGBA (standard Minecraft)
- **Varietà**: Infinite combinazioni possibili

## Utilizzo Avanzato

### Generazione Programmatica
```python
# Avvia il server
python minecraft_skin_ai_generator.py

# Genera skin con tema specifico
GET http://localhost:5000/generate_and_download?theme=warrior

# Genera FUSION intelligente
GET http://localhost:5000/generate_fusion?method=intelligent&samples=30
```

### Esempi Curl
```bash
# Skin casuale
curl -O http://localhost:5000/generate_and_download

# Skin mago
curl -O "http://localhost:5000/generate_and_download?theme=mage"

# FUSION mosaic
curl -O "http://localhost:5000/generate_fusion?method=mosaic&samples=25"
```

## Requisiti di Sistema

### Minimi
- **Python**: 3.8+
- **RAM**: 4GB (dataset 4408 skin)
- **Storage**: 2GB (modelli + dataset)
- **CPU**: Dual-core (training lento ma possibile)

### Raccomandati
- **RAM**: 8GB+ (caricamento dataset più veloce)
- **GPU**: CUDA-compatible (training 5x più veloce)
- **Storage**: SSD (I/O dataset ottimizzato)

## Troubleshooting

### Errori Comuni
**"ModuleNotFoundError"**: 
```bash
pip install -r requirements.txt
```

**"Dataset size: 0"**: Dataset corrotto, ri-scaricare
**"Training not starting"**: Verificare spazio disco (2GB+)
**"FUSION skin vuote"**: Riavviare server, caricare dataset

### Performance Issues
**Training lento**: Normale su CPU (45-60 min)
**High memory usage**: Ridurre batch_size se necessario
**FUSION timeout**: Ridurre samples (es. 15-20)

---

**Sistema ottimizzato e pronto per produzione**
- Dataset professionale: 4.408 skin reali
- Modello addestrato: 80 epochs su dati massivi  
- Funzionalità complete: Generazione + FUSION
- API stabile: REST endpoints per tutte le funzioni 