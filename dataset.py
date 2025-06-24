import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import config
import hashlib

class SkinDataset(Dataset):
    """
    Dataset personalizzato per le skin di Minecraft.
    
    Gestisce il caricamento efficiente delle skin dal filesystem,
    applicando le trasformazioni necessarie per il training della GAN.
    Include funzionalità di pulizia automatica per rimuovere file corrotti.
    """
    def __init__(self, dataset_path=None, transform=None):
        """
        Inizializza il dataset con path e trasformazioni opzionali.
        
        Args:
            dataset_path: Percorso alla cartella del dataset (default da config)
            transform: True per trasformazioni standard, o oggetto transforms personalizzato
        """
        if dataset_path is None:
            self.data_dir = config.DATASET_PATH
        else:
            self.data_dir = dataset_path
        
        # Filtra solo file PNG validi con dimensione minima
        # Questo aiuta a evitare problemi con file corrotti o incompleti
        # Cerca anche in sottocartelle come professional/
        self.image_files = []
        
        # File PNG nella cartella principale
        for f in os.listdir(self.data_dir):
            if f.endswith('.png'):
                full_path = os.path.join(self.data_dir, f)
                if os.path.getsize(full_path) > 100:
                    self.image_files.append(f)
        
        # File PNG nelle sottocartelle
        for subdir in os.listdir(self.data_dir):
            subdir_path = os.path.join(self.data_dir, subdir)
            if os.path.isdir(subdir_path):
                for f in os.listdir(subdir_path):
                    if f.endswith('.png'):
                        full_path = os.path.join(subdir_path, f)
                        if os.path.getsize(full_path) > 100:
                            # Usa path relativo dalla cartella principale
                            relative_path = os.path.join(subdir, f)
                            self.image_files.append(relative_path)
        
        # Configurazione delle trasformazioni per preprocessing
        if transform is True:
            # Trasformazione standard se transform=True
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        elif transform is not None:
            # Oggetto transform personalizzato
            self.transform = transform
        else:
            # Trasformazione di default: converte PIL in Tensor normalizzato [0,1]
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        
        print(f"Dataset caricato: {len(self.image_files)} skin.")

    def __len__(self):
        """Ritorna il numero totale di skin nel dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Carica e preprocessa una singola skin.
        
        Gestisce automaticamente:
        - Conversione a formato RGBA standard
        - Resize a 64x64 se necessario  
        - Aggiunta canale alpha se mancante
        - Rimozione automatica file corrotti
        """
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        
        try:
            # Carica immagine e forza conversione RGBA
            image = Image.open(img_path).convert('RGBA')
            
            # Verifica e correzione dimensioni
            if image.size != (64, 64):
                # Resize con algoritmo Lanczos per migliore qualità
                image = image.resize((64, 64), Image.Resampling.LANCZOS)
            
            # Converte in tensor PyTorch
            tensor = self.transform(image)
            
            # Assicura che abbiamo 4 canali RGBA
            if tensor.shape[0] == 3:
                # Aggiunge canale alpha opaco se mancante
                alpha = torch.ones(1, 64, 64)
                tensor = torch.cat([tensor, alpha], dim=0)
            
            return tensor
            
        except Exception as e:
            print(f"Errore caricamento skin: {img_path}, errore: {e}")
            
            # Rimozione automatica file corrotto per pulizia dataset
            try:
                os.remove(img_path)
            except:
                pass
            
            # Fallback: carica la skin successiva per evitare crash
            return self.__getitem__((idx + 1) % len(self))

    def add_skin(self, image_data):
        """
        Aggiunge una nuova skin al dataset evitando duplicati.
        
        Utilizza hash MD5 per identificare duplicati in modo efficiente.
        Utile per espansioni dinamiche del dataset durante il training.
        """
        try:
            # Calcola hash per rilevamento duplicati
            file_hash = hashlib.md5(image_data).hexdigest()
            filename = f"skin_{file_hash}.png"
            filepath = os.path.join(self.data_dir, filename)

            # Salva solo se non esiste già
            if not os.path.exists(filepath):
                with open(filepath, 'wb') as f:
                    f.write(image_data)
                self.image_files.append(filename)
                return True
                
        except Exception as e:
            print(f"Errore durante l'aggiunta della skin: {e}")
        
        return False 