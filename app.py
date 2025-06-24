#!/usr/bin/env python3
"""
Server Web per Generazione Skin Minecraft
Interfaccia web semplice per utilizzare il sistema AI tramite browser.

Il server fornisce endpoint REST per generare skin on-demand
utilizzando il modello addestrato.
"""

from flask import Flask, request, jsonify, send_file
import torch
import numpy as np
from PIL import Image
import io
import base64
import os
from datetime import datetime

# Import dei nostri moduli
from models import SkinGenerator
import config

# Inizializzazione dell'applicazione Flask
app = Flask(__name__)

class SkinAPI:
    """
    API handler per la generazione di skin.
    Gestisce il caricamento del modello e la generazione on-demand.
    """
    def __init__(self):
        # Configurazione hardware per inferenza
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"API Server avviato su device: {self.device}")
        
        # Caricamento del modello pre-addestrato
        self.generator = SkinGenerator().to(self.device)
        self.load_model()
        
    def load_model(self):
        """Carica il modello GAN addestrato."""
        try:
            if os.path.exists(config.MODEL_PATH):
                # Carica solo il generatore per inferenza
                checkpoint = torch.load(config.MODEL_PATH, map_location=self.device)
                self.generator.load_state_dict(checkpoint['generator_state_dict'])
                self.generator.eval()  # Modalità inferenza
                print("Modello AI caricato con successo!")
                return True
            else:
                print("Modello non trovato. Esegui prima il training!")
                return False
        except Exception as e:
            print(f"Errore caricamento modello: {e}")
            return False
    
    def generate_skin(self, num_attempts=5):
        """
        Genera una skin utilizzando il modello AI.
        
        Args:
            num_attempts: Numero di tentativi per ottenere una skin di qualità
            
        Returns:
            Array numpy della skin generata
        """
        best_skin = None
        best_quality = 0
        
        with torch.no_grad():  # Disabilita gradiente per inferenza
            for _ in range(num_attempts):
                # Genera rumore casuale nel latent space
                noise = torch.randn(1, config.LATENT_DIM, 1, 1, device=self.device)
                generated = self.generator(noise)
                
                # Conversione da tensor a array numpy
                skin_array = generated.squeeze().permute(1, 2, 0).cpu().numpy()
                skin_array = np.clip(skin_array * 255, 0, 255).astype(np.uint8)
                
                # Valutazione qualitativa semplice (varianza colori)
                quality = np.var(skin_array)
                
                if quality > best_quality:
                    best_quality = quality
                    best_skin = skin_array
        
        return best_skin

# Istanza globale dell'API
skin_api = SkinAPI()

@app.route('/')
def home():
    """Pagina principale con interfaccia utente."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Generatore Skin Minecraft AI</title>
        <style>
            body { font-family: Arial; margin: 40px; background: #f0f0f0; }
            .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
            button { background: #4CAF50; color: white; padding: 15px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background: #45a049; }
            .result { margin-top: 20px; text-align: center; }
            img { border: 3px solid #333; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Generatore Skin Minecraft AI</h1>
            <p>Clicca il pulsante per generare una skin unica tramite intelligenza artificiale!</p>
            <button onclick="generateSkin()">Genera Skin AI</button>
            <div id="result" class="result"></div>
        </div>
        
        <script>
            async function generateSkin() {
                document.getElementById('result').innerHTML = '<p>Generazione in corso...</p>';
                
                try {
                    const response = await fetch('/generate');
                    const data = await response.json();
                    
                    if (data.success) {
                        document.getElementById('result').innerHTML = 
                            '<h3>Skin Generata!</h3>' +
                            '<img src="data:image/png;base64,' + data.image + '" width="256" height="256" style="image-rendering: pixelated;">' +
                            '<p><a href="' + data.download_url + '" download="' + data.filename + '">Scarica Skin</a></p>';
                    } else {
                        document.getElementById('result').innerHTML = '<p style="color:red;">Errore: ' + data.error + '</p>';
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = '<p style="color:red;">Errore di rete</p>';
                }
            }
        </script>
    </body>
    </html>
    """

@app.route('/generate')
def generate_skin_endpoint():
    """
    Endpoint REST per generazione skin.
    Restituisce la skin in formato base64 per visualizzazione web.
    """
    try:
        # Genera skin usando il modello AI
        skin_array = skin_api.generate_skin()
        
        if skin_array is None:
            return jsonify({
                'success': False, 
                'error': 'Modello non disponibile'
            })
        
        # Conversione in immagine PIL e salvataggio
        img = Image.fromarray(skin_array, 'RGBA')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_skin_{timestamp}.png"
        
        # Salva su disco per download
        img.save(filename)
        
        # Converte in base64 per display web
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'filename': filename,
            'download_url': f'/download/{filename}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': str(e)
        })

@app.route('/download/<filename>')
def download_skin(filename):
    """Endpoint per download diretto delle skin generate."""
    try:
        if os.path.exists(filename):
            return send_file(filename, as_attachment=True)
        else:
            return "File non trovato", 404
    except Exception as e:
        return f"Errore download: {e}", 500

if __name__ == '__main__':
    print("\nAvvio Server Web per Generazione Skin")
    print("=" * 50)
    print("Accedi a: http://localhost:5000")
    print("Per fermare il server: Ctrl+C")
    print("=" * 50)
    
    # Avvio server Flask in modalità debug per sviluppo
    app.run(debug=True, host='0.0.0.0', port=5000) 