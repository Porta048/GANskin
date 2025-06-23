#!/usr/bin/env python3
"""
Script per scaricare automaticamente skin generate dall'AI
"""

import requests
import os
from datetime import datetime

def download_random_skin(output_dir="./downloaded_skins"):
    """Scarica una skin casuale generata dall'AI"""
    
    # Crea directory se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("🎲 Generando e scaricando skin casuale...")
        
        # Chiama endpoint per generazione + download
        response = requests.get("http://localhost:5000/generate_and_download", stream=True)
        
        if response.status_code == 200:
            # Estrai nome file dall'header
            filename = "minecraft_skin_random.png"
            if 'content-disposition' in response.headers:
                content_disp = response.headers['content-disposition']
                if 'filename=' in content_disp:
                    filename = content_disp.split('filename=')[1].strip('"')
            
            # Se non c'è nome specifico, crea uno basato su timestamp
            if filename == "minecraft_skin_random.png":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"minecraft_skin_{timestamp}.png"
            
            # Percorso completo
            filepath = os.path.join(output_dir, filename)
            
            # Salva file
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Skin scaricata: {filepath}")
            print(f"📊 Dimensione: {os.path.getsize(filepath)} bytes")
            return filepath
            
        else:
            print(f"❌ Errore: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Errore: {e}")
        return None

def download_multiple_skins(count=5, output_dir="./downloaded_skins"):
    """Scarica multiple skin casuali"""
    
    print(f"🎯 Scaricando {count} skin casuali...")
    
    successful = 0
    for i in range(count):
        print(f"\n📦 Skin {i+1}/{count}")
        
        if download_random_skin(output_dir):
            successful += 1
        
        # Piccola pausa tra download
        import time
        time.sleep(1)
    
    print(f"\n🎉 Completato! {successful}/{count} skin scaricate con successo")

def download_with_theme(theme, output_dir="./downloaded_skins"):
    """Scarica skin con tema specifico"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"🎨 Generando skin tema '{theme}'...")
        
        # Prima ottieni i dati JSON
        response = requests.get(f"http://localhost:5000/generate/{theme}")
        
        if response.status_code == 200:
            data = response.json()
            
            if data['success']:
                skin_id = data['skin_id']
                used_theme = data['theme']
                
                # Ora scarica il file
                download_response = requests.get(f"http://localhost:5000/download_skin/{skin_id}", stream=True)
                
                if download_response.status_code == 200:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"minecraft_skin_{used_theme}_{timestamp}.png"
                    filepath = os.path.join(output_dir, filename)
                    
                    with open(filepath, 'wb') as f:
                        for chunk in download_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    print(f"✅ Skin tema '{used_theme}' scaricata: {filepath}")
                    return filepath
                else:
                    print(f"❌ Errore download: {download_response.status_code}")
            else:
                print(f"❌ Errore generazione: {data.get('error', 'Unknown')}")
        else:
            print(f"❌ Errore API: {response.status_code}")
            
        return None
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        return None

def main():
    """Menu interattivo per scaricare skin"""
    
    print("🎮 === MINECRAFT SKIN DOWNLOADER ===")
    print("Scarica automaticamente skin generate dall'AI\n")
    
    while True:
        print("Opzioni disponibili:")
        print("1. 🎲 Scarica 1 skin casuale")
        print("2. 📦 Scarica multiple skin casuali")
        print("3. 🎨 Scarica skin con tema specifico")
        print("4. 📁 Apri cartella download")
        print("0. ❌ Esci")
        
        choice = input("\nScegli opzione: ").strip()
        
        try:
            if choice == "1":
                download_random_skin()
                
            elif choice == "2":
                try:
                    count = int(input("Quante skin scaricare? (default 5): ") or "5")
                    download_multiple_skins(count)
                except ValueError:
                    print("❌ Numero non valido!")
                    
            elif choice == "3":
                print("Temi disponibili: warrior, mage, nature, tech, shadow, random")
                theme = input("Inserisci tema: ").strip().lower()
                if theme:
                    download_with_theme(theme)
                else:
                    print("❌ Tema non specificato!")
                    
            elif choice == "4":
                output_dir = "./downloaded_skins"
                if os.path.exists(output_dir):
                    # Apri cartella nel file manager
                    import subprocess
                    import platform
                    
                    system = platform.system()
                    if system == "Windows":
                        os.startfile(output_dir)
                    elif system == "Darwin":  # macOS
                        subprocess.run(["open", output_dir])
                    else:  # Linux
                        subprocess.run(["xdg-open", output_dir])
                    
                    print(f"📁 Cartella aperta: {output_dir}")
                else:
                    print("❌ Nessuna skin scaricata ancora!")
                    
            elif choice == "0":
                print("👋 Arrivederci!")
                break
                
            else:
                print("❌ Opzione non valida!")
                
        except KeyboardInterrupt:
            print("\n👋 Uscita forzata!")
            break
        except Exception as e:
            print(f"❌ Errore: {e}")

if __name__ == "__main__":
    # Verifica che il server sia attivo
    try:
        response = requests.get("http://localhost:5000/status", timeout=5)
        if response.status_code == 200:
            print("✅ Server AI connesso!")
            main()
        else:
            print("❌ Server AI non risponde!")
    except:
        print("❌ Server AI offline!")
        print("Avvia prima: python minecraft_skin_ai_generator.py") 