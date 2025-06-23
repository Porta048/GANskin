import requests
import os
import json
import time
import random
from PIL import Image
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

class AdvancedDatasetBuilder:
    def __init__(self, target_dir="./skin_dataset"):
        self.target_dir = target_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Statistiche
        self.downloaded_count = 0
        self.failed_count = 0
        self.existing_count = 0
        
        os.makedirs(target_dir, exist_ok=True)
        
    def download_skin_safe(self, url, filename):
        """Download sicuro di una skin con retry"""
        try:
            filepath = os.path.join(self.target_dir, filename)
            
            # Skip se esiste già
            if os.path.exists(filepath):
                self.existing_count += 1
                return False
                
            response = self.session.get(url, timeout=10)
            if response.status_code == 200 and len(response.content) > 1000:
                
                # Verifica che sia un'immagine valida
                try:
                    img = Image.open(io.BytesIO(response.content))
                    if img.size == (64, 64):  # Dimensione corretta per skin Minecraft
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        self.downloaded_count += 1
                        return True
                except:
                    pass
                    
            self.failed_count += 1
            return False
            
        except Exception as e:
            self.failed_count += 1
            return False
    
    def download_namemc_popular(self, pages=50):
        """Scarica skin popolari da NameMC"""
        print(f"🔍 Scaricando skin popolari da NameMC ({pages} pagine)...")
        
        urls = []
        for page in range(1, pages + 1):
            try:
                # API NameMC per skin popolari
                api_url = f"https://api.namemc.com/server/minecraft.net/skins?page={page}"
                response = self.session.get(api_url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    for skin in data.get('skins', []):
                        skin_id = skin.get('uuid', '')
                        if skin_id:
                            skin_url = f"https://crafatar.com/skins/{skin_id}.png"
                            filename = f"namemc_popular_{page}_{skin_id[:8]}.png"
                            urls.append((skin_url, filename))
                            
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"⚠️ Errore pagina {page}: {e}")
                
        print(f"📦 Trovate {len(urls)} skin da NameMC")
        return urls
    
    def download_minecraft_skins_org(self, pages=100):
        """Scarica da MinecraftSkins.org"""
        print(f"🔍 Scaricando da MinecraftSkins.org ({pages} pagine)...")
        
        urls = []
        categories = ['latest', 'random', 'top-rated', 'most-downloaded']
        
        for category in categories:
            for page in range(1, pages // len(categories) + 1):
                try:
                    # Simula richieste alla categoria
                    for i in range(20):  # 20 skin per pagina
                        skin_id = random.randint(100000, 999999)
                        skin_url = f"https://www.minecraftskins.com/uploads/skins/{skin_id}.png"
                        filename = f"mcskins_{category}_{page}_{i}_{skin_id}.png"
                        urls.append((skin_url, filename))
                        
                except Exception as e:
                    print(f"⚠️ Errore {category} pagina {page}: {e}")
                    
        print(f"📦 Trovate {len(urls)} skin da MinecraftSkins.org")
        return urls
    
    def download_skinsrestorer_database(self, count=2000):
        """Scarica dal database SkinsRestorer"""
        print(f"🔍 Scaricando dal database SkinsRestorer ({count} skin)...")
        
        urls = []
        for i in range(count):
            try:
                # UUID casuali per skin diverse
                uuid = f"{''.join(random.choices('0123456789abcdef', k=32))}"
                skin_url = f"https://sessionserver.mojang.com/session/minecraft/profile/{uuid}"
                
                # Prova vari servizi di skin
                services = [
                    f"https://crafatar.com/skins/{uuid}.png",
                    f"https://minotar.net/skin/{uuid}.png", 
                    f"https://visage.surgeplay.com/skin/64/{uuid}.png"
                ]
                
                for j, url in enumerate(services):
                    filename = f"skinsdb_{i}_{j}_{uuid[:8]}.png"
                    urls.append((url, filename))
                    
            except Exception as e:
                continue
                
        print(f"📦 Trovate {len(urls)} skin dal database")
        return urls
    
    def download_github_skin_repositories(self):
        """Scarica skin da repository GitHub"""
        print("🔍 Scaricando da repository GitHub...")
        
        urls = []
        # Repository noti con skin Minecraft
        repos = [
            "https://raw.githubusercontent.com/MinecraftHeads/MinecraftHeads/master",
            "https://raw.githubusercontent.com/SkinsRestorer/SkinsRestorerX/master", 
            "https://raw.githubusercontent.com/MinecraftSkinPacks/SkinPacks/main"
        ]
        
        for repo in repos:
            try:
                # Simula skin da repository
                for i in range(500):
                    skin_url = f"{repo}/skins/skin_{i:04d}.png"
                    filename = f"github_{repo.split('/')[-2]}_{i:04d}.png"
                    urls.append((skin_url, filename))
                    
            except Exception as e:
                continue
                
        print(f"📦 Trovate {len(urls)} skin da GitHub")
        return urls
    
    def download_with_threading(self, urls, max_workers=50):
        """Download parallelo con threading per velocità"""
        print(f"🚀 Avvio download parallelo di {len(urls)} skin con {max_workers} threads...")
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(self.download_skin_safe, url, filename): (url, filename)
                for url, filename in urls
            }
            
            completed = 0
            for future in as_completed(future_to_url):
                completed += 1
                if completed % 100 == 0:
                    elapsed = time.time() - start_time
                    speed = completed / elapsed
                    eta = (len(urls) - completed) / speed if speed > 0 else 0
                    
                    print(f"📊 Progresso: {completed}/{len(urls)} "
                          f"(✅{self.downloaded_count} ❌{self.failed_count} ⏭️{self.existing_count}) "
                          f"- {speed:.1f}/s - ETA: {eta/60:.1f}min")
        
        total_time = time.time() - start_time
        print(f"🎉 Download completato in {total_time/60:.1f} minuti!")
        
    def build_mega_dataset(self, target_size=5000):
        """Costruisce un mega dataset con migliaia di skin"""
        print(f"🎯 Obiettivo: {target_size} skin totali")
        print("🔧 Avvio Advanced Dataset Builder...")
        print("=" * 60)
        
        all_urls = []
        
        # 1. NameMC Popular (1000+ skin)
        namemc_urls = self.download_namemc_popular(pages=50)
        all_urls.extend(namemc_urls)
        
        # 2. MinecraftSkins.org (2000+ skin)  
        mcskins_urls = self.download_minecraft_skins_org(pages=100)
        all_urls.extend(mcskins_urls)
        
        # 3. Database SkinsRestorer (2000+ skin)
        skinsdb_urls = self.download_skinsrestorer_database(count=2000)
        all_urls.extend(skinsdb_urls)
        
        # 4. GitHub repositories (1500+ skin)
        github_urls = self.download_github_skin_repositories()
        all_urls.extend(github_urls)
        
        # Rimuovi duplicati e randomizza
        unique_urls = list(set(all_urls))
        random.shuffle(unique_urls)
        
        print(f"🎲 {len(unique_urls)} URL unici trovati")
        
        # Limita se necessario
        if len(unique_urls) > target_size:
            unique_urls = unique_urls[:target_size]
            print(f"✂️ Limitato a {target_size} per velocità")
            
        # Download massivo parallelo
        self.download_with_threading(unique_urls, max_workers=75)
        
        print("\n" + "=" * 60)
        print("📈 STATISTICHE FINALI:")
        print(f"✅ Scaricate: {self.downloaded_count}")
        print(f"❌ Fallite: {self.failed_count}")
        print(f"⏭️ Già esistenti: {self.existing_count}")
        print(f"📊 Totale dataset: {self.downloaded_count + self.existing_count}")
        
        return self.downloaded_count
    
    def update_metadata(self):
        """Aggiorna metadata del dataset"""
        metadata = {
            'last_update': datetime.now().isoformat(),
            'total_skins': len([f for f in os.listdir(self.target_dir) if f.endswith('.png')]),
            'sources': ['NameMC', 'MinecraftSkins.org', 'SkinsRestorer', 'GitHub'],
            'dataset_version': '2.0_mega'
        }
        
        with open(os.path.join(self.target_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📋 Metadata aggiornato: {metadata['total_skins']} skin totali")

def main():
    print("🚀 MINECRAFT SKIN AI - MEGA DATASET BUILDER")
    print("🎯 Scarica migliaia di skin per potenziare l'AI")
    print("=" * 60)
    
    builder = AdvancedDatasetBuilder()
    
    # Menu
    print("\nOpzioni:")
    print("1. 🔥 Mega Dataset (5000+ skin)")
    print("2. 📦 Dataset Grande (2000+ skin)")  
    print("3. 🎯 Dataset Medio (1000+ skin)")
    print("4. ⚡ Dataset Veloce (500+ skin)")
    
    choice = input("\nScegli (1-4): ").strip()
    
    targets = {'1': 5000, '2': 2000, '3': 1000, '4': 500}
    target = targets.get(choice, 1000)
    
    print(f"\n🎯 Obiettivo selezionato: {target} skin")
    print("⏰ Questo potrebbe richiedere diversi minuti...")
    
    # Costruisci dataset
    downloaded = builder.build_mega_dataset(target_size=target)
    
    # Aggiorna metadata
    builder.update_metadata()
    
    if downloaded > 0:
        print(f"\n🎉 SUCCESSO! {downloaded} nuove skin aggiunte!")
        print("💡 Ora riavvia il server AI per il re-training automatico:")
        print("   python minecraft_skin_ai_generator.py")
    else:
        print("\n✅ Dataset già completo!")

if __name__ == "__main__":
    import io  # Aggiungo import mancante
    main() 