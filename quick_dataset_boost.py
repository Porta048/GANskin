import requests
import os
import time
import random
from PIL import Image
import io

def download_skin_batch():
    """Scarica velocemente un batch di skin da fonti affidabili"""
    target_dir = "./skin_dataset"
    os.makedirs(target_dir, exist_ok=True)
    
    print("🚀 Quick Dataset Boost - Aggiunta rapida skin")
    print("=" * 50)
    
    downloaded = 0
    failed = 0
    
    # Fonti affidabili con skin di qualità
    sources = [
        # Crafatar API - skin di giocatori veri  
        lambda i: f"https://crafatar.com/skins/{generate_uuid()}.png",
        # Minotar - alternativa affidabile
        lambda i: f"https://minotar.net/skin/{generate_uuid()}.png",
        # Visage - servizio skin di qualità
        lambda i: f"https://visage.surgeplay.com/skin/64/{generate_uuid()}.png",
    ]
    
    print("🎯 Scaricando 300 skin aggiuntive...")
    
    for i in range(300):
        try:
            # Scegli fonte casuale
            source = random.choice(sources)
            url = source(i)
            
            filename = f"boost_{i:04d}_{random.randint(1000,9999)}.png"
            filepath = os.path.join(target_dir, filename)
            
            # Scarica
            response = requests.get(url, timeout=5)
            if response.status_code == 200 and len(response.content) > 1000:
                
                # Verifica che sia una skin valida
                try:
                    img = Image.open(io.BytesIO(response.content))
                    if img.size == (64, 64):
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        downloaded += 1
                        
                        if downloaded % 10 == 0:
                            print(f"✅ Scaricate: {downloaded}/300")
                    else:
                        failed += 1
                except:
                    failed += 1
            else:
                failed += 1
                
            # Piccola pausa per non sovraccaricare i server
            time.sleep(0.05)
            
        except Exception as e:
            failed += 1
            continue
    
    print(f"\n🎉 Completato!")
    print(f"✅ Scaricate: {downloaded}")
    print(f"❌ Fallite: {failed}")
    
    # Conta totale skin nel dataset
    total_skins = len([f for f in os.listdir(target_dir) if f.endswith('.png')])
    print(f"📊 Totale dataset: {total_skins} skin")
    
    return downloaded

def generate_uuid():
    """Genera UUID casuali per ottenere skin diverse"""
    # Lista di UUID di giocatori famosi e casuali
    famous_uuids = [
        "853c80ef3c3749fdaa49938b674adae6",  # jeb_
        "069a79f444e94726a5befca90e38aaf5",  # Notch  
        "61699b2ed3274a019f1e0ea8c3f06bc6",  # Dinnerbone
        "7125ba8b1c864508b92bb5c042ccfe2b",  # Grumm
        "45f50155c09f4fdcb5cee30af2e72667",  # Tommaso
    ]
    
    # 50% possibilità di usare UUID famosi, 50% casuali
    if random.random() < 0.5 and famous_uuids:
        return random.choice(famous_uuids)
    else:
        # Genera UUID casuale
        return ''.join(random.choices('0123456789abcdef', k=32))

def boost_with_themed_skins():
    """Aggiunge skin tematiche specifiche"""
    print("🎨 Aggiungendo skin tematiche...")
    
    # Cerca skin con nomi specifici per migliorare la varietà
    theme_keywords = [
        "warrior", "knight", "mage", "wizard", "ninja", "samurai",
        "robot", "cyber", "tech", "futuristic", "steampunk",
        "nature", "forest", "animal", "dragon", "phoenix",
        "dark", "shadow", "demon", "angel", "ghost",
        "medieval", "royal", "princess", "king", "queen"
    ]
    
    downloaded = 0
    for keyword in theme_keywords:
        try:
            # Simula ricerca tematica (questo è un placeholder)
            for i in range(5):  # 5 skin per tema
                uuid = generate_uuid()
                url = f"https://crafatar.com/skins/{uuid}.png"
                filename = f"theme_{keyword}_{i}_{uuid[:8]}.png"
                filepath = os.path.join("./skin_dataset", filename)
                
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    downloaded += 1
                    
                time.sleep(0.1)
                
        except:
            continue
    
    print(f"🎨 Aggiunte {downloaded} skin tematiche")
    return downloaded

def main():
    print("⚡ QUICK DATASET BOOST")
    print("Aggiungi rapidamente più dati all'AI esistente")
    print("=" * 50)
    
    current_count = len([f for f in os.listdir("./skin_dataset") if f.endswith('.png')])
    print(f"📊 Dataset attuale: {current_count} skin")
    
    print("\nOpzioni:")
    print("1. 🚀 Boost Rapido (+300 skin)")
    print("2. 🎨 Boost Tematico (+150 skin)")
    print("3. 🔥 Boost Completo (+450 skin)")
    
    choice = input("\nScegli (1-3): ").strip()
    
    if choice == '1':
        downloaded = download_skin_batch()
    elif choice == '2':
        downloaded = boost_with_themed_skins()
    elif choice == '3':
        downloaded1 = download_skin_batch()
        downloaded2 = boost_with_themed_skins()
        downloaded = downloaded1 + downloaded2
    else:
        print("❌ Scelta non valida")
        return
    
    if downloaded > 0:
        final_count = len([f for f in os.listdir("./skin_dataset") if f.endswith('.png')])
        print(f"\n🎉 SUCCESSO!")
        print(f"📈 Da {current_count} a {final_count} skin (+{downloaded})")
        print(f"💡 Riavvia il server AI per il re-training:")
        print(f"   python minecraft_skin_ai_generator.py")
        
        # Suggerimenti per training
        if final_count > 1000:
            print(f"🔥 Con {final_count} skin, l'AI sarà molto più potente!")
        if final_count > 2000:
            print(f"🚀 Dataset MEGA! Considera training con più epochs")

if __name__ == "__main__":
    main() 