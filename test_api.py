from byteplussdkarkruntime import Ark
from dotenv import load_dotenv
load_dotenv()
import os
# ==========================================
# KONFIGURASI (ISI INI DULU)
# ==========================================

ARK_API_KEY = os.getenv("ARK_API_KEY")  
MODEL_ENDPOINT_ID = os.getenv("MODEL_ENDPOINT_ID") 

# 1. Inisialisasi Client
print("🔄 Menghubungkan ke BytePlus Ark...")
client = Ark(
    api_key=ARK_API_KEY,
    base_url="https://ark.ap-southeast.bytepluses.com/api/v3"
)

# 2. Kirim Pesan Sederhana
try:
    print("📨 Mengirim pesan tes...")
    
    completion = client.chat.completions.create(
        model=MODEL_ENDPOINT_ID,
        messages=[
            {"role": "system", "content": "Kamu adalah asisten tes."},
            {"role": "user", "content": "Halo! Cek koneksi, balas dengan singkat jika kamu menerima pesan ini."}
        ]
    )
    
    # 3. Tampilkan Hasil
    print("\n✅ KONEKSI BERHASIL!")
    print("🤖 Balasan AI:")
    print("------------------------------------------------")
    print(completion.choices[0].message.content)
    print("------------------------------------------------")

except Exception as e:
    print("\n❌ TERJADI ERROR:")
    print(e)
    print("\nTips:")
    print("1. Cek apakah API Key sudah benar.")
    print("2. Cek apakah Endpoint ID benar (formatnya biasanya 'ep-xxxx').")
    print("3. Pastikan Endpoint ID tersebut sudah 'Running' di console BytePlus.")