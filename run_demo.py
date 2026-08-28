import os
from pyngrok import ngrok
from app import app

# 1. Masukkan Authtoken resmi dari dashboard akun Ngrok Anda di sini
ngrok.set_auth_token("2z87KAhGQ6NXGrpqtK7aCLu4geU_tjmkH9gE924Jzywrnsj1")

if __name__ == '__main__':
    # 2. Masukkan nama domain statis gratis yang sudah Anda klaim dari dashboard Ngrok
    # Contoh: "poodle-casual-strongly.ngrok-free.app"
    DOMAIN_STATIS = " https://853d-2402-8780-1068-aaa1-4939-c74e-a36b-ea3.ngrok-free.app"
    
    # 3. Membuka terowongan publik aman yang terkunci pada domain tetap
    public_url = ngrok.connect(8080, domain=DOMAIN_STATIS)
    
    print("\n" + "="*60)
    print(f" WEBSITE LUNGSEE SEKARANG ONLINE PERMANEN (TIDAK BERUBAH)!")
    print(f" SILAKAN BAGIKAN URL RESMI INI KE DOSEN/PENGUJI:")
    print(f" {public_url.public_url}")
    print("="*60 + "\n")
    
    # 4. Menjalankan server Flask lokal Anda
    app.run(port=8080)