# Tutorial Pemula: Deploy V14 ABSA ke Server Ubuntu

Panduan ini dibuat untuk kondisi berikut:

- Laptop menggunakan Windows dan PowerShell.
- Server menggunakan Ubuntu 22.04 dengan user `ubuntu`.
- IP lokal server adalah `192.168.1.3`.
- Alamat aplikasi yang direncanakan adalah `http://10.13.13.90:8000` melalui WireGuard.
- Docker dan Git sudah tersedia di server.

> Penting: jangan menulis password di dalam command, file `.env`, Git, atau screenshot.
> Ketika SSH meminta password, ketik langsung pada prompt. Karakter password memang
> tidak akan terlihat saat diketik.

## Mengenali tempat menjalankan perintah

Panduan menggunakan dua label:

- **LAPTOP — PowerShell**: jalankan di PowerShell pada laptop Windows.
- **SERVER — Ubuntu**: jalankan setelah berhasil masuk melalui SSH. Prompt biasanya
  terlihat seperti `ubuntu@ciptadra-svr:~$`.

Jangan menjalankan perintah PowerShell di server Ubuntu dan jangan menjalankan
perintah Ubuntu langsung di PowerShell kecuali melalui SSH.

## Tahap 1 — Masuk dan periksa server

### 1. Pastikan laptop berada di jaringan kantor

Alamat `192.168.1.3` hanya dapat diakses dari jaringan lokal kantor. Sambungkan laptop
ke LAN atau Wi-Fi kantor terlebih dahulu.

### 2. Masuk ke server

**LAPTOP — PowerShell**

```powershell
ssh ubuntu@192.168.1.3
```

Jika muncul pertanyaan fingerprint seperti `Are you sure you want to continue`, periksa
fingerprint dengan pengelola server. Setelah dipastikan benar, ketik `yes`. Masukkan
password pada prompt berikutnya.

Berhasil jika prompt berubah menjadi kurang lebih:

```text
ubuntu@ciptadra-svr:~$
```

### 3. Periksa CPU, RAM, disk, Docker, IP, dan port

**SERVER — Ubuntu**

```bash
hostnamectl
nproc
free -h
df -h /
docker --version
docker compose version
ip -brief address
sudo ss -ltnp | grep -E ':(8000|8080)\b' || true
nvidia-smi || true
```

Saat `sudo` meminta password, masukkan password user Ubuntu.

Syarat minimum untuk melanjutkan:

- RAM total minimal 8 GB.
- Ruang kosong disk minimal 20 GB.
- `docker compose version` berhasil.
- Port 8000 tidak sedang dipakai aplikasi lain.

Jika RAM atau disk kurang, atau port 8000 menampilkan proses lain, berhenti dan
konsultasikan dengan pengelola server. Jangan menghentikan proses yang sudah ada secara
sembarangan.

### 4. Buat folder tujuan

**SERVER — Ubuntu**

```bash
mkdir -p /home/ubuntu/v14-absa
exit
```

Perintah `exit` mengembalikan terminal ke PowerShell laptop.

## Tahap 2 — Kemas aplikasi di laptop

### 5. Masuk ke folder proyek

**LAPTOP — PowerShell**

```powershell
cd "C:\Users\Asus\Downloads\ML"
```

### 6. Buat arsip deployment

**LAPTOP — PowerShell**

```powershell
tar -czf v14-absa-deploy.tar.gz `
  --exclude="frontend/node_modules" `
  --exclude="frontend/dist" `
  --exclude="backend/__pycache__" `
  --exclude="backend/.pytest_cache" `
  --exclude="absa_v14_context_hardened_seed_42/absa_v14/__pycache__" `
  -C "Ui test" `
  backend `
  frontend `
  absa_v14_context_hardened_seed_42 `
  Dockerfile.v14 `
  docker-compose.v14.yml `
  .dockerignore `
  .env.v14.example `
  DEPLOY_V14.md
```

Proses ini dapat memerlukan beberapa menit karena bundle model sekitar 3,12 GiB.
Periksa hasilnya:

```powershell
Get-Item .\v14-absa-deploy.tar.gz | Select-Object Name,Length,LastWriteTime
```

Berhasil jika file `v14-absa-deploy.tar.gz` muncul dan ukurannya tidak nol.

## Tahap 3 — Kirim arsip ke server

### 7. Upload dengan SCP

**LAPTOP — PowerShell**

```powershell
scp .\v14-absa-deploy.tar.gz ubuntu@192.168.1.3:/home/ubuntu/
```

Masukkan password saat diminta. Karena file besar, proses dapat berlangsung cukup lama.
Tunggu hingga progres mencapai `100%` dan prompt PowerShell kembali. Jika koneksi putus,
jalankan kembali perintah yang sama; file tujuan yang belum lengkap akan ditimpa.

## Tahap 4 — Ekstrak dan atur aplikasi

### 8. Masuk lagi ke server

**LAPTOP — PowerShell**

```powershell
ssh ubuntu@192.168.1.3
```

### 9. Ekstrak arsip

**SERVER — Ubuntu**

```bash
cd /home/ubuntu/v14-absa
tar -xzf /home/ubuntu/v14-absa-deploy.tar.gz
ls -lah
```

Daftar file harus memuat `Dockerfile.v14`, `docker-compose.v14.yml`, `backend`,
`frontend`, dan `absa_v14_context_hardened_seed_42`.

Periksa tiga checkpoint utama:

```bash
ls -lh \
  absa_v14_context_hardened_seed_42/aspect/model.safetensors \
  absa_v14_context_hardened_seed_42/opinion/model.safetensors \
  absa_v14_context_hardened_seed_42/sentiment/model.safetensors
```

Ketiganya harus ditemukan. Jangan lanjut jika ada pesan `No such file or directory`.

### 10. Buat konfigurasi server

**SERVER — Ubuntu**

```bash
cp .env.v14.example .env
nano .env
```

Ubah isi `.env` menjadi:

```dotenv
V14_ABSA_PORT=8000
ABSA_MODEL_LOADING=stream
API_MEMORY_LIMIT=12G
MAX_REVIEW_LENGTH=10000
MAX_CSV_FILE_SIZE_MB=100
BATCH_TIMEOUT_SECONDS=1800
CORS_ORIGINS=http://192.168.1.3:8000,http://10.13.13.90:8000
```

Cara menyimpan di Nano:

1. Tekan `Ctrl+O`.
2. Tekan `Enter` untuk mengonfirmasi nama file.
3. Tekan `Ctrl+X` untuk keluar.

Mode `stream` dipilih agar penggunaan RAM lebih rendah.

### 11. Validasi konfigurasi Docker

**SERVER — Ubuntu**

```bash
docker compose --env-file .env -f docker-compose.v14.yml config
```

Berhasil jika konfigurasi ditampilkan tanpa pesan error. Jika muncul error, jangan lanjut
ke build.

## Tahap 5 — Build dan jalankan

### 12. Build image Docker

**SERVER — Ubuntu**

```bash
docker compose --env-file .env -f docker-compose.v14.yml build
```

Build pertama dapat memakan waktu lama karena Docker perlu menarik base image dan
mengunduh dependency Python serta Node. Server membutuhkan akses internet saat build
pertama. Tunggu sampai proses selesai dan kembali ke prompt.

Jika terdapat `no space left on device`, hentikan proses dan hubungi pengelola server.
Jangan menghapus image atau container lain tanpa mengetahui pemiliknya.

### 13. Jalankan aplikasi

**SERVER — Ubuntu**

```bash
docker compose --env-file .env -f docker-compose.v14.yml up -d
```

Periksa status:

```bash
docker compose --env-file .env -f docker-compose.v14.yml ps
```

API memerlukan waktu untuk startup. Status akhirnya harus `healthy`, sedangkan web
harus `Up`. Tunggu 1–3 menit bila status masih `health: starting`.

Jika belum sehat, lihat log:

```bash
docker compose --env-file .env -f docker-compose.v14.yml logs --tail=200 api
```

## Tahap 6 — Tes dari server dan laptop

### 14. Tes health endpoint dari server

**SERVER — Ubuntu**

```bash
curl http://localhost:8000/api/health
```

Respons yang benar kurang lebih:

```json
{"status":"ok","service":"V14 ABSA API","supported_engines":["v14"]}
```

Periksa daftar engine:

```bash
curl http://localhost:8000/api/inference/engines
```

Hanya engine `v14` yang boleh muncul.

### 15. Jalankan smoke test model

**SERVER — Ubuntu**

```bash
curl -X POST http://localhost:8000/api/inference/single \
  -H 'Content-Type: application/json' \
  -d '{"review":"Makanannya enak tetapi pelayanannya lambat.","engine_version":"v14","profile":"maps_high_recall","confidence_threshold":0.1}'
```

Pada respons, pastikan terdapat:

```json
"engine_version": "v14"
"model_bundle_loaded": true
"inference_mode": "transformer_bundle"
```

Inference pertama bisa lebih lambat karena model baru dimuat.

### 16. Buka dari laptop kantor

Buka browser pada laptop yang masih berada di jaringan kantor:

```text
http://192.168.1.3:8000
```

Jika akses WireGuard sudah diarahkan dengan benar, tes juga:

```text
http://10.13.13.90:8000
```

Jika alamat lokal berhasil tetapi WireGuard gagal, aplikasi sudah berjalan dan masalah
berada pada routing atau firewall WireGuard. Hubungi pengelola infrastruktur; jangan
mengubah firewall server tanpa koordinasi.

## Tahap 7 — Operasional dasar

Semua perintah berikut dijalankan di folder `/home/ubuntu/v14-absa` pada server.

Melihat status:

```bash
docker compose --env-file .env -f docker-compose.v14.yml ps
```

Melihat log terbaru:

```bash
docker compose --env-file .env -f docker-compose.v14.yml logs --tail=200
```

Restart:

```bash
docker compose --env-file .env -f docker-compose.v14.yml restart
```

Menghentikan aplikasi tanpa menghapus data:

```bash
docker compose --env-file .env -f docker-compose.v14.yml down
```

Menjalankan kembali:

```bash
docker compose --env-file .env -f docker-compose.v14.yml up -d
```

Jangan menambahkan opsi `-v` pada perintah `down`. Opsi tersebut menghapus volume yang
berisi database history dan hasil batch.

## Tahap 8 — Keamanan setelah aplikasi berhasil

Password server yang pernah dikirim melalui pesan atau terlihat pada foto harus dianggap
sudah terekspos. Setelah memastikan deployment berhasil, ganti password:

**SERVER — Ubuntu**

```bash
passwd
```

Gunakan password baru yang kuat dan jangan kirimkan kembali melalui chat. Setelah itu,
minta pengelola infrastruktur memasang SSH key dan mempertimbangkan menonaktifkan login
SSH menggunakan password.

V14 masih berstatus **pre-production candidate**. Aspect, opinion, dan sentiment memakai
checkpoint terlatih. Relation dan taxonomy masih fallback, sehingga hasilnya tetap perlu
human review dan belum boleh menjadi satu-satunya dasar keputusan penting.

## Jika terjadi error

Jangan langsung mengubah banyak konfigurasi. Salin output dari tiga perintah berikut dan
kirimkan kepada pendamping teknis, tanpa menyertakan password atau isi `.env`:

```bash
docker compose --env-file .env -f docker-compose.v14.yml ps
docker compose --env-file .env -f docker-compose.v14.yml logs --tail=200 api
docker compose --env-file .env -f docker-compose.v14.yml logs --tail=100 web
```
