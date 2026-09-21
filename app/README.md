# V14 ABSA / Multi-Engine Development Workspace

Deployment khusus **V14 ABSA** tersedia di [DEPLOY_V14.md](DEPLOY_V14.md). Konfigurasi
tersebut hanya mengemas bundle V14 dan menjalankan web serta API melalui Docker Compose.

## Jalankan lokal

Backend membutuhkan Python 3.10+.

```powershell
cd backend
python -m pip install -r requirements.txt
$env:ABSA_ROOT_DIR=".."
python -m uvicorn app.main:app --reload --port 8000
```

Terminal kedua:

```powershell
cd frontend
npm install
npm run dev
```

Buka `http://localhost:5173`. API tersedia di `http://localhost:8000/docs`.

## Docker

```powershell
docker compose up --build
```

## Model

Inference Playground menyediakan empat pilihan engine:

- **V11** membaca bundle lokal `absa_v11_web_bundle` untuk Bahasa Indonesia.
- **V12** membaca bundle lokal `absa_v12_multilingual_bundle` untuk Indonesia, English, dan code-switch Indonesia-English.
- **V14** membaca bundle lokal `absa_v14_context_hardened_seed_42`. Adapter web memakai pemuatan komponen satu per satu pada mesin dengan RAM/VRAM terbatas. Bundle saat ini berisi model aspect, opinion, dan sentiment; relation serta taxonomy yang belum tersedia ditandai sebagai fallback/review, bukan diklaim sebagai keluaran model terlatih.
- **V15.1** membaca bundle lokal `absa_v15_1_production_fp16`. Tagger aspect/opinion V14 diteruskan ke head gabungan relation+sentiment dan taxonomy hierarkis domain→entity→issue. Keenam checkpoint dimuat streaming secara default. Bundle berstatus `candidate_unpromoted` dan belum memiliki Human GOLD, sehingga hasil selalu membawa penanda review dan tidak diklaim sebagai validasi produksi.

Secara default taxonomy V15 memakai kebijakan `best_effort`: label entity dan issue teratas tetap dikembalikan saat confidence rendah, dengan `taxonomy_low_confidence=true` dan routing human review. Untuk kembali ke perilaku aman `OTHER#ABSTAIN`, set `ABSA_V15_TAXONOMY_POLICY=strict`.

Pilihan UI diteruskan ke API melalui field `engine_version` (`v11`, `v12`, `v14`, atau `v15`). Endpoint `GET /api/inference/engines` menampilkan ketersediaan, komponen, dan status setiap bundle. Untuk menjaga penggunaan RAM/VRAM, backend hanya mengaktifkan satu engine pada satu waktu. V15 memakai pemuatan streaming kecuali mode dipaksa dengan environment variable `ABSA_MODEL_LOADING=resident`.

Tokenizer dan lima komponen model dimuat secara lokal menggunakan Hugging Face Transformers tanpa mengunduh model dari internet. Jika bundle tidak kompatibel atau dependency belum terpasang, sistem menggunakan fallback rule-based; respons inference selalu menyertakan `inference_mode` agar sumber hasil dapat diperiksa.

## Evaluasi

Mode saat ini `proxy_without_human_gold`. Human GOLD tidak wajib, tetapi metrik proxy tidak boleh dianggap sebagai validasi produksi.

## GPU

PyTorch memakai CUDA jika tersedia. CPU tetap dapat dipakai untuk inference ringan dan smoke test, tetapi training Transformer penuh membutuhkan GPU NVIDIA dengan CUDA yang sesuai.

## Catatan

Training saat ini masih berupa lifecycle runner MVP; integrasi inference bundle sudah disiapkan, sedangkan penggantian simulator training dengan fine-tuning multi-task V11 memerlukan label schema notebook V11 yang lengkap.

## Metadata enrichment

Endpoint lama `POST /api/inference/single` tetap kompatibel dengan payload `{"text": "..."}` dan field respons lama `text`/`results`. Payload juga menerima field opsional `customer_id`, `city`, dan `province` (alias input `review`, `kota`, dan `provinsi` juga diterima). Contoh:

```json
{
  "customer_id": "213",
  "city": "Bandung",
  "province": "Jawa Barat",
  "review": "aplikasi ini sangat bagus",
  "engine_version": "v12"
}
```

Respons menambahkan `raw_text`, `clean_review`, `customer`, `location`, `absa.aspects`, `model_version`, `warnings`, dan rincian timing. ID hanya diekstrak jika berasal dari field khusus atau memiliki penanda `no`, `id`, `customer`, atau `customer_id`; angka biasa tidak dianggap sebagai ID. Digit pertama dipetakan lewat satu konfigurasi: `1=VIP`, `2=SILVER`, `3=REGULER`, selain itu `UNKNOWN`. Field terstruktur selalu menang dan konflik dilaporkan sebagai warning.

Resolver lokasi menggunakan [master wilayah operasional](backend/app/data/indonesia_regions.json). File memuat seluruh 38 provinsi dan subset kota/kabupaten yang dipakai untuk operasional/test, dengan kode, nama kanonis, tipe, provinsi, dan alias. Penamaan mengikuti pola administrasi Kemendagri, tetapi versi bundle ini bukan salinan lengkap 514 kabupaten/kota; sinkronkan dengan master resmi terbaru sebelum cakupan nasional/regulasi. Exact match memperoleh skor dasar 0,42, konteks lokasi atau posisi metadata menaikkan skor, bentuk `kota/kabupaten` eksplisit mencapai confidence tinggi, dan threshold default adalah 0,80. Nama ganda seperti Bandung dan Serang dikembalikan sebagai `ambiguous`; pola opini seperti `sangat malang`, `seperti batu`, serta substring `menyerang` tidak dihapus dari teks ABSA.

## Batch CSV

UI memiliki mode **Batch CSV** dengan preview sebelum proses, progress, hasil per baris, ringkasan latency, cancel, dan download CSV/JSON. Alur API:

```text
POST /api/batch/upload                  multipart: file, engine_version, batch_size
POST /api/batch/{job_id}/start          header: X-Batch-Token
GET  /api/batch/{job_id}/status         header: X-Batch-Token
GET  /api/batch/{job_id}/results        header: X-Batch-Token, query: offset, limit
GET  /api/batch/{job_id}/download       header: X-Batch-Token, query: format=csv|json
POST /api/batch/{job_id}/cancel         header: X-Batch-Token
```

Upload mengembalikan `job_id`, token akses acak, metadata validasi, dan sepuluh baris preview. Token harus dirahasiakan dan dikirim kembali pada setiap operasi job. UI menyimpannya sementara agar polling dapat dilanjutkan setelah refresh pada proses backend yang sama.

Kolom review yang dikenali: `review`, `review_text`, `text`, `comment`, `komentar`, `ulasan`, dan `Isi Review`. Kolom metadata opsional: `customer_id/customer/id`, `city/kota/regency/kabupaten`, dan `province/provinsi`. CSV UTF-8/UTF-8-BOM, Windows-1252, atau Latin-1 dengan delimiter koma, titik koma, atau tab didukung. Format ekspor VOC yang membungkus seluruh record dan menambahkan `;;;` juga dideteksi serta dinormalisasi otomatis. File dibaca streaming; hasil ditulis sebagai JSONL dan endpoint hasil memakai pagination agar ribuan baris tidak dikumpulkan dalam satu respons. Baris kosong dilewati, kegagalan model diisolasi per baris, urutan asli dipertahankan, dan ekspor mengawali nilai `=`, `+`, `-`, atau `@` dengan apostrof untuk mencegah formula injection.

Job memakai worker thread per upload, sementara manager model memegang mutex sehingga checkpoint yang tidak thread-safe tetap dipanggil secara serial. `batch_size` saat ini mengatur ukuran unit kerja/API benchmark; inference engine V11/V12 yang ada masih mengeksekusi review secara serial. State job bersifat in-memory, hasil sementara disimpan di `storage/batch_jobs`, dan file upload mentah dihapus saat proses selesai/cancel. Restart backend menghapus kemampuan mengakses state job lama; production multi-instance sebaiknya mengganti store dengan Redis/database dan object storage berttl.

Konfigurasi environment:

```text
MAX_CSV_FILE_SIZE_MB=100
MAX_CSV_ROWS=100000
MAX_REVIEW_LENGTH=10000
DEFAULT_BATCH_SIZE=32
MAX_BATCH_SIZE=256
BATCH_TIMEOUT_SECONDS=1800
MAX_RETRY_PER_ROW=1
LOCATION_CONFIDENCE_THRESHOLD=0.80
```

Metrik memakai `perf_counter`: preprocessing, inference, postprocessing, end-to-end, rata-rata, median, p95, p99, min/max, total waktu, dan throughput (`successful_rows / total_processing_seconds`). Waktu upload tidak masuk ke inference. Cold start tidak dicampur dan ditandai `null`/`warmup_executed=false` pada job biasa.

## Test dan benchmark

```powershell
cd ..
$env:PYTHONPATH="backend"
python -m pytest backend/tests -q --basetemp=.pytest_run

python backend/benchmarks/generate_fixture.py backend/benchmarks/reviews_1000.csv --rows 1000
python backend/benchmarks/run_batch_benchmark.py backend/benchmarks/reviews_1000.csv --models v11 v12 --batch-sizes 1 8 16 32 64 --iterations 3
```

Generator mendukung 100, 1.000, 5.000, dan 10.000 baris. Benchmark melakukan warm-up terpisah, memakai dataset yang sama untuk setiap model, menjalankan beberapa iterasi, dan menghasilkan JSON berisi latency, throughput, jumlah sukses/gagal, device, serta perubahan RSS memory. Jangan membandingkan hasil dari hardware atau mode pemuatan model yang berbeda.
