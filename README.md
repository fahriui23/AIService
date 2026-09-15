# V14 ABSA

**Aspect-Based Sentiment Analysis untuk Bahasa Indonesia, Inggris, dan code-switch.**

Paket deployment khusus model V14 tersedia di
[`Ui test/DEPLOY_V14.md`](Ui%20test/DEPLOY_V14.md).

## Apa itu ABSA?

**Aspect-Based Sentiment Analysis (ABSA)** adalah metode dalam Natural Language Processing (NLP) yang digunakan untuk memahami sentimen secara lebih terperinci berdasarkan aspek tertentu yang dibahas dalam sebuah teks.

Analisis sentimen biasa umumnya hanya memberikan satu label untuk seluruh teks, misalnya **positif**, **negatif**, atau **netral**. Pendekatan tersebut kurang memadai ketika satu ulasan membahas beberapa hal dengan sentimen yang berbeda.

Contoh:

> Makanannya enak, tetapi pelayanannya lambat dan tempatnya cukup nyaman.

Analisis sentimen tingkat dokumen mungkin menyebut ulasan tersebut sebagai campuran atau netral. ABSA memecahnya menjadi informasi yang lebih spesifik:

| Aspek | Opini | Sentimen |
|---|---|---|
| makanan | enak | positif |
| pelayanan | lambat | negatif |
| tempat | cukup nyaman | positif |

Dengan demikian, ABSA tidak hanya menjawab **“bagaimana sentimen pengguna?”**, tetapi juga **“sentimen tersebut ditujukan kepada aspek apa dan didukung oleh opini yang mana?”**

## Mengapa ABSA diperlukan?

Sebuah ulasan sering mengandung lebih dari satu pendapat. Pelanggan dapat menyukai kualitas produk, tetapi tidak menyukai harga, proses pembayaran, layanan pelanggan, atau waktu tunggu. Jika seluruh ulasan diringkas menjadi satu label, informasi penting tersebut dapat hilang.

ABSA membantu organisasi untuk:

- menemukan aspek yang paling sering dipuji atau dikeluhkan;
- membedakan sentimen terhadap produk, layanan, harga, fasilitas, atau aspek lainnya;
- mengidentifikasi akar masalah secara lebih tepat;
- memantau perubahan pengalaman pelanggan dari waktu ke waktu;
- memprioritaskan perbaikan berdasarkan aspek dan tingkat keluhan;
- mengubah teks ulasan menjadi data terstruktur untuk dashboard dan laporan.

## Komponen utama ABSA

### 1. Aspect Term Extraction

**Aspect Term Extraction (ATE)** bertugas menemukan kata atau frasa yang menjadi objek pembicaraan.

Contoh:

> Koneksi internetnya cepat, tetapi aplikasinya sering error.

Aspek yang ditemukan:

- `koneksi internet`
- `aplikasi`

Aspek dapat bersifat eksplisit maupun implisit. Pada kalimat “terlalu mahal untuk kualitas seperti ini”, aspek harga dibicarakan secara implisit walaupun kata “harga” tidak muncul.

### 2. Opinion Term Extraction

**Opinion Term Extraction (OTE)** menemukan kata atau frasa yang menjadi bukti penilaian terhadap suatu aspek.

Pada contoh sebelumnya, opini yang ditemukan adalah:

- `cepat`
- `sering error`

Pemisahan aspek dan opini penting karena satu aspek dapat memiliki beberapa opini, sedangkan satu opini terkadang berkaitan dengan lebih dari satu aspek.

### 3. Aspect–Opinion Relation Extraction

Tahap ini menentukan hubungan antara aspek dan opini yang tepat.

Contoh:

> Kopinya enak, tetapi pelayanannya lambat.

Relasi yang benar:

- `kopi` ↔ `enak`
- `pelayanan` ↔ `lambat`

Model tidak boleh memasangkan `kopi` dengan `lambat` atau `pelayanan` dengan `enak`. Relasi menjadi semakin sulit pada kalimat panjang, banyak klausa, atau ulasan yang membahas beberapa aspek sekaligus.

### 4. Aspect Sentiment Classification

**Aspect Sentiment Classification (ASC)** memberikan label sentimen untuk setiap pasangan aspek dan opini.

Label yang umum digunakan:

- **positive** — menunjukkan kepuasan atau penilaian positif;
- **negative** — menunjukkan keluhan atau penilaian negatif;
- **neutral** — menunjukkan informasi faktual atau penilaian yang tidak jelas positif maupun negatif.

Beberapa sistem dapat menambahkan label seperti `conflict`, tetapi penggunaan tiga kelas lebih umum dan lebih mudah diinterpretasikan.

### 5. Aspect Category dan Taxonomy Classification

Istilah aspek yang berbeda dapat merujuk pada kategori masalah yang sama. Karena itu, hasil ABSA sering dipetakan ke dalam taksonomi atau kategori standar.

Contoh:

| Istilah aspek | Kategori |
|---|---|
| antrean, waktu tunggu, menunggu | Waiting Time |
| staf, kasir, petugas | Staff Service |
| tarif, biaya, harga | Price & Value |
| aplikasi, situs, sistem | Digital Service |

Taksonomi juga dapat dibuat secara hierarkis:

```text
Domain
└── Entity
    └── Issue
```

Contoh:

```text
telecommunication
└── network
    └── connection_stability
```

Struktur hierarkis membuat hasil lebih konsisten dan memudahkan agregasi lintas produk atau layanan.

## Bentuk keluaran ABSA

Keluaran ABSA biasanya berbentuk triplet atau tuple:

```text
(aspect, opinion, sentiment)
```

Contoh:

```text
("internet", "cukup cepat", "positive")
("koneksi", "sering terputus", "negative")
```

Dalam sistem yang lebih lengkap, hasil dapat memuat relasi, posisi karakter, confidence, dan taksonomi:

```json
{
  "aspect": "koneksi",
  "opinion": "sering terputus",
  "sentiment": "negative",
  "relation": true,
  "aspect_span": [42, 49],
  "opinion_span": [50, 65],
  "taxonomy": "network#connection_stability",
  "confidence": 0.91,
  "needs_human_review": false
}
```

Posisi karakter atau **span** menunjukkan lokasi aspek dan opini pada teks asli. Span penting untuk audit, penyorotan teks, dan pemeriksaan apakah hasil model benar-benar didukung oleh bukti yang tersedia.

## Alur kerja ABSA

Secara umum, sistem ABSA menjalankan tahapan berikut:

```text
Teks ulasan
    ↓
Pembersihan dan normalisasi teks
    ↓
Ekstraksi aspek dan opini
    ↓
Pembentukan pasangan kandidat
    ↓
Klasifikasi relasi aspek–opini
    ↓
Klasifikasi sentimen per aspek
    ↓
Pemetaan kategori atau taksonomi
    ↓
Confidence, validasi, dan hasil terstruktur
```

Normalisasi harus dilakukan dengan hati-hati. Mengubah teks dapat menyebabkan posisi karakter bergeser. Sistem yang memerlukan span sebaiknya mempertahankan teks asli dan menyimpan pemetaan antara teks asli dengan teks yang telah dinormalisasi.

## Pendekatan pembangunan ABSA

### Pendekatan berbasis aturan

Pendekatan ini menggunakan kamus aspek, kamus sentimen, pola bahasa, dan aturan jarak antarkata.

Kelebihan:

- mudah dijelaskan dan diaudit;
- tidak membutuhkan dataset besar;
- sesuai untuk domain yang sempit dan stabil.

Keterbatasan:

- sulit menangani variasi bahasa;
- sensitif terhadap slang, negasi, dan konteks;
- membutuhkan pemeliharaan aturan secara manual.

### Machine Learning klasik

Model seperti Logistic Regression, Support Vector Machine, atau Conditional Random Field dapat menggunakan fitur kata, n-gram, part-of-speech, dan dependency parsing.

Pendekatan ini lebih fleksibel daripada aturan, tetapi kualitasnya sangat bergantung pada rekayasa fitur dan data berlabel.

### Deep Learning dan Transformer

Model Transformer seperti BERT, IndoBERT, atau XLM-R dapat mempelajari konteks kata secara lebih baik. Sistem dapat dibangun sebagai beberapa model terpisah atau sebagai model multi-task yang mempelajari ekstraksi, relasi, sentimen, dan taksonomi secara bersama-sama.

Kelebihan:

- lebih baik dalam memahami konteks;
- mampu menangani variasi bahasa yang lebih luas;
- dapat digunakan untuk skenario multilingual dan code-switching.

Keterbatasan:

- membutuhkan data dan sumber daya komputasi lebih besar;
- confidence model belum tentu terkalibrasi;
- hasil tetap dapat salah pada domain atau gaya bahasa yang berbeda dari data pelatihan.

## Tantangan utama ABSA

### Banyak aspek dalam satu kalimat

Model harus menemukan semua aspek tanpa mencampurkan opini masing-masing.

### Negasi

Kalimat seperti “layanannya tidak buruk” tidak dapat dianalisis hanya berdasarkan kata “buruk”. Cakupan negasi harus diperhitungkan.

### Kontras

Kata seperti “tetapi”, “namun”, dan “walaupun” sering menandai perubahan sentimen antarklausa.

### Opini implisit

Kalimat “saya menunggu dua jam” dapat menunjukkan sentimen negatif meskipun tidak memiliki kata sentimen eksplisit.

### Sarkasme dan ironi

Kalimat “bagus sekali, baru dipakai sehari sudah rusak” menggunakan kata positif untuk menyampaikan keluhan.

### Slang dan code-switching

Ulasan Indonesia sering mencampurkan bahasa Indonesia, Inggris, singkatan, dan bahasa informal, misalnya “apps-nya smooth tapi CS-nya slow response”.

### Aspek dan opini bertumpuk

Pada beberapa frasa, bagian teks yang dianggap aspek dapat bertumpang tindih dengan opini. Sistem harus memiliki kebijakan yang jelas: memperbaiki span, memilih salah satu interpretasi, atau mengirim hasil untuk peninjauan manusia.

### Perbedaan domain

Kata yang sama dapat memiliki arti kategori berbeda pada domain yang berbeda. “Layanan” dalam restoran, rumah sakit, asuransi, dan telekomunikasi membutuhkan taksonomi yang berbeda.

## Data dan anotasi

Dataset ABSA idealnya menyimpan:

- teks asli;
- bahasa dan domain;
- aspect term beserta span;
- opinion term beserta span;
- relasi aspect–opinion;
- label sentimen;
- kategori atau taksonomi;
- status dan sumber anotasi;
- tingkat confidence atau kebutuhan human review.

Contoh sederhana:

```json
{
  "text": "Kopinya enak tetapi antreannya lama.",
  "annotations": [
    {
      "aspect": "Kopinya",
      "opinion": "enak",
      "sentiment": "positive"
    },
    {
      "aspect": "antreannya",
      "opinion": "lama",
      "sentiment": "negative"
    }
  ]
}
```

Kualitas anotasi sangat menentukan kualitas model. Pedoman anotasi harus menjelaskan penanganan aspek implisit, opini tanpa aspek, konflik sentimen, negasi, koordinasi, dan batas span.

## Evaluasi ABSA

Setiap subtask sebaiknya dievaluasi secara terpisah.

| Subtask | Metrik umum |
|---|---|
| Ekstraksi aspek | Precision, Recall, Exact/Overlap F1 |
| Ekstraksi opini | Precision, Recall, Exact/Overlap F1 |
| Relasi aspek–opini | Precision, Recall, Macro F1 |
| Sentimen | Accuracy, Macro F1, F1 per kelas |
| Taksonomi | Accuracy, Macro F1, Hierarchical F1 |
| Kalibrasi confidence | ECE, Brier Score |
| Hasil end-to-end | Triplet F1 atau Tuple F1 |

**Macro F1** penting ketika distribusi label tidak seimbang karena setiap kelas memperoleh bobot yang sama. Evaluasi end-to-end juga diperlukan; skor tinggi pada ekstraksi atau sentimen secara terpisah belum tentu menghasilkan pasangan akhir yang benar.

Dataset evaluasi sebaiknya merupakan **Human GOLD** yang terpisah dari data pelatihan dan data pemilihan model. Data proxy, silver, atau synthetic berguna untuk pengembangan, tetapi tidak boleh dianggap sebagai pengganti validasi manusia untuk klaim produksi.

## Confidence dan human review

Nilai confidence menunjukkan tingkat keyakinan model, bukan jaminan bahwa prediksi benar. Sistem produksi dapat menggunakan beberapa kebijakan:

- menerima otomatis hasil dengan confidence tinggi;
- menandai hasil confidence sedang untuk sampling audit;
- mengirim hasil confidence rendah atau konflik ke human review;
- melakukan abstain ketika tidak ada prediksi yang cukup aman.

Jika sistem tetap menampilkan prediksi confidence rendah, hasil tersebut harus diberi penanda **best effort** agar pengguna tidak menganggapnya sebagai fakta pasti.

## Penerapan ABSA

ABSA dapat digunakan untuk menganalisis:

- ulasan produk dan marketplace;
- ulasan restoran dan hotel;
- layanan kesehatan;
- layanan publik;
- telekomunikasi dan internet;
- perbankan dan asuransi;
- survei kepuasan pelanggan;
- komentar media sosial;
- tiket keluhan dan percakapan customer service.

Hasilnya dapat digunakan untuk dashboard tren, peringkat masalah, ringkasan suara pelanggan, deteksi penurunan kualitas layanan, dan penyusunan prioritas operasional.

## Batasan

ABSA tidak sepenuhnya memahami maksud manusia. Prediksi dapat terpengaruh oleh kualitas data, bias anotasi, perubahan bahasa, domain baru, teks yang ambigu, dan konteks yang tidak tersedia.

Karena itu, hasil ABSA sebaiknya:

- diperlakukan sebagai alat bantu analisis;
- disertai confidence dan metadata model;
- diaudit secara berkala menggunakan Human GOLD;
- dipantau untuk mendeteksi data drift;
- tidak digunakan sebagai satu-satunya dasar keputusan berisiko tinggi.

## Ringkasan

ABSA mengubah ulasan bebas menjadi informasi terstruktur pada tingkat aspek. Sistem yang lengkap tidak hanya menentukan sentimen, tetapi juga menemukan aspek, menemukan bukti opini, memasangkan keduanya, menentukan polaritas, dan memetakan hasil ke taksonomi yang relevan.

Tujuan akhirnya adalah menjawab tiga pertanyaan utama:

1. **Apa yang sedang dibicarakan?** — aspek.
2. **Apa pendapat pengguna tentang hal tersebut?** — opini dan sentimen.
3. **Masalah atau kategori apa yang diwakili?** — taksonomi.

Dengan informasi tersebut, organisasi dapat memahami pengalaman pengguna secara lebih rinci dan mengambil tindakan berdasarkan masalah yang benar-benar disebutkan dalam teks.
