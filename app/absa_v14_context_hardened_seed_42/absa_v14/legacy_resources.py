import re, unicodedata
SEED=42
CONFIG={"training_languages":["id","en"],"experimental_languages":["ja","ar"],"coordination_split_min_taxonomy_conf":.60}
GMAPS_SLANG_MAP = {'yg': 'yang', 'yng': 'yang', 'bgt': 'banget', 'bngt': 'banget', 'bgtt': 'banget', 'bgttt': 'banget', 'pdhl': 'padahal', 'pdahal': 'padahal', 'tp': 'tapi', 'tpi': 'tapi', 'krn': 'karena', 'karna': 'karena', 'dgn': 'dengan', 'dg': 'dengan', 'dengn': 'dengan', 'dri': 'dari', 'sm': 'sama', 'sma': 'sama', 'sama2': 'sama-sama', 'sampe': 'sampai', 'ampe': 'sampai', 'sampek': 'sampai', 'nyampe': 'sampai', 'gk': 'tidak', 'ga': 'tidak', 'gak': 'tidak', 'gx': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak', 'enggak': 'tidak', 'engga': 'tidak', 'kagak': 'tidak', 'tdk': 'tidak', 'blm': 'belum', 'blom': 'belum', 'belom': 'belum', 'udh': 'sudah', 'udah': 'sudah', 'sdh': 'sudah', 'dah': 'sudah', 'klo': 'kalau', 'kl': 'kalau', 'kalo': 'kalau', 'jd': 'jadi', 'jdi': 'jadi', 'trs': 'terus', 'trus': 'terus', 'trz': 'terus', 'cm': 'cuma', 'cman': 'cuma', 'cuman': 'cuma', 'gw': 'saya', 'gue': 'saya', 'gua': 'saya', 'sy': 'saya', 'sya': 'saya', 'aq': 'saya', 'kmrn': 'kemarin', 'kmaren': 'kemarin', 'kemaren': 'kemarin', 'skrg': 'sekarang', 'skr': 'sekarang', 'bs': 'bisa', 'bsa': 'bisa', 'hrs': 'harus', 'hrus': 'harus', 'emg': 'memang', 'emang': 'memang', 'bkn': 'bukan', 'org': 'orang', 'orng': 'orang', 'tmpt': 'tempat', 'tempet': 'tempat', 'jgn': 'jangan', 'jng': 'jangan', 'knp': 'kenapa', 'kenap': 'kenapa', 'gmn': 'bagaimana', 'gmna': 'bagaimana', 'gimana': 'bagaimana', 'bgmn': 'bagaimana', 'utk': 'untuk', 'u/': 'untuk', 'biar': 'supaya', 'spy': 'supaya', 'kyk': 'seperti', 'kayak': 'seperti', 'kek': 'seperti', 'smoga': 'semoga', 'moga': 'semoga', 'msh': 'masih', 'msih': 'masih', 'sblm': 'sebelum', 'seblm': 'sebelum', 'stlh': 'setelah', 'lg': 'lagi', 'lgi': 'lagi', 'dlu': 'dulu', 'dl': 'dulu', 'ntar': 'nanti', 'ntr': 'nanti', 'entar': 'nanti', 'bsk': 'besok', 'mlm': 'malam', 'malem': 'malam', 'pgi': 'pagi', 'org2': 'orang-orang', 'anak2': 'anak-anak', 'ibu2': 'ibu-ibu', 'bapak2': 'bapak-bapak', 'mba': 'mbak', 'mb': 'mbak', 'pak': 'bapak', 'bpk': 'bapak', 'bu': 'ibu', 'kak': 'kakak', 'min': 'admin', 'mimin': 'admin', 'minn': 'admin', 'adminn': 'admin', 'thx': 'terima kasih', 'thanks': 'terima kasih', 'thankyou': 'terima kasih', 'tq': 'terima kasih', 'tks': 'terima kasih', 'mksh': 'terima kasih', 'makasih': 'terima kasih', 'makasi': 'terima kasih', 'trims': 'terima kasih', 'trimakasih': 'terima kasih', 'pls': 'tolong', 'plis': 'tolong', 'please': 'tolong', 'sorry': 'maaf', 'sory': 'maaf', 'sori': 'maaf', 'maap': 'maaf', 'maf': 'maaf', 'bgs': 'bagus', 'bgus': 'bagus', 'mantab': 'mantap', 'mantabb': 'mantap', 'mantul': 'mantap', 'mantep': 'mantap', 'mantapp': 'mantap', 'toppp': 'bagus', 'topmarkotop': 'sangat bagus', 'recommended': 'direkomendasikan', 'recommend': 'direkomendasikan', 'rekomen': 'direkomendasikan', 'recomended': 'direkomendasikan', 'recomend': 'direkomendasikan', 'rekomended': 'direkomendasikan', 'worthit': 'sepadan', 'worth-it': 'sepadan', 'okee': 'baik', 'okey': 'baik', 'okay': 'baik', 'goodd': 'bagus', 'goood': 'bagus', 'great': 'sangat bagus', 'awesome': 'sangat bagus', 'amazing': 'sangat bagus', 'excellent': 'sangat bagus', 'perfect': 'sempurna', 'mantep': 'mantap', 'jos': 'bagus', 'joss': 'bagus', 'sip': 'bagus', 'sipp': 'bagus', 'kereen': 'bagus', 'kerenn': 'bagus', 'love': 'suka', 'luv': 'suka', 'gercep': 'cepat', 'satset': 'cepat', 'murmer': 'murah', 'bad': 'buruk', 'badd': 'buruk', 'verybad': 'sangat buruk', 'zonk': 'mengecewakan', 'zonkk': 'mengecewakan', 'lelet': 'lambat', 'lemot': 'lambat', 'jutek': 'tidak ramah', 'judes': 'tidak ramah', 'ketus': 'tidak ramah', 'songong': 'tidak ramah', 'cuek': 'tidak responsif', 'ribet': 'rumit', 'rempong': 'rumit', 'jorok': 'kotor', 'jorokk': 'kotor', 'overprice': 'terlalu mahal', 'overpriced': 'terlalu mahal', 'gaenak': 'tidak enak', 'gakenak': 'tidak enak', 'notrecommended': 'tidak direkomendasikan', 'pelayananya': 'pelayanannya', 'pelayannnya': 'pelayanannya', 'pelayannya': 'pelayanannya', 'pelayannannya': 'pelayanannya', 'pelayananannya': 'pelayanannya', 'pelayann': 'pelayanan', 'layanananya': 'layanannya', 'service': 'pelayanan', 'servis': 'pelayanan', 'servicenya': 'pelayanannya', 'servisnya': 'pelayanannya', 'staff': 'staf', 'staffnya': 'stafnya', 'petugass': 'petugas', 'pegawainyaa': 'pegawainya', 'slowrespon': 'lambat merespons', 'slowresponse': 'lambat merespons', 'slowresp': 'lambat merespons', 'fastrespon': 'cepat merespons', 'fastresponse': 'cepat merespons', 'responsive': 'responsif', 'helpfull': 'membantu', 'helpful': 'membantu', 'friendly': 'ramah', 'unfriendly': 'tidak ramah', 'antri': 'antre', 'antrian': 'antrean', 'antriannya': 'antreannya', 'antriann': 'antrean', 'ngantri': 'mengantre', 'ngantre': 'mengantre', 'queue': 'antrean', 'waiting': 'menunggu', 'wait': 'menunggu', 'nunggu': 'menunggu', 'nungguin': 'menunggu', 'kelamaan': 'terlalu lama', 'parkiran': 'area parkir', 'parkirannya': 'area parkirnya', 'parkiranya': 'area parkirnya', 'parking': 'parkir', 'vallet': 'valet', 'maps': 'peta', 'gmap': 'google maps', 'gmaps': 'google maps', 'nyasar': 'tersesat', 'kesasar': 'tersesat', 'acces': 'akses', 'access': 'akses', 'jln': 'jalan', 'jl': 'jalan', 'facilities': 'fasilitas', 'facility': 'fasilitas', 'wc': 'toilet', 'toilet nya': 'toiletnya', 'musholla': 'musala', 'mushola': 'musala', 'musholah': 'musala', 'wi-fi': 'wifi', 'wifi nya': 'wifi', 'clean': 'bersih', 'cleaning': 'kebersihan', 'debuan': 'berdebu', 'pesing': 'bau urine', 'coffeshop': 'coffee shop', 'coffeeshop': 'coffee shop', 'coffee-shop': 'coffee shop', 'coffe shop': 'coffee shop', 'cofee shop': 'coffee shop', 'caffe': 'kafe', 'café': 'kafe', 'resto': 'restoran', 'restaurant': 'restoran', 'restauran': 'restoran', 'makananya': 'makanannya', 'mknannya': 'makanannya', 'makannannya': 'makanannya', 'mkanan': 'makanan', 'mknan': 'makanan', 'minumanya': 'minumannya', 'mnman': 'minuman', 'food': 'makanan', 'foods': 'makanan', 'drink': 'minuman', 'drinks': 'minuman', 'taste': 'rasa', 'yummy': 'enak', 'yumm': 'enak', 'delicious': 'enak', 'coffee': 'kopi', 'cappucino': 'cappuccino', 'capucino': 'cappuccino', 'expresso': 'espresso', 'espreso': 'espresso', 'cashier': 'kasir', 'waiter': 'pramusaji', 'waitress': 'pramusaji', 'orderan': 'pesanan', 'pesen': 'pesan', 'mesen': 'memesan', 'pesenan': 'pesanan', 'takeaway': 'bawa pulang', 'take-away': 'bawa pulang', 'dinein': 'makan di tempat', 'dine-in': 'makan di tempat', 'ambience': 'suasana', 'ambiance': 'suasana', 'vibes': 'suasana', 'vibe': 'suasana', 'cozy': 'nyaman', 'comfy': 'nyaman', 'instagramable': 'instagrammable', 'aesthetic': 'estetis', 'aestetik': 'estetis', 'estetik': 'estetis', 'nongki': 'nongkrong', 'wfc': 'work from cafe', 'smokingarea': 'area merokok', 'nosmoking': 'dilarang merokok', 'outdor': 'outdoor', 'indor': 'indoor', 'portion': 'porsi', 'price': 'harga', 'hrg': 'harga', 'rmh sakit': 'rumah sakit', 'puskes': 'puskesmas', 'pusk': 'puskesmas', 'clinic': 'klinik', 'nurse': 'perawat', 'nakes': 'tenaga kesehatan', 'apotik': 'apotek', 'apotekk': 'apotek', 'pharmacy': 'apotek', 'laborat': 'laboratorium', 'xray': 'rontgen', 'x-ray': 'rontgen', 'medicalcheckup': 'pemeriksaan kesehatan', 'medical checkup': 'pemeriksaan kesehatan', 'checkup': 'pemeriksaan', 'rawatinap': 'rawat inap', 'rawatjalan': 'rawat jalan', 'ranap': 'rawat inap', 'rajal': 'rawat jalan', 'diagnosa': 'diagnosis', 'registration': 'pendaftaran', 'pemda': 'pemerintah daerah', 'pemkot': 'pemerintah kota', 'pemkab': 'pemerintah kabupaten', 'pemprov': 'pemerintah provinsi', 'dukcapil': 'dinas kependudukan dan pencatatan sipil', 'disdukcapil': 'dinas kependudukan dan pencatatan sipil', 'disduk': 'dinas kependudukan', 'dinkes': 'dinas kesehatan', 'disnaker': 'dinas tenaga kerja', 'dishub': 'dinas perhubungan', 'dinsos': 'dinas sosial', 'disdik': 'dinas pendidikan', 'ektp': 'KTP elektronik', 'e-ktp': 'KTP elektronik', 'e ktp': 'KTP elektronik', 'passport': 'paspor', 'ijin': 'izin', 'fotocopy': 'fotokopi', 'foto copy': 'fotokopi', 'mall pelayanan publik': 'mal pelayanan publik', 'banking': 'perbankan', 'custservice': 'layanan pelanggan', 'cust service': 'layanan pelanggan', 'customer-service': 'layanan pelanggan', 'mbanking': 'mobile banking', 'm-banking': 'mobile banking', 'm banking': 'mobile banking', 'mobilebanking': 'mobile banking', 'ibanking': 'internet banking', 'i-banking': 'internet banking', 'internetbanking': 'internet banking', 'creditcard': 'kartu kredit', 'credit card': 'kartu kredit', 'adminfee': 'biaya administrasi', 'admin fee': 'biaya administrasi', 'bansos': 'bantuan sosial', 'blt': 'bantuan langsung tunai', 'sembako': 'bantuan sembako', 'difabel': 'penyandang disabilitas', 'panti jompo': 'panti lansia', 'sekolahan': 'sekolah', 'univ': 'universitas', 'university': 'universitas', 'perpus': 'perpustakaan', 'library': 'perpustakaan', 'guesthouse': 'penginapan', 'guest house': 'penginapan', 'homestay': 'penginapan', 'room': 'kamar', 'receptionist': 'resepsionis', 'reception': 'resepsionis', 'checkin': 'check-in', 'check in': 'check-in', 'checkout': 'check-out', 'check out': 'check-out', 'breakfast': 'sarapan', 'pool': 'kolam renang', 'swimmingpool': 'kolam renang', 'store': 'toko', 'shop': 'toko', 'mall': 'pusat perbelanjaan', 'product': 'produk', 'stock': 'stok', 'discount': 'diskon', 'disc': 'diskon', 'promotion': 'promosi', 'service mobil': 'servis mobil', 'service motor': 'servis motor', 'mechanic': 'mekanik', 'sparepart': 'suku cadang', 'spare part': 'suku cadang', 'onderdil': 'suku cadang', 'carwash': 'pencucian mobil', 'car wash': 'pencucian mobil', 'station': 'stasiun', 'commuterline': 'kereta komuter', 'commuter line': 'kereta komuter', 'ojol': 'ojek online', 'ojekonline': 'ojek online', 'driver': 'pengemudi', 'supir': 'pengemudi', 'ticket': 'tiket', 'view': 'pemandangan', 'scenery': 'pemandangan', 'photo': 'foto', 'familyfriendly': 'ramah keluarga', 'family friendly': 'ramah keluarga', 'kidsfriendly': 'ramah anak', 'kids friendly': 'ramah anak', 'kidfriendly': 'ramah anak', 'app': 'aplikasi', 'apps': 'aplikasi', 'apk': 'aplikasi', 'web': 'situs web', 'system': 'sistem', 'eror': 'error', 'errorr': 'error', 'err': 'error', 'bug': 'gangguan sistem', 'bugs': 'gangguan sistem', 'server down': 'server tidak dapat diakses', 'loading lama': 'lambat memuat', 'log in': 'masuk', 'log out': 'keluar', 'notif': 'notifikasi', 'notification': 'notifikasi', 'callcenter': 'call center', 'custcare': 'layanan pelanggan', 'wa': 'WhatsApp', 'w.a': 'WhatsApp', 'telp': 'telepon', 'tlp': 'telepon', 'telfon': 'telepon', 'telefon': 'telepon', 'telpon': 'telepon', 'bales': 'balas', 'dibales': 'dibalas', 'komplain': 'keluhan', 'complain': 'keluhan', 'complaint': 'keluhan', 'komplenan': 'keluhan', 'kwalitas': 'kualitas', 'aktifitas': 'aktivitas', 'resiko': 'risiko', 'praktek': 'praktik', 'sekedar': 'sekadar', 'rapih': 'rapi', 'nafas': 'napas', 'jaman': 'zaman', 'silahkan': 'silakan', 'nomer': 'nomor'}

GMAPS_DOMAIN_SLANG_MAP = {'food_beverage': {'cs': 'coffee shop'}, 'healthcare': {'dr': 'dokter', 'drg': 'dokter gigi', 'rs': 'rumah sakit', 'rsu': 'rumah sakit umum', 'rsud': 'rumah sakit umum daerah', 'pkm': 'puskesmas', 'igd': 'instalasi gawat darurat', 'ugd': 'unit gawat darurat', 'lab': 'laboratorium', 'mcu': 'pemeriksaan kesehatan', 'bpjs': 'BPJS', 'jkn': 'JKN', 'kis': 'KIS'}, 'government': {'kk': 'kartu keluarga', 'ktp': 'kartu tanda penduduk', 'mpp': 'mal pelayanan publik', 'sim': 'surat izin mengemudi', 'stnk': 'surat tanda nomor kendaraan', 'skck': 'surat keterangan catatan kepolisian', 'fc': 'fotokopi'}, 'banking': {'cs': 'customer service', 'tf': 'transfer', 'rek': 'rekening', 'cc': 'kartu kredit', 'kpr': 'kredit pemilikan rumah', 'atm': 'ATM', 'qris': 'QRIS'}, 'social_service': {'pm': 'penerima manfaat', 'pkh': 'program keluarga harapan', 'bpnt': 'bantuan pangan non tunai', 'dtks': 'data terpadu kesejahteraan sosial', 'pbi': 'penerima bantuan iuran', 'kip': 'kartu indonesia pintar', 'kks': 'kartu keluarga sejahtera'}, 'transportation': {'tj': 'TransJakarta', 'krl': 'kereta rel listrik', 'mrt': 'MRT', 'lrt': 'LRT'}, 'education': {'tu': 'tata usaha', 'sd': 'sekolah dasar', 'smp': 'sekolah menengah pertama', 'sma': 'sekolah menengah atas', 'smk': 'sekolah menengah kejuruan'}}

DOMAIN_HINTS = {'food_beverage': ['coffee shop', 'kafe', 'restoran', 'kopi', 'barista', 'menu', 'makanan', 'minuman', 'kasir'], 'healthcare': ['rumah sakit', 'puskesmas', 'klinik', 'dokter', 'perawat', 'farmasi', 'apotek', 'bpjs', 'igd', 'poli'], 'government': ['dukcapil', 'kelurahan', 'kecamatan', 'pemkot', 'pemkab', 'pemprov', 'perizinan', 'ktp', 'paspor', 'samsat'], 'banking': ['bank', 'teller', 'rekening', 'atm', 'mobile banking', 'transfer', 'kredit', 'tabungan'], 'social_service': ['dinas sosial', 'bansos', 'pkh', 'dtks', 'pbi', 'bantuan sosial', 'penerima manfaat'], 'education': ['sekolah', 'kampus', 'universitas', 'guru', 'dosen', 'tata usaha', 'perpustakaan'], 'transportation': ['stasiun', 'terminal', 'halte', 'kereta', 'bus', 'transjakarta', 'mrt', 'lrt', 'ojek']}

def resolve_review_domain(text, explicit_domain=None):
    if explicit_domain and explicit_domain in GMAPS_DOMAIN_SLANG_MAP:
        return explicit_domain
    low = str(text).lower()
    scores = {d: sum((1 for term in terms if term in low)) for d, terms in DOMAIN_HINTS.items()}
    if not scores or max(scores.values()) == 0:
        return None
    return max(scores, key=scores.get)

WORD_OR_GAP_PATTERN = re.compile('\\w+|[^\\w]+', re.UNICODE)

# V14.1 layered, reversible normalizer. Raw text and raw offsets remain authoritative.
ID_SLANG_MAP = dict(GMAPS_SLANG_MAP)
EN_SLANG_MAP = {'luv':'love','gr8':'great','gud':'good','thx':'thanks','pls':'please','plz':'please','tho':'though','u':'you','ur':'your','b4':'before'}
TYPO_CANDIDATES_ID = {'kmar':'kamar','brsih':'bersih','servisnyaa':'servisnya','pelayannnya':'pelayanannya','antriann':'antrean'}
TYPO_CANDIDATES_EN = {'freindly':'friendly','reciept':'receipt','servce':'service','restarant':'restaurant'}
NEGATION_MAP = {'ga':'tidak','gak':'tidak','nggak':'tidak','ngga':'tidak','tdk':'tidak','never':'never','not':'not','bukan':'bukan','tidak':'tidak'}
INTENSIFIER_MAP = {'bgt':'banget','bngt':'banget','bgtt':'banget','sangat':'sangat','banget':'banget','super':'super','very':'very','too':'too'}
EMOJI_SENTIMENT_MAP = {'👍':'[EMOJI_POS]','❤️':'[EMOJI_POS]','❤':'[EMOJI_POS]','😍':'[EMOJI_POS]','😊':'[EMOJI_POS]','😁':'[EMOJI_POS]','👎':'[EMOJI_NEG]','😡':'[EMOJI_NEG]','😠':'[EMOJI_NEG]','😞':'[EMOJI_NEG]','😢':'[EMOJI_NEG]','🤮':'[EMOJI_NEG]','😐':'[EMOJI_NEU]','🤔':'[EMOJI_NEU]'}
UNIT_NORMALIZATION_MAP = {'mnt':'menit','min':'menit','dtk':'detik','rb':'ribu','k':'ribu','km':'km','meter':'meter','jam':'jam'}
DOMAIN_ALIAS_MAP = dict(GMAPS_DOMAIN_SLANG_MAP)
ABBREVIATION_MAP = {'w/':'with','w/o':'without'}
CODE_SWITCH_SUFFIX_RULES = {'nya':'possessive_or_definite_id','ku':'possessive_first_person_id','mu':'possessive_second_person_id'}
PROTECTED_TOKENS = {'[ASP]','[/ASP]','[OPN]','[/OPN]','[DOC]','[/DOC]','[CTX]','[/CTX]','[EMOJI_POS]','[EMOJI_NEG]','[EMOJI_NEU]'}
V12_EMOJI_TOKEN_MAP = EMOJI_SENTIMENT_MAP
V12_EN_SAFE_SLANG = EN_SLANG_MAP
V12_ID_MARKERS = {'yang','dan','tapi','tetapi','tidak','nggak','gak','ga','sangat','banget','pelayanan','makanan','minuman','harga','tempat','ramah','lambat','enak','mahal','murah','parkir','lokasi','saya','kami'}
V12_EN_MARKERS = {'the','and','but','not','very','service','food','drink','price','place','staff','friendly','slow','good','bad','great','expensive','cheap','parking','location','was','were','is','are','my','our'}
V12_TOKEN_RE = re.compile(r"\b[\w'-]+\b",re.UNICODE)

def infer_language_metadata_v12(text,declared_language=None):
    raw=str(text);matches=list(V12_TOKEN_RE.finditer(raw.casefold()))
    ids=[m for m in matches if m.group(0) in V12_ID_MARKERS];ens=[m for m in matches if m.group(0) in V12_EN_MARKERS]
    suffix=list(re.finditer(r'\b[A-Za-z]+-(?:nya|ku|mu)\b',raw,re.I))
    spans=[{'start':m.start(),'end':m.end(),'language':'id','method':'marker'} for m in ids]+[{'start':m.start(),'end':m.end(),'language':'en','method':'marker'} for m in ens]+[{'start':m.start(),'end':m.end(),'language':'id-en','method':'mixed_suffix'} for m in suffix]
    declared=str(declared_language or '').casefold();detector_label=None;detector_confidence=0.0
    if ids and ens or suffix:label='id-en';confidence=min(.95,.65+.04*min(len(ids),len(ens))+.1*bool(suffix))
    elif declared in {'id','en','id-en','ja','ar'}:label=declared;confidence=1.0
    elif len(ids)>len(ens):label='id';confidence=min(.95,.55+.06*len(ids))
    elif len(ens)>len(ids):label='en';confidence=min(.95,.55+.06*len(ens))
    else:
        label='unknown';confidence=.3
        if len(raw.strip())>=12:
            try:
                from langdetect import detect_langs
                c=detect_langs(raw)
                if c:detector_label=c[0].lang;detector_confidence=float(c[0].prob);label=detector_label if detector_label in {'id','en'} else label;confidence=detector_confidence
            except Exception:pass
    return {'language':label,'confidence':round(float(confidence),4),'is_code_switched':label=='id-en','language_spans':sorted(spans,key=lambda x:(x['start'],x['end'])),'detector_label':detector_label,'detector_confidence':round(float(detector_confidence),4),'routing_policy':'metadata_only_xlmr_receives_full_text'}

EMOJI_PATTERN = r'[\U0001F300-\U0001FAFF\u2600-\u27BF](?:[\uFE0F\U0001F3FB-\U0001F3FF]|\u200D[\U0001F300-\U0001FAFF\u2600-\u27BF][\uFE0F\U0001F3FB-\U0001F3FF]*)*'
LAYER_TOKEN_RE = re.compile(
    r'\[/?(?:ASP|OPN|DOC|CTX)\]|\[(?:EMOJI_POS|EMOJI_NEG|EMOJI_NEU)\]'
    r'|https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}'
    r'|#[\w_]+|\+?\d[\d() .-]{7,}\d|\b(?:w/|w/o)\b'
    r'|\b\d+(?:[.,]\d+)?(?:mnt|menit|jam|dtk|detik|km|meter|rb|ribu|k)\b'
    r'|\b[\w]+(?:-[\w]+)*\b|\s+|'+EMOJI_PATTERN+r'|.', re.UNICODE|re.I)


def _nfkc_with_alignment(raw):
    pieces=[];mapping=[]
    for i,ch in enumerate(str(raw)):
        value=unicodedata.normalize('NFKC',ch);pieces.append(value);mapping.extend([(i,i+1)]*len(value))
    return ''.join(pieces),mapping


def _append_piece(parts,mapping_out,value,raw_span):
    parts.append(value);mapping_out.extend([raw_span]*len(value))


def _strict_view(raw):
    text,mapping=_nfkc_with_alignment(raw);parts=[];out=[];pending=None
    for m in re.finditer(r'\s+|\S+',text,re.UNICODE):
        token=m.group();span=(min(x[0] for x in mapping[m.start():m.end()]),max(x[1] for x in mapping[m.start():m.end()]))
        if token.isspace():pending=span if pending is None else (min(pending[0],span[0]),max(pending[1],span[1]));continue
        if parts and pending is not None:_append_piece(parts,out,' ',pending)
        _append_piece(parts,out,token,span);pending=None
    return ''.join(parts),out


def _looks_protected(token,index,raw):
    if token in PROTECTED_TOKENS:return True,'annotation_marker'
    if re.fullmatch(r'https?://\S+|www\.\S+',token,re.I):return True,'url'
    if re.fullmatch(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',token):return True,'email'
    if token.startswith('#'):return True,'hashtag'
    if re.fullmatch(r'\+?\d[\d() .-]{7,}\d',token):return True,'phone'
    if len(token)>=6 and re.search('[A-Za-z]',token) and re.search(r'\d',token):return True,'transaction_or_product_id'
    if token.isupper() and len(token)>1:return True,'acronym_or_brand'
    if index>0 and token[:1].isupper():return True,'proper_name_candidate'
    return False,None


def _language_for_token(token,language_meta):
    low=token.casefold()
    if re.search(r'-(?:nya|ku|mu)$',low):return 'id-en'
    if low in EN_SLANG_MAP or low in V12_EN_MARKERS:return 'en'
    if low in ID_SLANG_MAP or low in V12_ID_MARKERS:return 'id'
    return language_meta.get('language','unknown')


def _transform_token(token,domain,language):
    low=token.casefold();kind=None;confidence=1.0
    if low in ABBREVIATION_MAP:return ABBREVIATION_MAP[low],'abbreviation',.98
    unit=re.fullmatch(r'(\d+(?:[.,]\d+)?)(mnt|menit|jam|dtk|detik|km|meter|rb|ribu|k)',low,re.I)
    if unit:return unit.group(1)+' '+UNIT_NORMALIZATION_MAP[unit.group(2)],'number_unit',.99
    if domain in DOMAIN_ALIAS_MAP and low in DOMAIN_ALIAS_MAP[domain]:return DOMAIN_ALIAS_MAP[domain][low],'domain_alias',.98
    if low in NEGATION_MAP:return NEGATION_MAP[low],'negation',1.0
    if low in INTENSIFIER_MAP:return INTENSIFIER_MAP[low],'intensifier',1.0
    if language in {'id','id-en','unknown'} and low in ID_SLANG_MAP:return ID_SLANG_MAP[low],'id_slang',.97
    if language in {'en','id-en','unknown'} and low in EN_SLANG_MAP:return EN_SLANG_MAP[low],'en_slang',.97
    if language in {'id','id-en','unknown'} and low in TYPO_CANDIDATES_ID:return TYPO_CANDIDATES_ID[low],'typo_candidate_id',.90
    if language in {'en','id-en','unknown'} and low in TYPO_CANDIDATES_EN:return TYPO_CANDIDATES_EN[low],'typo_candidate_en',.90
    elongated=re.sub(r'([A-Za-z])\1{2,}',r'\1\1',token)
    if elongated!=token:return elongated,'elongation',.98
    return token,None,1.0


def normalize_multilingual_text_with_alignment(text,explicit_domain=None,language_hint=None,mask_sensitive=False):
    raw=str(text);strict_text,strict_map=_strict_view(raw);nfkc,nfkc_map=_nfkc_with_alignment(raw)
    language_meta=infer_language_metadata_v12(raw,declared_language=language_hint)
    domain=resolve_review_domain(raw,explicit_domain=explicit_domain) or explicit_domain
    parts=[];mapping=[];trace=[];pending_space=None
    for index,m in enumerate(LAYER_TOKEN_RE.finditer(nfkc)):
        token=m.group();raw_slice=nfkc_map[m.start():m.end()]
        span=(min(x[0] for x in raw_slice),max(x[1] for x in raw_slice))
        if token.isspace():pending_space=span if pending_space is None else (min(pending_space[0],span[0]),max(pending_space[1],span[1]));continue
        if parts and pending_space is not None:_append_piece(parts,mapping,' ',pending_space)
        pending_space=None
        protected,protected_kind=_looks_protected(token,index,raw)
        language=_language_for_token(token,language_meta)
        replacement=token;kind=None;confidence=1.0
        if token in EMOJI_SENTIMENT_MAP:
            replacement=EMOJI_SENTIMENT_MAP[token];kind='emoji_semantic';confidence=.99
        elif re.fullmatch(EMOJI_PATTERN,token):
            replacement='[EMOJI]';kind='emoji_semantic_unknown';confidence=.70
        elif protected:
            if mask_sensitive and protected_kind in {'url','email','phone','transaction_or_product_id'}:
                replacement={'url':'[URL]','email':'[EMAIL]','phone':'[PHONE]','transaction_or_product_id':'[ID]'}[protected_kind];kind='privacy_mask'
            else:kind='protected_'+protected_kind
        else:replacement,kind,confidence=_transform_token(token,domain,language)
        _append_piece(parts,mapping,replacement,span)
        if replacement!=token or kind:
            trace.append({'raw':raw[span[0]:span[1]],'normalized':replacement,'raw_start':span[0],'raw_end':span[1],
                          'kind':kind,'confidence':float(confidence),'language':language,'domain':domain})
    normalized=''.join(parts)
    assert len(normalized)==len(mapping)
    result={'raw_text':raw,'normalized_text':normalized,'normalized_text_strict':strict_text,
            'norm_to_raw':mapping,'strict_norm_to_raw':strict_map,'normalization_trace':trace,'replacements':trace,
            'resolved_domain':domain,'language_metadata':language_meta,'alignment_contract':'normalized char -> raw [start,end)'}
    return result


def normalize_gmaps_text_with_alignment(text,explicit_domain=None):
    return normalize_multilingual_text_with_alignment(text,explicit_domain=explicit_domain,language_hint='id')


def normalize_gmaps_token(token,domain=None):
    value,kind,_=_transform_token(str(token),domain,'id');return value,kind


def normalized_span_to_raw(ns,ne,alignment,view='normalized_text'):
    mapping=alignment['strict_norm_to_raw'] if view=='normalized_text_strict' else alignment['norm_to_raw']
    if ns<0 or ne<=ns or ne>len(mapping):return None
    chunk=mapping[ns:ne];return min(x[0] for x in chunk),max(x[1] for x in chunk)


def raw_span_to_normalized(rs,re_,alignment,view='normalized_text'):
    if rs<0 or re_<=rs or re_>len(alignment['raw_text']):return None
    mapping=alignment['strict_norm_to_raw'] if view=='normalized_text_strict' else alignment['norm_to_raw']
    hits=[i for i,(a,b) in enumerate(mapping) if max(a,rs)<min(b,re_)]
    return (min(hits),max(hits)+1) if hits else None


def normalization_acceptance_tests():
    samples=['Servicenya lamaaa 😡 tapi food-nya enakkk 👍','Ｋｏｐｉ\t enakkk','Dr Anindhita ramah; ID AB12CD34 aman','not bad, gr8 service','[ASP] kopi [/ASP] [OPN] enak [/OPN]']
    results=[]
    for raw in samples:
        first=normalize_multilingual_text_with_alignment(raw,language_hint='id-en');second=normalize_multilingual_text_with_alignment(first['normalized_text'],language_hint='id-en')
        markers=[x for x in PROTECTED_TOKENS if x in raw]
        spans=[]
        for m in re.finditer(r'\w+',raw):
            ns=raw_span_to_normalized(m.start(),m.end(),first)
            spans.append(ns is not None and normalized_span_to_raw(*ns,first)[0]<=m.start() and normalized_span_to_raw(*ns,first)[1]>=m.end())
        results.append({'raw':raw,'raw_unchanged':first['raw_text']==raw,'alignment_length':len(first['normalized_text'])==len(first['norm_to_raw']),
                        'strict_alignment_length':len(first['normalized_text_strict'])==len(first['strict_norm_to_raw']),
                        'idempotent':second['normalized_text']==first['normalized_text'],'markers_preserved':all(x in first['normalized_text'] for x in markers),
                        'roundtrip_word_spans':all(spans),'negation_preserved':not re.search(r'\b(tidak|ga|gak|nggak|not|never)\b',raw,re.I) or bool(re.search(r'\b(tidak|not|never)\b',first['normalized_text'],re.I))})
    return {'passed':all(all(v for k,v in r.items() if k!='raw') for r in results),'cases':results}


TAXONOMY_RULES = {'generic.overall_experience': {'domain': 'generic', 'label': 'Overall Experience', 'keywords': ['overall', 'pengalaman', 'experience', 'kesan', 'puas', 'kecewa', 'worth', 'sepadan', 'recommended', 'direkomendasikan', 'balik lagi', 'datang lagi']}, 'generic.service_quality': {'domain': 'generic', 'label': 'Service Quality', 'keywords': ['pelayanan', 'layanan', 'service', 'petugas', 'pegawai', 'staf', 'staff', 'helpful', 'ramah', 'jutek', 'responsif', 'tidak responsif']}, 'generic.service_speed': {'domain': 'generic', 'label': 'Service Speed', 'keywords': ['cepat', 'lambat', 'lama', 'gercep', 'satset', 'waktu layanan', 'service time', 'proses cepat', 'proses lama']}, 'generic.waiting_time': {'domain': 'generic', 'label': 'Waiting Time', 'keywords': ['antre', 'antrean', 'menunggu', 'nunggu', 'queue', 'waiting time', 'nomor antrean', 'waktu tunggu']}, 'generic.staff_attitude': {'domain': 'generic', 'label': 'Staff Attitude', 'keywords': ['ramah', 'jutek', 'judes', 'ketus', 'sopan', 'tidak sopan', 'cuek', 'membantu', 'helpful', 'informatif', 'responsif']}, 'generic.cleanliness': {'domain': 'generic', 'label': 'Cleanliness', 'keywords': ['bersih', 'kotor', 'jorok', 'kebersihan', 'sampah', 'bau', 'pesing', 'berdebu']}, 'generic.comfort': {'domain': 'generic', 'label': 'Comfort', 'keywords': ['nyaman', 'tidak nyaman', 'panas', 'dingin', 'adem', 'sejuk', 'bising', 'berisik', 'tenang']}, 'generic.price_value': {'domain': 'generic', 'label': 'Price & Value', 'keywords': ['harga', 'mahal', 'murah', 'terjangkau', 'overpriced', 'sepadan', 'worth', 'value for money', 'biaya', 'tarif', 'fee']}, 'generic.location_access': {'domain': 'generic', 'label': 'Location & Access', 'keywords': ['lokasi', 'akses', 'jalan', 'strategis', 'mudah dicari', 'susah dicari', 'nyasar', 'maps']}, 'generic.parking': {'domain': 'generic', 'label': 'Parking', 'keywords': ['parkir', 'area parkir', 'parking', 'valet', 'motor', 'mobil', 'slot parkir']}, 'generic.facilities': {'domain': 'generic', 'label': 'Facilities', 'keywords': ['fasilitas', 'toilet', 'wifi', 'musala', 'mushola', 'ac', 'lift', 'eskalator', 'ruang tunggu', 'kursi', 'charging', 'stop kontak', 'cctv']}, 'generic.security': {'domain': 'generic', 'label': 'Security', 'keywords': ['aman', 'tidak aman', 'security', 'satpam', 'petugas keamanan', 'cctv', 'penjaga']}, 'generic.operating_hours': {'domain': 'generic', 'label': 'Operating Hours', 'keywords': ['jam buka', 'jam tutup', 'jam operasional', 'buka', 'tutup', '24 jam', 'opening hours']}, 'generic.digital_service': {'domain': 'generic', 'label': 'Digital Service', 'keywords': ['aplikasi', 'website', 'sistem', 'online', 'login', 'otp', 'error', 'down', 'loading', 'notifikasi', 'registrasi online', 'pendaftaran online']}, 'generic.communication_channel': {'domain': 'generic', 'label': 'Communication Channel', 'keywords': ['whatsapp', 'wa', 'telepon', 'call center', 'hotline', 'chat', 'email', 'dm', 'dibalas', 'tidak dibalas']}, 'fnb.food_quality': {'domain': 'food_beverage', 'label': 'Food Quality', 'keywords': ['makanan', 'menu', 'nasi', 'mie', 'pasta', 'roti', 'croissant', 'dessert', 'kue', 'snack', 'cemilan', 'cassava', 'casava', 'steak', 'ayam', 'burger', 'pizza']}, 'fnb.beverage_quality': {'domain': 'food_beverage', 'label': 'Beverage Quality', 'keywords': ['minuman', 'kopi', 'americano', 'latte', 'espresso', 'cappuccino', 'matcha', 'teh', 'jus', 'juice', 'milkshake', 'mocktail']}, 'fnb.taste': {'domain': 'food_beverage', 'label': 'Taste', 'keywords': ['rasa', 'enak', 'tidak enak', 'asin', 'keasinan', 'manis', 'kemanisan', 'pahit', 'hambar', 'gurih', 'pedas', 'asam']}, 'fnb.portion': {'domain': 'food_beverage', 'label': 'Portion', 'keywords': ['porsi', 'portion', 'banyak', 'sedikit', 'dikit', 'kecil', 'besar']}, 'fnb.presentation': {'domain': 'food_beverage', 'label': 'Food Presentation', 'keywords': ['penyajian', 'plating', 'presentasi makanan', 'tampilan makanan']}, 'fnb.menu_variety': {'domain': 'food_beverage', 'label': 'Menu Variety', 'keywords': ['variasi menu', 'pilihan menu', 'menu lengkap', 'menu sedikit', 'banyak pilihan']}, 'fnb.order_accuracy': {'domain': 'food_beverage', 'label': 'Order Accuracy', 'keywords': ['pesanan salah', 'order salah', 'salah pesanan', 'pesanan sesuai', 'order sesuai', 'kurang item', 'item kurang']}, 'fnb.order_speed': {'domain': 'food_beverage', 'label': 'Order Speed', 'keywords': ['pesanan lama', 'makanan lama', 'minuman lama', 'keluar lama', 'nunggu makanan', 'nunggu minuman', 'service time']}, 'fnb.cashier': {'domain': 'food_beverage', 'label': 'Cashier', 'keywords': ['kasir', 'cashier']}, 'fnb.barista': {'domain': 'food_beverage', 'label': 'Barista', 'keywords': ['barista']}, 'fnb.waiter': {'domain': 'food_beverage', 'label': 'Waiter / Server', 'keywords': ['waiter', 'waitress', 'pramusaji', 'pelayan']}, 'fnb.ambience': {'domain': 'food_beverage', 'label': 'Ambience', 'keywords': ['suasana', 'ambience', 'vibes', 'cozy', 'nyaman', 'estetis', 'instagrammable', 'outdoor', 'indoor', 'pohon', 'rindang', 'adem']}, 'fnb.wfc_support': {'domain': 'food_beverage', 'label': 'Work From Cafe Support', 'keywords': ['wfc', 'work from cafe', 'wifi', 'stop kontak', 'colokan', 'meja kerja', 'laptop']}, 'healthcare.medical_service': {'domain': 'healthcare', 'label': 'Medical Service', 'keywords': ['dokter', 'perawat', 'bidan', 'tenaga kesehatan', 'pelayanan medis', 'pemeriksaan', 'diagnosis', 'diagnosa', 'tindakan']}, 'healthcare.doctor': {'domain': 'healthcare', 'label': 'Doctor', 'keywords': ['dokter', 'dr ', 'dr.', 'drg', 'dokter gigi']}, 'healthcare.nurse': {'domain': 'healthcare', 'label': 'Nurse', 'keywords': ['perawat', 'suster', 'nurse']}, 'healthcare.registration': {'domain': 'healthcare', 'label': 'Registration', 'keywords': ['pendaftaran', 'registrasi', 'admisi', 'daftar online', 'pendaftaran online']}, 'healthcare.queue': {'domain': 'healthcare', 'label': 'Healthcare Queue', 'keywords': ['antrean dokter', 'nomor antrean', 'nunggu dokter', 'waktu tunggu dokter']}, 'healthcare.pharmacy': {'domain': 'healthcare', 'label': 'Pharmacy', 'keywords': ['farmasi', 'apotek', 'obat', 'resep']}, 'healthcare.lab': {'domain': 'healthcare', 'label': 'Laboratory', 'keywords': ['laboratorium', 'lab', 'cek darah', 'tes darah']}, 'healthcare.radiology': {'domain': 'healthcare', 'label': 'Radiology', 'keywords': ['rontgen', 'x-ray', 'xray', 'usg', 'radiologi']}, 'healthcare.emergency': {'domain': 'healthcare', 'label': 'Emergency / ER', 'keywords': ['igd', 'ugd', 'gawat darurat', 'emergency']}, 'healthcare.bpjs': {'domain': 'healthcare', 'label': 'BPJS / Insurance Administration', 'keywords': ['bpjs', 'jkn', 'kis', 'rujukan', 'jaminan', 'asuransi kesehatan']}, 'healthcare.inpatient': {'domain': 'healthcare', 'label': 'Inpatient Care', 'keywords': ['rawat inap', 'ranap', 'kamar inap']}, 'healthcare.outpatient': {'domain': 'healthcare', 'label': 'Outpatient Care', 'keywords': ['rawat jalan', 'rajal', 'kontrol', 'poliklinik', 'poli']}, 'government.counter_service': {'domain': 'government', 'label': 'Counter Service', 'keywords': ['loket', 'petugas loket', 'counter', 'pelayanan loket']}, 'government.document_process': {'domain': 'government', 'label': 'Document Processing', 'keywords': ['dokumen', 'berkas', 'surat', 'akta', 'legalisasi', 'legalisir', 'fotokopi']}, 'government.identity_document': {'domain': 'government', 'label': 'Population Document', 'keywords': ['ktp', 'e-ktp', 'kartu tanda penduduk', 'kartu keluarga', 'kk', 'akta kelahiran', 'dukcapil']}, 'government.licensing': {'domain': 'government', 'label': 'Licensing', 'keywords': ['izin', 'perizinan', 'izin online', 'mpp', 'mal pelayanan publik']}, 'government.immigration': {'domain': 'government', 'label': 'Immigration', 'keywords': ['imigrasi', 'paspor', 'passport', 'visa']}, 'government.vehicle_document': {'domain': 'government', 'label': 'Vehicle Document Service', 'keywords': ['samsat', 'sim', 'stnk', 'pajak kendaraan']}, 'government.bureaucracy': {'domain': 'government', 'label': 'Bureaucracy', 'keywords': ['birokrasi', 'prosedur', 'alur proses', 'persyaratan', 'syarat']}, 'banking.customer_service': {'domain': 'banking', 'label': 'Customer Service', 'keywords': ['customer service', 'cs bank', 'layanan pelanggan']}, 'banking.teller': {'domain': 'banking', 'label': 'Teller', 'keywords': ['teller']}, 'banking.security': {'domain': 'banking', 'label': 'Bank Security', 'keywords': ['satpam', 'security', 'petugas keamanan']}, 'banking.account_opening': {'domain': 'banking', 'label': 'Account Opening', 'keywords': ['buka rekening', 'pembukaan rekening', 'rekening baru']}, 'banking.atm': {'domain': 'banking', 'label': 'ATM', 'keywords': ['atm', 'mesin atm', 'tarik tunai', 'setor tunai', 'atm error', 'atm kosong']}, 'banking.mobile_banking': {'domain': 'banking', 'label': 'Mobile Banking', 'keywords': ['mobile banking', 'mbanking', 'm-banking', 'livin', 'brimo', 'bca mobile', 'wondr']}, 'banking.internet_banking': {'domain': 'banking', 'label': 'Internet Banking', 'keywords': ['internet banking', 'ibanking', 'i-banking']}, 'banking.transaction': {'domain': 'banking', 'label': 'Transaction', 'keywords': ['transfer', 'transaksi', 'saldo', 'pending', 'gagal transaksi', 'qris']}, 'banking.credit_loan': {'domain': 'banking', 'label': 'Credit / Loan', 'keywords': ['kredit', 'pinjaman', 'loan', 'kpr', 'kartu kredit']}, 'banking.fee': {'domain': 'banking', 'label': 'Bank Fee', 'keywords': ['biaya admin', 'biaya administrasi', 'admin fee', 'fee']}, 'social.benefit_service': {'domain': 'social_service', 'label': 'Social Benefit Service', 'keywords': ['bansos', 'bantuan sosial', 'blt', 'bpnt', 'pkh', 'sembako', 'penerima manfaat']}, 'social.data_registration': {'domain': 'social_service', 'label': 'Social Welfare Data', 'keywords': ['dtks', 'data sosial', 'data penerima', 'verifikasi data']}, 'social.case_worker': {'domain': 'social_service', 'label': 'Social Worker / Companion', 'keywords': ['pendamping', 'pendamping sosial', 'petugas sosial']}, 'social.disability_service': {'domain': 'social_service', 'label': 'Disability Service', 'keywords': ['disabilitas', 'difabel', 'penyandang disabilitas']}, 'social.elderly_service': {'domain': 'social_service', 'label': 'Elderly Service', 'keywords': ['lansia', 'panti lansia', 'panti jompo']}, 'hotel.room': {'domain': 'hotel', 'label': 'Room', 'keywords': ['kamar', 'room', 'kasur', 'bed', 'handuk', 'linen']}, 'hotel.cleanliness': {'domain': 'hotel', 'label': 'Room Cleanliness', 'keywords': ['kamar bersih', 'kamar kotor', 'housekeeping', 'bau kamar']}, 'hotel.reception': {'domain': 'hotel', 'label': 'Reception', 'keywords': ['resepsionis', 'receptionist', 'reception']}, 'hotel.checkin_checkout': {'domain': 'hotel', 'label': 'Check-in / Check-out', 'keywords': ['check-in', 'check-out', 'checkin', 'checkout']}, 'hotel.breakfast': {'domain': 'hotel', 'label': 'Breakfast', 'keywords': ['sarapan', 'breakfast']}, 'hotel.pool': {'domain': 'hotel', 'label': 'Swimming Pool', 'keywords': ['kolam renang', 'pool', 'swimming pool']}, 'retail.product_quality': {'domain': 'retail', 'label': 'Product Quality', 'keywords': ['produk', 'barang', 'kualitas produk', 'kualitas barang']}, 'retail.stock': {'domain': 'retail', 'label': 'Stock Availability', 'keywords': ['stok', 'stock', 'habis', 'restock', 'tersedia']}, 'retail.cashier': {'domain': 'retail', 'label': 'Cashier', 'keywords': ['kasir', 'cashier']}, 'retail.promotion': {'domain': 'retail', 'label': 'Promotion / Discount', 'keywords': ['promo', 'promosi', 'diskon', 'discount', 'voucher']}, 'retail.store_layout': {'domain': 'retail', 'label': 'Store Layout', 'keywords': ['layout toko', 'penataan', 'rak', 'lorong', 'aisle']}, 'education.teacher': {'domain': 'education', 'label': 'Teacher / Lecturer', 'keywords': ['guru', 'dosen', 'pengajar']}, 'education.administration': {'domain': 'education', 'label': 'Academic Administration', 'keywords': ['tata usaha', 'tu', 'admin kampus', 'administrasi akademik']}, 'education.classroom': {'domain': 'education', 'label': 'Classroom', 'keywords': ['kelas', 'ruang kelas', 'classroom']}, 'education.library': {'domain': 'education', 'label': 'Library', 'keywords': ['perpustakaan', 'perpus', 'library']}, 'education.lab': {'domain': 'education', 'label': 'Laboratory', 'keywords': ['laboratorium', 'lab komputer', 'lab']}, 'education.campus_facility': {'domain': 'education', 'label': 'Campus Facility', 'keywords': ['kantin', 'wifi kampus', 'toilet kampus', 'parkir kampus']}, 'transport.station': {'domain': 'transportation', 'label': 'Station / Terminal', 'keywords': ['stasiun', 'terminal', 'halte', 'station']}, 'transport.ticketing': {'domain': 'transportation', 'label': 'Ticketing', 'keywords': ['tiket', 'ticket', 'loket tiket', 'mesin tiket']}, 'transport.schedule': {'domain': 'transportation', 'label': 'Schedule', 'keywords': ['jadwal', 'schedule', 'keberangkatan', 'kedatangan', 'delay', 'terlambat']}, 'transport.cleanliness': {'domain': 'transportation', 'label': 'Transport Cleanliness', 'keywords': ['toilet stasiun', 'stasiun bersih', 'terminal kotor']}, 'transport.driver': {'domain': 'transportation', 'label': 'Driver', 'keywords': ['driver', 'pengemudi', 'sopir', 'supir']}, 'automotive.mechanic': {'domain': 'automotive', 'label': 'Mechanic', 'keywords': ['mekanik', 'mechanic']}, 'automotive.service_quality': {'domain': 'automotive', 'label': 'Workshop Service', 'keywords': ['servis mobil', 'servis motor', 'service mobil', 'service motor']}, 'automotive.sparepart': {'domain': 'automotive', 'label': 'Spare Part', 'keywords': ['suku cadang', 'sparepart', 'onderdil']}, 'automotive.carwash': {'domain': 'automotive', 'label': 'Car Wash', 'keywords': ['cuci mobil', 'car wash', 'carwash']}, 'tourism.scenery': {'domain': 'tourism', 'label': 'Scenery', 'keywords': ['pemandangan', 'view', 'scenery', 'sunset', 'gunung', 'danau', 'pantai']}, 'tourism.photo_spot': {'domain': 'tourism', 'label': 'Photo Spot', 'keywords': ['spot foto', 'instagrammable', 'foto', 'photo']}, 'tourism.family_friendly': {'domain': 'tourism', 'label': 'Family Friendly', 'keywords': ['ramah keluarga', 'family friendly', 'ramah anak', 'kids friendly']}, 'tourism.ticket_price': {'domain': 'tourism', 'label': 'Ticket Price', 'keywords': ['harga tiket', 'tiket masuk', 'HTM', 'biaya masuk']}}

def _taxonomy_text(aspect, context=None):
    return (str(aspect or '').lower(), str(context or '').lower())

def taxonomy_mapper(aspect, context=None, domain_hint=None):
    """
    V6 ASPECT-FIRST taxonomy.

    Aspect surface is the primary evidence.
    Context is only supporting/tie-break evidence, preventing:
      parkiran -> Ambience
      harga -> Food Quality
    """
    aspect_text, context_text = _taxonomy_text(aspect, context)
    candidates = []
    for taxonomy_id, spec in TAXONOMY_RULES.items():
        domain = spec['domain']
        aspect_matches = [kw for kw in spec['keywords'] if kw.lower() in aspect_text]
        context_matches = [kw for kw in spec['keywords'] if kw.lower() in context_text]
        if not aspect_matches and (not context_matches):
            continue
        aspect_score = sum((4.0 + min(len(kw.split()), 4) * 0.35 for kw in aspect_matches))
        context_score = sum((0.35 + min(len(kw.split()), 4) * 0.05 for kw in context_matches))
        domain_bonus = 0.8 if domain_hint and domain == domain_hint else 0.0
        total = aspect_score + context_score + domain_bonus
        candidates.append({'taxonomy_id': taxonomy_id, 'domain': domain, 'label': spec['label'], 'score': total, 'aspect_matches': aspect_matches, 'context_matches': context_matches})
    if not candidates:
        return {'taxonomy_id': 'generic.other', 'normalized_category': 'other', 'taxonomy_domain': domain_hint or 'generic', 'taxonomy_confidence': 0.0, 'taxonomy_method': 'heuristic_no_match_v6', 'taxonomy_evidence': []}
    candidates.sort(key=lambda x: (bool(x['aspect_matches']), x['score'], x['domain'] != 'generic', max([len(k) for k in x['aspect_matches'] + x['context_matches']] or [0])), reverse=True)
    best = candidates[0]
    conf = min(0.93, 0.56 + 0.1 * len(best['aspect_matches']) + 0.03 * len(best['context_matches']) + (0.04 if domain_hint == best['domain'] else 0.0))
    evidence = (best['aspect_matches'] + best['context_matches'])[:8]
    return {'taxonomy_id': best['taxonomy_id'], 'normalized_category': best['label'], 'taxonomy_domain': best['domain'], 'taxonomy_confidence': round(float(conf), 4), 'taxonomy_method': 'heuristic_aspect_first_taxonomy_v6', 'taxonomy_evidence': evidence}

def split_coordinated_aspect(text, span, domain_hint=None):
    """
    Conservative repair for merged coordination spans.
    Split only when >=2 fragments independently receive meaningful taxonomy.
    """
    s, e = (span['start'], span['end'])
    surface = str(text)[s:e]
    parts = list(re.finditer('\\b(?:dan|sama)\\b|&|,', surface, re.I))
    if not parts:
        return [span]
    boundaries = [0]
    for m in parts:
        boundaries.extend([m.start(), m.end()])
    boundaries.append(len(surface))
    fragments = []
    cursor = 0
    for m in parts:
        frag = surface[cursor:m.start()]
        fs = cursor
        cursor = m.end()
        frag_s = fs + len(frag) - len(frag.lstrip(' ,'))
        frag_e = m.start() - len(frag) + len(frag.rstrip(' ,'))
        stripped = frag.strip(' ,')
        if stripped:
            rel = surface.find(stripped, fs, m.start())
            fragments.append((rel, rel + len(stripped), stripped))
    tail = surface[cursor:]
    stripped = tail.strip(' ,')
    if stripped:
        rel = surface.find(stripped, cursor)
        fragments.append((rel, rel + len(stripped), stripped))
    valid = []
    for rs, re_, frag in fragments:
        tax = taxonomy_mapper(frag, '', domain_hint=domain_hint)
        if tax['taxonomy_id'] != 'generic.other' and tax['taxonomy_confidence'] >= CONFIG['coordination_split_min_taxonomy_conf']:
            valid.append({**span, 'start': s + rs, 'end': s + re_, 'score': float(span.get('score', 0)) * 0.97, 'coordination_repair': True})
    return valid if len(valid) >= 2 else [span]

V13_ASPECT_PATTERNS = {'Waiting Time': ['\\b(?:waktu\\s+tunggu(?:nya)?|masa\\s+tunggu|antrean(?:nya)?|antrian(?:nya)?|waiting\\s+time|wait(?:ing)?)\\b', '待ち時間|待ち|وقت\\s*الانتظار|الانتظار'], 'Staff Attitude': ['\\b(?:barista(?:nya)?|kasir(?:nya)?|staf(?:nya)?|staff|pegawai(?:nya)?|pelayan(?:nya)?|employee|waiter|waitress|server)\\b', 'スタッフ|店員|従業員|الموظف(?:ون|ين)?|العامل(?:ون|ين)?|النادل'], 'Price & Value': ['\\b(?:harga(?:nya)?|biaya(?:nya)?|tarif(?:nya)?|price|cost|charge|fee)\\b', '料金|価格|値段|رسوم|سعر|تكلفة'], 'Beverage Quality': ['\\b(?:kopi(?:nya)?|coffee|americano|latte|espresso|cappuccino|minuman(?:nya)?|drink)\\b', 'コーヒー|飲み物|القهوة|المشروب'], 'Food Quality': ['\\b(?:makanan(?:nya)?|masakan(?:nya)?|food|meal|dish|menu)\\b', '料理|食事|الطعام|الوجبة'], 'Service Quality': ['\\b(?:pelayanan(?:nya)?|layanan(?:nya)?|service|pesanan(?:nya)?|order)\\b', 'サービス|注文|الخدمة|الطلب']}

V13_OPINION_PATTERNS = [('negative_quality', '\\b(?:mengecewakan|kecewa|buruk|tidak\\s+enak|gak\\s+enak|bad|disappoint(?:ing|ed)?|awful|terrible)\\b|悪く|残念|سيئ|مخيب'), ('duration', '\\b(?:hampir\\s+)?(?:satu|dua|tiga|setengah|\\d+)\\s*(?:jam|menit|mnt)\\b|\\b(?:almost\\s+)?(?:one|two|three|half|thirty|forty(?:-?five)?|sixty|\\d+)(?:-|\\s)+(?:hours?|minutes?|hour|minute)\\b|\\d+\\s*時間|\\d+\\s*分|ساعة|دقيقة'), ('waiting', '\\b(?:lama|lambat|menunggu|nunggu|slow|late|delayed|waited)\\b|遅い|待た|بطيء|انتظر'), ('staff_action', '\\b(?:meminta|minta)\\s+maaf\\b|\\b(?:mengganti|menggantikan|membuat\\s+ulang|bikin\\s+ulang)\\s+(?:pesanan|minuman|makanan)?\\b|\\b(?:apologi[sz]ed?|said\\s+sorry|replaced?|remade|helped)\\b|謝罪|作り直|交換|اعتذر|استبدل'), ('free', '\\b(?:tanpa\\s+(?:biaya|charge)|gratis|for\\s+free|free\\s+of\\s+charge|no\\s+(?:extra\\s+)?charge)\\b|無料|追加料金なし|دون\\s+رسوم|بدون\\s+رسوم|مجانا'), ('positive', '\\b(?:ramah|cepat|bagus|enak|membantu|friendly|quick|good|helpful|excellent)\\b|親切|良い|おいしい|سريع|ودود|جيد|لذيذ'), ('neutral', '\\b(?:standar|biasa\\s+saja|standard|average)\\b|普通|عادي')]

def _v13_recover_aspects(text):
    out = []
    for taxonomy, patterns in V13_ASPECT_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, str(text), re.I):
                out.append({'start': match.start(), 'end': match.end(), 'score': 0.88, 'source_view': 'v13_multilingual_lexicon', 'aspect_method': 'v13_evidence_candidate_recovery', 'v13_taxonomy_hint': taxonomy})
    return out

def _v13_recover_opinions(text):
    out = []
    for family, pattern in V13_OPINION_PATTERNS:
        for match in re.finditer(pattern, str(text), re.I):
            out.append({'start': match.start(), 'end': match.end(), 'score': 0.84, 'source_view': 'v13_multilingual_cue', 'opinion_family': family})
    return out

recover_aspects = _v13_recover_aspects
recover_opinions = _v13_recover_opinions

GMAPS_EXPLICIT_CANDIDATE_PATTERNS = ['\\b(?:mba|mbak|mas|kak)\\s+(?:kasir(?:nya)?|barista(?:nya)?|dokter(?:nya)?|perawat(?:nya)?|bidan(?:nya)?)\\b', '\\b(?:kasir(?:nya)?|barista(?:nya)?|dokter(?:nya)?|perawat(?:nya)?|bidan(?:nya)?|farmasi(?:nya)?)\\b', '\\b(?:musholla(?:nya)?|mushola(?:nya)?|musala(?:nya)?|toilet(?:nya)?|parkiran(?:\\s+mobil)?|ruang\\s+tunggu)\\b', '\\b(?:service\\s*speed|servicenya|pelayanannya|antrean(?:nya)?|antrian(?:nya)?)\\b', '\\b(?:iced?\\s+)?americano(?:nya)?\\b', '\\b(?:cappuccino|cappucino|latte|espresso|matcha)(?:nya)?\\b', '\\b(?:chicken\\s+)?pasta(?:nya)?\\b']

def _v8_head(term):
    low = unicodedata.normalize('NFKC', str(term or '')).lower()
    low = re.sub('[^\\w\\s]', ' ', low)
    low = re.sub('\\s+', ' ', low).strip()
    low = re.sub('^(mba|mbak|mas|kak)\\s+', '', low)
    if re.match('^(harga|biaya|tarif|price)\\b', low):
        return 'price::' + low
    toks = [t for t in re.findall('[a-z0-9]+', low) if t]
    if not toks:
        return low
    head = toks[-1]
    if head.endswith('nya') and len(head) > 5:
        head = head[:-3]
    if head == 'pastanya':
        head = 'pasta'
    if head == 'kopinya':
        head = 'kopi'
    return head


def recover_aspects(text):
    out=_v13_recover_aspects(text)
    for pattern in GMAPS_EXPLICIT_CANDIDATE_PATTERNS:
        for match in re.finditer(pattern,str(text),re.I):
            out.append({'start':match.start(),'end':match.end(),'score':.60,'source_view':'v13_legacy_recovery',
                        'aspect_method':'legacy_recovery_review_only'})
    return out

def add_alias_metadata(rows):
    # Preserve repeated-mention assistance without merging conflicting triplet sentiments/offsets.
    for row in rows:
        key=_v8_head(row.get('aspect_term'))
        related=[x for x in rows if _v8_head(x.get('aspect_term'))==key and abs(x['aspect_start']-row['aspect_start'])<=420]
        aliases=list(dict.fromkeys(x.get('aspect_term') for x in related))
        row['alias_terms']=aliases
        row['alias_method']='legacy_surface_group_review_hint'
    return rows
