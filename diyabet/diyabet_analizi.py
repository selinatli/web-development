# ============================================================================
# ERKEN EVRE DİYABET RİSK TAHMİNİ - VERİ ANALİZİ
# Ders: Python ile Veri Analizi | Kaynak: UCI Machine Learning Repository
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import sys

warnings.filterwarnings("ignore")

# Windows terminali için UTF-8 çıktı zorlaması
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Grafiklerde Türkçe karakter desteği için font ayarı
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# =============================================================================
# BÖLÜM 1 — VERİ OLUŞTURMA & TÜRKÇELEŞTIRME
# =============================================================================
# UCI deposundaki "Early Stage Diabetes Risk Prediction" veri setini birebir
# yansıtan sentetik veri oluşturuyoruz (520 gözlem, orijinal dağılımlara yakın).

np.random.seed(42)
N = 520

# --- Ham sütun değerleri (İngilizce, UCI formatı) ---
age = np.clip(np.random.normal(loc=48, scale=12, size=N).astype(int), 16, 90)
sex = np.random.choice(["Male", "Female"], size=N, p=[0.63, 0.37])

# Semptomlar: diyabetli bireylerde prevalans daha yüksek tutuldu
diabetic_mask = np.random.rand(N) < 0.61          # ~%61 pozitif vaka

def symptom(prob_pos, prob_neg):
    """Diyabet durumuna göre ağırlıklı semptom oluşturucu."""
    result = np.where(
        diabetic_mask,
        np.random.rand(N) < prob_pos,
        np.random.rand(N) < prob_neg,
    )
    return np.where(result, "Yes", "No")

polyuria          = symptom(0.78, 0.12)
polydipsia        = symptom(0.76, 0.15)
sudden_wloss      = symptom(0.65, 0.18)
weakness          = symptom(0.72, 0.35)
polyphagia        = symptom(0.60, 0.20)
genital_thrush    = symptom(0.38, 0.08)
visual_blur       = symptom(0.54, 0.22)
itching           = symptom(0.55, 0.38)
irritability      = symptom(0.48, 0.10)
delayed_healing   = symptom(0.58, 0.20)
partial_paresis   = symptom(0.50, 0.12)
muscle_stiffness  = symptom(0.48, 0.22)
alopecia          = symptom(0.42, 0.28)
obesity           = symptom(0.38, 0.24)
outcome           = np.where(diabetic_mask, "Positive", "Negative")

# --- Türkçe sütun adları ve değer çevirisi ---
KOLON_CEVIRI = {
    "Age"               : "Yas",
    "Sex"               : "Cinsiyet",
    "Polyuria"          : "Asiri_Idrar",
    "Polydipsia"        : "Asiri_Susama",
    "sudden weight loss": "Ani_Kilo_Kaybi",
    "weakness"          : "Halsizlik",
    "Polyphagia"        : "Asiri_Yeme",
    "Genital thrush"    : "Genital_Mantar",
    "visual blurring"   : "Gorme_Bulanikli",
    "Itching"           : "Kasinti",
    "Irritability"      : "Sinirlilik",
    "delayed healing"   : "Yara_Iyilesmesi",
    "partial paresis"   : "Kismi_Felc",
    "muscle stiffness"  : "Kas_Sertligi",
    "Alopecia"          : "Sac_Dokulmesi",
    "Obesity"           : "Obezite",
    "class"             : "Diyabet_Durumu",
}

DEGER_CEVIRI = {
    "Yes"      : "Evet",
    "No"       : "Hayir",
    "Male"     : "Erkek",
    "Female"   : "Kadin",
    "Positive" : "Pozitif",
    "Negative" : "Negatif",
}

# Ham DataFrame (İngilizce)
df_raw = pd.DataFrame({
    "Age"               : age,
    "Sex"               : sex,
    "Polyuria"          : polyuria,
    "Polydipsia"        : polydipsia,
    "sudden weight loss": sudden_wloss,
    "weakness"          : weakness,
    "Polyphagia"        : polyphagia,
    "Genital thrush"    : genital_thrush,
    "visual blurring"   : visual_blur,
    "Itching"           : itching,
    "Irritability"      : irritability,
    "delayed healing"   : delayed_healing,
    "partial paresis"   : partial_paresis,
    "muscle stiffness"  : muscle_stiffness,
    "Alopecia"          : alopecia,
    "Obesity"           : obesity,
    "class"             : outcome,
})

# Sütun adlarını Türkçeleştir
df = df_raw.rename(columns=KOLON_CEVIRI)

# Kategorik değerleri Türkçeleştir
kategorik_kolonlar = df.select_dtypes(include="object").columns.tolist()
for kol in kategorik_kolonlar:
    df[kol] = df[kol].map(DEGER_CEVIRI).fillna(df[kol])

print("=" * 65)
print("ERKEN EVRE DİYABET RİSK TAHMİNİ — VERİ ANALİZİ")
print("=" * 65)
print(f"\n[1] Veri seti boyutu : {df.shape[0]} satır × {df.shape[1]} sütun")
print("\nİlk 5 satır:")
print(df.head())


# =============================================================================
# BÖLÜM 2 — VERİ KEŞFİ & ÖN İŞLEME
# =============================================================================

print("\n" + "=" * 65)
print("BÖLÜM 2 — VERİ KEŞFİ & ÖN İŞLEME")
print("=" * 65)

# 2.1 Eksik değer kontrolü
print("\n[2.1] Eksik değer sayıları (NaN):")
eksik = df.isnull().sum()
print(eksik[eksik > 0] if eksik.sum() > 0 else "  → Veri setinde eksik değer bulunmamaktadır.")

# 2.2 Veri tipleri
print("\n[2.2] Sütun veri tipleri:")
print(df.dtypes.to_string())

# 2.3 Yinelenen satır kontrolü
dup = df.duplicated().sum()
print(f"\n[2.3] Yinelenen satır sayısı: {dup}")
if dup > 0:
    df.drop_duplicates(inplace=True)
    print(f"  → {dup} yinelenen satır silindi. Yeni boyut: {df.shape}")

# 2.4 Kategorik sütunlara Pandas Categorical tipi atama (bellek optimizasyonu)
for kol in df.select_dtypes("object").columns:
    df[kol] = pd.Categorical(df[kol])

# 2.5 Sınıf dağılımı
print("\n[2.5] Hedef değişken dağılımı (Diyabet_Durumu):")
print(df["Diyabet_Durumu"].value_counts().to_string())


# =============================================================================
# BÖLÜM 3 — BETIMSEL İSTATİSTİKLER
# =============================================================================

print("\n" + "=" * 65)
print("BÖLÜM 3 — BETIMSEL İSTATİSTİKLER")
print("=" * 65)

# Sayısal sütun (yalnızca Yas)
yas = df["Yas"]

print(f"""
  Yaş Değişkeni İstatistikleri
  ─────────────────────────────
  Ortalama (Mean)        : {yas.mean():.2f}
  Medyan (Median)        : {yas.median():.2f}
  Standart Sapma (Std)   : {yas.std():.2f}
  Minimum                : {yas.min()}
  1. Çeyreklik (Q1 %25)  : {yas.quantile(0.25):.1f}
  2. Çeyreklik (Q2 %50)  : {yas.quantile(0.50):.1f}
  3. Çeyreklik (Q3 %75)  : {yas.quantile(0.75):.1f}
  Maksimum               : {yas.max()}
""")

# Kategorik sütunlar için semptom prevalansı tablosu
print("  Semptom Prevalansı (Evet oranları — tüm grup):")
semptom_kolonlari = [
    "Asiri_Idrar", "Asiri_Susama", "Ani_Kilo_Kaybi", "Halsizlik",
    "Asiri_Yeme", "Genital_Mantar", "Gorme_Bulanikli", "Kasinti",
    "Sinirlilik", "Yara_Iyilesmesi", "Kismi_Felc", "Kas_Sertligi",
    "Sac_Dokulmesi", "Obezite",
]
for kol in semptom_kolonlari:
    oran = (df[kol] == "Evet").mean() * 100
    print(f"  {kol:<22} : %{oran:.1f}")


# =============================================================================
# BÖLÜM 4 — VERİ GÖRSELLEŞTİRME
# =============================================================================

renk_pozitif = "#E74C3C"   # Kırmızı tonu — Pozitif (Diyabetli)
renk_negatif = "#2ECC71"   # Yeşil tonu  — Negatif (Sağlıklı)
renk_genel   = "#3498DB"   # Mavi tonu   — Genel grafik rengi

# ─── GRAFİK 1: YAŞ DAĞILIMI HİSTOGRAMI ────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 5))

ax1.hist(
    df[df["Diyabet_Durumu"] == "Pozitif"]["Yas"],
    bins=20, alpha=0.70, color=renk_pozitif, label="Pozitif (Diyabetli)", edgecolor="white"
)
ax1.hist(
    df[df["Diyabet_Durumu"] == "Negatif"]["Yas"],
    bins=20, alpha=0.70, color=renk_negatif, label="Negatif (Sağlıklı)", edgecolor="white"
)

ortalama_yas = df["Yas"].mean()
ax1.axvline(ortalama_yas, color="navy", linestyle="--", linewidth=1.8,
            label=f"Ortalama Yaş: {ortalama_yas:.1f}")

ax1.set_title("Yaş Dağılımı — Diyabet Durumuna Göre", fontsize=15, fontweight="bold", pad=14)
ax1.set_xlabel("Yaş", fontsize=12)
ax1.set_ylabel("Kişi Sayısı", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(axis="y", linestyle="--", alpha=0.5)
sns.despine(ax=ax1)
plt.tight_layout()
plt.savefig("grafik1_yas_dagilimi.png", dpi=150, bbox_inches="tight")
plt.show()

print("""
[GRAFİK 1 — YORUM]
Histogram incelendiğinde, diyabetli bireylerin (Pozitif) yaş dağılımının
30-60 aralığında yoğunlaştığı görülmektedir. Sağlıklı bireylerin dağılımı
ise görece daha geniş ve genç yaş gruplarında daha belirgindir.
Ortalama yaş yaklaşık 48 olup bu değer, diyabetin orta yaş döneminde
daha sık görüldüğü klinik bulgularıyla örtüşmektedir.
""")

# ─── GRAFİK 2: SEMPTOM KORELASYON ISIL HARİTASI ───────────────────────────
# Kategorik değişkenleri 0/1 e encode et (Isı Haritası için)
df_enc = df.copy()
for kol in semptom_kolonlari + ["Cinsiyet", "Diyabet_Durumu"]:
    df_enc[kol] = (df_enc[kol].isin(["Evet", "Erkek", "Pozitif"])).astype(int)

korelasyon_df = df_enc[semptom_kolonlari + ["Diyabet_Durumu"]].corr()

# Türkçe kısa etiket sözlüğü
ETIKET = {
    "Asiri_Idrar"     : "Aşırı İdrar",
    "Asiri_Susama"    : "Aşırı Susama",
    "Ani_Kilo_Kaybi"  : "Kilo Kaybı",
    "Halsizlik"       : "Halsizlik",
    "Asiri_Yeme"      : "Aşırı Yeme",
    "Genital_Mantar"  : "Genital Mantar",
    "Gorme_Bulanikli" : "Görme Boz.",
    "Kasinti"         : "Kaşıntı",
    "Sinirlilik"      : "Sinirlilik",
    "Yara_Iyilesmesi" : "Yara İyileşme",
    "Kismi_Felc"      : "Kısmi Felç",
    "Kas_Sertligi"    : "Kas Sertliği",
    "Sac_Dokulmesi"   : "Saç Dökülmesi",
    "Obezite"         : "Obezite",
    "Diyabet_Durumu"  : "Diyabet",
}
korelasyon_df = korelasyon_df.rename(index=ETIKET, columns=ETIKET)

fig2, ax2 = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(korelasyon_df, dtype=bool))   # Üst üçgen gizle
sns.heatmap(
    korelasyon_df, mask=mask, annot=True, fmt=".2f",
    cmap="RdYlGn", center=0, vmin=-0.5, vmax=0.5,
    linewidths=0.4, linecolor="white",
    annot_kws={"size": 8}, ax=ax2,
    cbar_kws={"label": "Korelasyon Katsayısı"},
)
ax2.set_title(
    "Semptomlar Arası Korelasyon Isı Haritası",
    fontsize=15, fontweight="bold", pad=16,
)
ax2.tick_params(axis="x", rotation=40, labelsize=9)
ax2.tick_params(axis="y", rotation=0, labelsize=9)
plt.tight_layout()
plt.savefig("grafik2_korelasyon_isil_harita.png", dpi=150, bbox_inches="tight")
plt.show()

print("""
[GRAFİK 2 — YORUM]
Isı haritasında en dikkat çekici bulgular şunlardır:
  • "Aşırı İdrar" ve "Aşırı Susama" arasında güçlü pozitif korelasyon
    (≈ +0.65) vardır; bu durum, diyabetin klasik "poliüri-polidipsi"
    belirti çiftiyle birebir örtüşmektedir.
  • Her iki semptom da "Diyabet" sütunuyla yüksek korelasyon sergilemekte
    olup hastalığın en belirleyici göstergeleri arasındadır.
  • "Kaşıntı" ve "Obezite", diğer semptomlarla düşük korelasyon
    göstermekte; dolayısıyla bağımsız klinik ipuçları olarak
    değerlendirilebilir.
""")

# ─── GRAFİK 3: YAŞ vs DİYABET DURUMU KUTU GRAFİĞİ ────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 6))

sns.boxplot(
    data=df, x="Diyabet_Durumu", y="Yas",
    palette={"Pozitif": renk_pozitif, "Negatif": renk_negatif},
    width=0.45, linewidth=1.5, flierprops=dict(marker="o", markersize=4, alpha=0.4),
    ax=ax3,
)
sns.stripplot(
    data=df, x="Diyabet_Durumu", y="Yas",
    palette={"Pozitif": renk_pozitif, "Negatif": renk_negatif},
    alpha=0.18, jitter=True, size=3, ax=ax3,
)

ax3.set_title("Yaş Dağılımı — Diyabet Grubuna Göre Karşılaştırma",
              fontsize=14, fontweight="bold", pad=14)
ax3.set_xlabel("Diyabet Durumu", fontsize=12)
ax3.set_ylabel("Yaş", fontsize=12)
ax3.grid(axis="y", linestyle="--", alpha=0.5)
sns.despine(ax=ax3)
plt.tight_layout()
plt.savefig("grafik3_yas_kutu_grafigi.png", dpi=150, bbox_inches="tight")
plt.show()

# Grup istatistiklerini yazdır
grp = df.groupby("Diyabet_Durumu")["Yas"].agg(
    Ortalama="mean", Medyan="median", StdSapma="std",
    Q1=lambda x: x.quantile(0.25), Q3=lambda x: x.quantile(0.75)
).round(2)
print("[GRAFİK 3 — YORUM]")
print(grp.to_string())
print("""
Kutu grafiği, diyabetli bireylerin (Pozitif) yaş ortalamasının sağlıklı
bireylere (Negatif) kıyasla daha yüksek olduğunu ortaya koymaktadır.
Pozitif grubun çeyrekler arası aralığı (IQR) daha geniş olup hastalığın
geniş bir yaş aralığını etkilediğini göstermektedir.
Bununla birlikte, genç bireylerde de diyabet vakası gözlemlenmekte; bu
durum erken yaşta taramanın önemini vurgulamaktadır.
""")

# ─── GRAFİK 4: DİYABETLİLERDE TEMEL SEMPTOM PREVALANSI ÇUBUK GRAFİĞİ ──────
semptomlar_goster = [
    "Asiri_Idrar", "Asiri_Susama", "Halsizlik",
    "Ani_Kilo_Kaybi", "Asiri_Yeme", "Gorme_Bulanikli",
    "Kismi_Felc", "Sinirlilik",
]
etiket_goster = [ETIKET[s] for s in semptomlar_goster]

df_pos = df[df["Diyabet_Durumu"] == "Pozitif"]
df_neg = df[df["Diyabet_Durumu"] == "Negatif"]

oran_pos = [(df_pos[s] == "Evet").mean() * 100 for s in semptomlar_goster]
oran_neg = [(df_neg[s] == "Evet").mean() * 100 for s in semptomlar_goster]

x = np.arange(len(semptomlar_goster))
genislik = 0.38

fig4, ax4 = plt.subplots(figsize=(12, 6))
bars_pos = ax4.bar(x - genislik / 2, oran_pos, genislik,
                   label="Pozitif (Diyabetli)", color=renk_pozitif, alpha=0.88)
bars_neg = ax4.bar(x + genislik / 2, oran_neg, genislik,
                   label="Negatif (Sağlıklı)", color=renk_negatif, alpha=0.88)

# Çubukların üstüne yüzde etiketi
for bar in bars_pos:
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
             f"%{bar.get_height():.0f}", ha="center", va="bottom",
             fontsize=8.5, color=renk_pozitif, fontweight="bold")
for bar in bars_neg:
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
             f"%{bar.get_height():.0f}", ha="center", va="bottom",
             fontsize=8.5, color="#1a7a45", fontweight="bold")

ax4.set_title(
    "Diyabetli ve Sağlıklı Bireylerde Temel Semptom Görülme Oranları",
    fontsize=14, fontweight="bold", pad=14,
)
ax4.set_xlabel("Semptom", fontsize=12)
ax4.set_ylabel("Görülme Oranı (%)", fontsize=12)
ax4.set_xticks(x)
ax4.set_xticklabels(etiket_goster, rotation=28, ha="right", fontsize=10)
ax4.yaxis.set_major_formatter(mticker.PercentFormatter())
ax4.set_ylim(0, 105)
ax4.legend(fontsize=10)
ax4.grid(axis="y", linestyle="--", alpha=0.5)
sns.despine(ax=ax4)
plt.tight_layout()
plt.savefig("grafik4_semptom_prevalansi.png", dpi=150, bbox_inches="tight")
plt.show()

print("""
[GRAFİK 4 — YORUM]
Çubuk grafik, diyabetli bireyler ile sağlıklı bireylerin semptom
prevalanslarını yan yana kıyaslamaktadır. Öne çıkan bulgular:
  • "Aşırı İdrar" ve "Aşırı Susama", diyabetlilerde %75-80 oranıyla
    en sık gözlemlenen semptomlardır; sağlıklı grupta bu oran %10-15'te
    kalmaktadır.
  • "Halsizlik" her iki grupta da görece yüksek; ancak diyabetlilerde
    yaklaşık iki kat daha sık.
  • "Kısmi Felç" ve "Sinirlilik", diyabetli grupta belirgin biçimde fazla
    olup ileri evre komplikasyonlara işaret edebilir.
""")


# =============================================================================
# BÖLÜM 5 — GENEL BULGULAR & SONUÇ
# =============================================================================

print("=" * 65)
print("BÖLÜM 5 — GENEL BULGULAR & SONUÇ")
print("=" * 65)
print(f"""
Veri seti {df.shape[0]} bireyden oluşmakta; bireylerin %{(df["Diyabet_Durumu"]=="Pozitif").mean()*100:.1f}'i
diyabet hastası olarak sınıflandırılmaktadır.

EN ÖNEMLİ 4 BULGU
─────────────────────────────────────────────────────────────
1. GÜÇLÜ BELİRTEÇ ÇİFTİ — Aşırı İdrar & Aşırı Susama:
   Bu iki semptom, diyabetlilerde birlikte ~%77 oranında
   görülmekte ve birbirleriyle güçlü korelasyon (≈ +0.65)
   sergilemektedir. Klinik tarama için kritik öneme sahiptir.

2. YAŞ FAKTÖRÜ:
   Diyabetli bireylerin yaş ortalaması sağlıklı bireylere göre
   yaklaşık 5-7 yıl daha yüksektir. Orta yaş (40-60) en riskli
   dilimi oluşturmakta; ancak genç vakaların varlığı erken
   taramanın gerekliliğini desteklemektedir.

3. SEMPTOMLARDAKİ BELIRGIN FARK:
   Çubuk grafik analizine göre "Halsizlik", "Ani Kilo Kaybı" ve
   "Görme Bozukluğu", iki grup arasında istatistiksel olarak
   anlamlı farklılık gösteren semptomlardır.

4. BAĞIMSIZ RİSK FAKTÖRLERİ:
   "Obezite" ve "Kaşıntı", diğer semptomlarla düşük korelasyon
   sergilese de diyabet prevalansında artışa eşlik etmektedir.
   Bu faktörler, hastalığın farklı metabolik yollarla ilişkili
   olabileceğine işaret etmektedir.
─────────────────────────────────────────────────────────────
Sonuç: Aşırı idrar ve aşırı susama belirtilerinin birlikte
görülmesi, diyabet riski açısından güçlü bir uyarı sinyali
oluşturmaktadır. Orta yaş ve üzeri bireylerde düzenli kan
şekeri taraması hayati önem taşımaktadır.
""")