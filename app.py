import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import xml.etree.ElementTree as ET
from urllib.request import urlopen, Request

warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# ==============================================================================
# GEREKLİ EDA (VERİ TANIMA) FONKSİYONLARI (SUNUM FORMATLI)
# ==============================================================================
def check_df(dataframe, head=5):
    border = "=" * 80
    print(f"\n{border}\n📊 VERİ SETİ BOYUTU (SHAPE)\n{border}")
    print(f"Toplam Gözlem (Satır) : {dataframe.shape[0]:,}")
    print(f"Toplam Öznitelik (Sütun): {dataframe.shape[1]:,}")

    print(f"\n{border}\n📋 İLK {head} GÖZLEM (HEAD)\n{border}")
    print(dataframe.head(head).to_string())

    print(f"\n{border}\n🧬 VERİ TİPLERİ (TYPES)\n{border}")
    print(dataframe.dtypes.to_string())

    print(f"\n{border}\n🔍 EKSİK VERİ ANALİZİ (NA)\n{border}")
    na_counts = dataframe.isnull().sum()
    na_pct = (dataframe.isnull().sum() / len(dataframe)) * 100
    na_df = pd.DataFrame({"Eksik Değer": na_counts, "Oran (%)": na_pct.round(2)})
    print(na_df.to_string())

    print(f"\n{border}\n📈 SAYISAL DAĞILIM ANALİZİ (QUANTILES)\n{border}")
    num_cols = dataframe.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        quantiles_df = dataframe[num_cols].quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T
        print(quantiles_df.round(4).to_string())
    else:
        print("💡 [Bilgi]: Veri setinde sayısal sütun bulunamadı.")
    print(f"\n{border}\n")


def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car


def correlation_matrix(df, cols):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, fmt=".2f", cmap='RdBu', annot_kws={'size': 10})
    plt.title("🔥 Öznitelik Korelasyon Sıcaklık Haritası (Feature Correlation Heatmap)", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show(block=True)


# ==============================================================================
# DATA LOADING
# ==============================================================================
CSV_PATH = r"C:\Users\tbardakc1\OneDrive - Philip Morris International\Desktop\bist_stock_data.csv"
if not os.path.exists(CSV_PATH):
    CSV_PATH = "bist_stock_data.csv"

print("📥 Veri yükleniyor...")
df_raw = pd.read_csv(CSV_PATH)
df_raw.columns = df_raw.columns.str.strip()
if 'Date' in df_raw.columns:
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])

# ==============================================================================
# 1. ADIM: VERİ TEMİZLEME
# ==============================================================================
ohlc_mask = (
        (df_raw["Low"] <= df_raw["Open"]) &
        (df_raw["Low"] <= df_raw["Close"]) &
        (df_raw["High"] >= df_raw["Open"]) &
        (df_raw["High"] >= df_raw["Close"])
)
df_ohlc_filtered = df_raw.loc[ohlc_mask].copy()
df_model_ready = df_ohlc_filtered[df_ohlc_filtered['Volume'] > 0].copy()

# ==============================================================================
# 2. ADIM: FEATURE ENGINEERING (ÖZNİTELİK MÜHENDİSLİĞİ)
# ==============================================================================
print("💡 Tüm havuz için finansal öznitelikler hesaplanıyor...")
df = df_model_ready.sort_values(["Stock", "Date"])
g = df.groupby("Stock")

df["Return_1D"] = g["Close"].pct_change()
df["Return_5D"] = g["Close"].pct_change(5)
df["Return_10D"] = g["Close"].pct_change(10)

df["MA_5"] = g["Close"].transform(lambda x: x.rolling(5).mean())
df["MA_20"] = g["Close"].transform(lambda x: x.rolling(20).mean())
df["MA_50"] = g["Close"].transform(lambda x: x.rolling(50).mean())
df["MA20_RATIO"] = df["Close"] / df["MA_20"]
df["MA50_RATIO"] = df["Close"] / df["MA_50"]

df["Volatility_5D"] = g["Return_1D"].transform(lambda x: x.rolling(5).std())
df["Volatility_20D"] = g["Return_1D"].transform(lambda x: x.rolling(20).std())

df["HL_PCT"] = (df["High"] - df["Low"]) / df["Close"]
df["CO_PCT"] = (df["Close"] - df["Open"]) / df["Open"]

df["VOL_MA_5"] = g["Volume"].transform(lambda x: x.rolling(5).mean())
df["VOL_MA_20"] = g["Volume"].transform(lambda x: x.rolling(20).mean())
df["VOL_RATIO"] = df["Volume"] / df["VOL_MA_20"]


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


df["RSI_14"] = g["Close"].transform(compute_rsi)

df["Lag_Close_1"] = g["Close"].shift(1)
df["Lag_Close_5"] = g["Close"].shift(5)
df["Lag_Return_1"] = g["Return_1D"].shift(1)

df["BB_MID"] = g["Close"].transform(lambda x: x.rolling(20).mean())
df["BB_STD"] = g["Close"].transform(lambda x: x.rolling(20).std())
df["BB_UPPER"] = df["BB_MID"] + 2 * df["BB_STD"]
df["BB_LOWER"] = df["BB_MID"] - 2 * df["BB_STD"]
df["BB_POSITION"] = (df["Close"] - df["BB_LOWER"]) / (df["BB_UPPER"] - df["BB_LOWER"] + 1e-9)

df["EMA_12"] = g["Close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
df["EMA_26"] = g["Close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
df["MACD"] = df["EMA_12"] - df["EMA_26"]
df["MACD_SIGNAL"] = g["MACD"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
df["MACD_DIFF"] = df["MACD"] - df["MACD_SIGNAL"]

df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.dayofweek

df["Target_Vol"] = g["Return_1D"].transform(lambda x: x.rolling(5).std()).shift(-1)
df["Stock_Code"] = df["Stock"].astype("category").cat.codes

features = [
    "Stock_Code", "Return_1D", "Return_5D", "Return_10D",
    "MA20_RATIO", "MA50_RATIO", "Volatility_5D", "Volatility_20D",
    "HL_PCT", "CO_PCT", "VOL_RATIO", "RSI_14",
    "Lag_Return_1", "BB_POSITION", "MACD_DIFF", "Month", "Day"
]

df = df.replace([np.inf, -np.inf], np.nan)
df_clean = df.dropna(subset=["Target_Vol"] + features)

# ==============================================================================
# 🎯 SUNUM BAŞLANGICI - KEŞİFÇİ VERİ ANALİZİ (EDA) SEKANSI
# ==============================================================================
print("\n" + "=" * 100)
print("📢 SUNUM 1. BÖLÜM: VERİ TANIMA VE KEŞİFÇİ VERİ ANALİZİ (EDA)")
print("=" * 100)

check_df(df_clean, head=5)

cat_cols, num_cols, cat_but_car = grab_col_names(df_clean, cat_th=5, car_th=20)

print("\n🔥 Özniteliklerin Korelasyon Haritası Çiziliyor... (Grafiği kapatarak devam edin.)")
correlation_matrix(df_clean, features)

# ==============================================================================
# 3. ADIM: SUNUM İÇİN MODEL YARIŞTIRMA VE LİDERLİK TABLOSU
# ==============================================================================
print("\n" + "=" * 100)
print("📢 SUNUM 2. BÖLÜM: DEV GLOBAL VERİ SETİNDE ALGORİTMA YARIŞI VE OPTİMİZASYON")
print("=" * 100)

train_pool = df_clean[df_clean["Date"] < "2023-01-01"].sort_values("Date")
test_pool = df_clean[df_clean["Date"] >= "2023-01-01"].sort_values("Date")

X_train = train_pool[features]
y_train = train_pool["Target_Vol"]
X_test = test_pool[features]
y_test = test_pool["Target_Vol"]

print(f"📊 Havuz Özet -> Eğitim Sınıfı: {X_train.shape[0]:,} satır | Test Sınıfı: {X_test.shape[0]:,} satır")
print("🏁 HİPERPARAMETRE OPTİMİZASYONU BAŞLADI...\n")

model_candidates = {
    'LightGBM': {
        'model': LGBMRegressor(random_state=42, verbose=-1),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [4, 6, 8],
            'subsample': [0.7, 0.9],
            'colsample_bytree': [0.7, 0.9]
        }
    },
    'XGBoost': {
        'model': XGBRegressor(random_state=42, verbosity=0),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [4, 6, 8],
            'subsample': [0.7, 0.9],
            'colsample_bytree': [0.7, 0.9]
        }
    },
    'CatBoost': {
        'model': CatBoostRegressor(random_state=42, verbose=0),
        'params': {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5]
        }
    }
}

tscv = TimeSeriesSplit(n_splits=3)
report_data = []
trained_models = {}

for model_name, config in model_candidates.items():
    print(f"🔄 {model_name} için en iyi parametreler aranıyor...")

    search = RandomizedSearchCV(
        estimator=config['model'],
        param_distributions=config['params'],
        n_iter=8,
        scoring='neg_mean_absolute_error',
        cv=tscv,
        random_state=42,
        n_jobs=-1
    )

    try:
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_

        best_model.fit(X_train, y_train)
        preds = best_model.predict(X_test)

        test_mae = mean_absolute_error(y_test, preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, preds))
        test_r2 = r2_score(y_test, preds)

        report_data.append({
            "Model": model_name,
            "En İyi Parametreler": best_params,
            "Test MAE": round(test_mae, 6),
            "Test RMSE": round(test_rmse, 6),
            "Test R2 (%)": round(test_r2 * 100, 2)
        })
        trained_models[model_name] = {"model_obj": best_model, "r2": test_r2}
        print(f"✅ {model_name} başarıyla optimize edildi!")

    except Exception as e:
        print(f"❌ {model_name} hatayla karşılaştı: {str(e)}")

# Liderlik Tablosu
df_report = pd.DataFrame(report_data)
df_report = df_report.sort_values(by="Test R2 (%)", ascending=False).reset_index(drop=True)

print("\n" + "=" * 95)
print("🏆 ALGORİTMA LİDERLİK TABLOSU (BENCHMARK REPORT)")
print("=" * 95)
for idx, row in df_report.iterrows():
    print(f"\n🥇 Sıra {idx + 1}: {row['Model']}")
    print(f"  -> Test MAE (Mutlak Hata):       {row['Test MAE']:.6f}")
    print(f"  -> Test RMSE (Büyük Hata Cezası): {row['Test RMSE']:.6f}")
    print(f"  -> Test R2 (Açıklayıcılık Oranı): %{row['Test R2 (%)']:.2f}")
    print(f"  ⚙️ En İyi Hiperparametreler: {row['En İyi Parametreler']}")
    print("-" * 95)

# ==============================================================================
# 4. ADIM: OTOMATİK ŞAMPİYON SEÇİMİ VE GELECEK SİNYAL ÜRETİMİ
# ==============================================================================
champion_name = df_report.loc[0, "Model"]
print("\n" + "=" * 100)
print(f"📢 SUNUM 3. BÖLÜM: CANLI UYGULAMA - ŞAMPİYON MODEL ({champion_name.upper()}) İLE SİNYAL ÜRETİMİ")
print("=" * 100)

X_all = df_clean[features]
y_all = df_clean["Target_Vol"]
final_champion = trained_models[champion_name]["model_obj"]

print(f"⚡ {champion_name} tüm geçmiş piyasa verisiyle nihai olarak eğitiliyor...")
final_champion.fit(X_all, y_all)

latest_rows = df.dropna(subset=features).groupby("Stock").tail(1).copy()
X_latest = latest_rows[features]

latest_rows["Yarın Beklenen Volatilite"] = final_champion.predict(X_latest)
latest_rows["Mevcut Volatilite"] = latest_rows["Volatility_5D"]

latest_rows["Değişim Beklentisi"] = np.where(
    latest_rows["Yarın Beklenen Volatilite"] > latest_rows["Mevcut Volatilite"],
    "YÜKSELİŞ 📈 (Hareket Başlıyor)",
    "DÜŞÜŞ 📉 (Tahta Sakinleşiyor)"
)

df_signals = latest_rows[["Stock", "Mevcut Volatilite", "Yarın Beklenen Volatilite", "Değişim Beklentisi"]]
df_signals["Mevcut Volatilite"] = df_signals["Mevcut Volatilite"].round(5)
df_signals["Yarın Beklenen Volatilite"] = df_signals["Yarın Beklenen Volatilite"].round(5)
df_signals = df_signals.sort_values(by="Yarın Beklenen Volatilite", ascending=False).reset_index(drop=True)

print("\n🔮 YARININ CANLI VOLATİLİTE VE RİSK SİNYAL RAPORU")
print("=" * 95)
print(df_signals.to_string(index=False))
print("=" * 95)

# ==============================================================================
# 🔥 GARANTİLİ YENİ YÖNTEM: GOOGLE NEWS ÜZERİNDEN TÜRKÇE HABER AKIŞI
# ==============================================================================
print("\n" + "=" * 100)
print("📰 SUNUM EK BÖLÜM: EN YÜKSEK HAREKET BEKLENEN İLK 10 HİSSENİN GOOGLE NEWS HABER AKIŞI")
print("=" * 100)

top_10_stocks = df_signals["Stock"].head(10).tolist()

for stock in top_10_stocks:
    # Arama terimini sadeleştiriyoruz (örn: THYAO.IS veya THYAO ise sadece THYAO alıp yanına 'hisse' ekliyoruz)
    clean_stock_name = stock.split('.')[0]
    search_query = f"{clean_stock_name}+hisse"

    print(f"\n🔍 {clean_stock_name} İle İlgili Güncel Finansal Gelişmeler:")
    print("-" * 50)

    # Google News RSS URL (Türkçe sonuçlar ve Türkiye lokasyonu için kurgulandı)
    rss_url = f"https://news.google.com/rss/search?q={search_query}&hl=tr&gl=TR&ceid=TR:tr"

    try:
        # Sunucu engeline takılmamak için tarayıcı gibi istek (Request) atıyoruz
        req = Request(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(req) as response:
            xml_data = response.read()

        # XML verisini parse ediyoruz
        root = ET.fromstring(xml_data)
        items = root.findall('.//item')

        if items and len(items) > 0:
            # Sunumu boğmamak adına en güncel 3 haberi listeliyoruz
            for i, item in enumerate(items[:3]):
                title = item.find('title').text if item.find('title') is not None else 'Başlık Yok'
                link = item.find('link').text if item.find('link') is not None else 'Link Yok'
                source = item.find('source').text if item.find('source') is not None else 'Google News'

                # Başlığın sonundaki kaynak bilgisini temizlemek (görsel şıklık için)
                if " - " in title:
                    title = " - ".join(title.split(" - ")[:-1])

                print(f" {i + 1}. [{source}] {title}")
                print(f"    🔗 {link}")
        else:
            print(" ⚠️ Bu hisseyle ilgili güncel Türkçe haber akışı bulunamadı.")

    except Exception as e:
        print(f" ❌ Haberler taranırken bir aksaklık yaşandı: {str(e)}")

print("\n" + "=" * 100)
print("🏁 SUNUM VE ANALİZ RAPORU BAŞARIYLA TAMAMLANDI!")
print("=" * 100)
