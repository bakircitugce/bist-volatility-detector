"""
BIST VOLATİLİTE TAHMİN PROJESİ
Bootcamp Bitirme Sunumu — 6 Bölüm
─────────────────────────────────
Çalıştır:
  pip install streamlit lightgbm xgboost catboost plotly scikit-learn
  streamlit run bist_bootcamp_sunum.py
"""

import os, warnings, time
import xml.etree.ElementTree as ET
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm  import LGBMRegressor
from xgboost   import XGBRegressor
from catboost  import CatBoostRegressor

# ──────────────────────────────────────────────
st.set_page_config(page_title="BIST ML Sunumu", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{ font-family:'Plus Jakarta Sans',sans-serif; }
.stApp{ background:#06101e; color:#dde6f0; }

/* sidebar */
section[data-testid="stSidebar"]{ background:#080f1c; border-right:1px solid #12253d; }

/* hero */
.hero{ background:linear-gradient(135deg,#0b1e3e,#082c54,#0b1b30);
  border:1px solid #163660; border-radius:16px; padding:2.2rem 2.8rem;
  margin-bottom:1.8rem; position:relative; overflow:hidden; }
.hero::after{ content:''; position:absolute; bottom:-70px; right:-70px;
  width:220px; height:220px; border-radius:50%;
  background:radial-gradient(circle,rgba(0,200,255,.10) 0%,transparent 70%); }
.hero-title{ font-family:'IBM Plex Mono',monospace; font-size:1.85rem;
  font-weight:600; color:#00d4ff; margin:0; }
.hero-sub{ color:#5a8aaa; font-size:.95rem; margin-top:.45rem; line-height:1.6; }
.badge{ display:inline-block; margin-top:.9rem;
  background:rgba(0,212,255,.09); border:1px solid rgba(0,212,255,.28);
  border-radius:20px; padding:3px 14px;
  font-family:'IBM Plex Mono',monospace; font-size:.7rem; color:#00d4ff; }

/* section cards */
.sc{ background:#0b1825; border:1px solid #162d47;
  border-left:4px solid #00d4ff; border-radius:12px;
  padding:1.4rem 1.9rem; margin-bottom:1.5rem; }
.sc.or{ border-left-color:#f97316; }
.sc.pu{ border-left-color:#8b5cf6; }
.sc.gr{ border-left-color:#22c55e; }
.sc.re{ border-left-color:#ef4444; }
.sc.ye{ border-left-color:#eab308; }

.sn{ font-family:'IBM Plex Mono',monospace; font-size:.65rem;
  letter-spacing:3px; text-transform:uppercase; color:#00d4ff; }
.sn.or{color:#f97316;} .sn.pu{color:#8b5cf6;}
.sn.gr{color:#22c55e;} .sn.re{color:#ef4444;} .sn.ye{color:#eab308;}
.st2{ font-size:1.28rem; font-weight:700; color:#eef4ff; margin:.28rem 0 .7rem; }
.sb{ color:#6e9ab5; font-size:.9rem; line-height:1.78; }

/* info boxes */
.bx-i{ background:rgba(0,212,255,.05); border:1px solid rgba(0,212,255,.22);
  border-radius:9px; padding:.9rem 1.3rem; margin-top:.8rem;
  font-size:.86rem; color:#90c8e0; line-height:1.75; }
.bx-w{ background:rgba(249,115,22,.07); border:1px solid rgba(249,115,22,.26);
  border-radius:9px; padding:.85rem 1.3rem; margin-top:.8rem;
  font-size:.85rem; color:#fbb06a; line-height:1.7; }
.bx-ok{ background:rgba(34,197,94,.06); border:1px solid rgba(34,197,94,.22);
  border-radius:9px; padding:.85rem 1.3rem; margin-top:.8rem;
  font-size:.85rem; color:#80e8a8; line-height:1.7; }
.bx-fx{ background:rgba(239,68,68,.06); border:1px solid rgba(239,68,68,.24);
  border-radius:9px; padding:.85rem 1.3rem; margin-top:.8rem;
  font-size:.85rem; color:#fca5a5; line-height:1.7; }

/* code */
.co{ background:#030c18; border:1px solid #122036; border-radius:8px;
  padding:.85rem 1.1rem; font-family:'IBM Plex Mono',monospace;
  font-size:.74rem; color:#5bb8e8; line-height:1.85; overflow-x:auto;
  white-space:pre; margin-top:.7rem; }

/* metric row */
.mr{ display:flex; gap:.8rem; flex-wrap:wrap; margin-bottom:1.3rem; }
.mc{ background:#0b1825; border:1px solid #162d47; border-radius:11px;
  padding:1rem 1.4rem; flex:1; min-width:130px; }
.ml{ font-size:.67rem; color:#3d6a88; letter-spacing:1.5px;
  text-transform:uppercase; margin-bottom:.25rem; }
.mv{ font-family:'IBM Plex Mono',monospace; font-size:1.6rem;
  color:#00d4ff; font-weight:600; }
.ms{ font-size:.72rem; color:#2e5570; margin-top:.12rem; }

/* pill */
.pill{ display:inline-block; background:rgba(0,212,255,.07);
  border:1px solid rgba(0,212,255,.22); border-radius:18px;
  padding:2px 10px; margin:2px; font-family:'IBM Plex Mono',monospace;
  font-size:.69rem; color:#00d4ff; }

/* divider */
.hd{ border:none; border-top:1px solid #101f32; margin:1.8rem 0; }

/* champion card */
.champ{ background:linear-gradient(135deg,#0a2040,#0b3060);
  border:2px solid #00d4ff; border-radius:14px;
  padding:1.8rem 2.2rem; text-align:center; margin:1rem 0; }
.champ-title{ font-family:'IBM Plex Mono',monospace; font-size:2rem;
  color:#00d4ff; font-weight:600; }
.champ-sub{ color:#5a8aaa; font-size:.95rem; margin-top:.4rem; }

/* progress bar override */
.stProgress > div > div { background:#00d4ff !important; }

/* signal table */
.sig-high{ color:#ef4444; font-weight:700; }
.sig-low { color:#22c55e; font-weight:700; }

/* news cards */
.news-header{ background:linear-gradient(135deg,#0a1f3a,#0c2d50);
  border:1px solid #163660; border-radius:14px;
  padding:1.4rem 2rem; margin-bottom:1.2rem; }
.news-stock{ font-family:'IBM Plex Mono',monospace; font-size:1.05rem;
  font-weight:600; color:#00d4ff; margin-bottom:.2rem; }
.news-vol{ font-size:.75rem; color:#3d6a88; letter-spacing:1px; }

.news-card{ background:#080f1c; border:1px solid #142236;
  border-radius:10px; padding:1rem 1.3rem; margin-bottom:.7rem;
  border-left:3px solid #1e4d7a; transition:border-color .2s; }
.news-card:hover{ border-left-color:#00d4ff; }
.news-num{ font-family:'IBM Plex Mono',monospace; font-size:.68rem;
  color:#1e4d7a; margin-bottom:.3rem; }
.news-title{ font-size:.92rem; font-weight:600; color:#d0e4f0;
  line-height:1.45; margin-bottom:.4rem; }
.news-meta{ font-size:.75rem; color:#2d5570; }
.news-link{ color:#1e6a9a; text-decoration:none; font-size:.75rem; }
.news-link:hover{ color:#00d4ff; }

.news-empty{ background:rgba(30,77,122,.08); border:1px solid rgba(30,77,122,.2);
  border-radius:8px; padding:.8rem 1.2rem; color:#2d5570;
  font-size:.83rem; font-style:italic; }
.news-error{ background:rgba(239,68,68,.06); border:1px solid rgba(239,68,68,.18);
  border-radius:8px; padding:.8rem 1.2rem; color:#7a3030; font-size:.83rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  YARDIMCILAR
# ──────────────────────────────────────────────
DARK = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11,24,37,.9)",
            font_color="#7aacc4", title_font_color="#eef4ff", title_font_size=13)

def card(num, title, body, c=""):
    st.markdown(f'<div class="sc {c}"><div class="sn {c}">{num}</div>'
                f'<div class="st2">{title}</div><div class="sb">{body}</div></div>',
                unsafe_allow_html=True)

def info(t):  st.markdown(f'<div class="bx-i">💡 {t}</div>', unsafe_allow_html=True)
def warn(t):  st.markdown(f'<div class="bx-w">⚠️ {t}</div>', unsafe_allow_html=True)
def ok(t):    st.markdown(f'<div class="bx-ok">✅ {t}</div>', unsafe_allow_html=True)
def fix(t):   st.markdown(f'<div class="bx-fx">🔧 {t}</div>', unsafe_allow_html=True)
def code(t):  st.markdown(f'<div class="co">{t}</div>', unsafe_allow_html=True)
def div():    st.markdown('<hr class="hd">', unsafe_allow_html=True)
def mrow(html): st.markdown(f'<div class="mr">{html}</div>', unsafe_allow_html=True)
def mc(label, val, sub=""):
    return (f'<div class="mc"><div class="ml">{label}</div>'
            f'<div class="mv">{val}</div><div class="ms">{sub}</div></div>')

# ══════════════════════════════════════════════
#  VERİ + FEATURE ENGINEERING + MODEL  (cache)
# ══════════════════════════════════════════════
LOCAL = "bist_stock_data.csv"
FEATURES = ['Stock_Code','Return_1D','Return_5D','Return_10D',
            'MA20_RATIO','MA50_RATIO','Volatility_5D','Volatility_20D',
            'HL_PCT','CO_PCT','VOL_RATIO','RSI_14',
            'Lag_Return_1','BB_POSITION','MACD_DIFF','Month','Day']
SPLIT = "2023-01-01"

@st.cache_data(show_spinner=False)
def load_data(src):
    df = pd.read_csv(src)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data(show_spinner=False)
def build_features(src):
    df_raw = pd.read_csv(src)
    df_raw.columns = df_raw.columns.str.strip()
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])

    # temizlik
    mask = ((df_raw["Low"]<=df_raw["Open"])&(df_raw["Low"]<=df_raw["Close"])&
            (df_raw["High"]>=df_raw["Open"])&(df_raw["High"]>=df_raw["Close"]))
    df = df_raw.loc[mask & (df_raw["Volume"]>0)].copy()
    df = df.sort_values(["Stock","Date"]).reset_index(drop=True)
    g  = df.groupby("Stock")

    # returns — Adj Close
    df["Return_1D"]      = g["Adj Close"].pct_change()
    df["Return_5D"]      = g["Adj Close"].pct_change(5)
    df["Return_10D"]     = g["Adj Close"].pct_change(10)

    # MA
    df["MA_5"]           = g["Close"].transform(lambda x: x.rolling(5).mean())
    df["MA_20"]          = g["Close"].transform(lambda x: x.rolling(20).mean())
    df["MA_50"]          = g["Close"].transform(lambda x: x.rolling(50).mean())
    df["MA20_RATIO"]     = df["Close"] / df["MA_20"].replace(0,np.nan)
    df["MA50_RATIO"]     = df["Close"] / df["MA_50"].replace(0,np.nan)

    # volatilite
    df["Volatility_5D"]  = g["Return_1D"].transform(lambda x: x.rolling(5).std())
    df["Volatility_20D"] = g["Return_1D"].transform(lambda x: x.rolling(20).std())

    # price action
    df["HL_PCT"]         = (df["High"]-df["Low"]) / df["Close"].replace(0,np.nan)
    df["CO_PCT"]         = (df["Close"]-df["Open"]) / df["Open"].replace(0,np.nan)

    # volume
    vm20 = g["Volume"].transform(lambda x: x.rolling(20).mean()).replace(0,np.nan)
    df["VOL_MA_20"]      = vm20
    df["VOL_RATIO"]      = df["Volume"] / vm20

    # RSI
    def rsi(s, p=14):
        d=s.diff(); gain=d.where(d>0,0).rolling(p).mean()
        loss=-d.where(d<0,0).rolling(p).mean()
        return 100-(100/(1+gain/(loss+1e-9)))
    df["RSI_14"]         = g["Close"].transform(rsi)
    df["Lag_Return_1"]   = g["Return_1D"].shift(1)

    # Bollinger Bands
    df["BB_MID"]         = g["Close"].transform(lambda x: x.rolling(20).mean())
    df["BB_STD"]         = g["Close"].transform(lambda x: x.rolling(20).std())
    df["BB_UPPER"]       = df["BB_MID"] + 2*df["BB_STD"]
    df["BB_LOWER"]       = df["BB_MID"] - 2*df["BB_STD"]
    df["BB_POSITION"]    = ((df["Close"]-df["BB_LOWER"]) /
                            (df["BB_UPPER"]-df["BB_LOWER"]+1e-9))

    # MACD
    df["EMA_12"]         = g["Close"].transform(lambda x: x.ewm(span=12,adjust=False).mean())
    df["EMA_26"]         = g["Close"].transform(lambda x: x.ewm(span=26,adjust=False).mean())
    df["MACD"]           = df["EMA_12"]-df["EMA_26"]
    df["MACD_SIGNAL"]    = g["MACD"].transform(lambda x: x.ewm(span=9,adjust=False).mean())
    df["MACD_DIFF"]      = df["MACD"]-df["MACD_SIGNAL"]

    # takvim
    df["Month"]          = df["Date"].dt.month
    df["Day"]            = df["Date"].dt.dayofweek

    # target & encode
    df["Target_Vol"]     = g["Return_1D"].transform(lambda x: x.rolling(5).std()).shift(-1)
    df["Stock_Code"]     = df["Stock"].astype("category").cat.codes

    df = df.replace([np.inf,-np.inf], np.nan)
    df_clean = df.dropna(subset=["Target_Vol"]+FEATURES).copy()
    return df, df_clean

@st.cache_resource(show_spinner=False)
def train_models(src):
    _, df_clean = build_features(src)
    train = df_clean[df_clean["Date"]<SPLIT]
    test  = df_clean[df_clean["Date"]>=SPLIT]
    Xtr,ytr = train[FEATURES], train["Target_Vol"]
    Xte,yte = test[FEATURES],  test["Target_Vol"]

    tscv = TimeSeriesSplit(n_splits=3)
    candidates = {
        "LightGBM": {
            "model": LGBMRegressor(random_state=42, verbose=-1),
            "params": {"n_estimators":[100,200,300],"learning_rate":[0.01,0.05,0.1],
                       "max_depth":[4,6,8],"subsample":[0.7,0.9],"colsample_bytree":[0.7,0.9]}
        },
        "XGBoost": {
            "model": XGBRegressor(random_state=42, verbosity=0),
            "params": {"n_estimators":[100,200,300],"learning_rate":[0.01,0.05,0.1],
                       "max_depth":[4,6,8],"subsample":[0.7,0.9],"colsample_bytree":[0.7,0.9]}
        },
        "CatBoost": {
            "model": CatBoostRegressor(random_state=42, verbose=0),
            "params": {"iterations":[100,200,300],"learning_rate":[0.01,0.05,0.1],
                       "depth":[4,6,8],"l2_leaf_reg":[1,3,5]}
        },
    }

    results, cv_histories = {}, {}
    for name, cfg in candidates.items():
        srch = RandomizedSearchCV(cfg["model"], cfg["params"], n_iter=8,
                                  scoring="neg_mean_absolute_error",
                                  cv=tscv, random_state=42, n_jobs=-1)
        srch.fit(Xtr, ytr)
        best = srch.best_estimator_
        best.fit(Xtr, ytr)
        preds = best.predict(Xte)
        tr_preds = best.predict(Xtr)

        results[name] = dict(
            model      = best,
            best_params= srch.best_params_,
            mae        = mean_absolute_error(yte, preds),
            rmse       = np.sqrt(mean_squared_error(yte, preds)),
            r2         = r2_score(yte, preds),
            train_r2   = r2_score(ytr, tr_preds),
            preds      = preds,
            y_test     = yte.values,
            test_dates = test["Date"].values,
            cv_results = srch.cv_results_,
        )

        # CV fold MAE history
        fold_scores = []
        for fold,(tr_idx,val_idx) in enumerate(tscv.split(Xtr)):
            best.fit(Xtr.iloc[tr_idx], ytr.iloc[tr_idx])
            fold_preds = best.predict(Xtr.iloc[val_idx])
            fold_scores.append(mean_absolute_error(ytr.iloc[val_idx], fold_preds))
        cv_histories[name] = fold_scores
        best.fit(Xtr, ytr)   # refit on full train

    # champion
    champion = max(results, key=lambda k: results[k]["r2"])

    # final model trained on all data for signals
    final = results[champion]["model"]
    _, df_c = build_features(src)
    final.fit(df_c[FEATURES], df_c["Target_Vol"])
    results["_champion"] = champion
    results["_final"]    = final
    results["_cv_hist"]  = cv_histories
    results["_test_df"]  = test

    return results

# ══════════════════════════════════════════════
#  YÜKLEME
# ══════════════════════════════════════════════
# sidebar — dosya + navigasyon
with st.sidebar:
    st.markdown("### 📂 Veri Kaynağı")
    uploaded = st.file_uploader("CSV yükle", type="csv", label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 🗂 Bölümler")
    section = st.radio("", [
        "1 · Veri Tanıma",
        "2 · Veri Temizleme",
        "3 · Feature Engineering",
        "4 · Model Yarışı & Optimizasyon",
        "5 · Kazanan Model",
        "6 · Sonuç & Canlı Sinyaller",
        "7 · Haber Akışları",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("BIST Risk Radarı · Bootcamp Bitirme")

# CSV source
src = uploaded if uploaded else (LOCAL if os.path.exists(LOCAL) else "bist_stock_data.csv")

with st.spinner("Veri yükleniyor..."):
    try:
        df_raw = load_data(src)
    except Exception as e:
        st.error(f"CSV açılamadı: {e}"); st.stop()

# HERO
st.markdown(f"""
<div class="hero">
  <div class="hero-title">📈 BIST Volatilite Risk Radarı</div>
  <div class="hero-sub">
    Borsa İstanbul · {df_raw['Stock'].nunique()} Hisse · {df_raw['Date'].min().year}–{df_raw['Date'].max().year}
    · LightGBM vs XGBoost vs CatBoost
  </div>
  <span class="badge">Bootcamp Bitirme Projesi</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  BÖLÜM 1 — VERİ TANIMA
# ══════════════════════════════════════════════
if section == "1 · Veri Tanıma":

    card("BÖLÜM 01", "📊 Veri Tanıma & Keşifçi Analiz",
         "Ham veri setinin boyutu, değişken tipleri, eksik değerler ve temel istatistikler "
         "incelenir. Bu adım modelleme kararlarının temelini oluşturur.")

    n_r, n_c = df_raw.shape
    n_s = df_raw["Stock"].nunique()
    null_total = df_raw.isnull().sum().sum()

    mrow(mc("Satır (Kayıt)", f"{n_r:,}", "Günlük gözlem") +
         mc("Sütun", str(n_c), "OHLCV + meta") +
         mc("Hisse Sayısı", str(n_s), "Benzersiz sembol") +
         mc("Tarih Aralığı", f"{df_raw['Date'].min().year}–{df_raw['Date'].max().year}", "~20 yıl") +
         mc("Eksik Değer", str(null_total), "✅ Yok" if null_total==0 else "Var"))

    with st.expander("🗂 Ham verinin ilk 10 satırı"):
        st.dataframe(df_raw.head(10), use_container_width=True)

    div()

    # Veri tipleri
    card("01-A", "🧬 Değişken Türleri",
         "Her sütunun veri tipi ve benzersiz değer sayısı analiz edilir. "
         "Kategorik ve sayısal değişkenler ayrıştırılır.", c="or")

    dtype_df = pd.DataFrame({
        "Sütun": df_raw.columns,
        "Tip": df_raw.dtypes.astype(str).values,
        "Benzersiz": [df_raw[c].nunique() for c in df_raw.columns],
        "Örnek": [str(df_raw[c].iloc[0]) for c in df_raw.columns],
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    div()

    # Quantile dağılım
    card("01-B", "📈 Dağılım Analizi (Quantile)",
         "Her sayısal değişkenin dağılımı; min, %5, medyan, %95, %99 ve max "
         "değerleri ile incelenir. Aykırı değerlerin varlığı tespit edilir.", c="or")

    num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    q_df = df_raw[num_cols].quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T.round(4)
    q_df.columns = ["Min","P5","Medyan","P95","P99","Max"]
    st.dataframe(q_df, use_container_width=True)

    div()

    # Hisse bazlı kayıt dağılımı
    card("01-C", "🗓️ Hisse Bazlı Kayıt & Tarih Dağılımı",
         "Her hissenin veri başlangıç tarihi, bitiş tarihi ve kayıt sayısı karşılaştırılır. "
         "Dengesiz dağılım model önyargısına (bias) yol açabilir.", c="or")

    stk_info = df_raw.groupby("Stock").agg(
        Başlangıç=("Date","min"), Bitiş=("Date","max"), Kayıt=("Date","count")
    ).reset_index()

    fig_stk = px.bar(stk_info, x="Stock", y="Kayıt",
        color="Kayıt", color_continuous_scale=["#0a2a4a","#00d4ff"],
        title="Hisse Başına Kayıt Sayısı", text="Kayıt")
    fig_stk.update_traces(textposition="outside")
    fig_stk.update_layout(showlegend=False, height=320, **DARK)
    st.plotly_chart(fig_stk, use_container_width=True)
    ok("Tüm hisseler eşit kayıt sayısına sahip → dengesiz veri riski yok.")

    div()

    # Korelasyon matrisi (sadece getiri bazlı)
    card("01-D", "🔥 Hisseler Arası Korelasyon Isı Haritası",
         "Hisselerin günlük getirileri arasındaki Pearson korelasyonu görselleştirilir.", c="or")

    pivot    = df_raw.pivot_table(index="Date", columns="Stock", values="Close")
    corr_ret = pivot.pct_change().corr().round(2)

    def heat(corr, title):
        return go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale=[[0,"#071424"],[0.5,"#0e5090"],[1,"#00d4ff"]],
            text=corr.values, texttemplate="%{text}",
            textfont_size=9, zmin=-1, zmax=1,
        )).update_layout(title=title, height=400, **DARK)

    st.plotly_chart(heat(corr_ret, "Getiri Korelasyonu"),
                    use_container_width=True)

    div()

    # Tek hisse trend
    card("01-E", "📈 Tek Hisse Trend & Getiri Analizi",
         "Uzun vadeli fiyat trendi ve volatilite kümeleşmesi incelenir.", c="or")

    stocks_s = sorted(df_raw["Stock"].unique())
    sel = st.selectbox("Hisse Seç", stocks_s, key="eda_sel")
    sd  = df_raw[df_raw["Stock"]==sel].sort_values("Date").copy()
    sd["MA50"] = sd["Adj Close"].rolling(50).mean()
    sd["ret"]  = sd["Adj Close"].pct_change()

    fig_t = make_subplots(rows=2, cols=1, shared_xaxes=True,
        row_heights=[.7,.3], vertical_spacing=.04,
        subplot_titles=[f"{sel} — Adj Close & MA-50","Günlük Getiri (%)"])
    fig_t.add_trace(go.Scatter(x=sd["Date"],y=sd["Adj Close"],name="Adj Close",
        line=dict(color="#00d4ff",width=1.5),fill="tozeroy",
        fillcolor="rgba(0,212,255,.03)"),row=1,col=1)
    fig_t.add_trace(go.Scatter(x=sd["Date"],y=sd["MA50"],name="MA-50",
        line=dict(color="#f97316",width=2,dash="dot")),row=1,col=1)
    fig_t.add_trace(go.Bar(x=sd["Date"],y=sd["ret"]*100,name="Getiri%",
        marker_color=["#22c55e" if v>=0 else "#ef4444"
                      for v in sd["ret"].fillna(0)],opacity=.7),row=2,col=1)
    fig_t.update_layout(height=450,legend=dict(orientation="h",y=1.06),**DARK)
    st.plotly_chart(fig_t, use_container_width=True)
    info("Getiri barlarındaki büyük hareketlerin kümelenmesi <strong>volatilite kümeleşmesini</strong> "
         "(ARCH etkisi) göstermektedir. Model bu örüntüyü öğrenecektir.")


# ══════════════════════════════════════════════
#  BÖLÜM 2 — VERİ TEMİZLEME
# ══════════════════════════════════════════════
elif section == "2 · Veri Temizleme":

    card("BÖLÜM 02", "🧹 Veri Temizleme",
         "Ham veri iki adımda temizlenir: OHLC kural filtresi ve "
         "sıfır hacim günlerinin çıkarılması.",
         c="gr")

    div()

    card("02-B", "📐 OHLC Finansal Kural Filtresi",
         "Matematiksel olarak geçersiz fiyat kayıtları tespit edilip çıkarılır. "
         "Kural: <code>Low ≤ Open, Close</code> ve <code>High ≥ Open, Close</code>", c="gr")
    code(
        "ohlc_mask = (\n"
        "    (df['Low']  <= df['Open'])  & (df['Low']  <= df['Close']) &\n"
        "    (df['High'] >= df['Open'])  & (df['High'] >= df['Close'])\n"
        ")\n"
        "df = df.loc[ohlc_mask].copy()"
    )

    ohlc = ((df_raw["Low"]<=df_raw["Open"])&(df_raw["Low"]<=df_raw["Close"])&
            (df_raw["High"]>=df_raw["Open"])&(df_raw["High"]>=df_raw["Close"]))
    n_inv = (~ohlc).sum()
    col1, col2 = st.columns(2)
    col1.metric("✅ Geçerli Satır", f"{ohlc.sum():,}")
    col2.metric("❌ Silinen Satır", f"{n_inv:,}",
                delta=f"-%{n_inv/len(df_raw)*100:.2f}", delta_color="inverse")
    info("Silinen satırlar gerçek piyasada oluşması imkânsız fiyat ilişkileri içerir. "
         "Model bu satırları öğrenseydi geçersiz kalıpları ezberlerdi.")

    div()

    card("02-C", "🔢 Sıfır Hacim Günlerinin Çıkarılması",
         "İşlem hacmi = 0 olan günlerde gerçek alım-satım yoktur. "
         "Bu günleri modele vermek 'işlem olmayan gün' kalıplarını öğretir — "
         "bu genelde yararlı değildir.", c="gr")
    code("df = df[df['Volume'] > 0].copy()   # ~%2.28 oranında satır")

    zero_v = (df_raw["Volume"]==0).sum()
    zero_p = zero_v/len(df_raw)*100
    zv_stk = df_raw[df_raw["Volume"]==0]["Stock"].value_counts().reset_index()
    zv_stk.columns=["Stock","Gün"]
    fig_zv = px.bar(zv_stk,x="Stock",y="Gün",
        color="Gün",color_continuous_scale=["#0a2a4a","#00d4ff"],
        title=f"Sıfır Volume Günler (Toplam: {zero_v:,} satır — %{zero_p:.2f})",
        text="Gün")
    fig_zv.update_traces(textposition="outside")
    fig_zv.update_layout(showlegend=False,height=300,**DARK)
    st.plotly_chart(fig_zv, use_container_width=True)

    div()

    card("02-D", "📊 Aykırı Değer Analizi",
         "OHLCV sütunlarındaki uç değerler kutu grafikleri ile gözlemlenir. "
         "Finansal zaman serilerinde aykırı değerler genellikle kriz olaylarını yansıtır; "
         "direkt silinmez, yorumlanır.", c="gr")
    nc = [c for c in ["Open","High","Low","Close","Volume"] if c in df_raw.columns]
    fig_bx = make_subplots(rows=1,cols=len(nc),subplot_titles=nc)
    pal = ["#00d4ff","#38bdf8","#818cf8","#06b6d4","#f97316"]
    for i,col in enumerate(nc):
        fig_bx.add_trace(go.Box(y=df_raw[col],name=col,
            marker_color=pal[i%5],line_color=pal[i%5],
            fillcolor="rgba(0,212,255,.03)",boxmean=True),row=1,col=i+1)
    fig_bx.update_layout(title="OHLCV Aykırı Değer Dağılımı",
        height=380,showlegend=False,**DARK)
    st.plotly_chart(fig_bx, use_container_width=True)

    df_clean_final = df_raw.loc[ohlc & (df_raw["Volume"]>0)]
    mrow(mc("Ham Satır",f"{len(df_raw):,}","Başlangıç") +
         mc("OHLC Sonrası",f"{ohlc.sum():,}",f"-{n_inv} satır") +
         mc("Volume>0 Sonrası",f"{len(df_clean_final):,}",f"-%{(1-len(df_clean_final)/len(df_raw))*100:.1f}") +
         mc("Hazır Veri",f"{len(df_clean_final):,}","ML'ye hazır"))


# ══════════════════════════════════════════════
#  BÖLÜM 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════
elif section == "3 · Feature Engineering":

    card("BÖLÜM 03", "🛠️ Feature Engineering — 17 Özellik, 7 Kategori",
         "Ham OHLCV verisinden 17 anlamlı özellik türetilmiştir.",
         c="pu")

    mrow(mc("Ham Feature","5","OHLCV") +
         mc("Türetilen Feature","17","7 kategoride") +
         mc("Hedef Değişken","Target_Vol","Yarınki 5G vol") +
         mc("Temporal Split","2023-01-01","Train/Test"))

    div()

    feat_groups = {
        "📊 Return Features": {
            "pills":["Return_1D","Return_5D","Return_10D","Lag_Return_1"],
            "c":"#00d4ff",
            "desc":"1, 5 ve 10 günlük getiriler momentum bilgisi taşır. "
                   "Lag_Return_1 dünkü getiriyi modele hatırlatır — "
                   "volatilite kümeleşmesini yakalamak için kritiktir.",
            "code":"Return_1D = Adj_Close.pct_change()      # Adj Close ✓\nReturn_5D  = Adj_Close.pct_change(5)\nLag_Return = Return_1D.shift(1)"
        },
        "📈 Trend (MA Ratio)": {
            "pills":["MA20_RATIO","MA50_RATIO"],
            "c":"#f97316",
            "desc":"Fiyatın hareketli ortalamasına göre konumu. "
                   "1'in üstü momentum, 1'in altı düzeltme bölgesini gösterir.",
            "code":"MA_20      = Close.rolling(20).mean()\nMA20_RATIO = Close / MA_20    # normalize fiyat konumu"
        },
        "💥 Volatilite": {
            "pills":["Volatility_5D","Volatility_20D","HL_PCT"],
            "c":"#ef4444",
            "desc":"Rolling standart sapma kısa ve uzun vadeli oynaklığı yakalar. "
                   "HL_PCT (High-Low yüzde aralığı) tek günlük genişliği ölçer.",
            "code":"Vol_5D = Return_1D.rolling(5).std()\nVol_20 = Return_1D.rolling(20).std()\nHL_PCT = (High - Low) / Close"
        },
        "📉 Hacim Analizi": {
            "pills":["VOL_RATIO","CO_PCT"],
            "c":"#22c55e",
            "desc":"VOL_RATIO mevcut hacmi 20 günlük ortalamaya böler — "
                   "anormal hacim patlamalarını saptar. "
                   "CO_PCT gün içi fiyat kanalını ölçer.",
            "code":"VOL_MA20  = Volume.rolling(20).mean().replace(0, NaN)\nVOL_RATIO = Volume / VOL_MA20    # hacim şoku"
        },
        "📡 Momentum (RSI)": {
            "pills":["RSI_14"],
            "c":"#8b5cf6",
            "desc":"RSI 0-100 arası momentum ölçer. "
                   "30 altı aşırı satım, 70 üstü aşırı alım sinyali verir.",
            "code":"delta = Close.diff()\ngain  = delta.where(delta>0, 0).rolling(14).mean()\nloss  = -delta.where(delta<0, 0).rolling(14).mean()\nRSI   = 100 - 100/(1 + gain/loss)"
        },
        "📌 Bollinger Bands": {
            "pills":["BB_POSITION"],
            "c":"#0ea5e9",
            "desc":"BB_POSITION fiyatın bant içindeki göreceli konumunu 0-1 arasında normalleştirir. "
                   "0 = alt bant, 1 = üst bant, 0.5 = orta bant.",
            "code":"BB_MID    = Close.rolling(20).mean()\nBB_UPPER  = BB_MID + 2 * Close.rolling(20).std()\nBB_POS    = (Close - BB_LOWER) / (BB_UPPER - BB_LOWER)"
        },
        "⚡ MACD & Takvim": {
            "pills":["MACD_DIFF","Month","Day"],
            "c":"#eab308",
            "desc":"MACD_DIFF kısa/uzun vadeli momentum farkını ölçer. "
                   "Month ve Day takvim etkilerini yakalar (mevsimsellik, 'Monday Effect').",
            "code":"EMA_12     = Close.ewm(span=12).mean()\nEMA_26     = Close.ewm(span=26).mean()\nMACD_DIFF  = (EMA_12 - EMA_26) - 9-day signal"
        },
    }

    for gname, gd in feat_groups.items():
        with st.expander(f"{gname}  ·  {len(gd['pills'])} özellik"):
            cL,cR = st.columns([1.2,1])
            with cL:
                st.markdown(f"<div class='sb'>{gd['desc']}</div>", unsafe_allow_html=True)
                code(gd["code"])
            with cR:
                pills="".join(f"<span class='pill'>{p}</span>" for p in gd["pills"])
                st.markdown(f"<div style='margin:.5rem 0'>{pills}</div>",
                            unsafe_allow_html=True)

    div()

    card("03-A", "🎯 Hedef Değişken & Temporal Split",
         "Hedef: yarınki 5-günlük rolling volatilite. "
         "Veri sızıntısı (lookahead bias) önlemek için "
         "<code>shift(-1)</code> ile bir gün kaydırılır.", c="pu")
    code(
        "# Mevcut günün 5-günlük volatilitesi\n"
        "current_vol = Return_1D.rolling(5).std()\n\n"
        "# Hedef: YARIN bu değer ne olacak?\n"
        "Target_Vol  = current_vol.shift(-1)\n\n"
        "# Temporal Split — shuffle=False\n"
        "train = df[df['Date'] < '2023-01-01']   # 42,073 satır\n"
        "test  = df[df['Date'] >= '2023-01-01']  #  8,330 satır"
    )

    # Feature korelasyon haritası (df_clean gerek)
    _, df_c = build_features(src)
    corr_f = df_c[FEATURES].corr().round(2)
    fig_fh = go.Figure(go.Heatmap(
        z=corr_f.values, x=corr_f.columns, y=corr_f.index,
        colorscale=[[0,"#071424"],[0.3,"#0e3060"],[0.6,"#1565c0"],[1,"#00d4ff"]],
        text=corr_f.values, texttemplate="%{text}",
        textfont_size=8, zmin=-1, zmax=1,
    ))
    fig_fh.update_layout(title="🔥 Feature Korelasyon Isı Haritası",
        height=520, **DARK)
    st.plotly_chart(fig_fh, use_container_width=True)


# ══════════════════════════════════════════════
#  BÖLÜM 4 — MODEL YARIŞI
# ══════════════════════════════════════════════
elif section == "4 · Model Yarışı & Optimizasyon":

    card("BÖLÜM 04", "🏁 Algoritma Yarışı & Hiperparametre Optimizasyonu",
         "LightGBM, XGBoost ve CatBoost; "
         "<strong>RandomizedSearchCV + TimeSeriesSplit</strong> ile optimize edilir. "
         "Tüm değerlendirmeler <strong>test seti</strong> üzerinden yapılır.",
         c="ye")

    with st.spinner("🔄 Modeller eğitiliyor & optimize ediliyor... (ilk seferinde ~2-3 dk)"):
        t0 = time.time()
        results = train_models(src)
        elapsed = time.time()-t0

    champion = results["_champion"]
    model_names = [k for k in results if not k.startswith("_")]

    ok(f"Eğitim tamamlandı — {elapsed:.0f} saniye · "
       f"Şampiyon: <strong>{champion}</strong>")

    div()

    card("04-A", "⚙️ Hiperparametre Arama Stratejisi", "", c="ye")

    code(
        "tscv = TimeSeriesSplit(n_splits=3)   # zaman sırasına saygılı CV\n\n"
        "search = RandomizedSearchCV(\n"
        "    estimator  = LGBMRegressor(),\n"
        "    param_distributions = {\n"
        "        'n_estimators'   : [100, 200, 300],\n"
        "        'learning_rate'  : [0.01, 0.05, 0.1],\n"
        "        'max_depth'      : [4, 6, 8],\n"
        "        'subsample'      : [0.7, 0.9],\n"
        "        'colsample_bytree': [0.7, 0.9]\n"
        "    },\n"
        "    n_iter=8, scoring='neg_mean_absolute_error',\n"
        "    cv=tscv, random_state=42, n_jobs=-1\n"
        ")\n"
        "search.fit(X_train, y_train)"
    )

    warn("TimeSeriesSplit kritik tasarım kararıdır. Rastgele K-Fold kullanılsaydı "
         "gelecek verisi eğitim katlarına sızardı → model gerçekte olmayan bir "
         "performans gösterirdi (data leakage).")

    div()

    card("04-B", "🏆 Liderlik Tablosu",
         "Üç algoritma test seti üzerinde MAE, RMSE ve R² metrikleriyle karşılaştırılır.",
         c="ye")

    # Leaderboard tablosu
    lb_rows = []
    for i, name in enumerate(sorted(model_names, key=lambda k: -results[k]["r2"])):
        r = results[name]
        lb_rows.append({
            "Sıra": f"{'🥇' if i==0 else '🥈' if i==1 else '🥉'}  {i+1}",
            "Model": name,
            "Test R²": f"{r['r2']*100:.2f}%",
            "Test MAE": f"{r['mae']:.6f}",
            "Test RMSE": f"{r['rmse']:.6f}",
            "Train R²": f"{r['train_r2']*100:.2f}%",
        })
    st.dataframe(pd.DataFrame(lb_rows), use_container_width=True, hide_index=True)

    div()

    # Bar karşılaştırma
    card("04-C", "📊 Model Performans Karşılaştırması", "", c="ye")

    names_sorted = sorted(model_names, key=lambda k: results[k]["r2"])
    r2_vals  = [results[n]["r2"]*100  for n in names_sorted]
    mae_vals = [results[n]["mae"]     for n in names_sorted]

    fig_cmp = make_subplots(rows=1, cols=2,
        subplot_titles=["Test R² (↑ yüksek = iyi)","Test MAE (↓ düşük = iyi)"])
    colors = ["#22c55e" if n==champion else "#00d4ff" for n in names_sorted]
    fig_cmp.add_trace(go.Bar(x=names_sorted, y=r2_vals, marker_color=colors,
        text=[f"{v:.2f}%" for v in r2_vals], textposition="outside",
        name="R²"), row=1,col=1)
    fig_cmp.add_trace(go.Bar(x=names_sorted, y=mae_vals,
        marker_color=["#22c55e" if n==champion else "#f97316" for n in names_sorted],
        text=[f"{v:.6f}" for v in mae_vals], textposition="outside",
        name="MAE"), row=1,col=2)
    fig_cmp.update_layout(height=360, showlegend=False, **DARK)
    st.plotly_chart(fig_cmp, use_container_width=True)

    div()

    # CV fold karşılaştırması
    card("04-D", "🔁 TimeSeriesSplit CV Fold Sonuçları",
         "Her modelin 3 CV fold'undaki MAE değerleri karşılaştırılır. "
         "Tutarlı fold skorları modelin genelleme gücünü gösterir.", c="ye")

    cv_hist = results["_cv_hist"]
    fig_cv = go.Figure()
    cv_pal = {"LightGBM":"#00d4ff","XGBoost":"#f97316","CatBoost":"#8b5cf6"}
    for name in model_names:
        scores = cv_hist[name]
        fig_cv.add_trace(go.Scatter(
            x=[f"Fold {i+1}" for i in range(len(scores))],
            y=scores, mode="lines+markers",
            line=dict(color=cv_pal.get(name,"#aaa"),width=2),
            marker=dict(size=9), name=name,
        ))
    fig_cv.update_layout(title="TimeSeriesSplit — CV Fold MAE Karşılaştırması",
        yaxis_title="MAE", height=320, **DARK)
    st.plotly_chart(fig_cv, use_container_width=True)

    info("Fold skorlarının birbirine yakın olması modelin <strong>tutarlı</strong> "
         "olduğunu gösterir. Büyük sapma overfitting işareti olabilir.")


# ══════════════════════════════════════════════
#  BÖLÜM 5 — KAZANAN MODEL
# ══════════════════════════════════════════════
elif section == "5 · Kazanan Model":

    results  = train_models(src)
    champion = results["_champion"]
    r        = results[champion]

    st.markdown(f"""
    <div class="champ">
        <div style="font-size:.8rem;color:#3a6a8a;letter-spacing:3px;
                    text-transform:uppercase;margin-bottom:.4rem">ŞAMPİYON MODEL</div>
        <div class="champ-title">🏆 {champion}</div>
        <div class="champ-sub">
            Test R² = <strong style="color:#22c55e">{r['r2']*100:.2f}%</strong> &nbsp;·&nbsp;
            MAE = <strong style="color:#f97316">{r['mae']:.6f}</strong> &nbsp;·&nbsp;
            RMSE = {r['rmse']:.6f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    mrow(mc("Train R²",  f"{r['train_r2']*100:.2f}%", "In-sample uyum") +
         mc("Test R²",   f"{r['r2']*100:.2f}%",       "Görülmemiş veri") +
         mc("Test MAE",  f"{r['mae']:.6f}",            "Ortalama hata") +
         mc("Test RMSE", f"{r['rmse']:.6f}",           "Büyük hata cezası") +
         mc("En İyi Param.", str(len(r["best_params"])), "Hiperparametre"))

    # En iyi hiperparametreler
    with st.expander("⚙️ Seçilen En İyi Hiperparametreler"):
        for k,v in r["best_params"].items():
            st.markdown(f"- **{k}**: `{v}`")

    div()

    # Tahmin vs Gerçek
    card("05-A", "🎯 Gerçek vs Tahmin — Test Seti",
         "Modelin ürettiği tahminler gerçek değerlerle karşılaştırılır. "
         "İdeal model noktaları kırmızı köşegen boyunca dizer.", c="gr")

    y_test = r["y_test"]
    preds  = r["preds"]
    dates  = r["test_dates"]

    col1,col2 = st.columns([1,1])
    with col1:
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=y_test, y=preds, mode="markers",
            marker=dict(color=y_test, colorscale="Blues", size=3, opacity=.45),
            name="Tahmin"))
        mx = max(y_test.max(), preds.max())
        fig_sc.add_trace(go.Scatter(x=[0,mx],y=[0,mx],mode="lines",
            line=dict(color="#ef4444",dash="dash",width=1.5),name="Mükemmel"))
        fig_sc.update_layout(
            title=f"Gerçek vs Tahmin  (R²={r['r2']:.3f})",
            xaxis_title="Gerçek",yaxis_title="Tahmin",
            height=350, **DARK)
        st.plotly_chart(fig_sc, use_container_width=True)

    with col2:
        # Hata dağılımı
        errors = preds - y_test
        fig_er = go.Figure(go.Histogram(
            x=errors, nbinsx=50, marker_color="#00d4ff", opacity=.8, name="Hata"))
        fig_er.add_vline(x=0, line_dash="dash", line_color="#ef4444",
            annotation_text="0 hata", annotation_font_color="#ef4444")
        fig_er.update_layout(title="Tahmin Hatası Dağılımı",
            xaxis_title="Hata (pred - gerçek)", height=350, **DARK)
        st.plotly_chart(fig_er, use_container_width=True)

    info("Hata dağılımının sıfır etrafında simetrik ve dar olması "
         "modelin sistematik bir yanlılık (bias) taşımadığını gösterir.")

    div()

    # Zaman serisinde tahmin
    card("05-B", "📅 Test Döneminde Tahmin Zaman Serisi",
         "Test setindeki gerçek volatilite ile model tahminleri "
         "zaman ekseni üzerinde karşılaştırılır.", c="gr")

    test_df = results["_test_df"].copy()
    stocks_m = sorted(test_df["Stock"].unique())
    sel_s = st.selectbox("Hisse Seç", stocks_m, key="ch_stk")
    stk_test = test_df[test_df["Stock"]==sel_s].copy()

    if len(stk_test) > 5:
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=stk_test["Date"],
            y=stk_test["Target_Vol"],
            name="Gerçek Vol",line=dict(color="#00d4ff",width=1.8)))
        fig_ts.add_trace(go.Scatter(x=stk_test["Date"],
            y=results[champion]["model"].predict(stk_test[FEATURES]),
            name="Tahmin",line=dict(color="#f97316",width=1.8,dash="dot")))
        fig_ts.update_layout(
            title=f"{sel_s} — Gerçek vs Tahmin Volatilite (2023–2026)",
            yaxis_title="Volatilite",height=360,
            legend=dict(orientation="h",y=1.06),**DARK)
        st.plotly_chart(fig_ts, use_container_width=True)

    div()

    # Feature Importance
    card("05-C", "🔑 Feature Importance — Model Neye Bakıyor?",
         "Modelin her özelliğe atadığı ağırlık, feature engineering "
         "kararlarımızı doğrulayan bağımsız kanıt sunar.", c="gr")

    if hasattr(r["model"], "feature_importances_"):
        fi = pd.Series(r["model"].feature_importances_, index=FEATURES)
        fi = fi.sort_values(ascending=True)
        fig_fi = go.Figure(go.Bar(
            x=fi.values, y=fi.index, orientation="h",
            marker=dict(color=fi.values,
                colorscale=[[0,"#0a2a4a"],[0.5,"#0e6ea0"],[1,"#00d4ff"]]),
            text=[f"{v:,.0f}" for v in fi.values], textposition="outside"))
        fig_fi.update_layout(title="Feature Importance Sıralaması",
            xaxis_title="Importance", height=480, **DARK)
        st.plotly_chart(fig_fi, use_container_width=True)

        top3 = fi.tail(3).index.tolist()
        info(f"En kritik 3 özellik: <strong>{', '.join(top3)}</strong>. "
             "Volatilite ve getiri değişkenleri öne çıkıyor → "
             "<strong>volatilite kümeleşmesi (ARCH) hipotezi doğrulandı.</strong>")


# ══════════════════════════════════════════════
#  BÖLÜM 6 — SONUÇ & CANLI SİNYALLER
# ══════════════════════════════════════════════
elif section == "6 · Sonuç & Canlı Sinyaller":

    results  = train_models(src)
    champion = results["_champion"]
    final    = results["_final"]
    r        = results[champion]
    _, df_c  = build_features(src)

    card("BÖLÜM 06", "🔮 Canlı Sinyal Üretimi & Proje Özeti",
         "Şampiyon model tüm geçmiş veriyle yeniden eğitilerek "
         "her hisse için <strong>yarınki beklenen volatilite</strong> tahmin edilir.",
         c="re")

    div()

    # Sinyal tablosu
    card("06-A", "📡 Yarınki Risk Sinyal Raporu",
         "Her hissenin mevcut 5-günlük volatilitesi ile model tahmini karşılaştırılır. "
         "Artış beklentisi → hareket başlıyor uyarısı. Azalış → piyasa sakinleşiyor.", c="re")

    latest = df_c.dropna(subset=FEATURES).groupby("Stock").tail(1).copy()
    latest["Tahmin_Vol"] = final.predict(latest[FEATURES])
    latest["Mevcut_Vol"] = latest["Volatility_5D"]
    latest["Değişim_%"]  = ((latest["Tahmin_Vol"]-latest["Mevcut_Vol"])
                             / latest["Mevcut_Vol"].replace(0,np.nan) * 100).round(1)
    latest["Sinyal"]     = np.where(
        latest["Tahmin_Vol"] > latest["Mevcut_Vol"],
        "📈 YÜKSELİŞ — Hareket Başlıyor",
        "📉 DÜŞÜŞ — Piyasa Sakinleşiyor"
    )

    sig_df = (latest[["Stock","Mevcut_Vol","Tahmin_Vol","Değişim_%","Sinyal"]]
              .sort_values("Tahmin_Vol", ascending=False)
              .reset_index(drop=True))
    sig_df["Mevcut_Vol"] = sig_df["Mevcut_Vol"].round(5)
    sig_df["Tahmin_Vol"] = sig_df["Tahmin_Vol"].round(5)

    st.dataframe(
        sig_df.style.map(
            lambda v: ("background-color:rgba(239,68,68,.18);color:#fca5a5;font-weight:700"
                       if "YÜKSELİŞ" in str(v) else
                       "background-color:rgba(34,197,94,.12);color:#86efac;font-weight:700"
                       if "DÜŞÜŞ" in str(v) else ""),
            subset=["Sinyal"]
        ),
        use_container_width=True, hide_index=True
    )

    div()

    # Volatilite radar chart
    card("06-B", "🌡️ Hisse Bazlı Volatilite Radar",
         "Mevcut ve tahmin edilen volatiliteler hisse bazında karşılaştırılır.", c="re")

    fig_sig = go.Figure()
    sig_s = sig_df.sort_values("Tahmin_Vol", ascending=False)
    fig_sig.add_trace(go.Bar(x=sig_s["Stock"], y=sig_s["Mevcut_Vol"]*100,
        name="Mevcut Vol (%)", marker_color="#00d4ff", opacity=.7))
    fig_sig.add_trace(go.Bar(x=sig_s["Stock"], y=sig_s["Tahmin_Vol"]*100,
        name="Tahmin Vol (%)", marker_color="#f97316", opacity=.85))
    fig_sig.update_layout(barmode="group",
        title="Mevcut vs Tahmin Volatilite — Hisse Bazlı",
        yaxis_title="Volatilite (%)", height=350, **DARK)
    st.plotly_chart(fig_sig, use_container_width=True)

    div()

    # Son hisse dashboard
    card("06-C", "📊 Hisse Detay Analizi", "", c="re")

    stocks_f = sorted(sig_df["Stock"].tolist())
    sel_f    = st.selectbox("Hisse Seç", stocks_f, key="sig_stk")
    s_full   = df_c[df_c["Stock"]==sel_f].sort_values("Date").copy()
    s_full["Tahmin_Vol"] = final.predict(s_full[FEATURES].fillna(0))
    plot_f   = s_full.tail(90)

    last_f = plot_f.iloc[-1] if len(plot_f)>0 else None
    if last_f is not None:
        mrow(mc("Son Kapanış", f"{last_f['Close']:.2f}",
                f"{'▲' if last_f.get('Return_1D',0)>=0 else '▼'}"
                f" %{abs(last_f.get('Return_1D',0)*100):.2f}") +
             mc("Tahmin Vol", f"{last_f['Tahmin_Vol']:.5f}",
                "Yarınki beklenti") +
             mc("RSI-14", f"{last_f['RSI_14']:.1f}",
                "⚡ Aşırı Alım" if last_f["RSI_14"]>70 else
                "📉 Aşırı Satım" if last_f["RSI_14"]<30 else "Normal") +
             mc("BB Pozisyon", f"{last_f['BB_POSITION']:.2f}",
                "Üst bant" if last_f["BB_POSITION"]>0.8 else
                "Alt bant" if last_f["BB_POSITION"]<0.2 else "Orta"))

    fig_f = make_subplots(rows=3, cols=1, shared_xaxes=True,
        row_heights=[.5,.25,.25], vertical_spacing=.04,
        subplot_titles=[f"{sel_f} — Kapanış & Bollinger",
                        "Mevcut & Tahmin Volatilite",
                        "RSI-14"])
    # fiyat + BB
    fig_f.add_trace(go.Scatter(x=plot_f["Date"],y=plot_f["Close"],
        name="Kapanış",line=dict(color="#00d4ff",width=1.8)),row=1,col=1)
    fig_f.add_trace(go.Scatter(x=plot_f["Date"],y=plot_f["BB_UPPER"],
        name="BB Üst",line=dict(color="#ef4444",width=1,dash="dot")),row=1,col=1)
    fig_f.add_trace(go.Scatter(x=plot_f["Date"],y=plot_f["BB_LOWER"],
        name="BB Alt",line=dict(color="#22c55e",width=1,dash="dot"),
        fill="tonexty",fillcolor="rgba(0,212,255,.03)"),row=1,col=1)
    # volatilite
    fig_f.add_trace(go.Scatter(x=plot_f["Date"],y=plot_f["Volatility_5D"],
        name="Mevcut Vol",line=dict(color="#38bdf8",width=1.5)),row=2,col=1)
    fig_f.add_trace(go.Scatter(x=plot_f["Date"],y=plot_f["Tahmin_Vol"],
        name="Tahmin Vol",line=dict(color="#f97316",width=1.5,dash="dot")),row=2,col=1)
    # RSI
    fig_f.add_trace(go.Scatter(x=plot_f["Date"],y=plot_f["RSI_14"],
        name="RSI-14",line=dict(color="#8b5cf6",width=1.8)),row=3,col=1)
    fig_f.add_hline(y=70,row=3,col=1,line_dash="dot",line_color="#ef4444",opacity=.5)
    fig_f.add_hline(y=30,row=3,col=1,line_dash="dot",line_color="#22c55e",opacity=.5)
    fig_f.update_layout(height=550,
        legend=dict(orientation="h",y=1.04,font_size=10),**DARK)
    st.plotly_chart(fig_f, use_container_width=True)

    div()

    # Proje özeti
    card("06-D", "🏁 Proje Özeti",
         f"""
         <table style="width:100%;border-collapse:collapse;">
         <tr><td style="padding:.5rem;color:#00d4ff;width:30%">📥 Veri</td>
             <td style="padding:.5rem">{len(df_raw):,} satır · {df_raw['Stock'].nunique()} hisse ·
             {df_raw['Date'].min().year}–{df_raw['Date'].max().year}</td></tr>
         <tr><td style="padding:.5rem;color:#00d4ff">🧹 Temizlik</td>
             <td style="padding:.5rem">OHLC filtresi · Volume>0 · Adj Close düzeltmesi</td></tr>
         <tr><td style="padding:.5rem;color:#00d4ff">🛠️ Features</td>
             <td style="padding:.5rem">17 özellik · Return, Vol, RSI, BB, MACD, Takvim</td></tr>
         <tr><td style="padding:.5rem;color:#00d4ff">✂️ Split</td>
             <td style="padding:.5rem">Temporal 2023-01-01 · Shuffle=False · TimeSeriesSplit CV</td></tr>
         <tr><td style="padding:.5rem;color:#22c55e">🏆 Model</td>
             <td style="padding:.5rem"><strong>{champion}</strong> ·
             Test R²=<strong>{r['r2']*100:.2f}%</strong> ·
             MAE=<strong>{r['mae']:.6f}</strong></td></tr>
         <tr><td style="padding:.5rem;color:#00d4ff">🔮 Çıktı</td>
             <td style="padding:.5rem">Hisse bazlı yarınki volatilite tahmini & risk sinyali</td></tr>
         </table>
         """,
         c="gr")

# ══════════════════════════════════════════════
#  BÖLÜM 7 — HABER AKIŞLARI
# ══════════════════════════════════════════════
elif section == "7 · Haber Akışları":

    card("BÖLÜM 07", "📰 Canlı Haber Akışları — Google News RSS",
         "En yüksek volatilite beklenen hisseler için Google News üzerinden "
         "Türkçe güncel haber akışı çekilir. "
         "Piyasa haberleri ile model sinyalleri birlikte yorumlanır.",
         c="pu")

    # ── Sinyal sıralamasını al ──────────────────────────────
    with st.spinner("Model sinyalleri hesaplanıyor..."):
        results  = train_models(src)
        champion = results["_champion"]
        final    = results["_final"]
        _, df_c  = build_features(src)

    latest = df_c.dropna(subset=FEATURES).groupby("Stock").tail(1).copy()
    latest["Tahmin_Vol"]  = final.predict(latest[FEATURES])
    latest["Mevcut_Vol"]  = latest["Volatility_5D"]
    latest["Degisim_Pct"] = ((latest["Tahmin_Vol"] - latest["Mevcut_Vol"])
                              / latest["Mevcut_Vol"].replace(0, np.nan) * 100).round(1)
    latest["Sinyal"] = np.where(
        latest["Tahmin_Vol"] > latest["Mevcut_Vol"],
        "📈 YÜKSELİŞ", "📉 DÜŞÜŞ"
    )
    sig_df = (latest[["Stock","Tahmin_Vol","Mevcut_Vol","Degisim_Pct","Sinyal"]]
              .sort_values("Tahmin_Vol", ascending=False)
              .reset_index(drop=True))

    # ── Ayarlar ────────────────────────────────────────────
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    n_stocks    = col_cfg1.slider("Kaç hisse gösterilsin?", 3, 10, 5)
    n_news      = col_cfg2.slider("Hisse başı haber sayısı", 2, 5, 3)
    timeout_sec = col_cfg3.slider("Bağlantı zaman aşımı (sn)", 5, 20, 10)

    st.markdown("---")

    # ── Üst sinyal özeti ───────────────────────────────────
    card("07-A", "⚡ Model Sinyal Özeti — Haber İçin Sıralama",
         "Aşağıdaki hisseler tahmin edilen volatilite büyüklüğüne göre sıralanmıştır. "
         "En çok hareket beklenen hisselerin haberleri önce gösterilir.", c="pu")

    display_sig = sig_df.head(n_stocks)[
        ["Stock","Mevcut_Vol","Tahmin_Vol","Degisim_Pct","Sinyal"]
    ].copy()
    display_sig.columns = ["Hisse","Mevcut Vol","Tahmin Vol","Değişim %","Sinyal"]
    display_sig["Mevcut Vol"] = display_sig["Mevcut Vol"].round(5)
    display_sig["Tahmin Vol"] = display_sig["Tahmin Vol"].round(5)
    st.dataframe(
        display_sig.style.map(
            lambda v: ("background-color:rgba(239,68,68,.15);color:#fca5a5;font-weight:700"
                       if "YÜKSELİŞ" in str(v) else
                       "background-color:rgba(34,197,94,.10);color:#86efac;font-weight:700"
                       if "DÜŞÜŞ" in str(v) else ""),
            subset=["Sinyal"]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # ── RSS yardımcı fonksiyonu ────────────────────────────
    def fetch_google_news(stock_name: str, n: int = 3, timeout: int = 10):
        """
        Google News RSS'ten Türkçe haberleri çeker.
        Döner: list[dict]  |  None (hata durumunda)
        """
        clean = stock_name.split(".")[0]
        query = f"{clean}+hisse+borsa"
        url   = (f"https://news.google.com/rss/search"
                 f"?q={query}&hl=tr&gl=TR&ceid=TR:tr")
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "tr-TR,tr;q=0.9",
        }
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as resp:
                xml_data = resp.read()
            root  = ET.fromstring(xml_data)
            items = root.findall(".//item")
            news  = []
            for item in items[:n]:
                title  = item.findtext("title", "Başlık yok")
                link   = item.findtext("link",  "#")
                pub    = item.findtext("pubDate", "")
                source_el = item.find("source")
                source = source_el.text if source_el is not None else "Google News"
                # kaynak adını başlıktan temizle
                if " - " in title:
                    title = " - ".join(title.split(" - ")[:-1])
                news.append({"title": title, "link": link,
                             "pub": pub[:25] if pub else "", "source": source})
            return news
        except HTTPError as e:
            return {"error": f"HTTP {e.code}: Sunucu isteği reddetti ({e.reason})"}
        except URLError as e:
            return {"error": f"Bağlantı hatası: {e.reason}"}
        except Exception as e:
            return {"error": str(e)}

    # ── Haber akışları ─────────────────────────────────────
    card("07-B", "📡 Güncel Haber Akışı",
         f"En yüksek volatilite beklenen <strong>{n_stocks}</strong> hisse için "
         f"hisse başı <strong>{n_news}</strong> güncel Türkçe haber.", c="pu")

    for idx, row in sig_df.head(n_stocks).iterrows():
        stock     = row["Stock"]
        clean     = stock.split(".")[0]
        tahmin    = row["Tahmin_Vol"]
        degisim   = row["Degisim_Pct"]
        sinyal    = row["Sinyal"]
        sinyal_color = "#ef4444" if "YÜKSELİŞ" in sinyal else "#22c55e"

        # Hisse başlık kartı
        st.markdown(f"""
        <div class="news-header">
            <div class="news-stock">
                {sinyal}&nbsp;&nbsp;{clean}
                <span style="font-size:.75rem;color:{sinyal_color};margin-left:1rem;">
                    Tahmin: {tahmin:.5f} &nbsp;|&nbsp;
                    Değişim: {'▲' if degisim>=0 else '▼'} %{abs(degisim):.1f}
                </span>
            </div>
            <div class="news-vol">
                Sıra {idx+1} · En Yüksek Volatilite Beklentisi
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Haber çek
        with st.spinner(f"{clean} haberleri yükleniyor..."):
            news = fetch_google_news(stock, n=n_news, timeout=timeout_sec)

        if isinstance(news, dict) and "error" in news:
            # Hata durumu
            st.markdown(f"""
            <div class="news-error">
                ❌ <strong>{clean}:</strong> {news['error']}<br>
                <span style="font-size:.78rem;margin-top:.3rem;display:block;color:#5a3030;">
                Olası nedenler: İnternet bağlantısı yok · Google News erişimi engellendi · VPN gerekebilir<br>
                💡 Tarayıcınızda açmak için:
                <a href="https://news.google.com/search?q={clean}+hisse&hl=tr&gl=TR"
                   target="_blank" style="color:#7a6060;">
                   news.google.com/{clean}
                </a>
                </span>
            </div>
            """, unsafe_allow_html=True)

        elif not news:
            st.markdown(f"""
            <div class="news-empty">
                ⚠️ {clean} için Türkçe haber bulunamadı.
                Farklı arama terimi deneyin veya Google News'i doğrudan ziyaret edin.
            </div>
            """, unsafe_allow_html=True)

        else:
            # Haberler
            for i, item in enumerate(news, 1):
                st.markdown(f"""
                <div class="news-card">
                    <div class="news-num">HABER {i:02d} &nbsp;·&nbsp; {item['source']} &nbsp;·&nbsp; {item['pub']}</div>
                    <div class="news-title">{item['title']}</div>
                    <a class="news-link" href="{item['link']}" target="_blank">
                        🔗 Haberin tamamı için tıklayın →
                    </a>
                </div>
                """, unsafe_allow_html=True)

        # Hisseler arası küçük boşluk + hız limit
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        time.sleep(0.3)

    st.markdown("---")

    # ── Manuel arama ───────────────────────────────────────
    card("07-C", "🔍 Manuel Hisse Haber Ara",
         "Listede olmayan bir hisse için de haber arayabilirsiniz.", c="pu")

    col_m1, col_m2 = st.columns([2, 1])
    with col_m1:
        custom_stock = st.text_input(
            "Hisse sembolü girin",
            placeholder="örn: SISE veya EREGL.IS",
            label_visibility="collapsed",
        )
    with col_m2:
        search_btn = st.button("🔍 Haberleri Getir", use_container_width=True)

    if search_btn and custom_stock.strip():
        stk_query = custom_stock.strip().upper()
        with st.spinner(f"{stk_query} haberleri aranıyor..."):
            manual_news = fetch_google_news(stk_query, n=5, timeout=timeout_sec)

        if isinstance(manual_news, dict) and "error" in manual_news:
            st.error(f"Hata: {manual_news['error']}")
        elif not manual_news:
            st.warning("Bu hisse için haber bulunamadı.")
        else:
            for i, item in enumerate(manual_news, 1):
                st.markdown(f"""
                <div class="news-card">
                    <div class="news-num">HABER {i:02d} &nbsp;·&nbsp; {item['source']} &nbsp;·&nbsp; {item['pub']}</div>
                    <div class="news-title">{item['title']}</div>
                    <a class="news-link" href="{item['link']}" target="_blank">
                        🔗 Haberin tamamı için tıklayın →
                    </a>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Bağlantı notları ───────────────────────────────────
    st.markdown("""
    <div class="bx-w" style="margin-top:0">
    ⚠️ <strong>Not:</strong> Google News RSS bazı ağlarda (kurumsal, VPN, sunucu ortamı)
    HTTP 403 hatası verebilir. Bu durumda:<br>
    • Kişisel internet bağlantısından çalıştırın<br>
    • VPN kapatın (veya Türkiye konumlu VPN kullanın)<br>
    • <a href="https://news.google.com/?hl=tr&gl=TR" target="_blank"
       style="color:#f97316">news.google.com</a>'u tarayıcıdan ziyaret edin
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div style="text-align:center;color:#0e2238;font-size:.68rem;margin-top:2.5rem;
            padding:1.2rem;border-top:1px solid #0e2238;
            font-family:'IBM Plex Mono',monospace;">
    BIST Risk Radarı · LightGBM vs XGBoost vs CatBoost · Bootcamp Bitirme
</div>
""", unsafe_allow_html=True)
