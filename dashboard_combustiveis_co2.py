from __future__ import annotations

import io
import json
import re
import unicodedata
from pathlib import Path
from urllib.request import urlopen, urlretrieve

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
FUEL_PARQUET_FILE = BASE_DIR / "dataset_automotivos_consolidado_2020_2025.parquet"
FUEL_CSV_FILE = BASE_DIR / "dataset_automotivos_consolidado_2023_2025.csv"
SALES_DEFAULT_FILE = BASE_DIR / "vendas-combustiveis-m3-1990-2025.parquet"
FUEL_ANNUAL_PROD_FILE = BASE_DIR / "outputs" / "fuel_annual_produto.parquet"
FUEL_ANNUAL_REGION_FILE = BASE_DIR / "outputs" / "fuel_annual_regiao.parquet"
FUEL_ANNUAL_UF_FILE = BASE_DIR / "outputs" / "fuel_annual_uf.parquet"
FUEL_MUN_DIST_FILE = BASE_DIR / "outputs" / "fuel_municipio_distribuicao.parquet"
FUEL_MUN_EG_FILE = BASE_DIR / "outputs" / "fuel_municipio_produto_etanol_gasolina.parquet"
SEEG_FILE = BASE_DIR / "outputs" / "Dados-nacionais-13.0.xlsx"
SEEG_URL = "https://seeg.eco.br/wp-content/uploads/2025/12/Dados-nacionais-13.0.xlsx"
CO2_PARQUET_FILE = BASE_DIR / "outputs" / "co2e_anual.parquet"
IPCA_PARQUET_FILE = BASE_DIR / "outputs" / "ipca_anual_2020_2025.parquet"
SIDRA_PARQUET_FILE = BASE_DIR / "outputs" / "sidra_ipca_grupos_anual_2020_2025.parquet"
EV_PARQUET_FILE = BASE_DIR / "outputs" / "abve_eletrificados_serie_anual.parquet"
EV_CSV_FILE = BASE_DIR / "outputs" / "abve_eletrificados_serie_anual.csv"
REPORT_FILE = BASE_DIR / "outputs" / "resumo_exploratorio_automotivos.md"

YEAR_MIN = 2020
YEAR_MAX = 2025

REGION_COLORS = {
    "N": "#1f77b4",
    "NE": "#ff7f0e",
    "CO": "#2ca02c",
    "SE": "#d62728",
    "S": "#9467bd",
}


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return text


def money_label(value: float) -> str:
    return f"R$ {value:.2f}"


def apply_legend(fig, title: str = "Legenda"):
    fig.update_layout(
        legend_title_text=title,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return fig


def auto_find_anp_sales_file() -> Path | None:
    patterns = [
        "*vendas*combust*.parquet",
        "*vendas*derivad*.parquet",
        "*vendas*combust*.csv",
        "*vendas*derivad*.csv",
        "*vendas*etanol*.csv",
        "*vendas*biocomb*.csv",
        "*comercializacao*combust*.csv",
        "*vendas*combust*.xlsx",
        "*vendas*derivad*.xlsx",
    ]
    roots = [BASE_DIR, BASE_DIR / "outputs"]
    for root in roots:
        for pattern in patterns:
            files = sorted(root.glob(pattern))
            if files:
                return files[0]
    if SALES_DEFAULT_FILE.exists():
        return SALES_DEFAULT_FILE
    return None


def read_tabular_bytes(filename: str, data: bytes) -> pd.DataFrame:
    lower = filename.lower()
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(io.BytesIO(data))

    best: pd.DataFrame | None = None
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin-1"]:
        for sep in [";", ",", "\t"]:
            try:
                df = pd.read_csv(io.BytesIO(data), encoding=enc, sep=sep, low_memory=False)
                if best is None or df.shape[1] > best.shape[1]:
                    best = df
            except Exception:
                continue
    if best is None:
        raise ValueError("Nao foi possivel ler o arquivo como CSV/XLS.")
    return best


def read_tabular_file(path: Path) -> pd.DataFrame:
    lower = path.name.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(path)

    best: pd.DataFrame | None = None
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin-1"]:
        for sep in [";", ",", "\t"]:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, low_memory=False)
                if best is None or df.shape[1] > best.shape[1]:
                    best = df
            except Exception:
                continue
    if best is None:
        raise ValueError("Nao foi possivel ler o arquivo como CSV/XLS.")
    return best


def prepare_anp_sales(df: pd.DataFrame) -> pd.DataFrame:
    raw = df.copy()
    raw = raw.rename(columns={c: normalize_text(c) for c in raw.columns})

    product_col = next((c for c in raw.columns if "produto" in c or "combust" in c), None)
    volume_col = next((c for c in raw.columns if "volume" in c or "vendas" in c or "quantidade" in c), None)
    date_col = next((c for c in raw.columns if c in ["data", "mes_ano", "ano_mes", "periodo"]), None)
    month_col = next((c for c in raw.columns if c in ["mes", "mes_numero"] or re.fullmatch(r"m_?s", c)), None)

    if product_col is None or volume_col is None:
        raise ValueError("Colunas de produto/volume nao encontradas.")

    if date_col is not None:
        raw["dt"] = pd.to_datetime(raw[date_col], errors="coerce", dayfirst=True)
    elif "ano" in raw.columns and month_col is not None:
        month_map = {
            "JAN": 1,
            "FEV": 2,
            "MAR": 3,
            "ABR": 4,
            "MAI": 5,
            "JUN": 6,
            "JUL": 7,
            "AGO": 8,
            "SET": 9,
            "OUT": 10,
            "NOV": 11,
            "DEZ": 12,
        }
        month_raw = raw[month_col].astype(str).str.strip().str.upper()
        month_num = pd.to_numeric(month_raw, errors="coerce").fillna(month_raw.map(month_map))
        raw["dt"] = pd.to_datetime(
            {
                "year": pd.to_numeric(raw["ano"], errors="coerce"),
                "month": month_num,
                "day": 1,
            },
            errors="coerce",
        )
    elif "ano" in raw.columns:
        raw["dt"] = pd.to_datetime(raw["ano"].astype(str) + "-01-01", errors="coerce")
    else:
        raise ValueError("Coluna temporal nao encontrada.")

    raw["volume"] = pd.to_numeric(raw[volume_col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    raw["produto"] = raw[product_col].astype(str)
    raw = raw.dropna(subset=["dt", "volume"]).copy()
    raw["produto_norm"] = raw["produto"].map(normalize_text)
    raw = raw[raw["produto_norm"].str.contains("etanol|gasolina", na=False)].copy()
    raw = raw[~raw["produto_norm"].str.contains("aviacao", na=False)].copy()
    raw["produto_grupo"] = raw["produto_norm"].map(lambda x: "etanol" if "etanol" in x else "gasolina")
    raw["ano"] = raw["dt"].dt.year
    return raw


def ensure_seeg_file() -> Path:
    SEEG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not SEEG_FILE.exists():
        urlretrieve(SEEG_URL, SEEG_FILE)
    return SEEG_FILE


@st.cache_data(show_spinner=False)
def load_fuel_full() -> pd.DataFrame:
    cols = [
        "Regiao - Sigla",
        "Estado - Sigla",
        "Municipio",
        "Produto",
        "Data da Coleta",
        "Valor de Venda",
        "Bandeira",
        "ano",
        "semestre",
        "valor_venda_num",
    ]
    if FUEL_PARQUET_FILE.exists():
        df = pd.read_parquet(FUEL_PARQUET_FILE, columns=cols)
    else:
        df = pd.read_csv(FUEL_CSV_FILE, encoding="utf-8-sig", low_memory=False, usecols=cols)

    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    df["semestre"] = pd.to_numeric(df["semestre"], errors="coerce")
    df["valor_venda_num"] = pd.to_numeric(df["valor_venda_num"], errors="coerce")
    df["data_coleta_dt"] = pd.to_datetime(df["Data da Coleta"], dayfirst=True, errors="coerce")
    df = df[df["ano"].between(YEAR_MIN, YEAR_MAX)].copy()
    return df


@st.cache_data(show_spinner=False)
def load_fuel_annual_metrics() -> pd.DataFrame:
    if FUEL_ANNUAL_PROD_FILE.exists():
        annual = pd.read_parquet(FUEL_ANNUAL_PROD_FILE, columns=["ano", "Produto", "valor_venda_num"])
    else:
        base = load_fuel_full()
        base = base[base["Produto"].isin(["ETANOL", "GASOLINA"]) & base["valor_venda_num"].notna()].copy()
        annual = base.groupby(["ano", "Produto"], as_index=False)["valor_venda_num"].mean()

    annual = annual[annual["Produto"].isin(["ETANOL", "GASOLINA"])].copy()
    pivot = annual.pivot(index="ano", columns="Produto", values="valor_venda_num").reset_index()
    pivot = pivot.rename(columns={"ETANOL": "preco_etanol", "GASOLINA": "preco_gasolina"})
    pivot["ratio_etanol_gasolina"] = pivot["preco_etanol"] / pivot["preco_gasolina"]
    pivot["etanol_competitivo"] = pivot["ratio_etanol_gasolina"] <= 0.70

    if FUEL_MUN_EG_FILE.exists():
        mun = pd.read_parquet(FUEL_MUN_EG_FILE, columns=["ano", "Municipio", "Produto", "valor_venda_num"])
    else:
        base = load_fuel_full()
        base = base[base["Produto"].isin(["ETANOL", "GASOLINA"]) & base["valor_venda_num"].notna()].copy()
        mun = base.groupby(["ano", "Municipio", "Produto"], as_index=False)["valor_venda_num"].mean()

    mun_p = mun.pivot(index=["ano", "Municipio"], columns="Produto", values="valor_venda_num").reset_index()
    mun_p = mun_p.dropna(subset=["ETANOL", "GASOLINA"])
    mun_p["ratio"] = mun_p["ETANOL"] / mun_p["GASOLINA"]
    share = (
        mun_p.assign(comp=mun_p["ratio"] <= 0.70)
        .groupby("ano", as_index=False)["comp"]
        .mean()
        .rename(columns={"comp": "pct_municipios_etanol_competitivo"})
    )
    share["pct_municipios_etanol_competitivo"] = share["pct_municipios_etanol_competitivo"] * 100

    return pivot.merge(share, on="ano", how="left").sort_values("ano")


@st.cache_data(show_spinner=False)
def load_ipca_annual() -> pd.DataFrame:
    if IPCA_PARQUET_FILE.exists():
        df = pd.read_parquet(IPCA_PARQUET_FILE, columns=["ano", "ipca_anual_pct"])
        return df[df["ano"].between(YEAR_MIN, YEAR_MAX)].sort_values("ano")

    ipca_url = (
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=json"
        f"&dataInicial=01/01/{YEAR_MIN}&dataFinal=31/12/{YEAR_MAX}"
    )
    data = pd.read_json(ipca_url)
    data["data"] = pd.to_datetime(data["data"], dayfirst=True, errors="coerce")
    data["valor"] = pd.to_numeric(data["valor"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    data = data.dropna(subset=["data", "valor"])
    data["ano"] = data["data"].dt.year
    annual = (
        data.groupby("ano", as_index=False)["valor"]
        .apply(lambda s: ((1 + s / 100).prod() - 1) * 100)
        .rename(columns={"valor": "ipca_anual_pct"})
    )
    return annual.sort_values("ano")


@st.cache_data(show_spinner=False)
def load_sidra_group_annual() -> pd.DataFrame:
    if SIDRA_PARQUET_FILE.exists():
        df = pd.read_parquet(SIDRA_PARQUET_FILE, columns=["ano", "inflacao_anual_pct", "serie"])
        return df[df["ano"].between(YEAR_MIN, YEAR_MAX)].sort_values(["ano", "serie"])

    base = f"https://apisidra.ibge.gov.br/values/t/7060/n1/all/v/63/p/{YEAR_MIN}01-{YEAR_MAX}12/c315/"
    code_map = {
        "7169": "IPCA geral",
        "7170": "IPCA alimentacao e bebidas",
        "7625": "IPCA transportes",
    }

    frames: list[pd.DataFrame] = []
    for code, name in code_map.items():
        url = base + code
        payload = json.loads(urlopen(url, timeout=30).read().decode("utf-8"))
        df = pd.DataFrame(payload[1:])
        if df.empty:
            continue
        df["mes_code"] = df["D3C"].astype(str)
        df["valor"] = pd.to_numeric(df["V"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        df = df[df["valor"].notna()].copy()
        df["ano"] = df["mes_code"].str[:4].astype(int)
        annual = (
            df.groupby("ano", as_index=False)["valor"]
            .apply(lambda s: ((1 + s / 100).prod() - 1) * 100)
            .rename(columns={"valor": "inflacao_anual_pct"})
        )
        annual["serie"] = name
        frames.append(annual)

    if not frames:
        return pd.DataFrame(columns=["ano", "inflacao_anual_pct", "serie"])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["ano", "serie"])


@st.cache_data(show_spinner=False)
def load_co2e_annual() -> pd.DataFrame:
    if CO2_PARQUET_FILE.exists():
        df = pd.read_parquet(CO2_PARQUET_FILE, columns=["ano", "co2e_t", "co2e_mt"])
        return df[df["ano"].between(YEAR_MIN, YEAR_MAX)].sort_values("ano")

    path = ensure_seeg_file()
    raw = pd.read_excel(path, sheet_name=3, header=None)

    header_row = raw.iloc[5]
    total_row = raw[raw.iloc[:, 0].astype(str).str.strip().eq("Total Geral")].iloc[0]

    rows: list[tuple[int, float]] = []
    for col in raw.columns[1:]:
        name = str(header_row[col])
        if name.startswith("Soma de "):
            year = int(name.replace("Soma de ", "").strip())
            value = float(total_row[col])
            rows.append((year, value))

    co2 = pd.DataFrame(rows, columns=["ano", "co2e_t"])
    co2["co2e_mt"] = co2["co2e_t"] / 1_000_000
    return co2[co2["ano"].between(YEAR_MIN, YEAR_MAX)].sort_values("ano")


@st.cache_data(show_spinner=False)
def load_ev_annual() -> pd.DataFrame:
    if EV_PARQUET_FILE.exists():
        ev = pd.read_parquet(EV_PARQUET_FILE, columns=["ano", "total_eletrificados"])
    elif EV_CSV_FILE.exists():
        ev = pd.read_csv(EV_CSV_FILE, usecols=["ano", "total_eletrificados"])
    else:
        return pd.DataFrame(columns=["ano", "total_eletrificados"])
    ev["ano"] = pd.to_numeric(ev["ano"], errors="coerce")
    ev["total_eletrificados"] = pd.to_numeric(ev["total_eletrificados"], errors="coerce")
    ev = ev.dropna(subset=["ano", "total_eletrificados"]).copy()
    ev["ano"] = ev["ano"].astype(int)
    return ev[ev["ano"].between(YEAR_MIN, YEAR_MAX)].sort_values("ano")


def render_summary() -> None:
    st.subheader("Resumo executivo")
    fuel = load_fuel_annual_metrics()
    ipca = load_ipca_annual()
    co2 = load_co2e_annual()
    ipca_media_periodo = ipca["ipca_anual_pct"].mean() if not ipca.empty else pd.NA
    ipca_ultimo_ano = ipca.sort_values("ano").iloc[-1]["ipca_anual_pct"] if not ipca.empty else pd.NA

    latest = fuel.sort_values("ano").iloc[-1]
    first = fuel.sort_values("ano").iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Janela analisada", f"{YEAR_MIN}-{YEAR_MAX}")
    c2.metric("Razao etanol/gasolina", f"{latest['ratio_etanol_gasolina']:.3f}", f"{(latest['ratio_etanol_gasolina']/first['ratio_etanol_gasolina']-1)*100:.1f}%")
    c3.metric("Municipios etanol competitivo", f"{latest['pct_municipios_etanol_competitivo']:.1f}%")
    c4.metric("IPCA medio anual (2020-2025)", f"{ipca_media_periodo:.1f}%" if pd.notna(ipca_media_periodo) else "n/d")

    if pd.notna(ipca_ultimo_ano):
        st.caption(f"Referencia: IPCA do ultimo ano da serie ({int(ipca.sort_values('ano').iloc[-1]['ano'])}) = {ipca_ultimo_ano:.1f}%.")

    trend = fuel[["ano", "preco_etanol", "preco_gasolina", "ratio_etanol_gasolina"]].copy()
    fig = px.line(
        trend,
        x="ano",
        y=["preco_etanol", "preco_gasolina"],
        markers=True,
        title="Preco medio anual: etanol vs gasolina",
        labels={"value": "Preco (R$/litro)", "variable": "Serie"},
    )
    fig.for_each_trace(lambda t: t.update(name=t.name.replace("preco_etanol", "Etanol").replace("preco_gasolina", "Gasolina")))
    apply_legend(fig, "Serie")
    st.plotly_chart(fig, use_container_width=True)

    if not co2.empty:
        cover = sorted(co2["ano"].tolist())
        st.caption(f"Cobertura de CO2e disponivel no painel: {cover[0]}-{cover[-1]}")



def render_market_views() -> None:
    st.subheader("Mercado de combustiveis")
    if FUEL_ANNUAL_PROD_FILE.exists():
        annual_prod = pd.read_parquet(FUEL_ANNUAL_PROD_FILE, columns=["ano", "Produto", "valor_venda_num"])
        annual_prod = annual_prod[annual_prod["ano"].between(YEAR_MIN, YEAR_MAX)].copy()
    else:
        fuel = load_fuel_full()
        view = fuel[fuel["valor_venda_num"].notna()].copy()
        annual_prod = view.groupby(["ano", "Produto"], as_index=False)["valor_venda_num"].mean()

    products = sorted(annual_prod["Produto"].dropna().unique().tolist())
    default_products = [p for p in ["ETANOL", "GASOLINA", "DIESEL", "DIESEL S10"] if p in products]
    selected = st.multiselect("Produtos", options=products, default=default_products if default_products else products[:4])
    if selected:
        annual_prod = annual_prod[annual_prod["Produto"].isin(selected)].copy()

    fig = px.line(
        annual_prod,
        x="ano",
        y="valor_venda_num",
        color="Produto",
        markers=True,
        title="Evolucao anual por produto",
        labels={"valor_venda_num": "Preco medio (R$/litro)", "ano": "Ano"},
    )
    apply_legend(fig, "Produto")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if FUEL_ANNUAL_REGION_FILE.exists():
            region = pd.read_parquet(FUEL_ANNUAL_REGION_FILE, columns=["ano", "Regiao - Sigla", "valor_venda_num"])
            region = region[region["ano"].between(YEAR_MIN, YEAR_MAX)].copy()
        else:
            fuel = load_fuel_full()
            region = fuel.groupby(["ano", "Regiao - Sigla"], as_index=False)["valor_venda_num"].mean()
        heat_data = region.pivot(index="Regiao - Sigla", columns="ano", values="valor_venda_num")
        fig = px.imshow(
            heat_data,
            text_auto=True,
            color_continuous_scale="YlOrRd",
            title="Mapa de calor: preco medio por regiao e ano",
            labels={"x": "Ano", "y": "Regiao", "color": "R$/litro"},
        )
        fig.update_layout(coloraxis_colorbar=dict(title="R$/litro"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if FUEL_ANNUAL_UF_FILE.exists():
            uf = pd.read_parquet(FUEL_ANNUAL_UF_FILE, columns=["ano", "Estado - Sigla", "valor_venda_num"])
            uf = uf[uf["ano"].between(YEAR_MIN, YEAR_MAX)].copy()
            ranking = uf.groupby("Estado - Sigla", as_index=False)["valor_venda_num"].mean().sort_values("valor_venda_num", ascending=False)
        else:
            fuel = load_fuel_full()
            ranking = fuel.groupby("Estado - Sigla", as_index=False)["valor_venda_num"].mean().sort_values("valor_venda_num", ascending=False)
        top = ranking.head(10).sort_values("valor_venda_num", ascending=True)
        top["rotulo"] = top["valor_venda_num"].map(money_label)
        fig = px.bar(
            top,
            y="Estado - Sigla",
            x="valor_venda_num",
            orientation="h",
            text="rotulo",
            color="valor_venda_num",
            color_continuous_scale="Reds",
            title="Top 10 UFs: maior preco medio",
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(coloraxis_colorbar=dict(title="Preco medio"), margin=dict(r=90), xaxis_title="Preco medio (R$/litro)")
        st.plotly_chart(fig, use_container_width=True)



def render_inflation_views() -> None:
    st.subheader("Inflacao e repasse")
    if FUEL_ANNUAL_PROD_FILE.exists():
        annual_prod = pd.read_parquet(FUEL_ANNUAL_PROD_FILE, columns=["ano", "Produto", "valor_venda_num"])
        annual_prod = annual_prod[annual_prod["ano"].between(YEAR_MIN, YEAR_MAX)].copy()
        fuel_year = annual_prod.groupby("ano", as_index=False)["valor_venda_num"].mean().rename(columns={"valor_venda_num": "preco_medio_combustivel"})
    else:
        fuel = load_fuel_full()
        fuel_year = fuel.groupby("ano", as_index=False)["valor_venda_num"].mean().rename(columns={"valor_venda_num": "preco_medio_combustivel"})

    fuel_year["var_preco_comb_anual_pct"] = fuel_year["preco_medio_combustivel"].pct_change() * 100
    fuel_year["var_preco_comb_anual_pct"] = fuel_year["var_preco_comb_anual_pct"].fillna(0)

    sidra = load_sidra_group_annual()
    pivot = sidra.pivot(index="ano", columns="serie", values="inflacao_anual_pct").reset_index()
    merged = fuel_year.merge(pivot, on="ano", how="inner")

    fig = px.line(
        merged,
        x="ano",
        y=["var_preco_comb_anual_pct", "IPCA transportes", "IPCA alimentacao e bebidas", "IPCA geral"],
        markers=True,
        title="Variacao anual: combustiveis x inflacao setorial",
    )
    fig.for_each_trace(lambda t: t.update(name=t.name.replace("var_preco_comb_anual_pct", "Combustiveis")))
    apply_legend(fig, "Serie")
    fig.update_layout(yaxis_title="Variacao anual (%)")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        scat = px.scatter(
            merged,
            x="var_preco_comb_anual_pct",
            y="IPCA transportes",
            text="ano",
            trendline="ols" if len(merged) >= 3 else None,
            title="Combustiveis x IPCA transportes",
            labels={"var_preco_comb_anual_pct": "Combustiveis (%)", "IPCA transportes": "IPCA transportes (%)"},
        )
        scat.update_traces(textposition="top center")
        apply_legend(scat, "Serie")
        st.plotly_chart(scat, use_container_width=True)

    with c2:
        corr_cols = ["var_preco_comb_anual_pct", "IPCA geral", "IPCA transportes", "IPCA alimentacao e bebidas"]
        corr = merged[corr_cols].corr().round(3)
        heat = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1, title="Matriz de correlacao")
        heat.update_layout(coloraxis_colorbar=dict(title="Correlacao"))
        st.plotly_chart(heat, use_container_width=True)



def render_sales_volume_views() -> None:
    st.subheader("Vendas reais ANP")
    st.caption("Este bloco usa volume comercializado para complementar as visoes baseadas em preco.")

    uploaded = st.file_uploader("Arquivo de vendas ANP (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="sales_upload")

    sales_df: pd.DataFrame | None = None
    source_name = ""
    try:
        if uploaded is not None:
            sales_df = read_tabular_bytes(uploaded.name, uploaded.getvalue())
            source_name = uploaded.name
        else:
            auto_file = auto_find_anp_sales_file()
            if auto_file is not None:
                sales_df = read_tabular_file(auto_file)
                source_name = auto_file.name
    except Exception:
        sales_df = None

    if sales_df is None:
        st.info("Nenhum arquivo de vendas identificado automaticamente. Faça upload para ativar esta secao.")
        return

    try:
        prepared = prepare_anp_sales(sales_df)
    except Exception as exc:
        st.warning(f"Falha na padronizacao da base de vendas: {exc}")
        return

    prepared = prepared[prepared["ano"].between(YEAR_MIN, YEAR_MAX)].copy()
    if prepared.empty:
        st.info("Arquivo de vendas sem dados no intervalo selecionado.")
        return

    st.success(f"Base carregada: {source_name}")

    annual = prepared.groupby(["ano", "produto_grupo"], as_index=False)["volume"].sum()
    pivot = annual.pivot(index="ano", columns="produto_grupo", values="volume").reset_index().fillna(0)
    if "etanol" not in pivot.columns:
        pivot["etanol"] = 0.0
    if "gasolina" not in pivot.columns:
        pivot["gasolina"] = 0.0
    pivot["share_etanol"] = pivot["etanol"] / (pivot["etanol"] + pivot["gasolina"]).replace(0, pd.NA)

    c1, c2 = st.columns(2)
    with c1:
        bar = px.bar(
            annual,
            x="ano",
            y="volume",
            color="produto_grupo",
            barmode="stack",
            title="Volume anual empilhado por combustivel",
            labels={"volume": "Volume", "ano": "Ano", "produto_grupo": "Produto"},
        )
        apply_legend(bar, "Produto")
        st.plotly_chart(bar, use_container_width=True)

    with c2:
        line = px.line(
            pivot,
            x="ano",
            y="share_etanol",
            markers=True,
            title="Share anual do etanol no volume",
            labels={"share_etanol": "Share etanol", "ano": "Ano"},
        )
        apply_legend(line, "Serie")
        st.plotly_chart(line, use_container_width=True)

    co2 = load_co2e_annual()[["ano", "co2e_mt"]]
    merged = pivot.merge(co2, on="ano", how="left")
    available = merged.dropna(subset=["co2e_mt", "share_etanol"]).copy()

    if available.empty:
        st.info("Sem anos sobrepostos com CO2e para o scatter de emissao.")
        return

    miss = sorted(set(merged["ano"]) - set(available["ano"]))
    if miss:
        st.caption(f"Anos sem CO2e para o cruzamento: {', '.join(str(x) for x in miss)}")

    scat = px.scatter(
        available,
        x="share_etanol",
        y="co2e_mt",
        text="ano",
        trendline="ols" if len(available) >= 3 else None,
        title="Share etanol (volume) vs CO2e",
        labels={"share_etanol": "Share etanol", "co2e_mt": "CO2e nacional (Mt)"},
    )
    scat.update_traces(textposition="top center")
    apply_legend(scat, "Serie")
    st.plotly_chart(scat, use_container_width=True)



def render_emission_transition_views() -> None:
    st.subheader("Emissoes e transicao")
    fuel = load_fuel_annual_metrics()
    co2 = load_co2e_annual()
    panel = fuel.merge(co2, on="ano", how="inner")

    if panel.empty:
        st.warning("Sem sobreposicao entre combustiveis e CO2e no periodo filtrado.")
        return

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(panel, x="ano", y="co2e_mt", markers=True, title="CO2e nacional")
        fig.update_layout(yaxis_title="Mt CO2e")
        apply_legend(fig, "Serie")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(panel, x="ano", y="ratio_etanol_gasolina", markers=True, title="Razao etanol/gasolina")
        fig.update_layout(yaxis_title="Razao de preco")
        apply_legend(fig, "Serie")
        st.plotly_chart(fig, use_container_width=True)

    scat = px.scatter(
        panel,
        x="ratio_etanol_gasolina",
        y="co2e_mt",
        text="ano",
        trendline="ols" if len(panel) >= 3 else None,
        title="Razao etanol/gasolina vs CO2e",
        labels={"ratio_etanol_gasolina": "Razao etanol/gasolina", "co2e_mt": "CO2e (Mt)"},
    )
    scat.update_traces(textposition="top center")
    apply_legend(scat, "Serie")
    st.plotly_chart(scat, use_container_width=True)

    ev = load_ev_annual()
    if ev.empty:
        st.caption("Base de eletrificados nao encontrada em outputs/abve_eletrificados_serie_anual.parquet (ou csv fallback).")
        return

    merged_ev = ev.merge(co2[["ano", "co2e_mt"]], on="ano", how="left")
    c3, c4 = st.columns(2)
    with c3:
        fig = px.line(merged_ev, x="ano", y="total_eletrificados", markers=True, title="Serie anual de eletrificados")
        fig.update_layout(yaxis_title="Total")
        apply_legend(fig, "Serie")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        avail = merged_ev.dropna(subset=["co2e_mt"])
        if not avail.empty:
            fig = px.scatter(avail, x="total_eletrificados", y="co2e_mt", text="ano", trendline="ols" if len(avail) >= 3 else None, title="Eletrificados vs CO2e")
            fig.update_traces(textposition="top center")
            apply_legend(fig, "Serie")
            st.plotly_chart(fig, use_container_width=True)



def render_regional_rankings() -> None:
    st.subheader("Diagnostico regional e rankings")
    if FUEL_MUN_DIST_FILE.exists():
        view = pd.read_parquet(FUEL_MUN_DIST_FILE, columns=["ano", "Regiao - Sigla", "Municipio", "valor_venda_num"])
        view = view[view["ano"].between(YEAR_MIN, YEAR_MAX)].copy()
    else:
        fuel = load_fuel_full()
        view = fuel[fuel["valor_venda_num"].notna()].copy()
    if view.empty:
        st.caption(f"Periodo alvo desta secao: {YEAR_MIN}-{YEAR_MAX}.")
        st.warning("Sem dados para montar esta secao.")
        return

    ano_ini = int(view["ano"].min())
    ano_fim = int(view["ano"].max())
    years = st.slider(
        "Intervalo de anos da board",
        min_value=ano_ini,
        max_value=ano_fim,
        value=(ano_ini, ano_fim),
        key="regional_year_range",
    )
    view = view[(view["ano"] >= years[0]) & (view["ano"] <= years[1])].copy()
    st.caption(f"Periodo considerado nesta secao: {years[0]}-{years[1]} (janela alvo do painel: {YEAR_MIN}-{YEAR_MAX}).")

    if view.empty:
        st.warning("Sem dados para o intervalo de anos selecionado.")
        return

    region_order = (
        view.groupby("Regiao - Sigla")["valor_venda_num"].median().sort_values(ascending=True).index.tolist()
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(
            view,
            x="Regiao - Sigla",
            y="valor_venda_num",
            color="Regiao - Sigla",
            points="outliers",
            category_orders={"Regiao - Sigla": region_order},
            color_discrete_map=REGION_COLORS,
            title="Distribuicao de precos por regiao",
        )
        apply_legend(fig, "Regiao")
        fig.update_layout(xaxis_title="Regiao", yaxis_title="Preco (R$/litro)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if FUEL_ANNUAL_UF_FILE.exists():
            uf = pd.read_parquet(FUEL_ANNUAL_UF_FILE, columns=["ano", "Estado - Sigla", "valor_venda_num"])
            uf = uf[(uf["ano"] >= years[0]) & (uf["ano"] <= years[1])].copy()
            rank = uf.groupby("Estado - Sigla", as_index=False)["valor_venda_num"].mean().sort_values("valor_venda_num", ascending=True)
        else:
            st.info("Ranking de UFs indisponivel sem a base agregada de UF.")
            return
        rank["rotulo"] = rank["valor_venda_num"].map(money_label)
        fig = px.bar(
            rank.tail(10),
            y="Estado - Sigla",
            x="valor_venda_num",
            orientation="h",
            color="valor_venda_num",
            color_continuous_scale="Reds",
            text="rotulo",
            title="Top 10 UFs com maior preco medio",
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(coloraxis_colorbar=dict(title="Preco medio"), margin=dict(r=90), xaxis_title="Preco medio (R$/litro)")
        st.plotly_chart(fig, use_container_width=True)



def main() -> None:
    st.set_page_config(page_title="Mercado de Combustiveis", layout="wide")
    st.title("📊 Dashboard Analítico - Combustíveis e Emissões")

    st.markdown("## 🎯 Objetivo")
    st.markdown("Analisar dados de combustíveis e emissões para identificar padrões, tendências e relações relevantes.")

    st.markdown("## ❓ Perguntas Analíticas")
    st.markdown("""
    1. Como evoluiu o consumo de combustíveis ao longo do tempo?
    2. Quais regiões apresentam maior consumo?
    3. Existe relação entre consumo de combustível e emissão de CO2?
    4. Qual combustível é mais utilizado?
    5. Há crescimento no uso de veículos eletrificados?
    6. Existem diferenças significativas entre estados?
    7. Há tendência de aumento ou redução de emissões?
    """)
    #st.caption(
    #    "Revisao completa das visoes do projeto com foco em leitura executiva, comparabilidade anual e exploracao interativa."
    #)

    tabs = st.tabs([
        "1. Resumo",
        "2. Mercado",
        "3. Inflacao",
        "4. Vendas ANP",
        "5. Emissoes e transicao",
        "6. Regioes e ranking",
    ])

    with tabs[0]:
        render_summary()
        if REPORT_FILE.exists():
            with st.expander("Resumo textual da analise"):
                st.markdown(REPORT_FILE.read_text(encoding="utf-8"))

    with tabs[1]:
        render_market_views()

    with tabs[2]:
        render_inflation_views()

    with tabs[3]:
        render_sales_volume_views()

    with tabs[4]:
        render_emission_transition_views()

    with tabs[5]:
        render_regional_rankings()


if __name__ == "__main__":
    main()
