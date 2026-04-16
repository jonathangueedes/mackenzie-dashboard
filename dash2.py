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


def brl_compact_label(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    abs_value = abs(float(value))
    if abs_value >= 1_000_000_000:
        return f"R$ {value / 1_000_000_000:.1f} bi"
    if abs_value >= 1_000_000:
        return f"R$ {value / 1_000_000:.1f} mi"
    return f"R$ {value:,.0f}".replace(",", ".")


def classify_fuel_group(produto_norm: str) -> str:
    if "etanol" in produto_norm:
        return "etanol"
    if "gasolina" in produto_norm and "aviacao" in produto_norm:
        return "gasolina aviacao"
    if "gasolina" in produto_norm:
        return "gasolina"
    if "diesel" in produto_norm:
        return "diesel"
    if "glp" in produto_norm:
        return "glp"
    if "querosene" in produto_norm and "aviacao" in produto_norm:
        return "querosene aviacao"
    if "querosene" in produto_norm:
        return "querosene iluminante"
    if "oleo_combustivel" in produto_norm:
        return "oleo combustivel"
    return "outros"


def classify_price_group(produto_norm: str) -> str:
    if "etanol" in produto_norm:
        return "etanol"
    if "gasolina" in produto_norm:
        return "gasolina"
    if "diesel" in produto_norm:
        return "diesel"
    if "gnv" in produto_norm:
        return "gnv"
    return "outros"


def pct_change(first: float, last: float) -> float | None:
    if pd.isna(first) or pd.isna(last) or first == 0:
        return None
    return (last / first - 1) * 100


def format_pct(value: float | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    return f"{value:.{digits}f}%"


def render_generated_text(title: str, lines: list[str]) -> None:
    valid = [line for line in lines if line]
    if not valid:
        return
    st.markdown(f"#### {title}")
    st.markdown("\n".join([f"- {line}" for line in valid]))


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
    raw["produto_grupo"] = raw["produto_norm"].map(classify_fuel_group)
    raw = raw[~raw["produto_grupo"].isin(["oleo combustivel", "querosene iluminante"])].copy()
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
def load_financial_proxy_annual() -> pd.DataFrame:
    auto_file = auto_find_anp_sales_file()
    if auto_file is None:
        return pd.DataFrame(columns=["ano", "mov_brl", "cobertura_volume", "volume_total_m3"])

    sales_raw = read_tabular_file(auto_file)
    sales = prepare_anp_sales(sales_raw)
    sales = sales[sales["ano"].between(YEAR_MIN, YEAR_MAX)].copy()
    if sales.empty:
        return pd.DataFrame(columns=["ano", "mov_brl", "cobertura_volume", "volume_total_m3"])

    volume_by_group = sales.groupby(["ano", "produto_grupo"], as_index=False)["volume"].sum()
    volume_total = volume_by_group.groupby("ano", as_index=False)["volume"].sum().rename(columns={"volume": "volume_total_m3"})

    price = load_fuel_full()[["ano", "Produto", "valor_venda_num"]].copy()
    price = price[price["valor_venda_num"].notna()].copy()
    price["produto_grupo"] = price["Produto"].map(normalize_text).map(classify_price_group)
    price = price[price["produto_grupo"].isin(["etanol", "gasolina", "diesel", "gnv"])].copy()
    price_by_group = price.groupby(["ano", "produto_grupo"], as_index=False)["valor_venda_num"].mean()

    merged = volume_by_group.merge(price_by_group, on=["ano", "produto_grupo"], how="inner")
    merged["mov_brl"] = merged["volume"] * 1000 * merged["valor_venda_num"]

    mov_by_year = merged.groupby("ano", as_index=False).agg(
        mov_brl=("mov_brl", "sum"),
        volume_precificado_m3=("volume", "sum"),
    )

    out = volume_total.merge(mov_by_year, on="ano", how="left")
    out["cobertura_volume"] = out["volume_precificado_m3"] / out["volume_total_m3"].replace(0, pd.NA)
    return out[["ano", "mov_brl", "cobertura_volume", "volume_total_m3"]].sort_values("ano")


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
    ipca = load_ipca_annual()
    co2 = load_co2e_annual()
    financial = load_financial_proxy_annual()
    ipca_media_periodo = ipca["ipca_anual_pct"].mean() if not ipca.empty else pd.NA
    ipca_ultimo_ano = ipca.sort_values("ano").iloc[-1]["ipca_anual_pct"] if not ipca.empty else pd.NA
    mov_media_anual = financial["mov_brl"].dropna().mean() if not financial.empty else pd.NA
    cobertura_media = financial["cobertura_volume"].dropna().mean() * 100 if not financial.empty else pd.NA
    co2_medio_anual = co2["co2e_mt"].dropna().mean() if not co2.empty else pd.NA
    emissao_media_card = f"{co2_medio_anual:.0f} Mt CO2e/ano" if pd.notna(co2_medio_anual) else "n/d"
    if not co2.empty:
        co2_cover = sorted(pd.to_numeric(co2["ano"], errors="coerce").dropna().astype(int).unique().tolist())
        emissao_label = f"Emissoes medias ({co2_cover[0]}-{co2_cover[-1]})"
    else:
        emissao_label = f"Emissoes medias ({YEAR_MIN}-{YEAR_MAX})"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Janela analisada", f"{YEAR_MIN}-{YEAR_MAX}")
    c2.metric("Movimentacao media anual estimada", brl_compact_label(mov_media_anual))
    c3.metric("IPCA medio anual (2020-2025)", f"{ipca_media_periodo:.1f}%" if pd.notna(ipca_media_periodo) else "n/d")
    c4.metric(emissao_label, emissao_media_card)

    if pd.notna(cobertura_media):
        st.caption(f"Estimativa financeira no periodo: soma anual de volume ANP (m3) x preco medio por combustivel (R$/litro), com cobertura media de volume precificado de {cobertura_media:.1f}%.")

    if pd.notna(ipca_ultimo_ano):
        st.caption(f"Referencia: IPCA do ultimo ano da serie ({int(ipca.sort_values('ano').iloc[-1]['ano'])}) = {ipca_ultimo_ano:.1f}%.")

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
    st.caption("Como ler: mostra a evolucao do preco medio por produto ao longo dos anos. Inclinações mais fortes indicam alta de preco mais acelerada.")
    st.plotly_chart(fig, use_container_width=True)

    var_prod = []
    for prod, grp in annual_prod.groupby("Produto"):
        grp_o = grp.sort_values("ano")
        if len(grp_o) < 2:
            continue
        var = pct_change(grp_o.iloc[0]["valor_venda_num"], grp_o.iloc[-1]["valor_venda_num"])
        if var is not None:
            var_prod.append((prod, var))

    if var_prod:
        top_alta = max(var_prod, key=lambda x: x[1])
        top_queda = min(var_prod, key=lambda x: x[1])
        render_generated_text(
            "Leitura automatica",
            [
                f"Produto com maior alta acumulada no periodo: {top_alta[0]} ({top_alta[1]:.1f}%).",
                f"Produto com menor variacao acumulada no periodo: {top_queda[0]} ({top_queda[1]:.1f}%).",
                f"Produtos analisados neste grafico: {', '.join(sorted(annual_prod['Produto'].dropna().unique().tolist()))}.",
            ],
        )

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
        st.caption("Como ler: tons mais quentes representam regioes/anos com preco medio mais alto; tons mais frios indicam preco menor.")
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
        st.caption("Como ler: ranking das 10 UFs com maior preco medio no periodo. Barras maiores significam combustivel mais caro, em media.")
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
    st.caption("Como ler: compara a variacao anual (%) dos combustiveis com os grupos de inflacao. Quando a linha de combustiveis fica acima, houve alta mais forte que o indice.")
    st.plotly_chart(fig, use_container_width=True)

    corr_transp = merged[["var_preco_comb_anual_pct", "IPCA transportes"]].corr().iloc[0, 1] if len(merged) >= 2 else pd.NA
    idx_max_comb = merged["var_preco_comb_anual_pct"].idxmax() if not merged.empty else None
    ano_pico = int(merged.loc[idx_max_comb, "ano"]) if idx_max_comb is not None else None
    pico_comb = merged.loc[idx_max_comb, "var_preco_comb_anual_pct"] if idx_max_comb is not None else pd.NA

    render_generated_text(
        "Leitura automatica",
        [
            f"A correlacao entre variacao de combustiveis e IPCA transportes foi {corr_transp:.2f}." if pd.notna(corr_transp) else "Correlacao indisponivel por falta de anos comparaveis.",
            f"O maior pico anual de variacao de combustiveis ocorreu em {ano_pico}, com {pico_comb:.1f}%." if ano_pico is not None and pd.notna(pico_comb) else "Pico anual de combustiveis indisponivel.",
            "Use este bloco para comparar pressao de preco setorial (combustiveis) versus inflacao oficial de transporte.",
        ],
    )

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
        st.caption("Como ler: cada ponto representa um ano. Padrão ascendente sugere que altas de combustiveis tendem a acompanhar altas no IPCA de transportes.")
        st.plotly_chart(scat, use_container_width=True)

    with c2:
        corr_cols = ["var_preco_comb_anual_pct", "IPCA geral", "IPCA transportes", "IPCA alimentacao e bebidas"]
        corr = merged[corr_cols].corr().round(3)
        heat = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1, title="Matriz de correlacao")
        heat.update_layout(coloraxis_colorbar=dict(title="Correlacao"))
        st.caption("Como ler: valores proximos de 1 indicam relacao positiva forte; proximos de -1, relacao inversa; proximos de 0, pouca associacao linear.")
        st.plotly_chart(heat, use_container_width=True)



def render_sales_volume_views() -> None:
    st.subheader("Vendas reais ANP")
    st.caption("Este bloco usa volume comercializado para complementar as visoes baseadas em preco.")

    sales_df: pd.DataFrame | None = None
    try:
        auto_file = auto_find_anp_sales_file()
        if auto_file is not None:
            sales_df = read_tabular_file(auto_file)
    except Exception:
        sales_df = None

    if sales_df is None:
        st.info("Base de vendas ANP nao encontrada no projeto.")
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

    st.caption("Unidade do volume: m3 (metros cubicos). Conversao: 1 m3 = 1.000 litros.")

    annual = prepared.groupby(["ano", "produto_grupo"], as_index=False)["volume"].sum()
    pivot = annual.pivot(index="ano", columns="produto_grupo", values="volume").reset_index().fillna(0)

    fuel_cols = [c for c in pivot.columns if c != "ano"]
    total_volume = pivot[fuel_cols].sum(axis=1).replace(0, pd.NA)
    if "etanol" in pivot.columns:
        pivot["share_etanol"] = pivot["etanol"] / total_volume
    else:
        pivot["share_etanol"] = pd.NA

    share_long = pivot.melt(id_vars="ano", value_vars=fuel_cols, var_name="produto_grupo", value_name="volume")
    share_long["share_volume"] = share_long["volume"] / share_long.groupby("ano")["volume"].transform("sum").replace(0, pd.NA)

    bar = px.bar(
        annual,
        x="ano",
        y="volume",
        color="produto_grupo",
        barmode="stack",
        title="Volume anual empilhado por combustivel (m3)",
        labels={"volume": "Volume (m3)", "ano": "Ano", "produto_grupo": "Produto"},
    )
    bar.update_layout(
        legend_title_text="Produto",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
        margin=dict(t=70, b=95),
        title=dict(x=0, xanchor="left"),
    )
    st.caption("Como ler: volume anual total em m3, empilhado por tipo de combustivel. A altura total da barra e o volume total do ano.")
    st.plotly_chart(bar, use_container_width=True)

    line = px.line(
        share_long,
        x="ano",
        y="share_volume",
        color="produto_grupo",
        markers=True,
        title="Share anual por combustivel no volume total",
        labels={"share_volume": "Share no volume total", "ano": "Ano", "produto_grupo": "Produto"},
    )
    line.update_layout(
        yaxis_tickformat=".0%",
        legend_title_text="Produto",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
        margin=dict(t=70, b=95),
        title=dict(x=0, xanchor="left"),
    )
    st.caption("Como ler: participacao anual de cada combustivel no volume total. Ex.: 0,40 significa 40% do volume do ano.")
    st.plotly_chart(line, use_container_width=True)

    pivot_o = pivot.sort_values("ano")
    if len(pivot_o) >= 2:
        vol_total_ini = total_volume.loc[pivot_o.index[0]]
        vol_total_fim = total_volume.loc[pivot_o.index[-1]]
        vol_var = pct_change(vol_total_ini, vol_total_fim)
        share_var = pct_change(pivot_o.iloc[0]["share_etanol"], pivot_o.iloc[-1]["share_etanol"])
    else:
        vol_var = None
        share_var = None

    render_generated_text(
        "Leitura automatica",
        [
            f"O volume total (todos os combustiveis) variou {format_pct(vol_var)} no periodo analisado.",
            f"O share do etanol mudou {format_pct(share_var)} no periodo, saindo de {pivot_o.iloc[0]['share_etanol']:.1%} para {pivot_o.iloc[-1]['share_etanol']:.1%}." if len(pivot_o) >= 2 and pd.notna(pivot_o.iloc[0]["share_etanol"]) and pd.notna(pivot_o.iloc[-1]["share_etanol"]) else "Share do etanol indisponivel para toda a janela.",
            "Unidade de volume: m3. Para leitura em litros, multiplique por 1.000.",
        ],
    )

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
        co2_bar = panel.sort_values("ano").copy()
        co2_bar["direcao"] = co2_bar["co2e_mt"].diff().map(lambda v: "Alta anual" if pd.notna(v) and v > 0 else "Queda/estavel")
        fig = px.bar(
            co2_bar,
            x="ano",
            y="co2e_mt",
            color="direcao",
            title="CO2e nacional por ano",
            labels={"co2e_mt": "Mt CO2e", "ano": "Ano", "direcao": "Direcao"},
            color_discrete_map={"Alta anual": "#c0392b", "Queda/estavel": "#1f77b4"},
        )
        fig.update_layout(yaxis_title="Mt CO2e")
        apply_legend(fig, "Direcao")
        st.caption("Como ler: barras facilitam comparar niveis anuais de emissao. Vermelho indica alta vs ano anterior e azul indica queda/estabilidade.")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        ratio_view = panel.sort_values("ano").copy()
        ratio_view["competitividade"] = ratio_view["ratio_etanol_gasolina"].map(lambda v: "Competitivo" if v <= 0.70 else "Nao competitivo")
        fig = px.scatter(
            ratio_view,
            x="ano",
            y="ratio_etanol_gasolina",
            color="competitividade",
            title="Razao etanol/gasolina (dot plot)",
            labels={"ratio_etanol_gasolina": "Razao de preco", "ano": "Ano", "competitividade": "Faixa"},
            color_discrete_map={"Competitivo": "#2ca02c", "Nao competitivo": "#d62728"},
        )
        for _, row in ratio_view.iterrows():
            fig.add_shape(
                type="line",
                x0=row["ano"],
                x1=row["ano"],
                y0=0.70,
                y1=row["ratio_etanol_gasolina"],
                line=dict(color="#9aa0a6", width=1),
            )
        fig.add_hline(y=0.70, line_dash="dash", line_color="#555", annotation_text="limiar 0,70", annotation_position="top left")
        fig.update_traces(marker=dict(size=10), hovertemplate="Ano=%{x}<br>Razao=%{y:.3f}<extra></extra>")
        fig.update_layout(yaxis_title="Razao de preco")
        apply_legend(fig, "Faixa")
        st.caption("Como ler: cada ponto e um ano; a linha tracejada marca 0,70 (regra pratica de competitividade do etanol).")
        st.plotly_chart(fig, use_container_width=True)

    panel_o = panel.sort_values("ano")
    co2_var = pct_change(panel_o.iloc[0]["co2e_mt"], panel_o.iloc[-1]["co2e_mt"]) if len(panel_o) >= 2 else None
    ratio_var = pct_change(panel_o.iloc[0]["ratio_etanol_gasolina"], panel_o.iloc[-1]["ratio_etanol_gasolina"]) if len(panel_o) >= 2 else None
    corr_ratio_co2 = panel_o[["ratio_etanol_gasolina", "co2e_mt"]].corr().iloc[0, 1] if len(panel_o) >= 2 else pd.NA

    render_generated_text(
        "Leitura automatica",
        [
            f"No periodo em comum, CO2e variou {format_pct(co2_var)} e a razao etanol/gasolina variou {format_pct(ratio_var)}.",
            f"A correlacao entre razao etanol/gasolina e CO2e foi {corr_ratio_co2:.2f}." if pd.notna(corr_ratio_co2) else "Correlacao razao x CO2e indisponivel.",
            "Essa leitura e exploratoria e nao implica causalidade direta entre variaveis.",
        ],
    )

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
        st.caption("Como ler: boxplot resume distribuicao de precos por regiao; linha central e a mediana, caixa e intervalo interquartil, pontos sao outliers.")
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
        st.caption("Como ler: ranking das UFs mais caras na janela selecionada. Use para comparar diferencas territoriais de preco medio.")
        st.plotly_chart(fig, use_container_width=True)

        med_reg = view.groupby("Regiao - Sigla", as_index=False)["valor_venda_num"].median().sort_values("valor_venda_num")
        reg_min = med_reg.iloc[0]
        reg_max = med_reg.iloc[-1]
        gap_reg = reg_max["valor_venda_num"] - reg_min["valor_venda_num"]

        uf_top = rank.iloc[-1]
        uf_bottom = rank.iloc[0]
        gap_uf = uf_top["valor_venda_num"] - uf_bottom["valor_venda_num"]

        render_generated_text(
            "Leitura automatica",
            [
                f"Entre regioes, a menor mediana foi {reg_min['Regiao - Sigla']} ({money_label(reg_min['valor_venda_num'])}) e a maior foi {reg_max['Regiao - Sigla']} ({money_label(reg_max['valor_venda_num'])}), com diferenca de {money_label(gap_reg)}.",
                f"No ranking de UFs, o gap entre a UF mais barata e a mais cara foi de {money_label(gap_uf)} por litro.",
                "Esses diferenciais ajudam a priorizar investigacoes de logistica, tributacao e concorrencia regional.",
            ],
        )



def main() -> None:
    st.set_page_config(page_title="Mercado de Combustiveis", layout="wide")
    st.title("📊 Dashboard Analítico - Combustíveis e Emissões")

    st.markdown("## 🎯 Objetivo")
    st.markdown("Analisar dados de combustíveis e emissões para identificar padrões, tendências e relações relevantes.")

    st.markdown("## ❓ Perguntas Analíticas")
    st.markdown("""
    1. Como o consumo de combustiveis evoluiu ao longo do tempo?
    2. Como evoluem os precos dos diferentes combustiveis ao longo do tempo?
    3. Qual combustivel apresenta maior estabilidade de precos?
    4. Existe diferenca de precos entre regioes ao longo do tempo?
    5. Quais regioes apresentam maiores niveis de precos?
    6. Quais estados apresentam os maiores precos medios?
    7. Como os combustiveis se comportam em relacao a inflacao?
    8. Os combustiveis sao mais instaveis que a inflacao?
    9. Existe relacao entre combustiveis e inflacao de transportes?
    10. Como evoluiu o volume total de combustiveis ao longo do tempo?
    11. Qual combustivel mais contribui para o volume total?
    12. Como evoluiram as emissoes de CO2e ao longo do tempo?
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
