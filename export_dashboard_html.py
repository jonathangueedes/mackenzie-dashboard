from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import pandas as pd
import plotly.express as px
import plotly.io as pio


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
OUT_FILE = OUT_DIR / "dashboard_compartilhavel.html"


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return text


def render_fig(fig, title: str) -> str:
    return f"<section><h2>{title}</h2>{pio.to_html(fig, include_plotlyjs=False, full_html=False)}</section>"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fuel_prod = pd.read_parquet(OUT_DIR / "fuel_annual_produto.parquet")
    fuel_reg = pd.read_parquet(OUT_DIR / "fuel_annual_regiao.parquet")
    fuel_uf = pd.read_parquet(OUT_DIR / "fuel_annual_uf.parquet")
    co2 = pd.read_parquet(OUT_DIR / "co2e_anual.parquet")
    ipca = pd.read_parquet(OUT_DIR / "ipca_anual_2020_2025.parquet")
    sidra = pd.read_parquet(OUT_DIR / "sidra_ipca_grupos_anual_2020_2025.parquet")
    ev = pd.read_parquet(OUT_DIR / "abve_eletrificados_serie_anual.parquet")

    sales = pd.read_parquet(BASE_DIR / "vendas-combustiveis-m3-1990-2025.parquet")
    sales = sales.rename(columns={c: normalize_text(c) for c in sales.columns})
    sales["ano"] = pd.to_numeric(sales.get("ano"), errors="coerce")
    sales["vendas"] = pd.to_numeric(sales.get("vendas").astype(str).str.replace(",", ".", regex=False), errors="coerce")
    sales["produto"] = sales.get("produto", pd.Series(dtype=str)).astype(str)
    sales["produto_norm"] = sales["produto"].map(normalize_text)
    sales = sales[sales["ano"].between(2020, 2025)]
    sales = sales[sales["produto_norm"].str.contains("etanol|gasolina", na=False)]
    sales["produto_grupo"] = sales["produto_norm"].map(lambda x: "etanol" if "etanol" in x else "gasolina")
    annual_sales = sales.groupby(["ano", "produto_grupo"], as_index=False)["vendas"].sum()
    pivot_sales = annual_sales.pivot(index="ano", columns="produto_grupo", values="vendas").reset_index().fillna(0)
    if "etanol" not in pivot_sales.columns:
        pivot_sales["etanol"] = 0.0
    if "gasolina" not in pivot_sales.columns:
        pivot_sales["gasolina"] = 0.0
    pivot_sales["share_etanol"] = pivot_sales["etanol"] / (pivot_sales["etanol"] + pivot_sales["gasolina"]).replace(0, pd.NA)

    # Competitive metric from fuel annual product means
    eg = fuel_prod[fuel_prod["Produto"].isin(["ETANOL", "GASOLINA"])].copy()
    egp = eg.pivot(index="ano", columns="Produto", values="valor_venda_num").reset_index()
    egp = egp.rename(columns={"ETANOL": "preco_etanol", "GASOLINA": "preco_gasolina"})
    egp["ratio_etanol_gasolina"] = egp["preco_etanol"] / egp["preco_gasolina"]

    # Figures
    fig_fuel = px.line(
        fuel_prod,
        x="ano",
        y="valor_venda_num",
        color="Produto",
        markers=True,
        title="Evolucao anual de preco por produto",
        labels={"valor_venda_num": "Preco medio (R$/litro)", "ano": "Ano"},
    )

    heat_data = fuel_reg.pivot(index="Regiao - Sigla", columns="ano", values="valor_venda_num")
    fig_region = px.imshow(
        heat_data,
        text_auto=True,
        color_continuous_scale="YlOrRd",
        title="Mapa de calor: preco medio por regiao e ano",
        labels={"x": "Ano", "y": "Regiao", "color": "R$/litro"},
    )

    uf_rank = fuel_uf.groupby("Estado - Sigla", as_index=False)["valor_venda_num"].mean().sort_values("valor_venda_num", ascending=False).head(10)
    uf_rank = uf_rank.sort_values("valor_venda_num", ascending=True)
    fig_uf = px.bar(
        uf_rank,
        y="Estado - Sigla",
        x="valor_venda_num",
        orientation="h",
        color="valor_venda_num",
        color_continuous_scale="Reds",
        title="Top 10 UFs com maior preco medio (2020-2025)",
        labels={"valor_venda_num": "Preco medio (R$/litro)", "Estado - Sigla": "UF"},
    )

    fuel_year = fuel_prod.groupby("ano", as_index=False)["valor_venda_num"].mean().rename(columns={"valor_venda_num": "preco_medio_combustivel"})
    fuel_year["var_preco_comb_anual_pct"] = fuel_year["preco_medio_combustivel"].pct_change() * 100
    fuel_year["var_preco_comb_anual_pct"] = fuel_year["var_preco_comb_anual_pct"].fillna(0)
    sidra_pivot = sidra.pivot(index="ano", columns="serie", values="inflacao_anual_pct").reset_index()
    merged = fuel_year.merge(sidra_pivot, on="ano", how="inner")
    fig_inf = px.line(
        merged,
        x="ano",
        y=["var_preco_comb_anual_pct", "IPCA transportes", "IPCA alimentacao e bebidas", "IPCA geral"],
        markers=True,
        title="Variacao anual: combustiveis x inflacao",
    )

    fig_sales = px.bar(
        annual_sales,
        x="ano",
        y="vendas",
        color="produto_grupo",
        barmode="stack",
        title="Volume anual de vendas ANP (etanol x gasolina)",
        labels={"vendas": "Volume", "produto_grupo": "Produto"},
    )

    co2_ratio = egp.merge(co2[["ano", "co2e_mt"]], on="ano", how="inner")
    fig_co2_ratio = px.scatter(
        co2_ratio,
        x="ratio_etanol_gasolina",
        y="co2e_mt",
        text="ano",
        trendline="ols" if len(co2_ratio) >= 3 else None,
        title="Razao etanol/gasolina vs CO2e",
        labels={"ratio_etanol_gasolina": "Razao etanol/gasolina", "co2e_mt": "CO2e (Mt)"},
    )
    fig_co2_ratio.update_traces(textposition="top center")

    ev_co2 = ev.merge(co2[["ano", "co2e_mt"]], on="ano", how="left")
    fig_ev = px.line(ev_co2, x="ano", y="total_eletrificados", markers=True, title="Serie anual de eletrificados")

    summary = f"""
    <h1>Painel Compartilhavel - Combustiveis, Inflacao e Emissoes</h1>
    <p>Janela: 2020-2025 | Gerado automaticamente a partir dos parquets locais do projeto.</p>
    <ul>
      <li>IPCA medio anual (2020-2025): {ipca['ipca_anual_pct'].mean():.2f}%</li>
      <li>CO2e disponivel ate: {int(co2['ano'].max()) if not co2.empty else 'n/d'}</li>
      <li>Share etanol no volume (ultimo ano): {pivot_sales.sort_values('ano').iloc[-1]['share_etanol']:.2%}</li>
    </ul>
    """

    html = [
        "<html><head><meta charset='utf-8'><title>Dashboard Compartilhavel</title>",
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;line-height:1.35}section{margin:28px 0}h1,h2{margin:0 0 10px 0}</style>",
        "</head><body>",
        summary,
        render_fig(fig_fuel, "Mercado por produto"),
        render_fig(fig_region, "Mapa regional"),
        render_fig(fig_uf, "Ranking de UFs"),
        render_fig(fig_inf, "Inflacao e repasse"),
        render_fig(fig_sales, "Vendas reais ANP"),
        render_fig(fig_co2_ratio, "Combustiveis e CO2e"),
        render_fig(fig_ev, "Eletrificados"),
        "</body></html>",
    ]

    OUT_FILE.write_text("\n".join(html), encoding="utf-8")
    print(f"HTML gerado: {OUT_FILE}")


if __name__ == "__main__":
    main()
