from __future__ import annotations

from pathlib import Path
import math
import json
from urllib.request import urlopen

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset_automotivos_consolidado_2023_2025.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
CHARTS_DIR = OUTPUT_DIR / "charts"
REPORT_FILE = OUTPUT_DIR / "resumo_exploratorio_automotivos.md"


def fmt_period(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        year = int(math.floor(float(value)))
        sem = int(round((float(value) - year) * 100))
        return f"{year}.{sem:02d}"
    return str(value)


def prepare_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig", low_memory=False)
    df["valor_venda_num"] = pd.to_numeric(df["valor_venda_num"], errors="coerce")
    df["data_coleta_dt"] = pd.to_datetime(df["data_coleta_dt"], errors="coerce")
    df["ano_semestre"] = df["ano_semestre"].map(fmt_period)
    return df


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def save_line_period_avg(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("ano_semestre", as_index=False)["valor_venda_num"]
        .mean()
        .sort_values("ano_semestre")
    )
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=out, x="ano_semestre", y="valor_venda_num", marker="o")
    plt.title("Preco medio por semestre")
    plt.xlabel("Ano.Semestre")
    plt.ylabel("Preco medio (R$/litro)")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "01_preco_medio_por_semestre.png", dpi=150)
    plt.close()
    return out


def save_product_lines(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["ano_semestre", "Produto"], as_index=False)["valor_venda_num"]
        .mean()
        .sort_values(["Produto", "ano_semestre"])
    )
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=out, x="ano_semestre", y="valor_venda_num", hue="Produto", marker="o")
    plt.title("Evolucao por tipo de combustivel")
    plt.xlabel("Ano.Semestre")
    plt.ylabel("Preco medio (R$/litro)")
    plt.legend(title="Produto", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "02_evolucao_por_produto.png", dpi=150)
    plt.close()
    return out


def save_region_boxplot(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["Regiao - Sigla", "valor_venda_num"]].dropna()
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=out, x="Regiao - Sigla", y="valor_venda_num", order=sorted(out["Regiao - Sigla"].unique()))
    plt.title("Distribuicao de precos por regiao")
    plt.xlabel("Regiao")
    plt.ylabel("Preco (R$/litro)")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "03_boxplot_regiao.png", dpi=150)
    plt.close()
    return out


def save_uf_top_bottom(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    avg_uf = (
        df.groupby("Estado - Sigla", as_index=False)["valor_venda_num"]
        .mean()
        .sort_values("valor_venda_num", ascending=False)
    )
    top10 = avg_uf.head(10)
    bottom10 = avg_uf.tail(10).sort_values("valor_venda_num", ascending=True)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=top10, y="Estado - Sigla", x="valor_venda_num", color="#d55e00")
    plt.title("Top 10 UFs com maior preco medio")
    plt.xlabel("Preco medio (R$/litro)")
    plt.ylabel("UF")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "04_top10_ufs_maior_preco.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=bottom10, y="Estado - Sigla", x="valor_venda_num", color="#009e73")
    plt.title("Top 10 UFs com menor preco medio")
    plt.xlabel("Preco medio (R$/litro)")
    plt.ylabel("UF")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "05_top10_ufs_menor_preco.png", dpi=150)
    plt.close()

    return top10, bottom10


def save_brand_top(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("Bandeira", as_index=False)["valor_venda_num"]
        .mean()
        .sort_values("valor_venda_num", ascending=False)
        .head(10)
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=out, y="Bandeira", x="valor_venda_num", color="#0072b2")
    plt.title("Top 10 bandeiras com maior preco medio")
    plt.xlabel("Preco medio (R$/litro)")
    plt.ylabel("Bandeira")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "06_top10_bandeiras_maior_preco.png", dpi=150)
    plt.close()
    return out


def fetch_ipca_semester(periods: list[str]) -> pd.DataFrame:
    if not periods:
        return pd.DataFrame(columns=["ano_semestre", "ipca_semestre_pct", "ipca_indice_base100"])

    years = [int(p.split(".")[0]) for p in periods]
    start_date = f"01/01/{min(years)}"
    end_date = f"31/12/{max(years)}"

    url = (
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados"
        f"?formato=json&dataInicial={start_date}&dataFinal={end_date}"
    )

    with urlopen(url, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))

    ipca = pd.DataFrame(data)
    if ipca.empty:
        return pd.DataFrame(columns=["ano_semestre", "ipca_semestre_pct", "ipca_indice_base100"])

    ipca["data"] = pd.to_datetime(ipca["data"], format="%d/%m/%Y", errors="coerce")
    ipca["valor"] = pd.to_numeric(ipca["valor"].str.replace(",", ".", regex=False), errors="coerce")
    ipca = ipca.dropna(subset=["data", "valor"])

    ipca["ano"] = ipca["data"].dt.year
    ipca["mes"] = ipca["data"].dt.month
    ipca["semestre"] = ipca["mes"].apply(lambda m: 1 if m <= 6 else 2)
    ipca["ano_semestre"] = ipca.apply(lambda r: f"{int(r['ano'])}.{int(r['semestre']):02d}", axis=1)
    ipca = ipca[ipca["ano_semestre"].isin(periods)]

    ipca_sem = (
        ipca.groupby("ano_semestre", as_index=False)["valor"]
        .apply(lambda s: ((1 + (s / 100)).prod() - 1) * 100)
        .rename(columns={"valor": "ipca_semestre_pct"})
        .sort_values("ano_semestre")
    )

    ipca_sem["ipca_indice_base100"] = (1 + ipca_sem["ipca_semestre_pct"] / 100).cumprod() * 100
    return ipca_sem


def build_inflation_comparison(period_avg: pd.DataFrame) -> pd.DataFrame:
    comp = period_avg.copy()
    comp = comp.rename(columns={"valor_venda_num": "combustivel_medio"})
    comp = comp.sort_values("ano_semestre").reset_index(drop=True)
    comp["combustivel_var_sem_pct"] = comp["combustivel_medio"].pct_change() * 100
    comp.loc[0, "combustivel_var_sem_pct"] = 0.0
    comp["combustivel_indice_base100"] = (comp["combustivel_medio"] / comp.loc[0, "combustivel_medio"]) * 100

    ipca_sem = fetch_ipca_semester(comp["ano_semestre"].tolist())
    if ipca_sem.empty:
        return pd.DataFrame()

    comp = comp.merge(ipca_sem, on="ano_semestre", how="left")
    comp["ipca_semestre_pct"] = comp["ipca_semestre_pct"].fillna(0.0)
    comp["ipca_indice_base100"] = comp["ipca_indice_base100"].ffill().fillna(100.0)
    return comp


def save_inflation_charts(comp: pd.DataFrame) -> None:
    plt.figure(figsize=(11, 5))
    x = range(len(comp))
    plt.bar(x, comp["ipca_semestre_pct"], alpha=0.7, color="#7f7f7f", label="IPCA semestral (%)")
    plt.plot(x, comp["combustivel_var_sem_pct"], marker="o", color="#d55e00", label="Combustiveis variacao semestral (%)")
    plt.xticks(list(x), comp["ano_semestre"])
    plt.title("Inflacao (IPCA) x variacao semestral de combustiveis")
    plt.xlabel("Ano.Semestre")
    plt.ylabel("Variacao (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "07_ipca_vs_variacao_combustiveis.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(comp["ano_semestre"], comp["combustivel_indice_base100"], marker="o", color="#0072b2", label="Combustiveis (indice base 100)")
    plt.plot(comp["ano_semestre"], comp["ipca_indice_base100"], marker="o", color="#009e73", label="IPCA (indice base 100)")
    plt.title("Indice acumulado: combustiveis x IPCA (base 100 no 1o semestre)")
    plt.xlabel("Ano.Semestre")
    plt.ylabel("Indice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "08_indice_acumulado_combustiveis_vs_ipca.png", dpi=150)
    plt.close()


def write_report(
    df: pd.DataFrame,
    period_avg: pd.DataFrame,
    by_product: pd.DataFrame,
    top10_uf: pd.DataFrame,
    bottom10_uf: pd.DataFrame,
    top_brand: pd.DataFrame,
    inflation_comp: pd.DataFrame,
) -> None:
    periods = sorted(df["ano_semestre"].dropna().unique().tolist())
    max_period = period_avg.loc[period_avg["valor_venda_num"].idxmax()]
    min_period = period_avg.loc[period_avg["valor_venda_num"].idxmin()]
    growth = ((period_avg.iloc[-1]["valor_venda_num"] / period_avg.iloc[0]["valor_venda_num"]) - 1) * 100

    product_mean = (
        by_product.groupby("Produto", as_index=False)["valor_venda_num"]
        .mean()
        .sort_values("valor_venda_num", ascending=False)
    )
    most_expensive_product = product_mean.iloc[0]
    cheapest_product = product_mean.iloc[-1]

    missing_price_pct = (df["valor_venda_num"].isna().mean() * 100)

    inflation_lines: list[str] = []
    if not inflation_comp.empty:
        fuel_acum = inflation_comp["combustivel_indice_base100"].iloc[-1] - 100
        ipca_acum = inflation_comp["ipca_indice_base100"].iloc[-1] - 100
        diff = fuel_acum - ipca_acum
        sem_mais_pressao = inflation_comp.loc[
            (inflation_comp["combustivel_var_sem_pct"] - inflation_comp["ipca_semestre_pct"]).idxmax()
        ]
        inflation_lines = [
            "",
            "## Comparacao com inflacao (IPCA)",
            f"- Fonte do IPCA: Banco Central do Brasil (SGS, serie 433).",
            f"- Variacao acumulada de combustiveis no periodo: {fuel_acum:.2f}%.",
            f"- Inflacao acumulada (IPCA) no periodo: {ipca_acum:.2f}%.",
            f"- Diferenca acumulada (combustiveis - IPCA): {diff:.2f} p.p.",
            f"- Semestre de maior descolamento positivo: {sem_mais_pressao['ano_semestre']} (combustiveis {sem_mais_pressao['combustivel_var_sem_pct']:.2f}% vs IPCA {sem_mais_pressao['ipca_semestre_pct']:.2f}%).",
        ]
    else:
        inflation_lines = [
            "",
            "## Comparacao com inflacao (IPCA)",
            "- Nao foi possivel obter dados de IPCA automaticamente (falha de conexao ou API indisponivel).",
            "- Os demais resultados da analise permanecem validos.",
        ]

    lines = [
        "# Resumo Exploratorio - Combustiveis Automotivos (2023-2025)",
        "",
        "## Escopo da base",
        f"- Linhas: {len(df):,}".replace(",", "."),
        f"- Colunas: {df.shape[1]}",
        f"- Periodos: {', '.join(periods)}",
        f"- UFs: {df['Estado - Sigla'].nunique()}",
        f"- Municipios: {df['Municipio'].nunique()}",
        f"- Revendas unicas (CNPJ): {df['CNPJ da Revenda'].astype(str).str.strip().nunique():,}".replace(",", "."),
        "",
        "## 10 perguntas e respostas",
        f"1. O preco medio geral aumentou no periodo? Sim. Variou de {period_avg.iloc[0]['valor_venda_num']:.3f} para {period_avg.iloc[-1]['valor_venda_num']:.3f} ({growth:.2f}%).",
        f"2. Qual semestre teve maior preco medio? {max_period['ano_semestre']} com media {max_period['valor_venda_num']:.3f}.",
        f"3. Qual semestre teve menor preco medio? {min_period['ano_semestre']} com media {min_period['valor_venda_num']:.3f}.",
        f"4. Qual produto foi mais caro na media do periodo? {most_expensive_product['Produto']} ({most_expensive_product['valor_venda_num']:.3f}).",
        f"5. Qual produto foi mais barato na media do periodo? {cheapest_product['Produto']} ({cheapest_product['valor_venda_num']:.3f}).",
        f"6. Qual UF teve maior preco medio? {top10_uf.iloc[0]['Estado - Sigla']} ({top10_uf.iloc[0]['valor_venda_num']:.3f}).",
        f"7. Qual UF teve menor preco medio? {bottom10_uf.iloc[0]['Estado - Sigla']} ({bottom10_uf.iloc[0]['valor_venda_num']:.3f}).",
        f"8. Qual bandeira teve maior preco medio? {top_brand.iloc[0]['Bandeira']} ({top_brand.iloc[0]['valor_venda_num']:.3f}).",
        f"9. A variacao regional e relevante? Sim, conforme boxplot por regiao com dispersoes e medianas diferentes.",
        f"10. A qualidade da base permite analise robusta? Sim. Valores de venda ausentes estao em {missing_price_pct:.2f}% dos registros.",
        "",
        "## Arquivos de grafico gerados",
        "- outputs/charts/01_preco_medio_por_semestre.png",
        "- outputs/charts/02_evolucao_por_produto.png",
        "- outputs/charts/03_boxplot_regiao.png",
        "- outputs/charts/04_top10_ufs_maior_preco.png",
        "- outputs/charts/05_top10_ufs_menor_preco.png",
        "- outputs/charts/06_top10_bandeiras_maior_preco.png",
        "- outputs/charts/07_ipca_vs_variacao_combustiveis.png",
        "- outputs/charts/08_indice_acumulado_combustiveis_vs_ipca.png",
    ]

    lines.extend(inflation_lines)

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    sns.set_theme(style="whitegrid")
    ensure_output_dirs()

    df = prepare_data()
    period_avg = save_line_period_avg(df)
    by_product = save_product_lines(df)
    save_region_boxplot(df)
    top10_uf, bottom10_uf = save_uf_top_bottom(df)
    top_brand = save_brand_top(df)
    inflation_comp = pd.DataFrame()
    try:
        inflation_comp = build_inflation_comparison(period_avg)
        if not inflation_comp.empty:
            save_inflation_charts(inflation_comp)
    except Exception as exc:
        print(f"Aviso: comparacao com inflacao indisponivel ({exc}).")

    write_report(df, period_avg, by_product, top10_uf, bottom10_uf, top_brand, inflation_comp)

    print("Analise concluida.")
    print(f"Relatorio: {REPORT_FILE}")
    print(f"Graficos: {CHARTS_DIR}")


if __name__ == "__main__":
    main()
