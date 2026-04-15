import streamlit as st
import pandas as pd
import plotly.express as px

# =============================
# CONFIGURAÇÃO DA PÁGINA
# =============================
st.set_page_config(page_title="Dashboard Combustíveis", layout="wide")

st.title("📊 Análise de Combustíveis no Brasil (1990–2025)")

st.markdown("""
Este dashboard tem como objetivo analisar o consumo de combustíveis ao longo do tempo,
identificando padrões, tendências e diferenças entre categorias.

As análises foram estruturadas a partir de perguntas orientadoras.
""")

# =============================
# CARREGAMENTO DOS DADOS
# =============================
df = pd.read_parquet("vendas-combustiveis-m3-1990-2025.parquet")

# Ajuste de colunas (se necessário)
df.columns = [col.lower() for col in df.columns]

# =============================
# PERGUNTAS
# =============================
st.header("❓ Perguntas de Análise")

st.markdown("""
1. Qual é a evolução do consumo ao longo do tempo?  
2. Qual combustível possui maior volume total?  
3. Existe variação significativa entre combustíveis?  
4. Há presença de outliers nos dados?  
5. Como os combustíveis se distribuem ao longo dos anos?  
""")

# =============================
# FILTROS
# =============================
st.sidebar.header("Filtros")

if "produto" in df.columns:
    combustiveis = st.sidebar.multiselect(
        "Selecione o combustível:",
        options=df["produto"].unique(),
        default=df["produto"].unique()
    )
    df = df[df["produto"].isin(combustiveis)]

# =============================
# PERGUNTA 1
# =============================
st.subheader("📈 Evolução do consumo ao longo do tempo")

if "ano" in df.columns and "volume" in df.columns:
    df_group = df.groupby("ano")["volume"].sum().reset_index()

    fig = px.line(df_group, x="ano", y="volume", title="Consumo ao longo do tempo")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Resposta:** Observa-se uma tendência de variação no consumo ao longo dos anos,
    indicando mudanças no padrão de uso de combustíveis.
    """)

# =============================
# PERGUNTA 2
# =============================
st.subheader("📊 Volume total por combustível")

if "produto" in df.columns and "volume" in df.columns:
    df_group = df.groupby("produto")["volume"].sum().reset_index()

    fig = px.bar(df_group, x="produto", y="volume", title="Volume total por combustível")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Resposta:** O combustível com maior volume total indica maior consumo relativo no período analisado.
    """)

# =============================
# PERGUNTA 3
# =============================
st.subheader("📉 Comparação entre combustíveis")

if "produto" in df.columns and "volume" in df.columns:
    fig = px.box(df, x="produto", y="volume", title="Distribuição por combustível")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Resposta:** A variação entre combustíveis mostra diferenças no comportamento de consumo.
    """)

# =============================
# PERGUNTA 4
# =============================
st.subheader("🚨 Identificação de outliers")

if "volume" in df.columns:
    fig = px.box(df, y="volume", title="Outliers no volume")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Resposta:** A presença de valores extremos pode indicar picos de consumo ou inconsistências.
    """)

# =============================
# PERGUNTA 5
# =============================
st.subheader("📊 Distribuição dos dados")

if "volume" in df.columns:
    fig = px.histogram(df, x="volume", nbins=50, title="Distribuição do volume")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Resposta:** A distribuição permite identificar concentração de valores e comportamento geral dos dados.
    """)

# =============================
# CONCLUSÃO
# =============================
st.header("📌 Conclusão")

st.markdown("""
A análise permitiu identificar padrões relevantes no consumo de combustíveis,
incluindo variações ao longo do tempo, diferenças entre categorias e presença de outliers.

Esses insights podem auxiliar na compreensão do comportamento do mercado e apoiar tomadas de decisão.
""")