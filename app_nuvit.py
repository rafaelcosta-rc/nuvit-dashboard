import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
import statsmodels.api as sm

st.set_page_config(layout="wide")
st.title("Nuvit - Inteligência Comercial")

file = st.file_uploader("Suba seu Excel", type=["xlsx"])

if file:

    df = pd.read_excel(file)

    # =========================
    # TRATAMENTO
    # =========================
    df["Data Emissão"] = pd.to_datetime(df["Data Emissão"], errors="coerce")
    df = df.dropna(subset=["Data Emissão"])

    df["AnoMes"] = df["Data Emissão"].dt.to_period("M").astype(str)
    df["Mes"] = df["Data Emissão"].dt.to_period("M")

    df["Receita"] = df["Valor Total"]
    df["Custo Total"] = df["Quantidade"] * df["Custo"]
    df["Margem"] = df["Receita"] - df["Custo Total"]
    df["Margem %"] = np.where(df["Receita"] != 0, df["Margem"] / df["Receita"], 0)

    # =========================
    # VISÃO EXECUTIVA
    # =========================
    st.header("📊 Visão Executiva")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Receita", f"R$ {df['Receita'].sum():,.0f}")
    col2.metric("Margem", f"R$ {df['Margem'].sum():,.0f}")
    col3.metric("Clientes", df["Cliente ID"].nunique())
    col4.metric("Ticket Médio", f"R$ {df['Receita'].mean():,.0f}")

    receita_mes = df.groupby("AnoMes")["Receita"].sum().reset_index()
    st.plotly_chart(px.line(receita_mes, x="AnoMes", y="Receita"), width='stretch')

    # =========================
    # CURVA ABC
    # =========================
    st.header("📈 Curva ABC")

    abc = df.groupby("Cliente ID")["Receita"].sum().reset_index()
    abc = abc.sort_values(by="Receita", ascending=False)

    abc["Acumulado"] = abc["Receita"].cumsum()
    abc["Perc"] = abc["Acumulado"] / abc["Receita"].sum()

    fig = go.Figure()
    fig.add_bar(x=abc["Cliente ID"], y=abc["Receita"], name="Receita")
    fig.add_trace(go.Scatter(x=abc["Cliente ID"], y=abc["Perc"], yaxis="y2", name="% Acumulado"))

    fig.update_layout(yaxis2=dict(overlaying='y', side='right', tickformat=".0%"))
    st.plotly_chart(fig, width='stretch')

    # =========================
    # MIX
    # =========================
    st.header("📦 Mix de Vendas")
    mix = df.groupby("Categoria")["Receita"].sum().reset_index()
    st.plotly_chart(px.bar(mix, x="Categoria", y="Receita"), width='stretch')

    # =========================
    # DISPERSÃO
    # =========================
    st.header("💰 Dispersão de Preços")
    st.plotly_chart(px.box(df, x="Produto ID", y="Preço Unitário Líquido"), width='stretch')

    # =========================
    # PERFORMANCE
    # =========================
    st.header("🧑‍💼 Performance Comercial")
    vendedor_df = df.groupby("Vendedor")["Receita"].sum().reset_index()
    st.plotly_chart(px.bar(vendedor_df, x="Vendedor", y="Receita"), width='stretch')

    # =========================
    # CLUSTER
    # =========================
    st.header("🧠 Clusterização")

    cluster_df = df.groupby("Cliente ID").agg({
        "Receita": "sum",
        "NF": "count"
    }).rename(columns={"NF": "Frequencia"})

    cluster_df["Ticket"] = cluster_df["Receita"] / cluster_df["Frequencia"]
    cluster_df = cluster_df.replace([np.inf, -np.inf], np.nan).dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(cluster_df)

    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    cluster_df["Cluster"] = kmeans.fit_predict(X)

    st.plotly_chart(px.scatter(cluster_df, x="Receita", y="Frequencia",
                               color=cluster_df["Cluster"].astype(str)), width='stretch')

    # =========================
    # COHORT
    # =========================
    st.header("📊 Cohort")

    primeira = df.groupby("Cliente ID")["Mes"].min().reset_index()
    primeira.columns = ["Cliente ID", "Cohort"]

    cohort_df = df.merge(primeira, on="Cliente ID")
    cohort_df["Periodo"] = (cohort_df["Mes"] - cohort_df["Cohort"]).apply(lambda x: x.n)

    cohort = cohort_df.groupby(["Cohort", "Periodo"])["Cliente ID"].nunique().unstack()

    fig, ax = plt.subplots()
    sns.heatmap(cohort, cmap="Blues", ax=ax)
    st.pyplot(fig)

    # =========================
    # ELASTICIDADE
    # =========================
    st.header("📉 Elasticidade")

    elasticidades = []

    for produto, grupo in df.groupby("Produto ID"):
        grupo = grupo[(grupo["Quantidade"] > 0) & (grupo["Preço Unitário Líquido"] > 0)]
        if len(grupo) < 15:
            continue
        try:
            y = np.log(grupo["Quantidade"])
            X = np.log(grupo["Preço Unitário Líquido"])
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            coef = model.params.iloc[1]

            elasticidades.append({
                "Produto": produto,
                "Elasticidade": coef,
                "Tipo": "Elástico" if coef < -1 else "Inelástico"
            })
        except:
            continue

    st.dataframe(pd.DataFrame(elasticidades))

# =========================
