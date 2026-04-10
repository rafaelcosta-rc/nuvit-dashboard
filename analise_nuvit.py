import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
import statsmodels.api as sm

# ==============================
# CONFIG
# ==============================
ARQUIVO = "base_nuvit_exemplo.xlsx"

print("🚀 Iniciando análise...")

# ==============================
# LOAD E TRATAMENTO
# ==============================
df = pd.read_excel(ARQUIVO)

df["Data Emissão"] = pd.to_datetime(df["Data Emissão"], errors="coerce")
df = df.dropna(subset=["Data Emissão"])

df["AnoMes"] = df["Data Emissão"].dt.to_period("M").astype(str)
df["Mes"] = df["Data Emissão"].dt.to_period("M")

df["Receita"] = df["Valor Total"]
df["Custo Total"] = df["Quantidade"] * df["Custo"]
df["Margem"] = df["Receita"] - df["Custo Total"]
df["Margem %"] = np.where(df["Receita"] != 0, df["Margem"] / df["Receita"], 0)

print(f"✅ Base carregada: {df.shape}")

# ==============================
# 1. CURVA ABC CLIENTES
# ==============================
abc_cliente = df.groupby("Cliente ID")["Receita"].sum().reset_index()
abc_cliente = abc_cliente.sort_values(by="Receita", ascending=False)

abc_cliente["% Acumulado"] = abc_cliente["Receita"].cumsum() / abc_cliente["Receita"].sum()

def classificar(p):
    if p <= 0.8:
        return "A"
    elif p <= 0.95:
        return "B"
    else:
        return "C"

abc_cliente["Classe"] = abc_cliente["% Acumulado"].apply(classificar)

print("✅ ABC clientes OK")

# ==============================
# 2. MIX DE VENDAS
# ==============================
mix = df.groupby("Categoria")["Receita"].sum().reset_index()
mix["%"] = mix["Receita"] / mix["Receita"].sum()

# ==============================
# 3. DISPERSÃO DE PREÇOS
# ==============================
preco = df.groupby("Produto ID")["Preço Unitário Líquido"].agg(
    preco_medio="mean",
    preco_min="min",
    preco_max="max",
    desvio="std"
).reset_index()

# ==============================
# 4. PERFORMANCE VENDEDOR
# ==============================
vendedor = df.groupby("Vendedor").agg(
    receita=("Receita", "sum"),
    margem=("Margem", "sum"),
    clientes=("Cliente ID", "nunique"),
    pedidos=("NF", "count"),
    desconto_medio=("Desconto (%)", "mean")
).reset_index()

print("✅ Performance vendedor OK")

# ==============================
# 5. COHORT
# ==============================
primeira_compra = df.groupby("Cliente ID")["Mes"].min().reset_index()
primeira_compra.columns = ["Cliente ID", "Cohort"]

df_cohort = df.merge(primeira_compra, on="Cliente ID")
df_cohort["Periodo"] = (df_cohort["Mes"] - df_cohort["Cohort"]).apply(lambda x: x.n)

cohort = df_cohort.groupby(["Cohort", "Periodo"])["Cliente ID"].nunique().unstack()

print("✅ Cohort OK")

# ==============================
# 6. CLUSTERIZAÇÃO
# ==============================
cliente_cluster = df.groupby("Cliente ID").agg(
    receita=("Receita", "sum"),
    frequencia=("NF", "count")
)

cliente_cluster["ticket"] = cliente_cluster["receita"] / cliente_cluster["frequencia"]
cliente_cluster = cliente_cluster.replace([np.inf, -np.inf], np.nan).dropna()

scaler = StandardScaler()
X = scaler.fit_transform(cliente_cluster)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cliente_cluster["cluster"] = kmeans.fit_predict(X)

print("✅ Clusterização OK")

# ==============================
# 7. ELASTICIDADE (ROBUSTA)
# ==============================
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
            "Produto ID": produto,
            "Elasticidade": coef
        })

    except:
        continue

elasticidade_df = pd.DataFrame(elasticidades)

print("✅ Elasticidade OK")

# ==============================
# 8. CROSS-SELL (CORRIGIDO)
# ==============================
basket = df.groupby(["NF", "Produto ID"])["Quantidade"].sum().unstack().fillna(0)
basket = (basket > 0)

freq = apriori(basket, min_support=0.005, use_colnames=True)

if freq.empty:
    print("⚠️ Nenhum padrão de cross-sell encontrado")
    rules = pd.DataFrame()
else:
    rules = association_rules(freq, metric="lift", min_threshold=1.0)

print("✅ Cross-sell OK")

# ==============================
# EXPORT
# ==============================
with pd.ExcelWriter("resultado_nuvit.xlsx") as writer:
    abc_cliente.to_excel(writer, sheet_name="ABC_Clientes", index=False)
    mix.to_excel(writer, sheet_name="Mix", index=False)
    preco.to_excel(writer, sheet_name="Preco", index=False)
    vendedor.to_excel(writer, sheet_name="Vendedor", index=False)
    cliente_cluster.to_excel(writer, sheet_name="Cluster")
    elasticidade_df.to_excel(writer, sheet_name="Elasticidade", index=False)
    rules.to_excel(writer, sheet_name="CrossSell", index=False)
    cohort.to_excel(writer, sheet_name="Cohort")

print("📁 Arquivo gerado: resultado_nuvit.xlsx")

# ==============================
# GRÁFICOS
# ==============================
plt.figure()
df.groupby("AnoMes")["Receita"].sum().plot(title="Receita ao longo do tempo")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
sns.barplot(data=mix, x="Categoria", y="Receita")
plt.title("Mix de vendas")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
sns.scatterplot(data=cliente_cluster, x="receita", y="frequencia", hue="cluster")
plt.title("Cluster de Clientes")
plt.tight_layout()
plt.show()

print("🏁 Análise finalizada com sucesso!")