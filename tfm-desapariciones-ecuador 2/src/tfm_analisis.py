# -*- coding: utf-8 -*-
"""Modelado estocástico y aprendizaje no supervisado en dinámicas de desaparición en Ecuador.
Trabajo Fin de Máster - Carlos Andrés Mendoza Bravo. Máster en Ingeniería Matemática y Computación (UNIR).
Uso: TFM_XLSX=ruta/al/archivo.xlsx TFM_HEADLESS=1 python tfm_analisis.py
"""

# ======================================================================
# # Modelado estocástico y aprendizaje no supervisado en dinámicas de desaparición en Ecuador
# Trabajo Fin de Máster · Máster en Ingeniería Matemática y Computación (UNIR) · Carlos Andrés Mendoza Bravo
# 
# Cuaderno reproducible con el análisis completo de la memoria. Datos: conjunto de datos abiertos *Personas desaparecidas* del Ministerio del Interior de Ecuador (2017-2025).
# 
# Contenido:
# 1. Carga y revisión de calidad de los datos.
# 2. Preparación: definición del evento (cierre del caso), de la censura y de la fecha de corte.
# 3. Análisis de supervivencia: Kaplan-Meier, riesgos competitivos (Aalen-Johansen), log-rank, Cox estratificado por año, RMST.
# 4. Agrupamiento espacial DBSCAN (distancia de Haversine) en Guayas: selección de parámetros, severidad estandarizada por año con corrección FDR y validación temporal.
# 5. Reglas de asociación (Apriori) perfil → desenlace con filtros de robustez y validación temporal.
# 6. Integración de zonas y perfiles en un modelo de Cox con datos posteriores.
# ======================================================================
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg") if os.environ.get("TFM_HEADLESS") else None
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

try:
    import lifelines, mlxtend, statsmodels
except ImportError:
    raise SystemExit("Instale: pip install lifelines mlxtend statsmodels")
    import lifelines, mlxtend, statsmodels

from lifelines import KaplanMeierFitter, CoxPHFitter, AalenJohansenFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test, proportional_hazard_test
from lifelines.utils import median_survival_times, restricted_mean_survival_time
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors, BallTree
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
from mlxtend.frequent_patterns import apriori, association_rules

RUTA_XLSX = os.environ.get("TFM_XLSX", "/content/mdi_personasdesaparecidas_pm_2017_2025.xlsx")
SEMILLA   = 42
KM_TIERRA = 6371.0088
os.makedirs("figuras", exist_ok=True); os.makedirs("tablas", exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3, "font.size": 10})
RES = {}
def mostrar(x):
    try: display(x)
    except NameError: print(x)

# ======================================================================
# ## 1. Carga y revisión de calidad de los datos
# ======================================================================
raw = pd.read_excel(RUTA_XLSX)
print(f"Registros: {raw.shape[0]:,} | Variables: {raw.shape[1]}")
RES["n_registros"] = int(raw.shape[0])
aud = pd.DataFrame({"tipo": raw.dtypes.astype(str), "nulos": raw.isnull().sum(), "% nulos": (raw.isnull().sum()/len(raw)*100).round(2)})
mostrar(aud)
print("\nsituacion_actual:\n", raw["situacion_actual"].value_counts())
# Hallazgo 1: los 'FALLECIDO' tienen fecha de localización (se hallaron) -> son un EVENTO, no censura
raw["f_loc"] = pd.to_datetime(raw["fecha_localizacion"].replace({"NO_APLICA": np.nan, "SIN_DATO": np.nan}), errors="coerce", format="mixed")
mostrar(pd.crosstab(raw["situacion_actual"], raw["f_loc"].notna(), rownames=["situación"], colnames=["tiene fecha de localización"]))

# Hallazgo 2: corte real de la base
FECHA_CORTE_ORIG = raw["fecha_desaparicion"].max()
FECHA_CORTE      = max(raw["f_loc"].max(), raw["fecha_denuncia"].max(), raw["fecha_desaparicion"].max())
print(f"\nÚltima fecha de desaparición: {FECHA_CORTE_ORIG.date()}  |  corte observado en la base: {FECHA_CORTE.date()}")
RES["fecha_corte"] = str(FECHA_CORTE.date())

# Hallazgo 3: 'motivo' y 'estado' son jerárquicos (EXTRAVIADO-* ⊂ EXTRAVÍO)
mot_extr = raw["motivo_desaparicion"].where(raw["motivo_desaparicion"].str.startswith("EXTRAV"), "otros motivos")
mostrar(pd.crosstab(mot_extr, raw["estado_desaparecido"]))

# ======================================================================
# ## 2. Preprocesamiento
# ======================================================================
df = raw.copy()
sin_fecha = (df["situacion_actual"] == "ENCONTRADO") & df["f_loc"].isna()
print("Registros 'ENCONTRADO' sin fecha de localización (excluidos):", int(sin_fecha.sum()))
df = df[~sin_fecha].copy()

# Evento = cierre del caso (localizado con vida o fallecido). Censura = sigue desaparecido al corte.
df["evento"] = (df["situacion_actual"] != "DESAPARECIDO").astype(int)
df["t"] = np.where(df["evento"] == 1, (df["f_loc"] - df["fecha_desaparicion"]).dt.days, (FECHA_CORTE - df["fecha_desaparicion"]).dt.days)
assert (df["t"] >= 0).all()
df["causa"] = np.select([df["situacion_actual"]=="ENCONTRADO", df["situacion_actual"]=="FALLECIDO"], [1, 2], 0)
df["anio"]  = df["fecha_desaparicion"].dt.year

num = lambda s: pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")
df["lat"], df["lon"] = num(df["latitud_desaparicion"]), num(df["longitud_desaparicion"])
en_rango = df["lat"].between(-5.1, 1.8) & df["lon"].between(-92.5, -75.0)
print("Coordenadas nulas:", int(df[["lat","lon"]].isna().any(axis=1).sum()), "| fuera de rango:", int((~en_rango).sum()))
df = df[en_rango].copy()
dup = df.groupby(["lat","lon"]).size()
print(f"Coordenadas únicas: {len(dup):,} de {len(df):,} casos; {dup[dup>=5].sum()/len(df)*100:.1f}% de los casos comparte coordenada con ≥4 más.")

costa = {"GUAYAS","MANABÍ","LOS RÍOS","EL ORO","ESMERALDAS","SANTA ELENA","STO DGO DE LOS TSÁCHILAS"}
amaz  = {"MORONA SANTIAGO","NAPO","ORELLANA","PASTAZA","SUCUMBÍOS","ZAMORA CHINCHIPE"}
df["region"] = np.where(df["provincia"].isin(costa), "COSTA", np.where(df["provincia"].isin(amaz), "AMAZONIA", np.where(df["provincia"]=="GALÁPAGOS", "INSULAR", "SIERRA")))
df["edad_g"] = df["rango_edad"].where(df["rango_edad"] != "SIN_DATO")
print(f"\nBase analítica: {len(df):,} casos | eventos: {int(df.evento.sum()):,} | censurados: {int((df.evento==0).sum()):,}")
RES["n_analitico"] = int(len(df)); RES["n_censurados"] = int((df.evento==0).sum())

# ======================================================================
# ## 3. Modelo estocástico de supervivencia (Kaplan-Meier)
# ### 3.1 Efecto de la corrección de la censura
# ======================================================================
def km_resumen(t, e, etiqueta):
    k = KaplanMeierFitter().fit(t, e)
    lo, hi = median_survival_times(k.confidence_interval_).values[0]
    return k, dict(modelo=etiqueta, n=len(t), eventos=int(e.sum()), censurados=int((e==0).sum()), mediana=k.median_survival_time_,
                   mediana_IC95=f"[{lo:.0f}; {hi:.0f}]", **{f"S({d})": round(float(k.predict(d)), 4) for d in (7, 30, 365)})

o = df.copy()
o["ev_o"] = (o["situacion_actual"]=="ENCONTRADO").astype(int)
o["t_o"]  = np.where(o["ev_o"]==1, (o["f_loc"]-o["fecha_desaparicion"]).dt.days, (FECHA_CORTE_ORIG-o["fecha_desaparicion"]).dt.days)
o = o[o["t_o"]>=0]
k_orig, r1 = km_resumen(o["t_o"], o["ev_o"], "Alternativa (fallecidos censurados; corte 31-dic-2025)")
k_corr, r2 = km_resumen(df["t"], df["evento"], "Principal (evento = cierre; corte 12-ago-2026)")
tab = pd.DataFrame([r1, r2]); mostrar(tab); tab.to_csv("tablas/km_original_vs_corregida.csv", index=False)
print("Fallecidos tratados como 'siguen desaparecidos' en la definición alternativa:", int((o.situacion_actual=='FALLECIDO').sum()),
      "| tiempo medio imputado:", round(o.loc[o.situacion_actual=='FALLECIDO','t_o'].mean()), "días")
RES["km"] = dict(original=r1, corregida=r2)

fig, ax = plt.subplots(figsize=(9, 5))
k_corr.plot_survival_function(ax=ax, ci_show=True, color="#1f77b4", label="Evento = cierre")
k_orig.plot_survival_function(ax=ax, ci_show=False, color="#d62728", ls="--", label="Fallecidos censurados")
ax.axhline(.5, color="gray", ls=":"); ax.set_xlim(0, 365)
ax.set(title="Probabilidad de que el caso siga abierto, S(t)", xlabel="Días desde la desaparición", ylabel="S(t)")
ax.legend(); fig.tight_layout(); fig.savefig("figuras/fig_km_general.png"); plt.show()

# ======================================================================
# ### 3.2 Riesgos competitivos: localizado con vida vs. localizado fallecido
# ======================================================================
cif = {}
for c, nm in [(1, "Localizado con vida"), (2, "Localizado fallecido")]:
    aj = AalenJohansenFitter(calculate_variance=False, seed=SEMILLA).fit(df["t"], df["causa"], event_of_interest=c)
    cif[nm] = aj.cumulative_density_.iloc[:, 0]
fig, ax = plt.subplots(figsize=(9, 5))
for nm, s in cif.items(): ax.step(s.index, s.values, where="post", label=nm)
ax.set(xlim=(0, 365), title="Incidencia acumulada por tipo de cierre (Aalen-Johansen)", xlabel="Días", ylabel="Probabilidad acumulada")
ax.legend(); fig.tight_layout(); fig.savefig("figuras/fig_cif.png"); plt.show()
at = lambda s, d: float(s.loc[:d].iloc[-1])
RES["cif"] = {nm: {f"{d}d": round(at(s, d), 4) for d in (7, 30, 365)} for nm, s in cif.items()}
mostrar(pd.DataFrame(RES["cif"]).T)

# ======================================================================
# ### 3.3 Sensibilidad al origen del tiempo (desaparición vs. denuncia)
# ======================================================================
s = df[df["fecha_denuncia"] >= df["fecha_desaparicion"]].copy()
s["t_d"] = np.where(s["evento"]==1, (s["f_loc"]-s["fecha_denuncia"]).dt.days, (FECHA_CORTE-s["fecha_denuncia"]).dt.days)
s = s[s["t_d"]>=0]
k_den, r3 = km_resumen(s["t_d"], s["evento"], "Origen = fecha de denuncia")
mostrar(pd.DataFrame([r2, r3])[["modelo","n","mediana","S(7)","S(30)","S(365)"]])
print("Casos con denuncia anterior a la desaparición (excluidos en esta sensibilidad):", int((df.fecha_denuncia < df.fecha_desaparicion).sum()))
RES["km_origen_denuncia"] = r3

# ======================================================================
# ### 3.4 La resolución se ha vuelto más lenta: tendencia 2017-2025
# Antes de comparar grupos hay que comprobar que la población es estable en el tiempo. No lo es.
# ======================================================================
filas = []
for a in range(2017, 2026):
    s_ = df[df.anio == a]; k = KaplanMeierFitter().fit(s_.t, s_.evento)
    filas.append(dict(anio=a, n=len(s_), mediana=k.median_survival_time_, **{"S(30)": round(float(k.predict(30)),3)},
                      pct_fallecido=round((s_.situacion_actual=="FALLECIDO").mean()*100,1), pct_sigue_desaparecido=round((s_.situacion_actual=="DESAPARECIDO").mean()*100,1)))
tend = pd.DataFrame(filas); mostrar(tend); tend.to_csv("tablas/tendencia_anual.csv", index=False)
RES["tendencia_anual"] = tend.to_dict("records")

fig, ax = plt.subplots(figsize=(9, 5))
cmap = plt.cm.viridis(np.linspace(0, .9, 9))
for a, c in zip(range(2017, 2026), cmap):
    s_ = df[df.anio == a]; KaplanMeierFitter().fit(s_.t, s_.evento, label=str(a)).plot_survival_function(ax=ax, ci_show=False, color=c)
ax.set(title="S(t) por año de desaparición", xlim=(0, 120), xlabel="Días", ylabel="S(t)"); ax.legend(ncol=3, fontsize=8)
fig.tight_layout(); fig.savefig("figuras/fig_km_por_anio.png"); plt.show()

# ======================================================================
# ### 3.5 Comparación por sexo y edad: significancia **y** tamaño de efecto
# Con n ≈ 75 000 el Log-Rank detecta diferencias mínimas. Se reporta además el *hazard ratio* (Cox) crudo, ajustado por edad y **estratificado por año**, que es la comparación comparable en el tiempo.
# ======================================================================
d2 = df[df["edad_g"].notna()].copy()
def km_grupo(col, valores):
    out = []
    for v in valores:
        m = d2[col] == v; k = KaplanMeierFitter().fit(d2.loc[m,"t"], d2.loc[m,"evento"])
        out.append(dict(grupo=v, n=int(m.sum()), mediana=k.median_survival_time_, **{"S(30)": round(float(k.predict(30)),3)}, RMST_30d=round(restricted_mean_survival_time(k, t=30), 2)))
    return pd.DataFrame(out)
tab_sexo = km_grupo("sexo", ["HOMBRE", "MUJER"]); tab_edad = km_grupo("edad_g", ["NIÑOS(AS)", "ADOLESCENTES", "ADULTO", "ADULTO MAYOR"])
mostrar(tab_sexo); mostrar(tab_edad); tab_sexo.to_csv("tablas/km_por_sexo.csv", index=False); tab_edad.to_csv("tablas/km_por_edad.csv", index=False)

h, m_ = d2[d2.sexo=="HOMBRE"], d2[d2.sexo=="MUJER"]
lr = logrank_test(h.t, m_.t, h.evento, m_.evento); lre = multivariate_logrank_test(d2.t, d2.edad_g, d2.evento)
print(f"Log-Rank sexo: χ²={lr.test_statistic:.1f}, p={lr.p_value:.2e} | Log-Rank edad: χ²={lre.test_statistic:.1f}, p<0,001")

X = d2[["t","evento"]].copy(); X["mujer"] = (d2.sexo=="MUJER").astype(int)
for g_ in ["NIÑOS(AS)","ADULTO","ADULTO MAYOR"]: X[g_] = (d2.edad_g==g_).astype(int)     # ref.: ADOLESCENTES
cox_crudo = CoxPHFitter().fit(X[["t","evento","mujer"]], "t", "evento")
cox_aj    = CoxPHFitter().fit(X, "t", "evento")
Xa = X.copy(); Xa["anio"] = d2["anio"].values
cox_anio  = CoxPHFitter().fit(Xa, "t", "evento", strata=["anio"])
f = lambda c: c.summary.loc["mujer", ["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%"]].round(3).tolist()
hr = pd.DataFrame({"modelo": ["Mujer vs. hombre (crudo)", "Mujer vs. hombre (ajustado por edad)", "Mujer vs. hombre (ajustado por edad, estratificado por año)"],
                   "HR (IC95%)": [f"{a[0]} [{a[1]}; {a[2]}]" for a in (f(cox_crudo), f(cox_aj), f(cox_anio))]})
mostrar(hr); hr.to_csv("tablas/cox_sexo_edad.csv", index=False)
mostrar(cox_anio.summary[["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","p"]].round(3))
RES["cox_sexo"] = dict(crudo=f(cox_crudo), ajustado_edad=f(cox_aj), ajustado_edad_estrat_anio=f(cox_anio))
RES["cox_edad_estrat_anio"] = cox_anio.summary[["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","p"]].round(3).reset_index().to_dict("records")

filas = []
for g_ in ["NIÑOS(AS)","ADOLESCENTES","ADULTO","ADULTO MAYOR"]:
    sub = d2[d2.edad_g==g_]; a, b = sub[sub.sexo=="HOMBRE"], sub[sub.sexo=="MUJER"]
    filas.append(dict(grupo=g_, p_logrank=logrank_test(a.t,b.t,a.evento,b.evento).p_value))
mostrar(pd.DataFrame(filas).round(4))
ph = proportional_hazard_test(cox_aj, X, time_transform="rank"); print("\nPrueba de riesgos proporcionales (Schoenfeld), modelo ajustado por edad:"); mostrar(ph.summary.round(4))
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for v, c in [("HOMBRE","#1f77b4"), ("MUJER","#e377c2")]:
    m = d2.sexo == v; KaplanMeierFitter().fit(d2.t[m], d2.evento[m], label=v).plot_survival_function(ax=axs[0], color=c)
axs[0].set(title="Por sexo", xlim=(0, 180), xlabel="Días", ylabel="S(t)")
for v, c in [("NIÑOS(AS)","#2ca02c"),("ADOLESCENTES","#ff7f0e"),("ADULTO","#1f77b4"),("ADULTO MAYOR","#9467bd")]:
    m = d2.edad_g == v; KaplanMeierFitter().fit(d2.t[m], d2.evento[m], label=v).plot_survival_function(ax=axs[1], color=c, ci_show=False)
axs[1].set(title="Por grupo de edad", xlim=(0, 180), xlabel="Días")
fig.suptitle("Probabilidad de que el caso siga abierto", y=1.02); fig.tight_layout(); fig.savefig("figuras/fig_km_sexo_edad.png", bbox_inches="tight"); plt.show()

# ======================================================================
# ## 4. Agrupamiento espacial basado en densidad (DBSCAN) — provincia piloto: Guayas
# ### 4.1 Selección de parámetros
# ======================================================================
g = df[df.provincia == "GUAYAS"].copy().reset_index(drop=True)
Xg = np.radians(g[["lat","lon"]].to_numpy())
print(f"Guayas: {len(g):,} casos | coordenadas únicas: {len(g[['lat','lon']].drop_duplicates()):,}")
fig, ax = plt.subplots(figsize=(8, 4.5))
for k in (15, 30, 50):
    d, _ = NearestNeighbors(n_neighbors=k+1, metric="haversine", algorithm="ball_tree").fit(Xg).kneighbors(Xg)
    ax.plot(np.sort(d[:, -1]) * KM_TIERRA, label=f"k = {k}")
ax.set(ylim=(0, 3), xlabel="Casos ordenados", ylabel="Distancia al k-ésimo vecino (km)", title="Gráfico k-distance (Guayas)")
ax.axhline(.35, color="k", ls=":"); ax.legend(); fig.tight_layout(); fig.savefig("figuras/fig_kdistance.png"); plt.show()
filas = []
for mp in (15, 30, 50):
    for eps in (0.25, 0.35, 0.5, 1.0, 1.5, 2.0):
        lab = DBSCAN(eps=eps/KM_TIERRA, min_samples=mp, metric="haversine", algorithm="ball_tree").fit(Xg).labels_
        m = lab != -1; s = pd.Series(lab[m]).value_counts()
        sil = silhouette_score(Xg[m], lab[m], metric="haversine", sample_size=min(5000, m.sum()), random_state=SEMILLA) if len(s) > 1 else np.nan
        filas.append(dict(MinPts=mp, eps_km=eps, clusters=len(s), pct_en_cluster=round(m.mean()*100,1), pct_cluster_mayor=round(s.iloc[0]/len(g)*100,1), silueta=round(sil,3)))
grid = pd.DataFrame(filas); mostrar(grid); grid.to_csv("tablas/dbscan_sensibilidad.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for mp, gg in grid.groupby("MinPts"):
    axs[0].plot(gg.eps_km, gg.clusters, marker="o", label=f"MinPts={mp}"); axs[1].plot(gg.eps_km, gg["pct_cluster_mayor"], marker="o", label=f"MinPts={mp}")
axs[0].set(title="Número de clústeres", xlabel="ε (km)"); axs[1].set(title="% de casos en el clúster mayor", xlabel="ε (km)")
axs[0].legend(); fig.tight_layout(); fig.savefig("figuras/fig_dbscan_sensibilidad.png"); plt.show()
RES["dbscan_original_1.5_15"] = grid[(grid.eps_km==1.5)&(grid.MinPts==15)].iloc[0].to_dict()

# ======================================================================
# **Decisión de parámetros.** Con ε = 1,5 km el clúster mayor concentra casi 7 de cada 10 casos: es toda el área metropolitana, no una geocerca. Se adopta **ε = 0,35 km** (mediana de la distancia al 30.º vecino, rodilla del k-distance) y **MinPts = 30**: zonas de escala de barrio, ninguna dominante.
# ======================================================================
EPS_KM, MINPTS = 0.35, 30
g["cl"] = DBSCAN(eps=EPS_KM/KM_TIERRA, min_samples=MINPTS, metric="haversine", algorithm="ball_tree").fit(Xg).labels_
m = g.cl != -1; tam = g.loc[m, "cl"].value_counts()
print(f"ε={EPS_KM} km, MinPts={MINPTS}: {len(tam)} zonas | {m.mean()*100:.1f}% de casos en zonas | ruido {100-m.mean()*100:.1f}% | zona mayor {tam.iloc[0]/len(g)*100:.1f}%")
RES["dbscan_final"] = dict(eps_km=EPS_KM, minpts=MINPTS, zonas=int(len(tam)), pct_en_zonas=round(m.mean()*100,1), pct_ruido=round(100-m.mean()*100,1), pct_zona_mayor=round(tam.iloc[0]/len(g)*100,1))

# ======================================================================
# ### 4.2 Densidad no es riesgo: severidad **estandarizada por año**
# La proporción de desenlace adverso (fallecido o aún desaparecido) en Guayas pasa de ≈ 2 % (2017) a ≈ 26 % (2025). Comparar una zona con la tasa global la penaliza o favorece según los años en que tenga casos. Se usa **estandarización indirecta**: la tasa esperada de cada zona es el promedio de las tasas anuales de la provincia ponderado por los casos de la zona en cada año. Se contrasta con una prueba binomial unilateral y corrección de Benjamini-Hochberg (FDR).
# ======================================================================
g["adverso"] = g.situacion_actual.isin(["FALLECIDO","DESAPARECIDO"]).astype(int)
tasa_anual = g.groupby("anio").adverso.mean(); mostrar(tasa_anual.round(3).to_frame("tasa adversa Guayas"))

def severidad_zonas(d, col_cl, tasa_ref):
    filas = []
    for c, s in d[d[col_cl] != -1].groupby(col_cl):
        n, k = len(s), int(s.adverso.sum()); esp = float(s.anio.map(tasa_ref).mean())
        ya = s.groupby("anio").size().reindex(range(2017, 2026), fill_value=0)
        filas.append(dict(zona=c, casos=n, adversos=k, tasa=k/n, esperada=esp, razon_vs_esperada=(k/n)/esp, p=binomtest(k, n, esp, alternative="greater").pvalue,
                          anios_activos=int((ya >= 3).sum()), lat=s.lat.mean(), lon=s.lon.mean()))
    z = pd.DataFrame(filas); z["p_BH"] = multipletests(z.p, method="fdr_bh")[1] if len(z) else []
    z["severa"] = z.p_BH < 0.05; z["cronica"] = z.anios_activos >= 8
    return z.sort_values("p")

zonas = severidad_zonas(g, "cl", tasa_anual); mostrar(zonas.head(8).round(4)); zonas.to_csv("tablas/dbscan_zonas.csv", index=False)
print(f"Zonas: {len(zonas)} | severidad elevada (FDR<0,05, estandarizada por año): {int(zonas.severa.sum())} | crónicas (≥ 3 casos en ≥ 8 de 9 años): {int(zonas.cronica.sum())} ({zonas[zonas.cronica].casos.sum():,} casos)")
RES["zonas"] = dict(total=int(len(zonas)), severas=int(zonas.severa.sum()), cronicas=int(zonas.cronica.sum()),
                    severas_detalle=zonas[zonas.severa][["zona","casos","adversos","tasa","esperada","razon_vs_esperada","p_BH","anios_activos"]].round(4).to_dict("records"))
g = g.merge(zonas[["zona","severa"]].rename(columns={"zona":"cl"}), on="cl", how="left"); g["severa"] = g.severa.fillna(False).astype(bool)
fig, axs = plt.subplots(1, 2, figsize=(14, 6.2))
ruido = g[g.cl == -1]; cz = g[g.cl != -1]
for ax, titulo in zip(axs, ["Zonas detectadas (ε=0,35 km; MinPts=30)", "Zonas con severidad elevada (FDR<0,05, estand. por año)"]):
    ax.scatter(ruido.lon, ruido.lat, s=3, c="lightgrey", label="Ruido"); ax.set(title=titulo, xlabel="Longitud", ylabel="Latitud")
axs[0].scatter(cz.lon, cz.lat, s=5, c=cz.cl, cmap="tab20")
axs[1].scatter(cz.lon, cz.lat, s=3, c="#c7d3e0", label="Otras zonas")
sv = g[g.severa]; axs[1].scatter(sv.lon, sv.lat, s=10, c=sv.cl, cmap="Set1", label="Severidad elevada")
for z in zonas[zonas.severa].itertuples(): axs[1].annotate(f"Z{z.zona}", (z.lon, z.lat), fontsize=9, weight="bold")
axs[1].legend(loc="lower left"); fig.tight_layout(); fig.savefig("figuras/fig_dbscan_mapa.png"); plt.show()

# ======================================================================
# ### 4.3 Validación temporal: ¿las zonas de 2017-2022 predicen 2023-2025?
# ======================================================================
ent, sal = g[g.anio <= 2022].copy().reset_index(drop=True), g[g.anio >= 2023].copy().reset_index(drop=True)
ent["cl"] = DBSCAN(eps=EPS_KM/KM_TIERRA, min_samples=MINPTS, metric="haversine", algorithm="ball_tree").fit(np.radians(ent[["lat","lon"]].to_numpy())).labels_
z_ent = severidad_zonas(ent, "cl", ent.groupby("anio").adverso.mean())
sev_ent = set(z_ent[z_ent.severa].zona); nom_ent = set(z_ent[z_ent.p < 0.05].zona)

core = ent[ent.cl != -1]; arbol = BallTree(np.radians(core[["lat","lon"]].to_numpy()), metric="haversine")
dist, idx = arbol.query(np.radians(sal[["lat","lon"]].to_numpy()), k=1)
dentro = dist[:, 0] * KM_TIERRA <= EPS_KM
sal["cl_ent"] = np.where(dentro, core.cl.to_numpy()[idx[:, 0]], -1)
tasa_sal = sal.groupby("anio").adverso.mean()
def evaluar(S, nombre):
    m = sal.cl_ent.isin(list(S)); n = int(m.sum())
    if n < 20: return dict(criterio=nombre, zonas=len(S), n_prueba=n, observados=np.nan, esperados=np.nan, razon=np.nan, p=np.nan)
    k = int(sal.loc[m, "adverso"].sum()); esp = float(sal.loc[m, "anio"].map(tasa_sal).mean())
    return dict(criterio=nombre, zonas=len(S), n_prueba=n, observados=k, esperados=round(esp*n,1), razon=round(k/(esp*n),2), p=binomtest(k, n, esp, alternative="greater").pvalue)
val_z = pd.DataFrame([evaluar(sev_ent, "Severas en 2017-2022 (FDR<0,05)"), evaluar(nom_ent, "Nominal p<0,05 en 2017-2022 (exploratorio)")])
print(f"Zonas de entrenamiento: {len(z_ent)} | cobertura: {dentro.mean()*100:.1f}% de los casos 2023-2025 caen en zonas detectadas con datos 2017-2022")
mostrar(val_z); val_z.to_csv("tablas/dbscan_validacion_temporal.csv", index=False)
RES["validacion_temporal_zonas"] = dict(zonas_entrenamiento=int(len(z_ent)), cobertura_pct=round(dentro.mean()*100,1), resultados=val_z.replace({np.nan: None}).to_dict("records"))

# ======================================================================
# ## 5. Minería de reglas de asociación (Apriori)
# ### 5.1 Jerarquía entre las variables `motivo` y `estado`
# `EXTRAVIADO - DISCAPACIDAD…` y `EXTRAVIADO - AUSENCIA TEMPORAL` (variable `motivo`) son *exactamente* los casos `EXTRAVÍO` (variable `estado`): 4 216 + 2 104 = 6 320. Una regla entre ambos tiene confianza 1,0 por construcción y su Lift (≈ 12) mide la taxonomía, no un patrón de riesgo. Además `FALLECIDO`/`NO_APLICA` en `motivo` y `EN INVESTIGACIÓN` en `estado` ya codifican el desenlace.
# 
# ### 5.2 Diseño de las transacciones
# * **Antecedentes** (perfil): sexo, grupo de edad, etnia, nacionalidad, región y motivo (sin categorías que codifican el desenlace; `EXTRAVIADO-*` fusionadas).
# * **Consecuentes** (desenlace): `FALLECIDO`, `SIGUE_DESAPARECIDO`, `RESOLUCION>30d`.
# * Soporte mínimo 0,1 %; se conservan reglas con ≥ 50 casos, **límite inferior del IC95 % de la confianza** que implique Lift > 1,5, y sin redundancia respecto de reglas más generales.
# ======================================================================
def construir_transacciones(d):
    mot = d.motivo_desaparicion.where(~d.motivo_desaparicion.str.startswith("EXTRAVIADO"), "EXTRAVIADO")
    mot = mot.where(~mot.isin(["FALLECIDO","NO_APLICA","CERRADO POR FISCALÍA / DELITO REFORMULADO","DISCRIMINACIÓN"]), np.nan)
    et  = d.etnia.where(d.etnia.isin(["MESTIZO/A","INDIGENA","AFRO","MONTUBIO/A"]), "OTRA_ETNIA").where(d.etnia != "SIN_DATO", np.nan)
    nac = pd.Series(np.where(d.nacionalidad=="ECUADOR","ECUATORIANA","EXTRANJERA"), index=d.index).where(d.nacionalidad != "SIN_DATO", np.nan)
    perfil = pd.DataFrame({"sexo": d.sexo, "edad": d.edad_g, "etnia": et, "nac": nac, "region": d.region, "motivo": mot})
    items = pd.get_dummies(perfil, prefix_sep="=").astype(bool)
    sal_ = pd.DataFrame({"SALE=FALLECIDO": d.situacion_actual=="FALLECIDO", "SALE=SIGUE_DESAPARECIDO": d.situacion_actual=="DESAPARECIDO", "SALE=RESOLUCION>30d": d.t > 30})
    return pd.concat([items, sal_], axis=1), list(sal_.columns)

def minar_reglas(T, cols_salida, min_support=0.001, n_min=50, lift_min=1.5):
    N = len(T); fi = apriori(T, min_support=min_support, max_len=4, use_colnames=True)
    R = association_rules(fi, metric="lift", min_threshold=lift_min)
    R = R[R.apply(lambda r: len(r.consequents)==1 and list(r.consequents)[0] in cols_salida and not (set(r.antecedents) & set(cols_salida)), axis=1)].copy()
    R["n_regla"] = (R.support*N).round().astype(int); R["n_ant"] = (R["antecedent support"]*N).round().astype(int)
    R["conf_lo"], R["conf_hi"] = proportion_confint(R.n_regla, R.n_ant, method="wilson")
    R["ant"] = R.antecedents.apply(lambda s: " & ".join(sorted(s))); R["con"] = R.consequents.apply(lambda s: list(s)[0]); R["len_ant"] = R.antecedents.apply(len)
    R = R[(R.n_regla >= n_min) & (R.conf_lo / R["consequent support"] > lift_min)]
    keep = []
    for _, r in R.iterrows():
        gen = R[(R.con==r.con) & (R.len_ant < r.len_ant) & R.antecedents.apply(lambda s: s < r.antecedents)]
        keep.append(not (gen.lift >= 0.9*r.lift).any())
    return R[keep].sort_values(["con","lift"], ascending=[True, False]), fi

T, cols_salida = construir_transacciones(df)
print(f"Transacciones: {len(T):,} | ítems: {T.shape[1]}"); mostrar(T[cols_salida].mean().round(4).to_frame("prevalencia"))
reglas, fi = minar_reglas(T, cols_salida)
print(f"Conjuntos frecuentes: {len(fi):,} | reglas finales: {len(reglas)}")
cols = ["ant","con","n_ant","n_regla","confidence","conf_lo","conf_hi","lift"]
for c in cols_salida:
    print(f"\n### {c}  (prevalencia {T[c].mean()*100:.1f}%)"); mostrar(reglas[reglas.con==c][cols].head(8).round(3))
reglas[["ant","con","n_ant","n_regla","support","confidence","conf_lo","conf_hi","lift"]].round(4).to_csv("tablas/reglas_apriori.csv", index=False)
RES["apriori"] = dict(n_reglas=int(len(reglas)), top=reglas.groupby("con").head(3)[["ant","con","n_ant","confidence","lift"]].round(3).to_dict("records"))

# ======================================================================
# ### 5.3 Validación temporal de las reglas (minadas en 2017-2022, evaluadas en 2023-2025)
# ======================================================================
dtr, dte = df[df.anio <= 2022], df[df.anio >= 2023]
Ttr, _ = construir_transacciones(dtr); Tte, _ = construir_transacciones(dte)
Tte = Tte.reindex(columns=Ttr.columns, fill_value=False)
r_tr, _ = minar_reglas(Ttr, cols_salida)
filas = []
for r in r_tr.itertuples():
    ant = list(r.antecedents); m = Tte[ant].all(axis=1); n = int(m.sum())
    conf = Tte.loc[m, r.con].mean() if n else np.nan; base = Tte[r.con].mean()
    filas.append(dict(regla=f"{r.ant} → {r.con}", lift_entrenamiento=round(r.lift,2), n_prueba=n, confianza_prueba=round(conf,3), lift_prueba=round(conf/base,2) if n else np.nan))
val = pd.DataFrame(filas)
if len(val):
    mostrar(val.head(12))
    persist = (val.lift_prueba > 1.5).mean()
    print(f"Reglas minadas en 2017-2022: {len(val)} | con Lift>1,5 en 2023-2025: {int((val.lift_prueba>1.5).sum())} ({persist*100:.0f}%) | Lift>1: {(val.lift_prueba>1).mean()*100:.0f}%")
    val.to_csv("tablas/reglas_validacion_temporal.csv", index=False)
    RES["apriori_validacion"] = dict(n=int(len(val)), pct_lift_gt_1_5=round(persist*100,1), pct_lift_gt_1=round((val.lift_prueba>1).mean()*100,1))
    por_con = val.assign(con=val.regla.str.split(" → ").str[1]).groupby("con").lift_prueba.agg(["count", "median", lambda s: (s>1.5).mean()]).round(2)
    por_con.columns = ["reglas", "lift_mediano_prueba", "prop_lift>1,5"]; mostrar(por_con)
    RES["apriori_validacion_por_consecuente"] = por_con.reset_index().to_dict("records")
else:
    print("Ninguna regla supera los filtros en el periodo de entrenamiento.")

# ======================================================================
# ## 6. Integración de los tres motores (Cox fuera de muestra, estratificado por año)
# Para Guayas 2023-2025 se estima el efecto sobre la **velocidad de cierre** de (i) estar en una zona de densidad detectada con datos 2017-2022 (DBSCAN), (ii) tener un perfil de riesgo (etnia o nacionalidad, Lift ≥ 2) señalado por las reglas minadas en 2017-2022 (Apriori), y (iii) sexo y edad. HR < 1 = cierre más lento. Zonas y perfiles se definen con datos anteriores para evitar circularidad, y el modelo se **estratifica por año** para no confundir con la tendencia temporal.
# ======================================================================
# Perfil de riesgo = reglas minadas en 2017-2022 cuyo antecedente incluye etnia o nacionalidad (sexo y edad entran como covariables
# propias del Cox; una unión de perfiles demográficos amplios marcaría a ~79 % de los casos y se solaparía con ellas) y Lift de entrenamiento >= 2
cand = r_tr[r_tr.con.isin(["SALE=SIGUE_DESAPARECIDO","SALE=RESOLUCION>30d"])]
sel  = cand[cand.antecedents.apply(lambda s: any(i.startswith(("etnia=","nac=")) for i in s)) & (cand.lift >= 2)]
ants = [set(a) for a in sel.antecedents]
perfiles = sorted({" & ".join(sorted(a)) for a in ants}); print("Perfiles de riesgo (entrenamiento):", perfiles)
Tg, _ = construir_transacciones(df); Tg = Tg.reindex(columns=Ttr.columns, fill_value=False)
perfil = np.zeros(len(df), dtype=bool)
for a in ants: perfil |= Tg[list(a)].all(axis=1).to_numpy()
df["perfil_riesgo"] = perfil.astype(int)

gt = df.loc[(df.provincia=="GUAYAS") & (df.anio >= 2023) & df.edad_g.notna()].copy()
dist, idx = arbol.query(np.radians(gt[["lat","lon"]].to_numpy()), k=1)
gt["en_zona_densa"] = (dist[:,0]*KM_TIERRA <= EPS_KM).astype(int)
gt["en_zona_severa"] = (np.isin(core.cl.to_numpy()[idx[:,0]], list(sev_ent)) & (dist[:,0]*KM_TIERRA <= EPS_KM)).astype(int)
gt["mujer"] = (gt.sexo=="MUJER").astype(int)
for g_ in ["NIÑOS(AS)","ADULTO","ADULTO MAYOR"]: gt[g_] = (gt.edad_g==g_).astype(int)
print(f"Prevalencia del perfil de riesgo en Guayas 2023-2025: {gt.perfil_riesgo.mean()*100:.1f}%")
print(f"Casos Guayas 2023-2025: {len(gt):,} | en zona densa: {int(gt.en_zona_densa.sum())} | en zona severa: {int(gt.en_zona_severa.sum())} | con perfil de riesgo: {int(gt.perfil_riesgo.sum())}")

covs = ["en_zona_densa","perfil_riesgo","mujer","NIÑOS(AS)","ADULTO","ADULTO MAYOR"]
if gt.en_zona_severa.sum() >= 20: covs.insert(1, "en_zona_severa")
mod = CoxPHFitter().fit(gt[["t","evento","anio"] + covs], "t", "evento", strata=["anio"])
tab_cox = mod.summary[["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","p"]].round(3); mostrar(tab_cox); tab_cox.to_csv("tablas/cox_integrado.csv")
RES["cox_integrado"] = tab_cox.reset_index().to_dict("records")
RES["perfiles_riesgo"] = perfiles

# ======================================================================
# ## 7. Cifras clave para la memoria
# ======================================================================
def limpiar(o):
    if isinstance(o, dict):  return {str(k): limpiar(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)): return [limpiar(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return o
json.dump(limpiar(RES), open("tablas/resultados_clave.json", "w"), ensure_ascii=False, indent=2)
print("Resultados guardados en tablas/resultados_clave.json")
