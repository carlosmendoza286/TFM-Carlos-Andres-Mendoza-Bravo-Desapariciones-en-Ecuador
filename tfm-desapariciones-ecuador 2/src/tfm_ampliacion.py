# -*- coding: utf-8 -*-
"""Análisis complementarios del TFM (Capítulo 4): descripción de los casos, supervivencia por región,
Cox multivariable, incidencia de desenlace fatal por grupo, motivos registrados, distancia de localización,
caracterización de zonas, extensión de DBSCAN a Pichincha y sensibilidad de Apriori.
Ejecutar después de tfm_analisis.py (usa tablas/dbscan_zonas.csv).
Uso: TFM_XLSX=ruta/al/archivo.xlsx TFM_HEADLESS=1 python tfm_ampliacion.py
"""
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
#mostrar(aud)
print("\nsituacion_actual:\n", raw["situacion_actual"].value_counts())
# Hallazgo 1: los 'FALLECIDO' tienen fecha de localización (se hallaron) -> son un EVENTO, no censura
raw["f_loc"] = pd.to_datetime(raw["fecha_localizacion"].replace({"NO_APLICA": np.nan, "SIN_DATO": np.nan}), errors="coerce", format="mixed")
#mostrar(pd.crosstab(raw["situacion_actual"], raw["f_loc"].notna(), rownames=["situación"], colnames=["tiene fecha de localización"]))

# Hallazgo 2: corte real de la base
FECHA_CORTE_ORIG = raw["fecha_desaparicion"].max()
FECHA_CORTE      = max(raw["f_loc"].max(), raw["fecha_denuncia"].max(), raw["fecha_desaparicion"].max())
print(f"\nÚltima fecha de desaparición: {FECHA_CORTE_ORIG.date()}  |  corte observado en la base: {FECHA_CORTE.date()}")
RES["fecha_corte"] = str(FECHA_CORTE.date())

# Hallazgo 3: 'motivo' y 'estado' son jerárquicos (EXTRAVIADO-* ⊂ EXTRAVÍO)
mot_extr = raw["motivo_desaparicion"].where(raw["motivo_desaparicion"].str.startswith("EXTRAV"), "otros motivos")
#mostrar(pd.crosstab(mot_extr, raw["estado_desaparecido"]))

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


# ===== Funciones de Apriori (idénticas a tfm_analisis.py) =====
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



# ===== A-E. Descriptivo, región, Cox multivariable, distancia, zonas =====
# -*- coding: utf-8 -*-
# Análisis complementarios del TFM: descriptivo, supervivencia por región,
# Cox multivariable, distancia de localización y caracterización de zonas.
# Se ejecuta después de la preparación de datos de tfm_analisis.py (prep.py).
from lifelines.statistics import multivariate_logrank_test
os.makedirs("figuras", exist_ok=True); os.makedirs("tablas", exist_ok=True)
AMP = {}
df["adverso"] = (df["situacion_actual"] != "ENCONTRADO").astype(int)
df["fallecido"] = (df["situacion_actual"] == "FALLECIDO").astype(int)
df["sigue"] = (df["situacion_actual"] == "DESAPARECIDO").astype(int)
df["etnia_g"] = df.etnia.where(df.etnia.isin(["MESTIZO/A","INDIGENA","AFRO","MONTUBIO/A"]), "OTRA").where(df.etnia != "SIN_DATO", "SIN_DATO")
df["nac_g"] = np.where(df.nacionalidad == "ECUADOR", "ECUATORIANA", np.where(df.nacionalidad == "SIN_DATO", "SIN_DATO", "EXTRANJERA"))

# ---------- A. Perfil descriptivo ----------
filas = []
for var, nom in [("sexo","Sexo"),("edad_g","Grupo de edad"),("etnia_g","Etnia"),("nac_g","Nacionalidad"),("region","Región")]:
    for cat, sub in df.groupby(var, dropna=True):
        filas.append(dict(variable=nom, categoria=cat, casos=len(sub), pct=round(len(sub)/len(df)*100,1),
                          pct_fallecido=round(sub.fallecido.mean()*100,1), pct_sigue=round(sub.sigue.mean()*100,1)))
perfil = pd.DataFrame(filas); perfil.to_csv("tablas/descriptivo_perfil.csv", index=False); print(perfil.to_string())

# provincias
prov = []
for p_, sub in df.groupby("provincia"):
    k = KaplanMeierFitter().fit(sub.t, sub.evento)
    prov.append(dict(provincia=p_, casos=len(sub), pct=round(len(sub)/len(df)*100,1), mediana=k.median_survival_time_,
                     S30=round(float(k.predict(30)),3), pct_adverso=round(sub.adverso.mean()*100,1)))
prov = pd.DataFrame(prov).sort_values("casos", ascending=False); prov.to_csv("tablas/descriptivo_provincias.csv", index=False)
print(prov.head(12).to_string()); AMP["top_prov_pct"] = float(prov.head(2).pct.sum())

# figura: casos por año y situación
ct = pd.crosstab(df.anio, df.situacion_actual)[["ENCONTRADO","FALLECIDO","DESAPARECIDO"]]
fig, ax = plt.subplots(figsize=(9,5.3))
ax.bar(ct.index, ct.ENCONTRADO, color="#9DB9D5", label="Localizado con vida")
ax.bar(ct.index, ct.FALLECIDO, bottom=ct.ENCONTRADO, color="#14213D", label="Localizado sin vida")
ax.bar(ct.index, ct.DESAPARECIDO, bottom=ct.ENCONTRADO+ct.FALLECIDO, color="#D9822B", label="Sigue desaparecido")
ax2 = ax.twinx(); ax2.plot(ct.index, (ct.FALLECIDO+ct.DESAPARECIDO)/ct.sum(axis=1)*100, "o-", color="#9E2A2B", label="% desenlace adverso"); ax2.set_ylabel("% desenlace adverso"); ax2.grid(False); ax2.set_ylim(0, 20)
ax.set(xlabel="Año de desaparición", ylabel="Casos", title="Casos por año y situación actual"); ax.set_ylim(0, 12000)
h1,l1 = ax.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels(); fig.legend(h1+h2, l1+l2, loc="lower center", ncol=4, fontsize=9, frameon=False)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig("figuras/fig_casos_anio.png", dpi=110); plt.close(fig)
ct.to_csv("tablas/casos_anio_situacion.csv")

# figura: pirámide sexo x edad
orden = ["NIÑOS(AS)","ADOLESCENTES","ADULTO","ADULTO MAYOR"]
pir = pd.crosstab(df.edad_g, df.sexo).reindex(orden)[["HOMBRE","MUJER"]]
fig, ax = plt.subplots(figsize=(9,4.5))
ax.barh(orden, -pir.HOMBRE, color="#2E5E8C", label="Hombres"); ax.barh(orden, pir.MUJER, color="#D9822B", label="Mujeres")
for i,(h,m) in enumerate(zip(pir.HOMBRE, pir.MUJER)):
    ax.text(-h-400, i, f"{h:,}".replace(",", " "), va="center", ha="right", fontsize=9); ax.text(m+400, i, f"{m:,}".replace(",", " "), va="center", ha="left", fontsize=9)
lim = pir.values.max()*1.25; ax.set_xlim(-lim, lim); ax.set_xticks([])
ax.set(title="Casos por grupo de edad y sexo"); ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig("figuras/fig_piramide.png", dpi=110); plt.close(fig)
pir.to_csv("tablas/piramide_sexo_edad.csv")

# figura: mapa nacional
cont = df[df.lon > -82]
fig, ax = plt.subplots(figsize=(7,7))
hb = ax.hexbin(cont.lon, cont.lat, gridsize=70, bins="log", cmap="Blues", mincnt=1)
fig.colorbar(hb, ax=ax, label="Casos (escala logarítmica)")
ax.set(xlabel="Longitud", ylabel="Latitud", title="Distribución espacial de los casos (Ecuador continental)"); ax.set_aspect("equal")
fig.tight_layout(); fig.savefig("figuras/fig_mapa_nacional.png", dpi=110); plt.close(fig)

# ---------- B. Supervivencia por región ----------
regs = ["COSTA","SIERRA","AMAZONIA","INSULAR"]; rr = []
fig, ax = plt.subplots(figsize=(9,5))
for r_, col in zip(regs, ["#D9822B","#2E5E8C","#3A7D44","#9E2A2B"]):
    sub = df[df.region == r_]
    k = KaplanMeierFitter().fit(sub.t, sub.evento, label=f"{r_.capitalize()} (n={len(sub):,})".replace(",", " "))
    k.plot_survival_function(ax=ax, ci_show=False, color=col)
    rm = restricted_mean_survival_time(k, t=30)
    rr.append(dict(region=r_, casos=len(sub), mediana=k.median_survival_time_, S30=round(float(k.predict(30)),3),
                   S365=round(float(k.predict(365)),3), RMST30=round(float(rm),2), pct_fallecido=round(sub.fallecido.mean()*100,1)))
ax.set_xlim(0, 120); ax.set(title="Probabilidad de que el caso siga abierto por región", xlabel="Días desde la desaparición", ylabel="S(t)")
fig.tight_layout(); fig.savefig("figuras/fig_km_region.png", dpi=110); plt.close(fig)
rr = pd.DataFrame(rr); rr.to_csv("tablas/km_por_region.csv", index=False); print(rr.to_string())
lr = multivariate_logrank_test(df.t, df.region, df.evento); AMP["logrank_region"] = dict(chi2=round(float(lr.test_statistic),1), p=float(lr.p_value))
print(AMP["logrank_region"])

# ---------- C. Cox multivariable estratificado por año ----------
d = df[df.edad_g.notna() & df.sexo.isin(["HOMBRE","MUJER"]) & (df.etnia_g != "SIN_DATO") & (df.nac_g != "SIN_DATO")].copy()
X = pd.DataFrame({"t": d.t, "e": d.evento, "anio": d.anio, "mujer": (d.sexo=="MUJER").astype(int)})
for g_ in ["NIÑOS(AS)","ADULTO","ADULTO MAYOR"]: X["edad_"+g_] = (d.edad_g==g_).astype(int)
for r_ in ["SIERRA","AMAZONIA","INSULAR"]: X["region_"+r_] = (d.region==r_).astype(int)
for e_ in ["INDIGENA","AFRO","MONTUBIO/A","OTRA"]: X["etnia_"+e_] = (d.etnia_g==e_).astype(int)
X["extranjero"] = (d.nac_g=="EXTRANJERA").astype(int)
cph = CoxPHFitter().fit(X, "t", "e", strata=["anio"])
s = cph.summary[["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","p"]].round(4)
s.to_csv("tablas/cox_multivariable.csv"); print(s.to_string()); AMP["cox_multi_n"] = int(len(X))
print("C-index:", round(cph.concordance_index_, 3)); AMP["cox_multi_cindex"] = round(float(cph.concordance_index_), 3)

# ---------- D. Distancia entre el lugar de desaparición y el de localización ----------
la = num(df.latitud_localizacion); lo = num(df.longitud_localizacion)
ok = (df.situacion_actual != "DESAPARECIDO") & la.between(-5.1,1.8) & lo.between(-92.5,-75)
def hav(a1,o1,a2,o2):
    a1,o1,a2,o2 = map(np.radians,(a1,o1,a2,o2))
    h = np.sin((a2-a1)/2)**2 + np.cos(a1)*np.cos(a2)*np.sin((o2-o1)/2)**2
    return 2*KM_TIERRA*np.arcsin(np.sqrt(h))
dd = df[ok].copy(); dd["dist"] = hav(dd.lat, dd.lon, la[ok], lo[ok])
filas = []
for nom, sub in [("Total", dd)] + [(c, dd[dd.edad_g==c]) for c in orden] + [("Con vida", dd[dd.situacion_actual=="ENCONTRADO"]), ("Sin vida", dd[dd.situacion_actual=="FALLECIDO"])]:
    filas.append(dict(grupo=nom, casos=len(sub), mediana_km=round(sub.dist.median(),2), p75_km=round(sub.dist.quantile(.75),2),
                      p90_km=round(sub.dist.quantile(.9),2), pct_mismo_punto=round((sub.dist<0.05).mean()*100,1), pct_mas_10km=round((sub.dist>10).mean()*100,1)))
dist_t = pd.DataFrame(filas); dist_t.to_csv("tablas/distancia_localizacion.csv", index=False); print(dist_t.to_string())
AMP["dist_n"] = int(len(dd)); AMP["dist_validos_pct"] = round(ok.sum()/ (df.situacion_actual != "DESAPARECIDO").sum()*100, 1)

# ---------- E. Caracterización de las zonas de Guayas ----------
g = df[df.provincia == "GUAYAS"].copy().reset_index(drop=True)
g["cl"] = DBSCAN(eps=0.35/KM_TIERRA, min_samples=30, metric="haversine", algorithm="ball_tree").fit(np.radians(g[["lat","lon"]].to_numpy())).labels_
zonas_prev = pd.read_csv("tablas/dbscan_zonas.csv")
z = g[g.cl != -1].groupby("cl").agg(casos=("t","size"), canton=("canton", lambda s: s.mode().iat[0]), distrito=("distrito", lambda s: s.mode().iat[0]),
        circuitos=("circuito","nunique"), pct_mujer=("sexo", lambda s: round((s=="MUJER").mean()*100,1)),
        pct_adolesc=("edad_g", lambda s: round((s=="ADOLESCENTES").mean()*100,1)), anios=("anio","nunique")).reset_index().rename(columns={"cl":"zona"})
z = z.merge(zonas_prev[["zona","casos","tasa","severa","cronica"]].rename(columns={"casos":"casos_prev"}), on="zona", how="left")
print("coinciden tamaños con análisis principal:", bool((z.casos == z.casos_prev).all()))
z = z.sort_values("casos", ascending=False); z.to_csv("tablas/dbscan_caracterizacion.csv", index=False); print(z.head(12).to_string())
AMP["zonas_canton_gye"] = int((z.canton=="GUAYAQUIL").sum()); AMP["casos_zona_gye_pct"] = round(z[z.canton=="GUAYAQUIL"].casos.sum()/z.casos.sum()*100,1)
AMP["distritos_zonas"] = int(z.distrito.nunique())
print(AMP)
json.dump(AMP, open("tablas/ampliacion.json","w"), ensure_ascii=False, indent=1, default=str)


# ===== F. Incidencia de desenlace fatal por grupo y motivos =====
pd.set_option("display.width",250)
rows=[]
grupos=[("Total",df)]+[(f"Sexo: {s}",df[df.sexo==s]) for s in ["HOMBRE","MUJER"]]+[(f"Edad: {g_}",df[df.edad_g==g_]) for g_ in ["NIÑOS(AS)","ADOLESCENTES","ADULTO","ADULTO MAYOR"]]
fig,ax=plt.subplots(figsize=(9,5))
cols={"NIÑOS(AS)":"#3A7D44","ADOLESCENTES":"#2E5E8C","ADULTO":"#D9822B","ADULTO MAYOR":"#9E2A2B"}
for nom,sub in grupos:
    aj=AalenJohansenFitter(calculate_variance=False).fit(sub.t, sub.causa, event_of_interest=2)
    cif=aj.cumulative_density_.iloc[:,0]
    v=lambda d: float(cif[cif.index<=d].iloc[-1]) if (cif.index<=d).any() else 0.0
    rows.append(dict(grupo=nom,n=len(sub),F7=round(v(7)*100,2),F30=round(v(30)*100,2),F365=round(v(365)*100,2), share7=round(v(7)/v(365)*100,1) if v(365)>0 else None))
    if nom.startswith("Edad"):
        g_=nom.split(": ")[1]; ax.step(cif.index, cif.values*100, where="post", color=cols[g_], label=g_.capitalize())
ax.set_xlim(0,365); ax.set(xlabel="Días desde la desaparición", ylabel="Incidencia acumulada (%)", title="Incidencia acumulada de localización sin vida por grupo de edad"); ax.legend()
fig.tight_layout(); fig.savefig("figuras/fig_cif_edad.png", dpi=110)
cf=pd.DataFrame(rows); print(cf.to_string()); cf.to_csv("tablas/cif_fallecido_grupos.csv",index=False)
# motivos (casos cerrados con vida)
enc=df[df.situacion_actual=="ENCONTRADO"]
m=enc.motivo_desaparicion.value_counts(); mt=pd.DataFrame({"casos":m,"pct":(m/len(enc)*100).round(1)})
mt["mediana_dias"]=[enc[enc.motivo_desaparicion==k].t.median() for k in mt.index]
print(mt.to_string()); mt.to_csv("tablas/motivos_encontrados.csv")
print(pd.crosstab(df.estado_desaparecido, df.situacion_actual))
print(pd.crosstab(df.motivo_desaparicion, df.situacion_actual).loc[["FALLECIDO","NO_APLICA","VIOLENCIA","POSIBLE VINCULACIÓN A DELITO"]])


# ===== G. Extensión de DBSCAN a Pichincha =====
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
from sklearn.neighbors import BallTree
df["adverso"]=(df.situacion_actual!="ENCONTRADO").astype(int)
out={}
for prov in ["PICHINCHA","GUAYAS"]:
    g=df[df.provincia==prov].copy().reset_index(drop=True)
    X=np.radians(g[["lat","lon"]].to_numpy())
    g["cl"]=DBSCAN(eps=0.35/KM_TIERRA,min_samples=30,metric="haversine",algorithm="ball_tree").fit(X).labels_
    m=g.cl!=-1; tam=g[m].cl.value_counts()
    tasa=g.groupby("anio").adverso.mean()
    filas=[]
    for z,sub in g[m].groupby("cl"):
        n=len(sub); x=int(sub.adverso.sum()); e=float((sub.anio.map(tasa)).mean())
        filas.append(dict(zona=z,casos=n,adv=x,tasa=x/n,esp=e,razon=(x/n)/e if e>0 else np.nan,p=binomtest(x,n,e,alternative="greater").pvalue,
                          canton=sub.canton.mode().iat[0],distrito=sub.distrito.mode().iat[0]))
    Z=pd.DataFrame(filas); Z["pBH"]=multipletests(Z.p,method="fdr_bh")[1]; Z["severa"]=Z.pBH<0.05
    cron=sum(1 for z,sub in g[m].groupby("cl") if (sub.groupby("anio").size()>=3).sum()>=8)
    # validación temporal: cobertura
    ent=g[g.anio<=2022].copy(); sal=g[g.anio>=2023].copy()
    ent["cl"]=DBSCAN(eps=0.35/KM_TIERRA,min_samples=30,metric="haversine",algorithm="ball_tree").fit(np.radians(ent[["lat","lon"]].to_numpy())).labels_
    core=ent[ent.cl!=-1]; arbol=BallTree(np.radians(core[["lat","lon"]].to_numpy()),metric="haversine")
    dist,idx=arbol.query(np.radians(sal[["lat","lon"]].to_numpy()),k=1); cov=(dist[:,0]*KM_TIERRA<=0.35).mean()
    k=KaplanMeierFitter()
    out[prov]=dict(casos=len(g),coords=len(g[["lat","lon"]].drop_duplicates()),zonas=len(tam),pct_en=round(m.mean()*100,1),mayor=round(tam.iloc[0]/len(g)*100,1),
                   severas=int(Z.severa.sum()),cronicas=cron,zonas_ent=int(core.cl.nunique()),cobertura=round(cov*100,1),
                   tasa17=round(tasa.iloc[0]*100,1),tasa25=round(tasa.iloc[-1]*100,1))
    print(prov,out[prov]); print(Z.sort_values("casos",ascending=False).head(8)[["zona","casos","canton","distrito","tasa","razon","pBH","severa"]].to_string())
    print(Z[Z.severa][["zona","casos","canton","distrito","tasa","esp","razon","pBH"]].to_string())
    if prov=="PICHINCHA":
        fig,ax=plt.subplots(figsize=(7,7))
        ax.scatter(g[~m].lon,g[~m].lat,s=2,c="#C9CED6",label="Ruido")
        cmap=plt.get_cmap("tab20")
        for i,(z,sub) in enumerate(g[m].groupby("cl")): ax.scatter(sub.lon,sub.lat,s=3,color=cmap(i%20))
        sv=Z[Z.severa].zona.tolist()
        for z in sv:
            sub=g[g.cl==z]; ax.scatter(sub.lon.mean(),sub.lat.mean(),s=160,facecolors="none",edgecolors="#9E2A2B",linewidths=2)
        ax.set(xlabel="Longitud",ylabel="Latitud",title=f"Zonas DBSCAN en Pichincha (ε = 0,35 km, MinPts = 30)")
        ax.set_xlim(-78.65,-78.30); ax.set_ylim(-0.42,0.05); ax.set_aspect("equal")
        fig.tight_layout(); fig.savefig("figuras/fig_dbscan_pichincha.png",dpi=110)
        Z.to_csv("tablas/dbscan_zonas_pichincha.csv",index=False)
json.dump(out,open("tablas/dbscan_pichincha.json","w"),indent=1,default=str)


# ===== H. Schoenfeld y sensibilidad de Apriori =====
from lifelines.statistics import proportional_hazard_test
d2 = df[df.edad_g.notna() & df.sexo.isin(["HOMBRE","MUJER"])].copy()
X = d2[["t","evento"]].copy(); X["mujer"] = (d2.sexo=="MUJER").astype(int)
for g_ in ["NIÑOS(AS)","ADULTO","ADULTO MAYOR"]: X[g_] = (d2.edad_g==g_).astype(int)
cox_aj = CoxPHFitter().fit(X, "t", "evento")
ph = proportional_hazard_test(cox_aj, X, time_transform="rank"); print(ph.summary.round(4).to_string())
ph.summary.to_csv("tablas/schoenfeld.csv")
T, cols = construir_transacciones(df)
filas=[]
for ms in (0.0005, 0.001, 0.002):
    for lm in (1.2, 1.5, 2.0):
        R,_ = minar_reglas(T, cols, min_support=ms, lift_min=lm)
        c = R.con.value_counts()
        filas.append(dict(min_support=ms, lift_min=lm, total=len(R), fallecido=int(c.get("SALE=FALLECIDO",0)), sigue=int(c.get("SALE=SIGUE_DESAPARECIDO",0)), mas30=int(c.get("SALE=RESOLUCION>30d",0))))
        print(filas[-1])
pd.DataFrame(filas).to_csv("tablas/apriori_sensibilidad.csv", index=False)
