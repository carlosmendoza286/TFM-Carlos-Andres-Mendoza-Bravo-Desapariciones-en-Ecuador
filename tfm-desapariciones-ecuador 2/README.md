# Modelado estocástico y aprendizaje no supervisado en dinámicas de desaparición en Ecuador

Código del Trabajo Fin de Máster de **Carlos Andrés Mendoza Bravo**  
Máster Universitario en Ingeniería Matemática y Computación, Universidad Internacional de La Rioja (UNIR)  
Directora: Paula de la Lama Zubiran

## Descripción

Análisis de los registros de personas desaparecidas en Ecuador entre 2017 y 2025 (datos abiertos del Ministerio del Interior) con tres técnicas:

1. **Análisis de supervivencia**: Kaplan-Meier, incidencia acumulada con riesgos competitivos (Aalen-Johansen), prueba de rangos logarítmicos, modelo de Cox estratificado por año y RMST.
2. **Agrupamiento espacial DBSCAN** con distancia de Haversine (provincia de Guayas): selección de parámetros, severidad de las zonas estandarizada por año con corrección de Benjamini-Hochberg y validación temporal.
3. **Reglas de asociación (Apriori)** entre el perfil de la persona y el desenlace del caso, con intervalos de Wilson y validación temporal.

Las zonas y los perfiles obtenidos con datos de 2017-2022 se integran como covariables de un modelo de Cox ajustado con los casos de 2023-2025.

## Estructura

```
notebooks/tfm_desapariciones.ipynb   cuaderno ejecutado con todas las salidas
src/tfm_analisis.py                  el mismo análisis como script
src/tfm_ampliacion.py                análisis complementarios del Capítulo 4
resultados/figuras/                  figuras de la memoria
resultados/tablas/                   tablas de resultados (resultados_clave.json reúne las cifras)
datos/                               ubicación del archivo de datos (no incluido)
requirements.txt                     dependencias
```

## Reproducción

```bash
pip install -r requirements.txt
# descargar el archivo de datos en datos/ (ver datos/LEEME.md)
TFM_XLSX=datos/mdi_personasdesaparecidas_pm_2017_2025.xlsx TFM_HEADLESS=1 python src/tfm_analisis.py
# análisis complementarios (descripción de los casos, región, Cox multivariable,
# incidencia de desenlace fatal por grupo, motivos, distancia de localización,
# caracterización de zonas, extensión a Pichincha y sensibilidad de Apriori)
TFM_XLSX=datos/mdi_personasdesaparecidas_pm_2017_2025.xlsx TFM_HEADLESS=1 python src/tfm_ampliacion.py
```

También puede ejecutarse el cuaderno en Jupyter o Google Colab, indicando la ruta del archivo en la variable `RUTA_XLSX`.

## Protección de datos

Se emplean únicamente datos abiertos sin identificadores personales y los resultados se presentan de forma agregada, conforme a la Ley Orgánica de Protección de Datos Personales del Ecuador.
