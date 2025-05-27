import pandas as pd
import numpy as np
import subprocess
from io import StringIO

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                       --- DATA IA ADOPTION ---
path_file = "SDMXE_2017-2024 v250409.mdb"

tables = subprocess.check_output(["mdb-tables", "-1", path_file]).decode().splitlines()

# Table - DataWithAggregates
table_name = tables[4]
csv_output = subprocess.check_output(["mdb-export", path_file, table_name]).decode()

# Convertir la salida CSV en un DataFrame - Database Original
df = pd.read_csv(StringIO(csv_output))
df

# Table - Countries
table_name = tables[2]
csv_output = subprocess.check_output(["mdb-export", path_file, table_name]).decode()
df_country = pd.read_csv(StringIO(csv_output))
df_country

# Eliminamos paises agregados de nuestro dataset:

# 32       EU15          European Union - 15 countries (1995-2004)
# 33         EA  Euro area (EA11-1999, EA12-2001, EA13-2007, EA...
# 35       EU25          European Union - 25 countries (2004-2006)
# 37  EU27_2007          European Union - 27 countries (2007-2013)
# 53       EU28          European Union - 28 countries (2013-2020)
# 45  EU27_2020          European Union - 27 countries (from 2020)

country_num = [32,33,35,37,53,45]
df_filtered = df[~df['IdCountry'].isin(country_num)]

# Table - Indicators
table_name = tables[8]
csv_output = subprocess.check_output(["mdb-export", path_file, table_name]).decode()
df_indicators = pd.read_csv(StringIO(csv_output))
df_indicators

# Filtramos los indicadores de interes:

# IdIndicator  ExpIndicator  IdIndicatorGroup  ExpIndicatorCaption
#3160  E_DI3_HI_AI_TANY    111.0  Enterprises with high digital intensity index, which use any artificial intelligence technology
#3165  E_DI3_LO_AI_TANY    111.0  Enterprises with low digital intensity index, which use any artificial intelligence technology
#3170  E_DI3_VHI_AI_TANY   111.0  Enterprises with very high digital intensity index, which use any artificial intelligence technology
#3175  E_DI3_VLO_AI_TANY   111.0  Enterprises with very low digital intensity index, which use any artificial intelligence technology
#3375  E_DI4_HI_AI_TANY    111.0  Enterprises with high digital intensity index (Version 4), which use any artificial intelligence technology
#3383  E_DI4_LO_AI_TANY    111.0  Enterprises with low digital intensity index (Version 4), which use any artificial intelligence technology
#3391  E_DI4_VHI_AI_TANY   111.0  Enterprises with very high digital intensity index (Version 4), which use any artificial intelligence technology
#3399  E_DI4_VLO_AI_TANY   111.0  Enterprises with very low digital intensity index (Version 4), which use any artificial intelligence technology
#pd.set_option('display.max_colwidth', None)

df_ia = df_indicators[df_indicators['ExpIndicatorCaption'].str.contains('use any Artificial Int', case=False, na=False)]
df_ia['IdIndicator'].unique()
df_ia

Any_AI_Tech=[3160, 3165, 3170, 3175, 3375, 3383, 3391, 3399]

# Filtramos por Any_AI_Tech:
df_filtered = df_filtered[df_filtered['IdIndicator'].isin(Any_AI_Tech)]
df_filtered['IdIndicator'].unique()
df_filtered

# Table - Activities
table_name = tables[0]
csv_output = subprocess.check_output(["mdb-export", path_file, table_name]).decode()
df_Activity = pd.read_csv(StringIO(csv_output))
df_Activity['IdActivity'].unique()
df_Activity

# Table - Units
table_name = tables[14]
csv_output = subprocess.check_output(["mdb-export", path_file, table_name]).decode()
df_Units = pd.read_csv(StringIO(csv_output))
df_Units.head(20)

# Comprobamos que todos tienen un Id=16, Percentage of enterprises:
df_filtered['IdUnit'].unique() 

# Eliminamos columnas
df_filtered = df_filtered.merge(df_country, on="IdCountry",how='inner').drop(columns=['IdCustBrkdwn','IdNote','Flags','IdCountry','IdUnit','IdRegion'])
df_filtered = df_filtered.merge(df_Activity, on="IdActivity",how='inner').drop(columns=['IdActivityGroup','ExpActivityCaption','IdActivity'])
df_filtered = df_filtered.rename(columns={'ExpCountryCaption':'Country', 'ExpCountry':'IdCountry'}).replace('Bosnia and Herzegovina',value='Bosnia')
df_filtered = df_filtered.dropna(subset=['Value'])
df_filtered

# Dataset original (empresa)
df_filtered = df_filtered.rename(columns={'ExpActivity':'NACE'})
df_filtered #dataset original de adopción de IA - 13649

# Agregamos sectores:
data_sector = df_filtered.copy()
data_sector['NACE'] = data_sector['NACE'].replace({
    'C10-C12': 'C',
    'C10-C18': 'C',
    'C16-C18': 'C',
    'C19-C23': 'C',
    'C20': 'C',
    'C21': 'C',
    'C22_C23': 'C',
    'C24_C25': 'C',
    'C26': 'C',
    'C26-C33': 'C',
    'C27': 'C',
    'C27_C28': 'C',
    'C28': 'C',
    'C29_C30': 'C',
    'C31-C33': 'C',
    'C13-C15': 'C', 
    'C19':'C'  
})
data_sector['NACE'] = data_sector['NACE'].replace({'D35': 'D'})
data_sector['NACE'] = data_sector['NACE'].replace({'G45': 'G','G46': 'G','G47': 'G'})
data_sector['NACE'] = data_sector['NACE'].replace({'I55': 'I',})
data_sector['NACE'] = data_sector['NACE'].replace({'J58-J60': 'J','J62_J63': 'J','J61': 'J'})
data_sector['NACE'] = data_sector['NACE'].replace({'L68': 'L'})
data_sector['NACE'] = data_sector['NACE'].replace({'M69-M71': 'M','M72': 'M','M73-M75': 'M'})
data_sector['NACE'] = data_sector['NACE'].replace({'N77-N82_X_N79': 'N','N79': 'N'})
data_sector['NACE'] = data_sector['NACE'].replace({'S951': 'S'})

# Eliminamos sectores que no se pueden agregar:
categorias_a_eliminar = ['L_M', 'C-F', 'C10-S951_X_K', 'G45-S951_X_K','C-E','D_E','ICT']
data_sector = data_sector[~data_sector['NACE'].isin(categorias_a_eliminar)]

data_sector['NACE'].unique()
data_sector['NACE'].value_counts()

# Comprovamos si existen duplicados:
cols_clave = ['IdYear','IdIndicator', 'IdEntSize', 'Value', 'Country', 'NACE']
duplicados = data_sector[data_sector.duplicated(subset=cols_clave, keep=False)]
duplicados_exactos = data_sector[data_sector.duplicated(keep=False)]
len(duplicados_exactos)==0 #hay duplicados

# Eliminamos duplicados:
data_sector = data_sector.drop_duplicates(subset=cols_clave)
data_IA = data_sector.copy()
data_IA #7388 obs

# Finalmente agrupamos por sector:
data_sector = data_sector.groupby(['Country','IdYear','NACE'])['Value'].mean().reset_index()
data_sector[data_sector['Country']=='Spain']
data_sector #dataset agregado por NACE
#687 obs

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                       --- DATA EMPLOYMENT ---
path = "lfsa_eisn2.xlsx"

NACE_Id = {"Agriculture, forestry and fishing":"A",
"Mining and quarrying":"B",
"Manufacturing":"C",
"Electricity, gas, steam and air conditioning supply":"D",
"Water supply; sewerage, waste management and remediation activities":"E",
"Construction":"F",
"Wholesale and retail trade; repair of motor vehicles and motorcycles":"G",
"Transportation and storage":"H",
"Accommodation and food service activities":"I",
"Information and communication":"J",
"Financial and insurance activities":"K",
"Real estate activities":"L",
"Professional, scientific and technical activities":"M",
"Administrative and support service activities":"N",
"Public administration and defence; compulsory social security":"O",
"Education":"P",
"Human health and social work activities":"Q",
"Arts, entertainment and recreation":"R",
"Other service activities":"S",
"Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use":"T",
"Activities of extraterritorial organisations and bodies":"U"}

ISCO_Id = {"Managers":"OC1",
"Professionals":"OC2",
"Technicians": "OC3",
"Clerical support workers":"OC4",
"Service and sales workers":"OC5",
"Skilled agricultural, forestry and fishery workers":"OC6",
"Craft and related trades workers":"OC7",
"Plant and machine operators and assemblers":"OC8",
"Elementary occupations":"OC9",
"Armed forces occupations":"OC0"}

dataset = None
sheet_num = 2

while sheet_num < 212:
    df = pd.read_excel(path, sheet_num, header=None)
    sheet_num += 1
    print(f"Hoja {sheet_num-1}")

    # — Detectar ISCO —
    for key, value in ISCO_Id.items():
        if df[2].astype(str).str.contains(key, na=False, case=True, regex=False).any():
            df['ISCO_08'] = value
            break

    # — Detectar NACE —
    for key, value in NACE_Id.items():
        if df[2].astype(str).str.contains(key, na=False, case=False, regex=False).any():
            df['NACE'] = value
            break

    # — Renombrar y limpiar columnas —
    df.rename(columns={0: "Country", 1: "23", 3: "24"}, inplace=True)
    df.drop(columns=[2, 4], errors='ignore', inplace=True)

    # — Filtrar filas relevantes —
    df = df.iloc[13:49].reset_index(drop=True)

    # — Concatenar en dataset —
    if dataset is None:
        # primera asignación
        dataset = df.copy()
    else:
        # ya existe, concatenamos
        dataset = pd.concat([dataset, df], ignore_index=True)

# Función melt (crear columna IdYear y occ, a partir de columnas 23, 24):
id_vars2= ['Country','ISCO_08', 'NACE'] #columnas que mantenemos
year_cols = ["23","24"] #columnas que transformaremos
data_ISCO = dataset.melt(id_vars=id_vars2,value_vars=year_cols,var_name='IdYear',value_name='occ')
data_ISCO

# Establecer datos NA a valores cero
data_ISCO.replace(':',0, inplace=True)

# Transformación de datos:
total_sector = data_ISCO.groupby(['Country', 'NACE','IdYear'])['occ'].sum().reset_index()
total_sector = total_sector.rename(columns={'occ': 'occ_sector'})

data_ISCO = data_ISCO.merge(total_sector, on=['Country', 'NACE','IdYear'], how='left')
data_ISCO['Share_Occupation'] = data_ISCO['occ']/data_ISCO['occ_sector']
data_ISCO

total_occupation = data_ISCO.groupby(['Country','IdYear'])['occ'].sum().reset_index()
total_occupation = total_occupation.rename(columns={'occ':'occ_total'})

data_ISCO = data_ISCO.merge(total_occupation,on=['Country','IdYear'], how="left")
data_ISCO['Share_Sector'] = data_ISCO['occ_sector']/data_ISCO['occ_total']
data_ISCO

# Limpieza de datos:
data_ISCO =data_ISCO[data_ISCO['Country']!='Montenegro']
data_ISCO =data_ISCO[data_ISCO['Country']!='United Kingdom']
data_ISCO =data_ISCO[data_ISCO['Country']!='North Macedonia']
data_ISCO.replace("Bosnia and Herzegovina","Bosnia",inplace=True)
data_ISCO.drop(['occ','occ_sector','occ_total'], axis=1, inplace=True)
data_ISCO = data_ISCO.dropna()

# Comprovamos datos correctos:
data_ISCO['Country'].unique()
data_ISCO['ISCO_08'].unique()
data_ISCO['NACE'].unique()
data_ISCO

# Agrupamos por Sector:
data_NACE = data_ISCO.groupby(['Country','IdYear','NACE'])['Share_Sector'].mean().reset_index()

# Corregimos formato:
data_NACE['IdYear'] = data_NACE['IdYear'].astype(int)
data_ISCO['IdYear'] = data_ISCO['IdYear'].astype(int)

# Comprovamos datos correctos:
data_NACE[data_NACE['Country']=='Germany']['Share_Sector'].sum() #ejemplo

data_NACE #datos agrupados por sectores
data_ISCO #1269*10 ocupaciones = 12690 obs

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                       --- MERGING DATA ---
file_path = "macro data.xlsx"
df_agreggdata = pd.read_excel(file_path, sheet_name=0)
df_agreggdata

#                          -- MODELO 1 - SECTORIAL --
data_sector
# Combinamos con datos macroeconomicos:
data_sector = data_sector.merge(df_agreggdata, on=["Country", "IdYear"], how="left")
data_sector.drop('RD_exp',axis=1,inplace=True)
data_sector = data_sector.dropna()

# Creamos copia:
data_occupation = data_sector.copy()

# Combinamos con datos de empleo:
data_sector = data_sector.merge(data_NACE, on=["Country", "IdYear","NACE"], how="left")
data_sector = data_sector.dropna()
data_sector #644

# Comprovamos:
test_sector = data_sector.groupby(['Country','IdYear','NACE'])['Share_Sector'].mean()
test_sector['Spain'].sum()
test_sector

#                          -- MODELO 2 - OCUPACIONAL --
data_occupation
# Combinamos con datos de empleo
data_occupation = data_occupation.merge(data_ISCO, on=["Country", "IdYear",'NACE'], how="left")
data_occupation = data_occupation.dropna()
data_occupation #6440

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                       --- REGRESIONES ---
# MODELO 1 - SECTORIAL
data_sector

# Algunas correcciones:
data_sector.rename(columns={'Value':"IA_Adoption",'Unempl_rate':'Unemp_Rate'},inplace=True)

# Establecer 'M' como la categoría base:
categorias = list(data_sector['NACE'].unique())
categorias.remove('M')
nueva_orden = ['M'] + sorted(categorias)  # Poner 'M' al inicio, luego el resto ordenado
data_sector['NACE'] = pd.Categorical(data_sector['NACE'], categories=nueva_orden, ordered=True)
data_sector['NACE']
print("-----------------------------------------------------------------------------------------------------------------------------------------------------")
#                       PANEL SECTORIAL
data_sector

from linearmodels.panel import PanelOLS

# Asegúrate de que tanto 'Country' como 'IdYear' estén en el índice
data_panel_sector = data_sector.set_index(['Country', 'IdYear'])

# Estimación con efectos fijos por entidad (país) y tiempo (año)
modelo_panel = PanelOLS.from_formula(
    'Share_Sector ~ IA_Adoption + GDPCap + Educ_Rate + Unemp_Rate + C(NACE)*IA_Adoption + EntityEffects + TimeEffects',
    data=data_panel_sector
)
results_sector = modelo_panel.fit(cov_type='clustered', cluster_entity=True)
print(results_sector.summary)
print("-----------------------------------------------------------------------------------------------------------------------------------------------------")
#                       EFECTOS DIRECTOS SECTORIAL
from scipy.stats import norm

sectores = ['C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'N', 'S']
resultados = {}

for sector in sectores:
    coef_interaccion = f'C(NACE)[T.{sector}]:IA_Adoption'
    
    if ('IA_Adoption' not in results_sector.params.index):
        print("No existe coef_base")
    if (coef_interaccion not in results_sector.params.index):
        print(f"No existe coef_interaccion para sector {sector}")
        continue
    coef_base = results_sector.params['IA_Adoption']
    coef_sector = results_sector.params[coef_interaccion]
    print(f"Coef_base: {coef_base}, Coef_sector: {coef_sector}")  # Debug
    
    efecto_total = coef_base + coef_sector

    cov = results_sector.cov
    se_total = np.sqrt(
        cov.loc['IA_Adoption', 'IA_Adoption'] +
        cov.loc[coef_interaccion, coef_interaccion] +
        2 * cov.loc['IA_Adoption', coef_interaccion]
    )
    t_stat = efecto_total / se_total
    p_value = 2 * (1 - norm.cdf(np.abs(t_stat)))
    resultados[sector] = {
        'Efecto Total': efecto_total,
        'Error estándar': se_total,
        't': t_stat,
        'p-value': p_value
    }

df_resultados = pd.DataFrame(resultados).T
print(df_resultados)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# MODELO 2 - OCUPACIONAL

data_occupation = data_occupation.dropna()
data_occupation

# Establecer la categoría base
categorias = list(data_occupation['ISCO_08'].unique())
categorias.remove('OC2')
nueva_orden = ['OC2'] + sorted(categorias)
data_occupation['ISCO_08'] = pd.Categorical(data_occupation['ISCO_08'], categories=nueva_orden, ordered=True)
data_occupation['ISCO_08']

data_occupation.rename(columns={'Value':"IA_Adoption",'Unempl_rate':'Unemp_Rate'},inplace=True)
data_occupation
print("-----------------------------------------------------------------------------------------------------------------------------------------------------")
#                       PANEL OCUPACIONAL

# Asegúrate de que la base YA está agregada a nivel sector-país-año-ocupación
# y de que Country, NACE y IdYear están en columnas.

data_panel_occ = data_occupation.set_index(['Country', 'IdYear'])

# data_panel_occ ya tiene MultiIndex (Country, IdYear) y columna NACE

# 1) Captura los niveles del índice
paises = data_panel_occ.index.get_level_values('Country')
años   = data_panel_occ.index.get_level_values('IdYear')

# 2) Crea el vector de cluster concatenando Country, NACE y IdYear
clusters = paises.astype(str) + '_' + \
           data_panel_occ['NACE'].astype(str) + '_' + \
           años.astype(str)

# 3) Estima el modelo, pasando clusters como cluster_entity
from linearmodels.panel import PanelOLS

formula = (
    'Share_Occupation ~ IA_Adoption + GDPCap + Educ_Rate + Unemp_Rate + C(NACE) + C(ISCO_08)*IA_Adoption + EntityEffects + TimeEffects'
)
modelo_occ = PanelOLS.from_formula(formula, data=data_panel_occ)

results_occ = modelo_occ.fit(
    cov_type='clustered',
    clusters=clusters
)
print(results_occ.summary)
print("-----------------------------------------------------------------------------------------------------------------------------------------------------")
#                       EFECTOS DIRECTOS OCUPACIONAL
from scipy import stats

# Supongamos que tu resultado de PanelOLS está en la variable results_occ
# y que las categorías ISCO-08 están en la lista siguiente (incluyendo la base 'OC2'):
ocupaciones = ['OC0', 'OC1', 'OC2', 'OC3', 'OC4', 'OC5', 'OC6', 'OC7', 'OC8', 'OC9']

# Extraemos coeficiente base de IA_Adoption y la matriz var-cov
beta_base = results_occ.params['IA_Adoption']
vcov = results_occ.cov

# Prepara lista para guardar resultados
filas = []

for oc in ocupaciones:
    if oc == 'OC2':
        # Para la categoría base, el efecto total es simplemente beta_base
        total = beta_base
        var_total = vcov.loc['IA_Adoption','IA_Adoption']
    else:
        term = f'C(ISCO_08)[T.{oc}]:IA_Adoption'
        beta_int = results_occ.params[term]
        # var(total) = var(base) + var(int) + 2 cov(base,int)
        var_total = (
            vcov.loc['IA_Adoption','IA_Adoption'] +
            vcov.loc[term,term] +
            2 * vcov.loc['IA_Adoption',term]
        )
        total = beta_base + beta_int

    se = np.sqrt(var_total)
    t_stat = total / se
    # gls approx: con muchos grados, usamos normal aprox o t con df large
    p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    filas.append({
        'ISCO': oc,
        'Efecto Total': total,
        'Error estándar': se,
        't': t_stat,
        'p-value': p_val
    })

df_effects_isco = pd.DataFrame(filas).set_index('ISCO')
print(df_effects_isco)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
print("-----------------------------------------------------------------------------------------------------------------------------------------------------")
# MODELO 3 - TEORICO:
data_sector
data_occupation

# 1. Carga el Excel (ajusta sheet_name si fuera necesario)
path = "RTI_ISCO08_major_groups.xlsx"
# Asumimos que la hoja que contiene NRA, NRI, RC, RM, NRM y RTI es la segunda (index 1)
df = pd.read_excel(path, sheet_name=1)
df
df_2 = pd.read_excel(path, sheet_name=0)
df_2 = df_2[['Mean_RTI','ISCO_08']]
df_2

# 2. Agrupa y calcula medias
agg = (
    df
    .groupby(["ISCO_08"], as_index=False)
    .agg({
        "NRA": "mean",
        "NRI": "mean",
        "RC" : "mean",
        "RM" : "mean",
        "NRM":"mean",
        "RTI":"mean"
    })
)

# 3. Formatea el código de grupo OC1…OC9 y las medias a 2 decimales con coma decimal
agg["ISCO_08"] = agg["ISCO_08"].astype(str).apply(lambda x: f"{x}")
for col in ["NRA","NRI","RC","RM","NRM","RTI"]:
    agg[col] = (
        agg[col]
        .round(2)                    # dos decimales
        .astype(str)
        .str.replace(".", ",")       # coma decimal
    )

# 4. Ordena por código para que quede OC1…OC9
agg = agg.sort_values("ISCO_08").reset_index(drop=True)
agg

# 1) Convertir comas a puntos y a float en agg
for col in ['NRA','NRI','RC','RM','NRM','RTI']:
    agg[col] = agg[col].str.replace(',','.').astype(float)

# 2) Hacer merge con data_occupation (10 filas por sector-país-año)
data_occ = data_occupation.merge(agg, on='ISCO_08', how='left')
data_occ = data_occ.merge(df_2,how='left', on="ISCO_08")
data_occ
data_occ = data_occ.dropna()
data_occ
# 3) Agrupar para obtener un solo RTI ponderado por sector-país-año
#    Usamos Share_Occupation como peso, y recuperamos los controles (tomamos el primero)
data_theory = (
    data_occ
    .groupby(['Country','IdYear','NACE'], as_index=False)
    .agg({
        'IA_Adoption':    'first',
        'GDPCap':         'first',
        'Educ_Rate':      'first',
        'Unemp_Rate':     'first',
        'Share_Sector':   'first',
        # Media ponderada:
        'RTI':            lambda x: (x * data_occ.loc[x.index, 'Share_Occupation']).sum(),
        # Si quieres también los componentes:
        'NRA':            lambda x: (x * data_occ.loc[x.index, 'Share_Occupation']).sum(),
        'NRI':            lambda x: (x * data_occ.loc[x.index, 'Share_Occupation']).sum(),
        'RC':             lambda x: (x * data_occ.loc[x.index, 'Share_Occupation']).sum(),
        'RM':             lambda x: (x * data_occ.loc[x.index, 'Share_Occupation']).sum(),
        'NRM':            lambda x: (x * data_occ.loc[x.index, 'Share_Occupation']).sum(),
        'Mean_RTI':       lambda x: (x * data_occ.loc[x.index, 'Share_Occupation']).sum()
    })
)
data_theory #dataset para modelos teoricos
print("-----------------------------------------------------------------------------------------------------------------------------------------------------")
#                               PANEL TEORICO 1

from linearmodels.panel import PanelOLS

# 1. Prepara el panel con country y year como índice
data_panel_theory = data_theory.set_index(['Country', 'IdYear'])

# 2. Fórmula: IA, RTI y su interacción, más los controles y efectos fijos
formula = (
    'Share_Sector ~ IA_Adoption + Mean_RTI + IA_Adoption:Mean_RTI + GDPCap + Educ_Rate + Unemp_Rate+ EntityEffects + TimeEffects'
)

# 3. Estima con efectos fijos por entidad y tiempo, errores clusterizados
modelo_theory = PanelOLS.from_formula(formula, data=data_panel_theory)
results_theory = modelo_theory.fit(
    cov_type='clustered',
    cluster_entity=True
)

# 4. Muestra resultados
print(results_theory.summary)
print("-----------------------------------------------------------------------------------------------------------------------------------------------------")
#                              PANEL TEORICO 2:

from linearmodels.panel import PanelOLS

# 1. Creamos el panel con Country y IdYear como índice
data_panel = data_theory.set_index(['Country', 'IdYear'])

# 2. Definimos la fórmula del modelo de sub-tareas (omitimos RC como categoría base)
formula_sub = (
    'Share_Sector ~ IA_Adoption '
    '+ IA_Adoption:NRA + IA_Adoption:NRI '
    '+ IA_Adoption:RM  + IA_Adoption:NRM '
    '+ NRA + NRI + RM + NRM '
    '+ GDPCap + Educ_Rate + Unemp_Rate '
    '+ EntityEffects + TimeEffects'
)

# 3. Estimamos con PanelOLS y errores estándar clusterizados por entidad (país)
modelo_sub = PanelOLS.from_formula(formula_sub, data=data_panel)
results_sub = modelo_sub.fit(
    cov_type='clustered',
    cluster_entity=True
)

# 4. Imprimimos el resumen de resultados
print(results_sub.summary)
print("-----------------------------------------------------------------------------------------------------------------------------------------------------")
#                       EFECTOS DIRECTOS TEORICO SUB-TAREAS

from scipy.stats import norm

# Lista de subtareas y la categoría base (RC)
subtareas = ['RC', 'NRA', 'NRI', 'RM', 'NRM']

# Extraemos parámetros y matriz de var-cov
b = results_sub.params
V = results_sub.cov

resultados = {}

# Función helper para cada subtarea
for tarea in subtareas:
    if tarea == 'RC':
        # Efecto puro IA en la categoría omitida (RC=1)
        efecto = b['IA_Adoption']
        var = V.loc['IA_Adoption','IA_Adoption']
    else:
        # Interacción IA × tarea
        inter = f'IA_Adoption:{tarea}'
        coef_int = b.get(inter, 0.0)
        efecto = b['IA_Adoption'] + coef_int

        # var(β1 + β_int) = var(β1) + var(β_int) + 2 cov(β1, β_int)
        var = (
            V.loc['IA_Adoption','IA_Adoption']
            + V.loc[inter, inter]
            + 2 * V.loc['IA_Adoption', inter]
        )

    se = np.sqrt(var)
    t  = efecto / se
    p  = 2 * (1 - norm.cdf(abs(t)))

    resultados[tarea] = {
        'Efecto_total': efecto,
        'SE':            se,
        't':             t,
        'p-value':       p
    }

df_ef = pd.DataFrame(resultados).T
print(df_ef.round(4))


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                       --- PRUEBA DE ROBUSTEZ ---
#                          --MODELO 1 - SECTORIAL --
print("-----------------------------------------------------------------------------------------------------------------------------------------------------")
# - VIF
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Reset índice y limpiar NAs
df = data_panel_sector.reset_index()
vars_reg = ['IA_Adoption', 'GDPCap', 'Educ_Rate', 'Unemp_Rate', 'NACE']
df = df.dropna(subset=vars_reg)

# 2. Dummies NACE e interacciones
nace_dummies = pd.get_dummies(df['NACE'], prefix='NACE', drop_first=True)
df_vif = pd.concat([df, nace_dummies], axis=1)
interaction_cols = []
for col in nace_dummies.columns:
    inter = f"{col}_IA"
    df_vif[inter] = df_vif['IA_Adoption'] * df_vif[col]
    interaction_cols.append(inter)

# 3. Montar X con solo numéricas
exog_cols = ['IA_Adoption', 'GDPCap', 'Educ_Rate', 'Unemp_Rate'] + nace_dummies.columns.tolist() + interaction_cols
X = df_vif[exog_cols]

# 4. Convertir todo a float y eliminar columnas que no se puedan convertir
X = X.apply(pd.to_numeric, errors='coerce')
X = X.dropna(axis=1, how='all')  # elimina cols que quedaron NaN completo
X = X.astype(float)

# 5. Añadir constante
X_const = sm.add_constant(X, has_constant='add')

# 6. Calcular VIF
vif_data = []
for i, col in enumerate(X_const.columns):
    vif = variance_inflation_factor(X_const.values, i)
    vif_data.append({'Variable': col, 'VIF': vif})
vif_df = pd.DataFrame(vif_data)

print(vif_df)
print("-----------------------------------------------------------------------------------------------------------------------------------------------------")
# - Breusch-Pagan en panel (residuos FE)

import numpy as np
from statsmodels.stats.diagnostic import het_breuschpagan

# 1. Extrae residuos como array 1D
resid = results_sector.resids.values

# 2. Extrae exógenas como array 2D (sin constante)
exog_raw = results_sector.model.exog.values2d

# 3. Añade manualmente una columna de unos al principio
ones = np.ones((exog_raw.shape[0], 1))
exog_bp = np.hstack([ones, exog_raw])

# 4. Ejecuta la prueba de Breusch–Pagan
bp_stat, bp_pvalue, f_stat, f_pvalue = het_breuschpagan(resid, exog_bp)

print(f"Breusch–Pagan LM statistic: {bp_stat:.3f}, p-value: {bp_pvalue:.3f}")


#                          --MODELO 2 - OCUPACIONAL --
print("-----------------------------------------------------------------------------------------------------------------------------------------------------")
# - VIF
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Reset del índice para tener Country, IdYear e ISCO_08 como columnas
df = data_occupation.reset_index()

# 2. Eliminar filas con NA en las variables base
vars_reg = ['IA_Adoption', 'GDPCap', 'Educ_Rate', 'Unemp_Rate', 'ISCO_08']
df = df.dropna(subset=vars_reg)

# 3. Crear dummies para ISCO_08 (sin la primera categoría)
isco_dummies = pd.get_dummies(df['ISCO_08'], prefix='ISCO', drop_first=True)

# 4. Unir esas dummies al DataFrame
df_vif = pd.concat([df, isco_dummies], axis=1)

# 5. Construir la lista de dummies efectivas
isco_cols = isco_dummies.columns.tolist()

# 6. Crear interacciones IA × cada dummy de ISCO
interaction_cols = []
for col in isco_cols:
    inter = f'{col}_IA'
    df_vif[inter] = df_vif[col] * df_vif['IA_Adoption']
    interaction_cols.append(inter)

# 7. Definir todas las variables explicativas
exog_cols = (
    ['IA_Adoption', 'GDPCap', 'Educ_Rate', 'Unemp_Rate']
    + isco_cols
    + interaction_cols
)

# 8. Construir la matriz X y convertir todo a float
X = df_vif[exog_cols].apply(pd.to_numeric, errors='coerce')
X = X.dropna(axis=1, how='all').astype(float)

# 9. Añadir constante
X_const = sm.add_constant(X, has_constant='add')

# 10. Calcular VIF para cada variable
vif_data = []
for i, col in enumerate(X_const.columns):
    vif = variance_inflation_factor(X_const.values, i)
    vif_data.append({'Variable': col, 'VIF': vif})
vif_df = pd.DataFrame(vif_data)

print(vif_df)
print("-----------------------------------------------------------------------------------------------------------------------------------------------------")
# - Breusch-Pagan en panel (residuos FE)

from statsmodels.stats.diagnostic import het_breuschpagan
import numpy as np

resid_occ = results_occ.resids.values
exog_occ  = np.hstack([np.ones((results_occ.model.exog.values2d.shape[0], 1)),
                       results_occ.model.exog.values2d])

bp_stat_occ, bp_pvalue_occ, _, _ = het_breuschpagan(resid_occ, exog_occ)
print(f"BP ocupacional:  LM={bp_stat_occ:.3f},  p={bp_pvalue_occ:.3f}")
print("-----------------------------------------------------------------------------------------------------------------------------------------------------")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                       --- ANALISIS DESCRIPTIVO ---
#                          --MODELO 1 - SECTORIAL --
data_sector

# Selecciona las variables clave para el descriptivo
variables = ['IA_Adoption', 'Share_Sector', 'GDPCap', 'Educ_Rate', 'Unemp_Rate']

# Calcula los estadísticos descriptivos básicos (media, desv. estándar, mínimo, máximo)
tabla_descriptiva = data_sector[variables].describe().T[['mean', 'std', 'min', 'max']]
tabla_descriptiva = tabla_descriptiva.rename(columns={
    'mean': 'Media',
    'std': 'Desv. Est.',
    'min': 'Mínimo',
    'max': 'Máximo'
})
print(tabla_descriptiva)

import matplotlib.pyplot as plt
import seaborn as sns

# Histograma de IA_Adoption
plt.figure(figsize=(6,4))
sns.histplot(data_sector['IA_Adoption'], bins=20, kde=True)
plt.title('Distribución de IA_Adoption (Sectorial)')
plt.xlabel('IA_Adoption')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# Histograma de Share_Sector (variable dependiente)
plt.figure(figsize=(6,4))
sns.histplot(data_sector['Share_Sector'], bins=20, kde=True)
plt.title('Distribución de Share_Sector')
plt.xlabel('Share_Sector')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# Observaciones por sector

counts = data_sector['NACE'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(counts.index, counts.values)
ax.set_title('Número de Observaciones por Sector Agregado (NACE)')
ax.set_xlabel('Sector (NACE)')
ax.set_ylabel('Número de Observaciones')
plt.tight_layout()
plt.show()

counts = data_IA['NACE'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(counts.index, counts.values)
ax.set_title('Número de Observaciones por Sector (NACE)')
ax.set_xlabel('Sector (NACE)')
ax.set_ylabel('Número de Observaciones')
plt.tight_layout()
plt.show()

# Crecimiento IA
import matplotlib.pyplot as plt

# 1. Pivot para tener columnas separadas de IA_Adoption en 2023 y 2024
df_wide = (
    data_sector
    .pivot_table(index=['Country', 'NACE'],
                 columns='IdYear',
                 values='IA_Adoption',
                 aggfunc='first')
    .reset_index()
    .rename(columns={23: 'IA_2023', 24: 'IA_2024'})
)

# 2. Eliminar filas con algún NA
df_wide = df_wide.dropna(subset=['IA_2023', 'IA_2024'])

# 3. Gráfica scatter mostrando cada observación
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_wide['IA_2023'], df_wide['IA_2024'])
# Línea 45° para ver crecimiento
lim = max(df_wide['IA_2023'].max(), df_wide['IA_2024'].max())
ax.plot([0, lim], [0, lim], linestyle='--')
ax.set_title('IA_Adoption: 2023 vs 2024 por Sector')
ax.set_xlabel('IA_Adoption 2023')
ax.set_ylabel('IA_Adoption 2024')
plt.tight_layout()
plt.show()
print("-----------------------------------------------------------------------------------------------------------------------------------------------------")
#                          --MODELO 2 - OCUPACIONAL --
data_occupation

# Selecciona las variables clave para el descriptivo
variables_ocup = ['IA_Adoption', 'Share_Occupation', 'GDPCap', 'Educ_Rate', 'Unemp_Rate','Share_Sector']

# Calcula los estadísticos descriptivos básicos (media, desv. estándar, mínimo, máximo)
tabla_descriptiva_ocup = data_occupation[variables_ocup].describe().T[['mean', 'std', 'min', 'max']]
tabla_descriptiva_ocup = tabla_descriptiva_ocup.rename(columns={
    'mean': 'Media',
    'std': 'Desv. Est.',
    'min': 'Mínimo',
    'max': 'Máximo'
})
print(tabla_descriptiva_ocup)
data_occupation
data_occupation[data_occupation['Share_Occupation'] == 1] #dejamos de momento estos

import matplotlib.pyplot as plt
import seaborn as sns

# Histograma de IA_Adoption
plt.figure(figsize=(6,4))
sns.histplot(data_occupation['IA_Adoption'], bins=20, kde=True)
plt.title('Distribución de IA_Adoption (Ocupacional)')
plt.xlabel('IA_Adoption')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# Histograma de Share_Occupation (variable dependiente)
plt.figure(figsize=(6,4))
sns.histplot(data_occupation['Share_Occupation'], bins=20, kde=True)
plt.title('Distribución de Share_Occupation')
plt.xlabel('Share_Occupation')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

print("-----------------------------------------------------------------------------------------------------------------------------------------------------")
#Proporciones de ocupaciones

# 1. Agrupa por sector (NACE) y ocupación (ISCO_08), calculando la media de Share_Occupation
df_group = (
    data_occupation
    .groupby(['NACE', 'ISCO_08'], as_index=False)['Share_Occupation']
    .mean()
)

# 2. Pivot a formato ancho: filas = NACE, columnas = OC1…OC9
tabla = (
    df_group
    .pivot(index='NACE', columns='ISCO_08', values='Share_Occupation')
    .fillna(0)
)

# 3. Asegura que las columnas estén en orden OC1…OC9
cols = [f'OC{i}' for i in range(1, 10)]
tabla = tabla.reindex(columns=cols, fill_value=0)

# 4. Añade columna Total que suma las proporciones por fila
tabla['Total'] = tabla.sum(axis=1)

print(tabla.round(2))
print("-----------------------------------------------------------------------------------------------------------------------------------------------------")
#                          --MODELO 3 - TEORICO --

data_theory
# Selecciona las variables clave para el descriptivo
variables = ['IA_Adoption', 'Share_Sector', 'GDPCap', 'Educ_Rate', 'Unemp_Rate','RTI', 'NRA', 'NRI', 'RC', 'RM', 'NRM', 'Mean_RTI']

# Calcula los estadísticos descriptivos básicos (media, desv. estándar, mínimo, máximo)
tabla_descriptiva = data_theory[variables].describe().T[['mean', 'std', 'min', 'max']]
tabla_descriptiva = tabla_descriptiva.rename(columns={
    'mean': 'Media',
    'std': 'Desv. Est.',
    'min': 'Mínimo',
    'max': 'Máximo'
})
print(tabla_descriptiva)
print("-----------------------------------------------------------------------------------------------------------------------------------------------------")
import matplotlib.pyplot as plt
import seaborn as sns

# Histograma de IA_Adoption
plt.figure(figsize=(6,4))
sns.histplot(data_sector['IA_Adoption'], bins=20, kde=True)
plt.title('Distribución de IA_Adoption (Sectorial)')
plt.xlabel('IA_Adoption')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# Histograma de Share_Sector (variable dependiente)
plt.figure(figsize=(6,4))
sns.histplot(data_sector['Share_Sector'], bins=20, kde=True)
plt.title('Distribución de Share_Sector')
plt.xlabel('Share_Sector')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()


# Supón que data_sector ya está cargado como DataFrame
df_counts = data_sector['Country'].value_counts() \
    .rename_axis('Country') \
    .reset_index(name='Observations')

print(df_counts)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dowload csv:
data_sector.to_csv('data_sector.csv', index=True)
data_occupation.to_csv('data_occupation.csv',index=True)
data_theory.to_csv('data_theory.csv', index=True)




