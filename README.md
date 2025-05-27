Impacto de la IA en el Empleo Europeo (TFG)
Repositorio con el código y los datos necesarios para replicar el análisis de
“Impacto de la adopción de Inteligencia Artificial en la estructura del empleo europeo (2008–2023)”
de Eduardo Fernández Dionicio.

1. Clonar el repositorio
git clone https://github.com/eduardo-fd/IA_empleo_Europa_TFG.git
cd IA_empleo_Europa_TFG

2. Preparar el entorno
Recomendado: Python 3.10 o superior.

python -m venv venv

Linux/Mac
source venv/bin/activate

Windows
venv\Scripts\activate

pip install -r requirements.txt

3. Descargar los datos grandes (Eurostat – Abril 2025)
El archivo principal es SDMXE_2017-2024_v250409.mdb, que contiene la base de datos de Eurostat.

Accede a
https://ec.europa.eu/eurostat/web/digital-economy-and-society/database/comprehensive-database

Descarga el fichero SDMXE_2017-2024_v250409.mdb.

Sitúalo en la carpeta raíz del proyecto:

arduino
Copy
Edit
IA_empleo_Europa_TFG/
├── data_tfg.py
├── RTI_ISCO08_major_groups.xlsx
├── lfsa_eisn2.xlsx
├── "macro data.xlsx"
├── SDMXE_2017-2024_v250409.mdb   ← este archivo
├── requirements.txt
└── README.md

4. Ejecutar el análisis
python data_tfg.py
--input SDMXE_2017-2024_v250409.mdb
--output results/

Esto generará en results/ todos los paneles y salidas de los modelos estimados.

5. Resultados y outputs
results/panel.csv: panel final

results/modelos/: tablas de resultados de efectos fijos

results/figuras/: gráficos de evolución temporal

Contacto
Eduardo Fernández Dionicio
– GitHub: https://github.com/eduardo-fd

