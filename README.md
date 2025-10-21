#  Dashboard Montacargas

**Proyecto desarrollado durante la práctica en Renault Sofasa (Bodega CKD)**  
Automatiza la exportación de datos desde SAP y genera un dashboard en **Python (Streamlit)** para el análisis de indicadores logísticos de montacargas.

---

## 🎯 Objetivo
Centralizar la información operativa proveniente de SAP y archivos Excel, transformándola en indicadores visuales para la toma de decisiones en tiempo real.

---

## 🧠 Tecnologías utilizadas
- Python 3.11  
- Streamlit  
- Pandas / SQLite  
- PowerShell y VBScript (automatización SAP)  
- Excel para respaldo de datos

---

## ⚙️ Estructura
app.py # Dashboard principal
rules.py, metrics.py # Módulos de lógica y visualización
style.py, widgets.py # Componentes visuales
requirements.txt # Dependencias

---

## ▶️ Ejecución
1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt

Ejecutar el dashboard:
   streamlit run app.py

