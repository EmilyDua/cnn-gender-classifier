# 🧠 CNN Gender Classification con XAI

Proyecto de clasificación de género usando Redes Neuronales Convolucionales (CNN) con técnicas de interpretabilidad explicable (XAI).

## 📋 Descripción

Este proyecto implementa una CNN para clasificar rostros en dos categorías: masculino y femenino, utilizando el dataset **Male and Female Faces** de Kaggle. Además, integra técnicas de interpretabilidad visual (Saliency Maps y Grad-CAM) para entender las decisiones del modelo.

## 🎯 Características

- ✅ Clasificación binaria de género con CNN
- ✅ Técnicas de interpretabilidad (XAI):
  - Saliency Maps
  - Grad-CAM (Gradient-weighted Class Activation Mapping)
- ✅ Aplicación web interactiva con Streamlit
- ✅ Análisis completo de métricas y visualizaciones
- ✅ Comparación de múltiples arquitecturas

## 🛠️ Tecnologías

- Python 3.8+
- TensorFlow 2.15
- Streamlit
- OpenCV
- NumPy, Pandas, Matplotlib, Seaborn

## 📦 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/cnn-gender-classifier.git
cd cnn-gender-classifier
```

### 2. Crear entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Organizar el dataset

Descarga el dataset desde [Kaggle](https://www.kaggle.com/datasets/ashwingupta3012/male-and-female-faces-dataset) y coloca la carpeta `male-and-female-faces-dataset` en la raíz del proyecto.

Estructura esperada:
```
male-and-female-faces-dataset/
├── Male Faces/
│   ├── imagen1.jpg
│   ├── imagen2.jpg
│   └── ...
└── Female Faces/
    ├── imagen1.jpg
    ├── imagen2.jpg
    └── ...
```

### 5. Configurar estructura del proyecto

```bash
python setup_project.py
```

## 🚀 Uso

### Entrenamiento del Modelo

Ejecuta el script principal que incluye todos los ejercicios (1-5):

```bash
# Si usas Jupyter
jupyter notebook laboratorio_cnn_xai.ipynb

# Si usas VS Code con extensión de Python
# Abre laboratorio_cnn_xai.py y ejecuta las celdas
```

El script realizará:
1. Organización del dataset
2. Exploración y visualización
3. Preprocesamiento y partición
4. Construcción y entrenamiento de CNN
5. Ajuste de hiperparámetros
6. Generación de mapas de interpretabilidad

### Aplicación Streamlit

Una vez entrenado el modelo:

```bash
streamlit run streamlit_app.py
```

La aplicación se abrirá en `http://localhost:8501`

## 📊 Estructura del Proyecto

```
proyecto/
├── data/
│   ├── male/           # Imágenes masculinas
│   └── female/         # Imágenes femeninas
├── models/
│   ├── model.keras     # Modelo principal
│   └── model_v2.keras  # Modelo alternativo
├── visualizations/     # Gráficos y análisis
├── laboratorio_cnn_xai.py    # Script principal (Ejercicios 1-5)
├── streamlit_app.py    # Aplicación web (Ejercicio 6)
├── requirements.txt    # Dependencias
├── setup_project.py    # Configuración inicial
└── README.md          # Este archivo
```

## 🎓 Ejercicios del Laboratorio

### Ejercicio 1: Exploración del Dataset
- Organización de carpetas
- Análisis estadístico
- Visualización de muestras

### Ejercicio 2: Preprocesamiento
- Redimensionamiento a 224x224
- Normalización [0, 1]
- Partición: 70% train, 15% val, 15% test

### Ejercicio 3: Construcción de CNN
- Arquitectura: 3 bloques Conv + MaxPool
- Batch Normalization y Dropout
- Entrenamiento con callbacks

### Ejercicio 4: Hiperparámetros
- Comparación de arquitecturas
- Análisis de métricas
- Selección del mejor modelo

### Ejercicio 5: Interpretabilidad (XAI)
- Saliency Maps
- Grad-CAM
- Visualización de regiones importantes

### Ejercicio 6: Despliegue con Streamlit
- Aplicación web interactiva
- Predicción en tiempo real
- Visualización de mapas XAI

## 📈 Resultados Esperados

- **Accuracy:** > 85% en conjunto de test
- **Interpretabilidad:** Visualización clara de regiones faciales importantes
- **Aplicación:** Interfaz funcional y desplegable en la nube

## 🌐 Despliegue en Streamlit Cloud

### Preparación

1. Asegúrate de que el modelo pesa < 100 MB
2. Verifica que todos los archivos estén en el repositorio

### Pasos para desplegar

1. Sube el proyecto a GitHub
2. Ve a [Streamlit Cloud](https://share.streamlit.io)
3. Conecta tu repositorio
4. Configura:
   - Main file: `streamlit_app.py`
   - Python version: 3.9
5. Despliega

Tu app estará disponible en: `https://tu-usuario-proyecto.streamlit.app`

## 🔧 Solución de Problemas

### Error al cargar el modelo
```bash
# Verifica que el archivo existe
ls models/model.keras

# Verifica la versión de TensorFlow
pip show tensorflow
```

### Error de memoria
```bash
# Reduce el batch size en el entrenamiento
BATCH_SIZE = 16  # en lugar de 32
```

### Error con OpenCV
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

## 📚 Referencias

- Dataset: [Male and Female Faces Dataset](https://www.kaggle.com/datasets/ashwingupta3012/male-and-female-faces-dataset)
- Grad-CAM Paper: [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- Streamlit: [Documentación oficial](https://docs.streamlit.io)

## 👥 Autor

Desarrollado como parte del Laboratorio CNNs-XAI

## 📄 Licencia

Este proyecto es con fines educativos.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Haz fork del proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## ⚠️ Notas Importantes

- El modelo está entrenado con imágenes de rostros. Los resultados pueden variar según la calidad de la imagen.
- Las técnicas XAI proporcionan interpretabilidad pero no garantizan explicaciones perfectas.
- Este proyecto es para fines educativos y de investigación.

## 📞 Soporte

Si tienes problemas o preguntas, abre un issue en GitHub.

---

**¡Buena suerte con tu proyecto! 🚀**
