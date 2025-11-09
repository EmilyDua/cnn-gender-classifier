"""
Aplicación Streamlit para Clasificación de Género con Interpretabilidad (XAI)
Laboratorio CNNs - Male and Female Faces Dataset
"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
import io

# Configuración de la página
st.set_page_config(
    page_title="Gender Classification + XAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-male {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
    }
    .prediction-female {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Funciones de carga
@st.cache_resource
def load_model():
    """Carga el modelo entrenado"""
    try:
        model = keras.models.load_model('models/model.keras')
        
        # CRÍTICO: Construir el modelo con un input dummy
        dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = model(dummy_input, training=False)
        
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocesa la imagen para el modelo"""
    # Convertir PIL Image a array numpy
    img_array = np.array(image)
    
    # Asegurar que está en RGB
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Redimensionar
    img_resized = cv2.resize(img_array, target_size)
    
    # Normalizar
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Agregar dimensión de batch
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, img_normalized

def get_saliency_map(model, image):
    """Genera Saliency Map"""
    image_tensor = tf.convert_to_tensor(image)
    
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        predictions = model(image_tensor)
        loss = predictions[0][0]
    
    gradients = tape.gradient(loss, image_tensor)
    saliency = tf.reduce_max(tf.abs(gradients), axis=-1)
    saliency = saliency.numpy()[0]
    
    # Normalizar
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    return saliency

def get_gradcam(model, image, layer_name=None):
    """Genera Grad-CAM - Compatible con Keras 3.x Sequential"""
    
    # Encontrar la última capa convolucional
    if layer_name is None:
        conv_layers = []
        for layer in model.layers:
            if 'conv2d' in layer.__class__.__name__.lower():
                conv_layers.append(layer.name)
        
        if not conv_layers:
            st.error("No se encontraron capas convolucionales")
            return None
        layer_name = conv_layers[-1]
    
    try:
        # Encontrar el índice de la capa
        target_layer_idx = None
        for idx, layer in enumerate(model.layers):
            if layer.name == layer_name:
                target_layer_idx = idx
                break
        
        if target_layer_idx is None:
            st.error(f"No se encontró la capa: {layer_name}")
            return None
        
        # Crear un modelo funcional RECONSTRUYENDO las capas
        # Input
        inputs = keras.Input(shape=(224, 224, 3))
        x = inputs
        
        # Aplicar capas hasta la capa objetivo
        for i in range(target_layer_idx + 1):
            x = model.layers[i](x)
        conv_output = x
        
        # Aplicar el resto de capas
        for i in range(target_layer_idx + 1, len(model.layers)):
            x = model.layers[i](x)
        final_output = x
        
        # Crear modelo funcional
        grad_model = keras.Model(inputs=inputs, outputs=[conv_output, final_output])
        
        # Convertir imagen a tensor
        img_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        
        # Calcular gradientes
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            loss = predictions[0, 0]
        
        # Gradientes
        grads = tape.gradient(loss, conv_outputs)
        
        if grads is None:
            st.error("No se pudieron calcular los gradientes")
            return None
        
        # Pooled gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weighted combination
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
        
        # Normalización
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        else:
            heatmap = tf.zeros_like(heatmap)
        
        return heatmap.numpy()
        
    except Exception as e:
        st.error(f"Error generando Grad-CAM: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def overlay_heatmap(image, heatmap, alpha=0.4):
    """Superpone heatmap sobre imagen"""
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    image_uint8 = np.uint8(255 * image)
    superimposed = cv2.addWeighted(image_uint8, 1-alpha, heatmap_colored, alpha, 0)
    
    return superimposed

# Header
st.markdown('<p class="main-header">🧠 Gender Classification with XAI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Clasificación de Género con Interpretabilidad Explicable</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/_static/logo.png", width=100)
    st.title("⚙️ Configuración")
    
    st.markdown("---")
    st.markdown("### 📊 Información del Modelo")
    st.info("""
    **Arquitectura:** CNN Secuencial
    - 3 bloques convolucionales
    - Batch Normalization
    - Dropout regularization
    - Clasificación binaria
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Clases")
    st.write("🔵 **Male** (Masculino)")
    st.write("🔴 **Female** (Femenino)")
    
    st.markdown("---")
    alpha = st.slider("Transparencia de Grad-CAM", 0.0, 1.0, 0.4, 0.05)
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Desarrollado por")
    st.write("Laboratorio CNNs-XAI")
    st.write("Dataset: Male and Female Faces")

# Cargar modelo
model = load_model()

if model is None:
    st.error("⚠️ No se pudo cargar el modelo. Asegúrate de que el archivo 'models/model.keras' existe.")
    st.stop()

st.success("✅ Modelo cargado exitosamente")

# Upload de imagen
st.markdown("---")
st.markdown("## 📤 Subir Imagen")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Selecciona una imagen de un rostro",
        type=['jpg', 'jpeg', 'png'],
        help="Formatos soportados: JPG, JPEG, PNG"
    )

if uploaded_file is not None:
    # Cargar y mostrar imagen original
    image = Image.open(uploaded_file)
    
    with col2:
        st.image(image, caption="Imagen cargada", use_column_width=True)
    
    # Preprocesar
    with st.spinner("🔄 Procesando imagen..."):
        img_batch, img_normalized = preprocess_image(image)
    
    # Predicción
    with st.spinner("🤔 Analizando..."):
        prediction = model.predict(img_batch, verbose=0)[0][0]
    
    # Mostrar predicción
    st.markdown("---")
    st.markdown("## 🎯 Resultado de la Predicción")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if prediction > 0.5:
            confidence = prediction * 100
            st.markdown(f'<div class="prediction-male">👨 MALE<br>{confidence:.2f}% confianza</div>', 
                       unsafe_allow_html=True)
        else:
            confidence = (1 - prediction) * 100
            st.markdown(f'<div class="prediction-female">👩 FEMALE<br>{confidence:.2f}% confianza</div>', 
                       unsafe_allow_html=True)
    
    # Métricas detalladas
    st.markdown("---")
    st.markdown("## 📊 Probabilidades Detalladas")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probabilidad Male", f"{prediction*100:.2f}%", 
                 delta=f"{(prediction-0.5)*100:.2f}%" if prediction > 0.5 else None)
    with col2:
        st.metric("Probabilidad Female", f"{(1-prediction)*100:.2f}%",
                 delta=f"{(0.5-prediction)*100:.2f}%" if prediction < 0.5 else None)
    
    # Barra de progreso
    st.progress(float(prediction))
    
    # Interpretabilidad
    st.markdown("---")
    st.markdown("## 🔍 Interpretabilidad Visual (XAI)")
    
    with st.spinner("🎨 Generando mapas de interpretabilidad..."):
        # Saliency Map
        saliency_map = get_saliency_map(model, img_batch)
        
        # Grad-CAM
        gradcam_heatmap = get_gradcam(model, img_batch)
        
        if gradcam_heatmap is not None:
            gradcam_overlay = overlay_heatmap(img_normalized, gradcam_heatmap, alpha=alpha)
    
    # Mostrar mapas
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Original", "🔥 Saliency Map", "🎯 Grad-CAM", "🔬 Grad-CAM Overlay"])
    
    with tab1:
        st.image(img_normalized, caption="Imagen Original Preprocesada", use_column_width=True)
        st.info("Esta es la imagen tal como la ve el modelo (224x224 píxeles, normalizada).")
    
    with tab2:
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(saliency_map, cmap='hot')
        ax.axis('off')
        ax.set_title('Saliency Map - Gradientes de Entrada', fontsize=14, fontweight='bold', pad=20)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
        plt.close(fig)
        st.info("""
        **Saliency Map:** Muestra los píxeles que más influyen en la predicción mediante el cálculo
        de gradientes. Las regiones más brillantes (rojas/amarillas) son las más importantes para la decisión.
        """)
    
    with tab3:
        if gradcam_heatmap is not None:
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(gradcam_heatmap, cmap='jet')
            ax.axis('off')
            ax.set_title('Grad-CAM - Mapa de Activación', fontsize=14, fontweight='bold', pad=20)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            plt.close(fig)
            st.info("""
            **Grad-CAM:** Visualiza las regiones de la imagen que activan fuertemente las capas
            convolucionales profundas. Indica qué partes del rostro son más relevantes.
            """)
    
    with tab4:
        if gradcam_heatmap is not None:
            st.image(gradcam_overlay, caption="Grad-CAM Superpuesto", use_column_width=True)
            st.info("""
            **Interpretación:** Las zonas rojas/amarillas indican las regiones del rostro que más
            contribuyeron a la clasificación. Típicamente: ojos, nariz, boca, estructura facial.
            """)
    
    # Análisis textual
    st.markdown("---")
    st.markdown("## 💡 Análisis de la Predicción")
    
    st.write(f"""
    ### Resultado: **{"Male" if prediction > 0.5 else "Female"}** ({confidence:.2f}% de confianza)
    
    El modelo ha analizado la imagen y ha determinado que el rostro corresponde a una persona 
    {"masculina" if prediction > 0.5 else "femenina"} con una confianza del {confidence:.2f}%.
    
    **Regiones clave identificadas:**
    - Los mapas de interpretabilidad muestran las áreas del rostro que más influyeron en esta decisión
    - Las zonas más resaltadas (colores cálidos) son las características faciales más determinantes
    - Esto permite entender *por qué* el modelo llegó a esta conclusión
    """)
    
    # Diferencia entre métodos
    with st.expander("📚 ¿Cuál es la diferencia entre Saliency Map y Grad-CAM?"):
        st.markdown("""
        ### Saliency Map
        - **Método:** Calcula los gradientes de la predicción respecto a los píxeles de entrada
        - **Ventaja:** Muestra píxeles individuales importantes con alta resolución
        - **Limitación:** Puede ser ruidoso y difícil de interpretar
        - **Uso:** Identifica detalles finos que afectan la decisión
        
        ### Grad-CAM (Gradient-weighted Class Activation Mapping)
        - **Método:** Pondera las activaciones de capas convolucionales con sus gradientes
        - **Ventaja:** Produce mapas más suaves y semánticamente significativos
        - **Limitación:** Menor resolución espacial
        - **Uso:** Muestra regiones generales importantes del rostro
        
        **Recomendación:** Usar ambos métodos en conjunto para obtener una interpretación completa.
        """)

else:
    # Instrucciones iniciales
    st.info("👆 Sube una imagen de un rostro para comenzar el análisis")
    
    st.markdown("---")
    st.markdown("## 🎓 Acerca de esta aplicación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Objetivo
        Esta aplicación demuestra:
        - Clasificación de género usando CNNs
        - Técnicas de interpretabilidad (XAI)
        - Despliegue de modelos con Streamlit
        """)
    
    with col2:
        st.markdown("""
        ### 🛠️ Tecnologías
        - TensorFlow/Keras
        - Streamlit
        - OpenCV
        - Grad-CAM & Saliency Maps
        """)
    
    st.markdown("---")
    st.markdown("### 📖 Cómo usar")
    st.markdown("""
    1. **Sube una imagen** usando el botón de arriba
    2. **Espera el análisis** - el modelo procesará la imagen
    3. **Revisa la predicción** - verás el género predicho y la confianza
    4. **Explora los mapas XAI** - entiende por qué el modelo hizo esa predicción
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Laboratorio CNNs - XAI</strong></p>
    <p>Dataset: Male and Female Faces Dataset (Ashwin Gupta, Kaggle)</p>
    <p>Técnicas de Interpretabilidad: Saliency Maps & Grad-CAM</p>
</div>
""", unsafe_allow_html=True)
