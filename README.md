# 📘 Clasificador de Símbolos Musicales con CNN  
Este proyecto implementa una **Red Neuronal Convolucional (CNN)** para clasificar símbolos musicales (banderines, silencios, etc.) a partir de imágenes. Además incluye generación de *Grad-CAM heatmaps* para visualizar qué zonas de la imagen influyen más en la predicción del modelo.

---
├── MUSIMA/ # Dataset MUSCIMA++
│
├── Prueba/ # Carpeta para pruebas de inferencia
│ ├── imagen1.jpg
│ └── ...
│
├── outputs/
│ ├── symbol_cnn.keras # Modelo entrenado
│ ├── label_encoder.pkl # Codificador de clases
│
├── deepv2dl.py # Pipeline principal (train + infer + gradcam)
├── README.md

---

# 🚀 **Características del Proyecto**

✔ Entrenamiento de una **CNN profunda** desde cero  
✔ Preprocesamiento automático: gris, invert, resize  
✔ Evaluación con métricas completas (precision, recall, F1-score)  
✔ Predicción de imágenes individuales o carpetas  
✔ Visualización **Grad-CAM** para interpretabilidad  
✔ Código compacto centralizado en **deepv2dl.py**

## 📊 Resultados Obtenidos

A continuación se presentan los resultados reales logrados por el modelo CNN entrenado sobre el dataset MUSCIMA++:

            precision    recall  f1-score   support

    f-clef       1.00      1.00      1.00        57
      flat       1.00      0.99      0.99       222
    g-clef       1.00      1.00      1.00        80
   natural       1.00      0.99      0.99       218

   
   notehead-empty 0.99     0.99     0.99       334
   notehead-full  1.00     1.00     1.00       4267
   sharp          0.99     1.00     1.00       414

✔ **Accuracy global: 1.00**  
✔ **Balance excelente por clase**  
✔ El modelo demuestra una **generalización sobresaliente**, apoyada en un preprocesamiento robusto y una arquitectura profunda bien ajustada.


