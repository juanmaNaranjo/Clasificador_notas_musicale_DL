import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import cv2

from muscima.io import parse_cropobject_list
from skimage.transform import resize
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import invert

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


CROPOBJECT_DIR = r'Clasificador_notas_musicale_DL\MUSCIMA-pp_v1.0\v1.0\data\cropobjects_manual'

# Parámetros de imagen
TARGET_SIZE = (64, 64)
MIN_SAMPLES = 100        
BATCH_SIZE = 32
EPOCHS = 1000             
RANDOM_STATE = 42

# Salidas
OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# Símbolos de interés
SYMBOLS_TO_CLASSIFY = [
    'notehead-full', 'notehead-empty',
    'sharp', 'flat', 'natural',
    'rest-quarter', 'rest-half', 'rest-whole',
    'g-clef', 'f-clef',
    'time-signature',
    'barline'
]

def load_docs(cropobject_dir):
    fnames = [os.path.join(cropobject_dir, f) for f in os.listdir(cropobject_dir) if f.lower().endswith('.xml')]
    docs = [parse_cropobject_list(f) for f in fnames]
    return docs

def extract_symbols_from_doc(cropobjects):
    _cropobj_dict = {c.objid: c for c in cropobjects}
    symbols = []
    for c in cropobjects:
        if c.clsname in SYMBOLS_TO_CLASSIFY:
            if c.clsname.startswith('notehead'):
                stem_obj = None
                for o in c.outlinks:
                    _o_obj = _cropobj_dict.get(o)
                    if _o_obj and _o_obj.clsname == 'stem':
                        stem_obj = _o_obj
                        break
                if stem_obj:
                    symbols.append(([c, stem_obj], c.clsname))
                else:
                    symbols.append(([c], c.clsname))
            else:
                symbols.append(([c], c.clsname))
    return symbols

print("Cargando documentos...")
docs = load_docs(CROPOBJECT_DIR)

# Extraer todos los símbolos
all_symbols = list(itertools.chain(*[extract_symbols_from_doc(doc) for doc in docs]))

# Contar por clase
symbol_counts = {}
for cropobj_list, cls in all_symbols:
    symbol_counts[cls] = symbol_counts.get(cls, 0) + 1

print("Conteo de símbolos por clase (total):")
for k, v in symbol_counts.items():
    print(f"  {k}: {v}")

# Filtrar clases con suficientes muestras
valid_classes = {k for k, v in symbol_counts.items() if v >= MIN_SAMPLES}
print(f"\nClases con >= {MIN_SAMPLES} muestras: {valid_classes}")

filtered_symbols = [(objs, cls) for objs, cls in all_symbols if cls in valid_classes]

def get_image_from_cropobjects(cropobjects, margin=5, context=10):
    top = min([c.top for c in cropobjects]) - context
    left = min([c.left for c in cropobjects]) - context
    bottom = max([c.bottom for c in cropobjects]) + context
    right = max([c.right for c in cropobjects]) + context

    top = max(0, top)
    left = max(0, left)

    height = bottom - top + 2 * margin
    width = right - left + 2 * margin
    if height <= 0 or width <= 0:
        raise ValueError("Bounding box inválido.")

    canvas = np.zeros((height, width), dtype='float32')

    for c in cropobjects:
        _pt = c.top - top + margin
        _pl = c.left - left + margin
        mask_h = min(c.height, height - _pt)
        mask_w = min(c.width, width - _pl)
        if mask_h > 0 and mask_w > 0:
            canvas[_pt:_pt+mask_h, _pl:_pl+mask_w] += c.mask[:mask_h, :mask_w].astype('float32')

    canvas[canvas > 0] = 1.0
    return canvas

def preprocess_image(img, target_size=TARGET_SIZE):
    h, w = img.shape
    if h == 0 or w == 0:
        return np.zeros(target_size, dtype='float32')
    scale = min(target_size[0]/h, target_size[1]/w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    resized = resize(img, (new_h, new_w), anti_aliasing=True)
    canvas = np.zeros(target_size, dtype='float32')
    y_start = (target_size[0] - new_h) // 2
    x_start = (target_size[1] - new_w) // 2
    canvas[y_start:y_start+new_h, x_start:x_start+new_w] = resized
    return canvas

images = []
labels = []
errors = 0
for crop_objs, cls in filtered_symbols:
    try:
        img = get_image_from_cropobjects(crop_objs)
        img_p = preprocess_image(img)
        images.append(img_p)
        labels.append(cls)
    except Exception as e:
        errors += 1
print(f"Dataset generado: {len(images)} imágenes (errores: {errors})")

X = np.array(images)  
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
y = to_categorical(y_encoded)


X_train, X_test, y_train, y_test, y_train_idx, y_test_idx = train_test_split(
    X, y, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)


X_train_cnn = X_train[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]

print("Formas de datos:")
print("  X_train:", X_train_cnn.shape, " y_train:", y_train.shape)
print("  X_test:", X_test_cnn.shape, " y_test:", y_test.shape)
print("Clases:", label_encoder.classes_)


datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train_cnn)


def build_cnn(input_shape, n_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape, padding='same', name='conv_1'),
        MaxPooling2D((2,2), name='mp_1'),
        Conv2D(64, (3,3), activation='relu', padding='same', name='conv_2'),
        MaxPooling2D((2,2), name='mp_2'),
        Conv2D(128, (3,3), activation='relu', padding='same', name='conv_3'),
        MaxPooling2D((2,2), name='mp_3'),
        Flatten(name='flatten'),
        Dense(256, activation='relu', name='dense_1'),
        Dropout(0.5, name='dropout'),
        Dense(n_classes, activation='softmax', name='predictions')
    ])
    return model

n_classes = len(label_encoder.classes_)
model = build_cnn(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 1), n_classes=n_classes)
model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

start_time = time.time()
history = model.fit(
    datagen.flow(X_train_cnn, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=max(1, len(X_train_cnn) // BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(X_test_cnn, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
train_time = time.time() - start_time
print(f"Tiempo entrenamiento: {train_time:.2f} s")


plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Loss')
plt.savefig(os.path.join(OUT_DIR, 'training_loss.png'))
plt.close()

plt.figure()
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title('Accuracy')
plt.savefig(os.path.join(OUT_DIR, 'training_acc.png'))
plt.close()


test_loss, test_acc = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"Precisión en test: {test_acc:.4f}")

y_pred_prob = model.predict(X_test_cnn)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# Reporte de clasificación
report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
print("Reporte de clasificación:\n", report)
with open(os.path.join(OUT_DIR, 'classification_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Matriz de Confusión - CNN')
plt.ylabel('Verdadero')
plt.xlabel('Predicho')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix_cnn.png'))
plt.close()

# Guardar modelo
MODEL_PATH = os.path.join(OUT_DIR, 'symbol_cnn.keras')
model.save(MODEL_PATH)
print(f"Modelo guardado en: {MODEL_PATH}")


def find_last_conv_layer(model):
    # devuelve el nombre de la última capa Conv2D en el modelo
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No se encontró Conv2D en el modelo.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
   
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.outputs[0]]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros_like(heatmap.numpy())
    heatmap /= max_val
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, out_path, alpha=0.4):
    
    # img: array HxW en rango [0,1]
    hmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    hmap_uint8 = np.uint8(255 * hmap)
    hmap_color = cv2.applyColorMap(hmap_uint8, cv2.COLORMAP_JET)
    # convertir img a 3 canales uint8
    img_rgb = np.uint8(img * 255)
    img_rgb_3 = np.stack([img_rgb]*3, axis=-1)
    superimposed = cv2.addWeighted(img_rgb_3, 1-alpha, hmap_color, alpha, 0)
    cv2.imwrite(out_path, superimposed)
    return superimposed


last_conv = find_last_conv_layer(model)
print("Última capa conv detectada:", last_conv)


N_PER_CLASS = 3
examples = []  


from collections import defaultdict
idx_by_class = defaultdict(list)
for i, lbl in enumerate(y_true):
    idx_by_class[lbl].append(i)

for cls_idx, idxs in idx_by_class.items():
    for i in idxs[:N_PER_CLASS]:
        examples.append(i)


gradcam_folder = os.path.join(OUT_DIR, 'gradcams')
os.makedirs(gradcam_folder, exist_ok=True)

for i in examples:
    img = X_test[i]            
    img_cnn = X_test_cnn[i:i+1] 
    pred_probs = model.predict(img_cnn)
    pred_idx = np.argmax(pred_probs[0])
    model.predict(np.zeros((1, 64, 64, 1)))
    heatmap = make_gradcam_heatmap(img_cnn, model, last_conv, pred_idx)
    out_file = os.path.join(gradcam_folder, f"gradcam_idx{i}_true{y_true[i]}_pred{pred_idx}.png")
    save_and_display_gradcam(img, heatmap, out_file)
    print(f"Grad-CAM guardado: {out_file}")


def predict_single_image(path, model, encoder, size=TARGET_SIZE):
    img = imread(path)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = rgb2gray(img)
    if np.mean(img) < 0.5:
        img = invert(img)
    img_p = preprocess_image(img, target_size=size)
    img_cnn = img_p[np.newaxis, ..., np.newaxis]
    probs = model.predict(img_cnn)
    pred = np.argmax(probs[0])
    pred_label = encoder.inverse_transform([pred])[0]
    last_conv = find_last_conv_layer(model)
    heatmap = make_gradcam_heatmap(img_cnn, model, last_conv, pred)
    out_path = os.path.join(OUT_DIR, f"pred_{os.path.basename(path)}_gradcam.png")
    save_and_display_gradcam(img_p, heatmap, out_path)
    return pred_label, out_path


'''ejemplo_path = r'imagen1.jpg'
if os.path.exists(ejemplo_path):
    plabel, gout = predict_single_image(ejemplo_path, model, label_encoder)
    print("Predicción:", plabel, "GradCAM guardado en:", gout)

print("Script finalizado. Revisa la carpeta 'outputs' para resultados (modelos, imágenes y reportes).")
'''


FOLDER_PATH = r"Clasificador_notas_musicale_DL\Prueba"

if os.path.exists(FOLDER_PATH) and os.path.isdir(FOLDER_PATH):
    # Iterar por cada archivo dentro de la carpeta
    for filename in os.listdir(FOLDER_PATH):
        image_path = os.path.join(FOLDER_PATH, filename)
        
        # Validar que sea una imagen por extensión
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            try:
                plabel, gout = predict_single_image(image_path, model, label_encoder)
                print(f"Imagen: {filename}")
                print(f"  → Predicción: {plabel}")
                print(f"  → GradCAM guardado en: {gout}")
                print("-" * 50)
            except Exception as e:
                print(f"Error procesando {filename}: {e}")
else:
    print("La ruta especificada no existe o no es una carpeta válida.")

print("Procesamiento finalizado. Revisa la carpeta 'outputs' para resultados (modelos, imágenes y reportes).")
