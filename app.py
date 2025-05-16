import streamlit as st
import cv2
import dlib
import numpy as np
from PIL import Image
import os

# Configuración inicial
st.set_page_config(page_title="Reemplazo de Caras", layout="wide")
st.title("Reemplazo de Caras en Imágenes")

# Descargar el modelo de dlib
def download_shape_predictor():
    import urllib.request
    predictor_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    predictor_path = "shape_predictor_68_face_landmarks.dat"

    if not os.path.exists(predictor_path):
        st.info("Descargando modelo de reconocimiento facial...")
        urllib.request.urlretrieve(predictor_url, predictor_path + ".bz2")

        import bz2
        with bz2.BZ2File(predictor_path + ".bz2", "rb") as f_in:
            with open(predictor_path, "wb") as f_out:
                f_out.write(f_in.read())
        os.remove(predictor_path + ".bz2")

    return predictor_path

# Función para detectar caras automáticamente
def detect_faces_auto(image, predictor_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    landmarks = []
    for face in faces:
        shape = predictor(gray, face)
        landmarks.append(np.array([[p.x, p.y] for p in shape.parts()], dtype=np.int32))

    return faces, landmarks

# Función para reemplazar caras
def replace_faces(image, replacement_img, faces, landmarks, mode="auto"):
    result = image.copy()

    for i, (face, landmark) in enumerate(zip(faces, landmarks)):
        if mode == "auto":
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        else:  # modo manual
            (x, y, w, h) = face

        # Redimensionar imagen de reemplazo
        replacement = cv2.resize(replacement_img, (w, h))

        # Crear máscara
        if mode == "auto":
            mask = np.zeros(replacement.shape[:2], dtype=np.uint8)
            cv2.convexHull(landmark, returnPoints=True)
            cv2.fillConvexPoly(mask, landmark, 255)
        else:
            mask = np.ones(replacement.shape[:2], dtype=np.uint8) * 255

        # Aplicar reemplazo
        replacement = cv2.bitwise_and(replacement, replacement, mask=mask)
        inv_mask = cv2.bitwise_not(mask)
        face_region = result[y:y+h, x:x+w]
        face_region = cv2.bitwise_and(face_region, face_region, mask=inv_mask)
        combined = cv2.add(face_region, replacement)
        result[y:y+h, x:x+w] = combined

    return result

# Interfaz de usuario
uploaded_file = st.file_uploader("Sube una imagen para detectar caras", type=["jpg", "jpeg", "png"])
replacement_file = st.file_uploader("Sube la imagen de reemplazo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and replacement_file is not None:
    # Cargar imágenes
    original_image = np.array(Image.open(uploaded_file))
    replacement_image = np.array(Image.open(replacement_file))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    replacement_image = cv2.cvtColor(replacement_image, cv2.COLOR_RGB2BGR)

    # Selección de modo
    mode = st.radio("Modo de operación:", ("Automático", "Manual"))

    if mode == "Automático":
        predictor_path = download_shape_predictor()
        faces, landmarks = detect_faces_auto(original_image, predictor_path)

        if len(faces) > 0:
            st.success(f"Se detectaron {len(faces)} caras en la imagen")
            result_image = replace_faces(original_image, replacement_image, faces, landmarks, "auto")
        else:
            st.warning("No se detectaron caras. Prueba el modo manual.")
            result_image = original_image.copy()
    else:  # Modo manual
        st.info("Selecciona la región de la cara en la imagen")

        # Mostrar imagen para selección
        display_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        st.image(display_image, caption="Selecciona la región de la cara", use_column_width=True)

        # Coordenadas de selección
        x = st.slider("Posición X", 0, original_image.shape[1], original_image.shape[1]//2)
        y = st.slider("Posición Y", 0, original_image.shape[0], original_image.shape[0]//2)
        w = st.slider("Ancho", 10, min(300, original_image.shape[1]-x), 150)
        h = st.slider("Alto", 10, min(300, original_image.shape[0]-y), 150)

        # Crear datos para el reemplazo
        faces = [(x, y, w, h)]
        landmarks = [np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])]  # Rectángulo simple
        result_image = replace_faces(original_image, replacement_image, faces, landmarks, "manual")

    # Mostrar resultados
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
    with col2:
        st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), caption="Modificada", use_column_width=True)

    # Opción para descargar
    st.download_button(
        label="Descargar imagen modificada",
        data=cv2.imencode(".jpg", result_image)[1].tobytes(),
        file_name="imagen_modificada.jpg",
        mime="image/jpeg"
    )
