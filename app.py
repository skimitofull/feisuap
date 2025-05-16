import streamlit as st
from PIL import Image
import numpy as np
import face_recognition

def replace_faces(base_img, face_img, scale_factor, x_offset, y_offset):
    # Convertir imágenes a arrays numpy
    base_img_np = np.array(base_img)
    face_img_np = np.array(face_img)

    # Detectar ubicaciones de caras en la imagen base
    face_locations = face_recognition.face_locations(base_img_np)

    if not face_locations:
        st.warning("No se detectaron caras en la imagen base.")
        return base_img

    # Redimensionar la imagen de la cara a poner para que encaje en cada cara detectada
    for (top, right, bottom, left) in face_locations:
        face_width = right - left
        face_height = bottom - top

        # Calcular el nuevo tamaño de la cara basada en el factor de escala
        new_face_width = int(face_width * scale_factor)
        new_face_height = int(face_height * scale_factor)

        # Redimensionar la cara a poner
        face_resized = face_img.resize((new_face_width, new_face_height))
        face_resized_np = np.array(face_resized)

        # Calcular la posición ajustada con los offsets
        x_pos = left + x_offset
        y_pos = top + y_offset

        # Asegurarse de que la cara no se salga de los límites de la imagen
        x_pos = max(0, min(x_pos, base_img_np.shape[1] - new_face_width))
        y_pos = max(0, min(y_pos, base_img_np.shape[0] - new_face_height))

        # Reemplazar la cara en la imagen base
        base_img_np[y_pos:y_pos + new_face_height, x_pos:x_pos + new_face_width] = face_resized_np

    # Convertir de nuevo a imagen PIL
    result_img = Image.fromarray(base_img_np)
    return result_img

st.title("Reemplazo de caras en imágenes")

uploaded_base = st.file_uploader("Sube la imagen base", type=["jpg", "jpeg", "png"])
uploaded_face = st.file_uploader("Sube la imagen de la cara para reemplazar", type=["jpg", "jpeg", "png"])

if uploaded_base and uploaded_face:
    base_img = Image.open(uploaded_base).convert("RGB")
    face_img = Image.open(uploaded_face).convert("RGB")

    st.image(base_img, caption="Imagen base", use_column_width=True)
    st.image(face_img, caption="Imagen de la cara a poner", use_column_width=True)

    # Controles deslizantes para ajustar la escala y la posición
    scale_factor = st.slider("Factor de escala", min_value=0.1, max_value=2.0, value=1.0, step=0.05)
    x_offset = st.slider("Desplazamiento horizontal", min_value=-100, max_value=100, value=0, step=1)
    y_offset = st.slider("Desplazamiento vertical", min_value=-100, max_value=100, value=0, step=1)

    if st.button("Reemplazar caras"):
        result = replace_faces(base_img, face_img, scale_factor, x_offset, y_offset)
        st.image(result, caption="Imagen con caras reemplazadas", use_column_width=True)