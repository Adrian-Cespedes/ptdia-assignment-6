import gradio as gr
import requests
import numpy as np
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000/predict/"

def predict(img):
    img = img["composite"]
    # Convertir canvas del Sketchpad a PNG y mandarlo a FastAPI
    if img is not None and img.size > 0:
        # Convertir el array numpy a imagen PIL y redimensionar a 28x28
        if isinstance(img, np.ndarray):
            # Si viene como array numpy del Sketchpad
            pil_img = Image.fromarray(img.astype("uint8")).convert("L")
        else:
            # Si viene en otro formato
            pil_img = img.convert("L")
        
        # Redimensionar a 28x28 para el modelo
        pil_img = pil_img.resize((28, 28))
        
        # Convertir a bytes para enviar
        file_bytes = io.BytesIO()
        pil_img.save(file_bytes, format="PNG")
        file_bytes.seek(0)

        try:
            response = requests.post(API_URL, files={"image": ("digit.png", file_bytes, "image/png")})
            if response.status_code == 200:
                pred = response.json()["prediction"]
                return pred
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return f"Error en API: {response.status_code}"
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            return f"Error de conexión: {str(e)}"
        except KeyError as e:
            print(f"Response parsing error: {e}")
            print(f"Response content: {response.text}")
            return f"Error en respuesta: {str(e)}"
    return "No input"

with gr.Blocks() as demo:
    gr.Markdown("# ✍️ Reconocimiento de Dígitos (Frontend Test)")

    with gr.Row():
        with gr.Column():
            brush = gr.Brush(default_size=8,colors=["black"])
            im = gr.Sketchpad(
                type="numpy",
                image_mode="L",
                brush=brush,
                fixed_canvas=True,
                canvas_size=(280, 280),
                label="Dibuja un dígito (blanco sobre negro)"
            )
            btn = gr.Button("Enviar a FastAPI")
        with gr.Column():
            output = gr.Label(label="Dígito detectado", num_top_classes=1)

    btn.click(predict, inputs=[im], outputs=[output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
