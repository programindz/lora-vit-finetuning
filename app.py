import gradio as gr
import PIL
from inference import inference_model, predict_image_class

def predict(adaptor, image):
    if not image or not adaptor:
        return "Check whether you have selected classification type or uploaded image"
    model, image_processor = inference_model(adaptor)
    return predict_image_class(image, model, image_processor)



if __name__ == "__main__":
    app = gr.Interface(
        fn = predict,
        inputs=[
            gr.Dropdown(choices=['Food', 'Human Actions'], label="Classification Type"),
            gr.Image(type='pil', label="Upload Image")
        ],
        outputs=gr.Textbox(label="Predicted Class"),
        title="LoRA Adaptor based Classification",
        description="Choose a LoRA adaptor for particular type of classification"
    )
    app.launch()