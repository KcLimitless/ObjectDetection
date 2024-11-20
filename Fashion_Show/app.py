import gradio as gr
import torch
from PIL import Image
import matplotlib.pyplot as plt
import io
from model import load_model, get_val_transform  # Import functions from model.py
import numpy as np
<<<<<<< HEAD
=======

# Load the model on GPU if available
model = load_model(device=0 if torch.cuda.is_available() else -1)
>>>>>>> 16e22fbe27ebb36b6090c462c63a4d127310b2b8

val_transform = get_val_transform()

# Define colors for bounding boxes
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def get_output_figure(pil_img, results, threshold):
    plt.figure(figsize=(12, 8))
    plt.imshow(pil_img)
    ax = plt.gca()

    for result in results:
        score = result['score']
        label = result['label']
        box = list(result['box'].values())
        if score > threshold:
            color = COLORS[hash(label) % len(COLORS)]
            ax.add_patch(
                plt.Rectangle(
                    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                    fill=False, color=color, linewidth=2
                )
            )
            text = f'{label}: {score:.2f}'
            ax.text(
                box[0], box[1] - 5, text, fontsize=10,
                bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none')
            )
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close()
    return Image.open(buf)

def detect(image, threshold=0.5):
    results = model(image)
    output_image = get_output_figure(image, results, threshold)
    return output_image

# Build the Gradio app
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Fashion Object Detection
        Detect fashion-related objects in images using a fine-tuned DETR model.  
        You can load or select an image then adjust the detection threshold using the slider for better results.
        """
    )

    with gr.Row():
        image_input = gr.Image(label="Input Image", type="pil")
        threshold_slider = gr.Slider(
            minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="Detection Threshold"
        )

    output_image = gr.Image(label="Output Prediction", type="pil")

    detect_button = gr.Button("Run Detection")

    detect_button.click(detect, inputs=[image_input, threshold_slider], outputs=output_image)

    gr.Markdown(
        """
        ### About the Model
        This app uses the DETR model fine-tuned on the Fashionpedia dataset, which includes diverse fashion-related objects.
        """
    )
    gr.Markdown(
        """
        ### Created by Kelechi Osuji.
        """
    )

    # Add example images
    example_images = [
        "examples/fashion_image_223.jpg",
        "examples/fashion_image_1094.jpg",
        "examples/fashion_image_1113.jpg",
        "examples/fashion_image_508.jpg"
    ]
    gr.Examples(
        examples=example_images,
        inputs=[image_input]
    )

demo.launch(show_error=True)
