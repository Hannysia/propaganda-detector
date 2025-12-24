import gradio as gr
from transformers import pipeline

MODEL_PATH = "hannusia123123/propaganda-baseline-bert"

try:
    classifier = pipeline("text-classification", model=MODEL_PATH, top_k=None)
except Exception as e:
    classifier = None
    print(f"Model not found yet: {e}")

def predict(text):
    if classifier is None:
        return {"Error": "Model not loaded. Please upload model to HF."}
    
    results = classifier(text)[0]
    
    output = {item['label']: item['score'] for item in results}
    return output

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕵️ Propaganda Detector (Baseline)")
    gr.Markdown("Enter a piece of news, and the model will try to detect propaganda techniques.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="News Text", placeholder="Paste text here...", lines=5)
            submit_btn = gr.Button("Analyze", variant="primary")
        
        with gr.Column():
            output_label = gr.Label(label="Detected Techniques", num_top_classes=5)
    
    submit_btn.click(fn=predict, inputs=input_text, outputs=output_label)
    
    gr.Examples(
        examples=[
            ["The failing New York Times is enemy of the people!"],
            ["We must protect our borders or we will lose our country."],
            ["Scientists say that climate change is real."]
        ],
        inputs=input_text
    )

if __name__ == "__main__":
    demo.launch()