import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- CONFIGURATION ---
MODEL_ID = "hannusia123123/propaganda-technique-detector"

print(f"⏳ Loading model from Hub: {MODEL_ID}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("💡 Hint: Check if the repository name is correct and model is pushed.")

if 'model' in locals():
    id2label = model.config.id2label
else:
    id2label = {}

# --- PREDICTION LOGIC ---
def predict(context, fragment):
    if not context:
        return None, "⚠️ Please enter the text of the article."
    
    if not fragment:
        return None, "⚠️ Insert the text fragment you want to check."

    if fragment in context:
        marked_text = context.replace(fragment, f" <E> {fragment} </E> ", 1)
    else:
        return None, "❌ Error: Fragment not found in the Context text."

    if 'model' not in locals():
        return None, "❌ Error: Model not loaded."

    inputs = tokenizer(
        marked_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=200
    )

    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    results = {}
    for i, prob in enumerate(probabilities[0]):
        label = id2label[i]
        results[label] = float(prob)
    
    return results, marked_text

# --- UI SETUP ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
        # 🕵️‍♀️ Propaganda Technique Detector
        **Model:** `{MODEL_ID}`
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_context = gr.Textbox(
                label="Article Context", 
                lines=5, 
                placeholder="Full text here..."
            )
            input_fragment = gr.Textbox(
                label="Suspicious Fragment", 
                placeholder="Paste the specific phrase..."
            )
            submit_btn = gr.Button("🔍 Analyze", variant="primary")
            
        with gr.Column():
            label_output = gr.Label(label="Predicted Technique", num_top_classes=3)
            debug_output = gr.Textbox(label="Model Input (Internal View)", interactive=False)

    submit_btn.click(
        fn=predict, 
        inputs=[input_context, input_fragment], 
        outputs=[label_output, debug_output]
    )
    
    gr.Examples(
        examples=[
            ["These idiots represent a danger to democracy.", "idiots"],
            ["We must fight this horrifying threat to our children.", "horrifying threat"],
        ],
        inputs=[input_context, input_fragment]
    )

if __name__ == "__main__":
    demo.launch()