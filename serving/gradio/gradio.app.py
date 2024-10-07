import gradio as gr
from src.trim_rag.inference import DataInference
import time

def generate(prompt, pdf, image, audio):
    time.sleep(2)
    data_inference = DataInference()
    output = data_inference.embed_query(prompt)
    return output

with gr.Blocks(theme="johnSmith9982/small_and_pretty") as demo:

    gr.Markdown("<center> <h1>MULTIPLE MODELS APPLICATION BASED ON GRADIO</h1> </center>")
    gr.Markdown("<p align='center'> <i>End to End your Retrieval Augmented Generation (RAG) pipelines integrating LLM Models (SOTA) </i></p>")
    prompt = gr.Textbox(label="Prompt", lines=3, max_lines=5)
    pdf = gr.UploadButton(label="Upload PDF", file_types=[".pdf"])
    image = gr.UploadButton(label="Upload Image", file_types=[".png", ".jpg", ".jpeg"])
    audio = gr.UploadButton(label="Upload Audio", file_types=[".mp3"])
    btn = gr.Button("Generate")
    output = gr.Textbox(label="Output")
    generate_fn = generate(prompt, pdf, image, audio)
    btn.click(fn=generate_fn, 
              inputs=[prompt, pdf, image, audio], 
              outputs=output,
              show_progress="hidden")


if __name__ == "__main__":
    demo.launch(show_api=False)