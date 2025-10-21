import gradio as gr
from huggingface_hub import InferenceClient
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'movies_knowledge_base')))
from src.application.search_cloud import search_movies_cloud as sear

def welcome(name):
    return f"Welcome to Gradio, {name}!"

css = """
body {
    background: #f5f5f5;
}
.gradio-container {
    background: #f5f5f5 !important;
}
#warning {background-color: #f3dcd}
.feedback textarea {font-size: 24px }
.title {font-size: 32px !important; text-align: center;}
"""



def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    """
    For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
    """
    response = ""

    results = sear(message, n_results=5)

    if results['documents'] and len(results['documents'][0]) > 0:
        documents = results['documents'][0]
        for i, doc in enumerate(documents, 1):
            response += f"{i}. {doc}\n"
    else:
        response = "No results found."

    yield response


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
chatbot = gr.ChatInterface(
    respond,
    type="messages",
)

with gr.Blocks( css=css) as demo:


    gr.HTML("""
<center>
    <img src='https://i.pinimg.com/originals/36/5e/68/365e6851d51814a090210f47911147ce.gif' 
         alt='Chatbot Animation' 
         width='300'
         style='border-radius: 20px;'>
</center>
""")
    gr.HTML("</center>")
    chatbot.render()


if __name__ == "__main__":
    demo.launch()

