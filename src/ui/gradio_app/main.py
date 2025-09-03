import gradio as gr

def greet(name):
    return f"Hello, {name}! Welcome to the healthcare Gradio app."

iface = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="Your Name"),
    outputs=gr.Textbox(label="Greeting"),
    title="Healthcare Gradio App",
    description="A simple Gradio demo for healthcare applications."
)

if __name__ == "__main__":
    iface.launch(server_port=7860, server_name="0.0.0.0", show_api=False)
