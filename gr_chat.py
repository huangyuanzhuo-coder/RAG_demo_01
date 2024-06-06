import gradio as gr


def echo(message, history):
    return message


if __name__ == '__main__':
    demo = gr.ChatInterface(fn=echo, examples=["hello", "hola", "merhaba"], title="Echo Bot")
    demo.launch()

