import random
import time

import gradio as gr


def print_random(query):
    return query + str(random.randint(1, 100))


def slow_echo(message, history):
    for i in range(len(message)):
        time.sleep(0.05)
        yield print_random(message)


demo = gr.ChatInterface(slow_echo).queue()

if __name__ == "__main__":
    demo.launch()
