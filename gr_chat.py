import gradio as gr

from Agent_test import Chat_Agent


def slow_echo(query, history):
    print(query)
    return Chat_Agent().run(query)


demo = gr.ChatInterface(slow_echo, examples=["安徽黄山胶囊股份有限公司的董事长是谁？", "公司地址在哪里？", "2016年上半年的营业收入是多少"]).queue()

if __name__ == "__main__":
    demo.launch()
