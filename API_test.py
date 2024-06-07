import os
from argparse import ArgumentParser
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Optional
import uvicorn, json, datetime
import torch
from typing import List, Dict, Any


def torch_gc():
    '''清空GPU缓存避免Out of Memory'''
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            with torch.cuda.device(device):
                torch.cuda.empty_cache()  # 清空PyTorch中的GPU缓存
                torch.cuda.ipc_collect()  # 收集并释放未使用的内存

def _chat_stream(model, tokenizer, query, history):
    # dict
    conversation = history
    conversation.append({'role': 'user', 'content': query})
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors='pt',
    )
    inputs = inputs.to(model.device)
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=inputs,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text

app = FastAPI()  # Web框架，用于构建API


# 格式是[{"role": "user","content": "你好"}]
class ChatRequest(BaseModel):
    prompt: str
    history: List[Dict[str, Any]]  # 使用自定义的模型


@app.post("/")      # 指定了API的URL路径为根路径"/"和HTTP方法为POST
async def create_item(chat_request: ChatRequest):  # 接收一个Request对象作为参数，该对象表示接收到的HTTP请求
    global model, tokenizer
    _query = chat_request.prompt
    _task_history = chat_request.history

    full_response = ""
    response = ""
    for new_text in _chat_stream(model, tokenizer, _query, history=_task_history):
        response += new_text
        full_response = response
    print(full_response)

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": full_response,
        "history": _task_history,
        "status": 200,
        "time": time
    }

    torch_gc()

    return answer

if __name__ == '__main__':
    DEFAULT_CKPT_PATH = '/home/ph/LLM/Qwen1.5/Qwen1.5-32B-Chat-AWQ'
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_CKPT_PATH, resume_download=True,
    )
    device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_CKPT_PATH,
        device_map=device_map,
        resume_download=True,
    ).eval()
    model.generation_config.max_new_tokens = 2048

    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=3344, workers=1)