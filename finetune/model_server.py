"""Small local server for the fine-tuned model (Llama-3.2-3B + QLoRA adapter v2).

The website backend sends it the same numbered context chunks it sends to GPT, and shows both answers.
Run it in the finetune folder with ft-venv (needs the GPU), and keep this window open:
    python model_server.py
It listens only on this computer: http://127.0.0.1:8001
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from unsloth import FastLanguageModel

ADAPTER_DIR = r"C:\Users\preri\finetune\lora_adapter_v2"
HOST, PORT = "127.0.0.1", 8001
MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 200

print("Loading the fine-tuned model on the GPU (about 1 minute) ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_DIR,  # the adapter folder knows its base model (Llama-3.2-3B-Instruct, 4-bit)
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
gpu_lock = threading.Lock()  # one answer at a time on the GPU


def generate(messages):
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")
    input_len = inputs["input_ids"].shape[1]
    if input_len > MAX_SEQ_LENGTH - MAX_NEW_TOKENS:
        raise ValueError("The question and context are too long for the local model.")
    with gpu_lock, torch.no_grad():
        start = time.time()
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        seconds = round(time.time() - start, 2)
    answer = tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
    return answer, seconds


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "model": "Llama-3.2-3B + QLoRA v2"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            answer, seconds = generate(body["messages"])
            self._send(200, {"answer": answer, "seconds": seconds})
        except Exception as e:
            print("generate failed:", e)
            self._send(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        pass  # keep the window quiet; answers are printed below

    def log_request(self, code="-", size="-"):
        pass


if __name__ == "__main__":
    print(f"Fine-tuned model ready on http://{HOST}:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
