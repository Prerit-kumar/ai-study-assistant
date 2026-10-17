"""Small local server for the fine-tuned models: one Llama-3.2-3B base with two QLoRA adapters.

  v2 - trained on research papers        -> used for questions about the demo library
  v3 - trained on everyday documents too -> used for questions about the user's own files

Benchmark (eval_judge_v3.py): on unseen everyday documents v3 hallucinated 6% vs v2's 17% and got
70% of reasoning questions right vs 40%; on research papers v2 stayed slightly better. So both are kept,
and the backend picks one per question. Both share the same base model, so this costs little extra GPU memory.

Run it in the finetune folder with ft-venv (needs the GPU), and keep this window open:
    python model_server.py
It listens only on this computer: http://127.0.0.1:8001
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from unsloth import FastLanguageModel

ADAPTERS = {
    "v2": r"C:\Users\preri\finetune\lora_adapter_v2",
    "v3": r"C:\Users\preri\finetune\lora_adapter_v3",
}
DEFAULT_ADAPTER = "v2"
HOST, PORT = "127.0.0.1", 8001
MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 200

print("Loading the base model + adapter v2 on the GPU (about 1 minute) ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTERS["v2"],  # the adapter folder knows its base model (Llama-3.2-3B-Instruct, 4-bit)
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)
# The first adapter is loaded under PEFT's name "default".
ADAPTER_NAMES = {"v2": "default"}

if os.path.isdir(ADAPTERS["v3"]):
    try:
        model.load_adapter(ADAPTERS["v3"], adapter_name="v3")
        ADAPTER_NAMES["v3"] = "v3"
        print("Adapter v3 loaded too.")
    except Exception as e:
        print(f"Could not load adapter v3, using v2 for everything: {e}")
else:
    print("Adapter v3 not found, using v2 for everything.")

FastLanguageModel.for_inference(model)
gpu_lock = threading.Lock()  # one answer at a time on the GPU (and one active adapter)


def generate(messages, adapter):
    if adapter not in ADAPTER_NAMES:
        adapter = DEFAULT_ADAPTER
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")
    input_len = inputs["input_ids"].shape[1]
    if input_len > MAX_SEQ_LENGTH - MAX_NEW_TOKENS:
        raise ValueError("The question and context are too long for the local model.")
    with gpu_lock, torch.no_grad():
        model.set_adapter(ADAPTER_NAMES[adapter])
        start = time.time()
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        seconds = round(time.time() - start, 2)
    answer = tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
    return answer, seconds, adapter


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
            self._send(200, {"ok": True, "model": "Llama-3.2-3B + QLoRA", "adapters": sorted(ADAPTER_NAMES)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            answer, seconds, adapter = generate(body["messages"], body.get("adapter", DEFAULT_ADAPTER))
            print(f"[{adapter}] {seconds}s")
            self._send(200, {"answer": answer, "seconds": seconds, "adapter": adapter})
        except Exception as e:
            print("generate failed:", e)
            self._send(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        pass  # keep the window quiet

    def log_request(self, code="-", size="-"):
        pass


if __name__ == "__main__":
    print(f"Fine-tuned models ready on http://{HOST}:{PORT}  (adapters: {', '.join(sorted(ADAPTER_NAMES))})  Ctrl+C to stop")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
