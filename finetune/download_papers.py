import os
import time
import urllib.request

SAVE_DIR = r"C:\Users\preri\ai-study-assistant\papers"

PAPERS = {
    "1706.03762": "Attention_Is_All_You_Need",
    "1810.04805": "BERT",
    "1907.11692": "RoBERTa",
    "1909.11942": "ALBERT",
    "1910.01108": "DistilBERT",
    "2003.10555": "ELECTRA",
    "1910.10683": "T5",
    "2005.14165": "GPT3_Few_Shot_Learners",
    "2302.13971": "LLaMA",
    "2307.09288": "Llama2",
    "2310.06825": "Mistral_7B",
    "2401.04088": "Mixtral",
    "2005.11401": "RAG_Retrieval_Augmented_Generation",
    "2002.08909": "REALM",
    "2004.04906": "DPR_Dense_Passage_Retrieval",
    "2004.12832": "ColBERT",
    "1908.10084": "Sentence_BERT",
    "2310.11511": "Self_RAG",
    "2112.04426": "RETRO",
    "2307.03172": "Lost_in_the_Middle",
    "2106.09685": "LoRA",
    "2305.14314": "QLoRA",
    "2101.00190": "Prefix_Tuning",
    "1902.00751": "Adapters_Parameter_Efficient",
    "2305.18290": "DPO",
    "2203.02155": "InstructGPT",
    "2201.11903": "Chain_of_Thought",
    "2210.03629": "ReAct",
    "2302.04761": "Toolformer",
    "2004.05150": "Longformer",
    "2001.04451": "Reformer",
    "2006.04768": "Linformer",
    "2009.14794": "Performer",
    "2007.14062": "BigBird",
    "2205.14135": "FlashAttention",
    "2312.00752": "Mamba",
    "2101.03961": "Switch_Transformers",
    "2001.08361": "Scaling_Laws",
    "2203.15556": "Chinchilla",
    "1901.02860": "Transformer_XL",
}


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    total = len(PAPERS)

    for i, (arxiv_id, name) in enumerate(PAPERS.items(), start=1):
        path = os.path.join(SAVE_DIR, f"{name}.pdf")

        if os.path.exists(path):
            print(f"[{i}/{total}] Already exists, skipping: {name}")
            continue

        url = f"https://arxiv.org/pdf/{arxiv_id}"
        print(f"[{i}/{total}] Downloading {name} ...")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                data = response.read()
            with open(path, "wb") as f:
                f.write(data)
        except Exception as e:
            print(f"   FAILED: {name} ({e})")

        time.sleep(3)

    print(f"\nDone. Papers saved in: {SAVE_DIR}")


if __name__ == "__main__":
    main()