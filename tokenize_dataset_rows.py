import argparse
import json
from tqdm import tqdm

import datasets
import transformers


def preprocess(tokenizer, example, max_seq_length=512):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target, max_length=max_seq_length, truncation=True, add_special_tokens=False
    )
    input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def read_jsonl(path, max_seq_length):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/root/autodl-tmp/chatglm-6b", trust_remote_code=True
    )
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            yield preprocess(tokenizer, example, max_seq_length)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, default="data/alpaca_gpt4_data_zh.jsonl")
    parser.add_argument("--save_path", type=str, default="data/alpaca_zh")
    parser.add_argument("--max_seq_length", type=int, default=320)
    args = parser.parse_args()

    dataset = datasets.Dataset.from_generator(
        lambda: read_jsonl(args.jsonl_path, args.max_seq_length)
    )
    dataset.save_to_disk(args.save_path)
    print(dataset.to_pandas().head())

if __name__ == "__main__":
    main()
