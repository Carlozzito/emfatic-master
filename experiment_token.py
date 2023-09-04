import torch
import logging
import os

from transformers import GPT2LMHeadModel
from transformers import pipeline

model_path = "./runs/gpt2-modelset_token-128/best_model"

model = GPT2LMHeadModel.from_pretrained(model_path)
logging.getLogger("transformers").setLevel(logging.ERROR)
generator = pipeline("text-generation", model=model_path, max_new_tokens=1)

output_file = "predictions.txt"

#file_path = "./modelset_token/test.txt"
file_path = "./qq.txt"
cnt = 0

"""with open(file_path, "r") as file, open(output_file, "w") as out_file:
    for line in file:
        cnt += 1
        tokens = line.split()
        prefix = ""
        out_tokens = "<s>"
        # Iteramos sobre todos los prefijos
        # (salvo la línea entera)
        for i in range(0, len(tokens)-1):
            prefix += ' ' + tokens[i]
            prediction = generator(prefix)[0]['generated_text']
            out_tokens += ' ' + prediction.split()[-1]
            #print(f"Prefix____: {prefix}")
            #print(f"Prediction: {prediction}")

        out_file.write(out_tokens + '\n')
        print("frase: " + str(cnt))
"""

with open(file_path, "r") as file, open(output_file, "r") as out_file:
    preds = out_file.readlines()
    gts = file.readlines()
    assert len(preds) == len(gts), f"Samples of predictions and answers are not equal, {len(preds)}: {len(gts)}"

    total = 0
    correct = 0.0
    for pred, gt in zip(preds, gts):
        pred = pred.split()
        gt = gt.split()
        assert len(pred) == len(gt), f"Sequence length of prediction and answer are not equal, {len(pred)}: {len(gt)}"
        for x, y in zip(pred, gt):
            if y not in ["<s>", "</s>", "<EOL>", "<pad>"]:
                total += 1
                if x == y:
                    correct += 1

    print(f"Total {total} tokens, accuracy: {round(correct / total * 100, 2)}")