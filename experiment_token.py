import torch
import logging
import os

from transformers import GPT2LMHeadModel
from transformers import pipeline

# Path del modelo pre-entrenado.
model_path = "./runs/gpt2-modelset_token-128/best_model"

# Carga del modelo pre-entrenado.
model = GPT2LMHeadModel.from_pretrained(model_path)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Usamos la GPU si esta disponible.
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Creamos un pipeline con el tipo de tarea que vamos a resolver
# y el path al modelo cargado.
# max_new_tokens para generar siempre 1 token mas.
generator = pipeline("text-generation", model=model_path, max_new_tokens=1)

# Fichero donde guardamos las predicciones.
#output_file = "predictions.txt"
output_file = "alonePrediction.txt"

# Fichero con el conjunto de test.
#test_path = "./modelset_token/test.txt"
test_path = "alone.txt"

# Fichero reducido
#file_path = "./test200.txt"
file_path = "alone.txt"


# Contador para saber por que frase vamos.
cnt = 0

# Abrimos el conjunto de test y el fichero para las predicciones.
# longitudes: [678, 1477]

if False:
    with open(test_path, "r") as file, open("test200.txt", "w") as test200:
        for line in file:
            tokens = line.split()
            if (len(tokens) <= 200):
                test200.write(line)


with open(test_path, "r") as file, open(output_file, "w") as out_file:
    for line in file:
        cnt += 1
        tokens = line.split()
        if(len(tokens)>200):
            continue
        prefix = ""
        out_tokens = "<s>"
        # Iteramos sobre todos los prefijos
        # (salvo la línea entera)
        words = line.split()
        print("Empiezan predicciones:" + str(len(words)))
        prefixes = [' '.join(words[:i + 1]) for i in range(len(words)-1)]
        stopped = False
        try:
            predictions = [generator(prefix, max_new_tokens = 1)[0]['generated_text'].strip() for prefix in prefixes]
        except IndexError:
            print("Error encontrado en la frase ", cnt)
            stopped = True
            break

        if stopped:
            continue

        for prefix, prediction in zip(prefixes, predictions):
            if len(prefix.split()) == len(prediction.split()):
                out_tokens += ' ' + 'NO_PRED'
            else:
                assert( len(prefix.split()) + 1 == len(prediction.split()) )
                out_tokens += ' ' + prediction.split()[-1]

            #print(f"Prefix: '{prefix}', Prediction: '{prediction}'")

        out_file.write(out_tokens + '\n')
        print("terminada frase: " + str(cnt))


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
            if y not in ["<s>", "</s>", "<EOL>", "<pad>", '[', ']', '{', '}', ';', 'class', 'extends']:
                total += 1
                if x == y:
                    correct += 1

    print(f"Total {total} tokens, accuracy: {round(correct / total * 100, 2)}")
