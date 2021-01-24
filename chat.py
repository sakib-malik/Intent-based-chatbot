import random
import json
import torch
from model import Net
from nltk_utils import tokenize, stem, bag_of_words

with open('intents.json', 'r') as f:
    intents = json.load(f)


FILE = 'data.pth'
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = Net(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "SAKIB"
print("Let's Chat! (type 'quit' to exit)")

while True:
    sentence = input("YOU: ")
    if sentence == 'quit':
        break
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, pred = torch.max(output, dim=1)
    tag = tags[pred.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][pred.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand ...")
