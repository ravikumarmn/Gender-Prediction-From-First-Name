import torch
from model import GenderClassifier
# import json

# vocab_size = 95024
# file_path ="vocabulary.json"

# with open(file_path, 'r') as f:
#     vocab = json.load(f)
    
checkpoint = torch.load("checkpoints/gender_classifier_using_fname_char_level_lstm_bs_64_embd_300_device_cpu.pt")
char_to_index = checkpoint['char_to_index']
params = checkpoint['params']
vocab_size = len(char_to_index)+1

model = GenderClassifier(vocab_size)
model.load_state_dict(checkpoint['state_dict'])

label_mapping = params['MAPPING']


def inference(model,input_id,fname,label_mapping):
    model.eval()
    with torch.no_grad():
        logits = model(input_id)
        _,predicted_indices = torch.max(logits.data,1)
        mapping_dict = {v:k for k,v in label_mapping.items()}
        out = mapping_dict[predicted_indices.item()]
        if out == "F":
            print(f"Given Name {fname} Is Female\n")
            print("-"*80)
        elif out == "M":
            print(f"Given Name {fname} Is Male\n")
            print("-"*80)
        else:
            raise NotImplementedError

while True:
    fname = input("Enter Your First Name: ").lower().strip()
    if fname in  ["q","bye","see you","quit","exit"]:
        print("\nCome back soon.")
        break
    else:
        try:
            input_ids = torch.tensor([char_to_index[name] for name in fname]).unsqueeze(0)
        except KeyError:
            print("Name not found in vocabulary.")
            continue

        inference(model, input_ids, fname, label_mapping)
