from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
import config
import pandas as pd

df = pd.read_csv("dataset/name_gender.csv").rename(columns={"Aaban":"name","M":"gender"})[['name','gender']]
import torch

df['name'] = df.name.apply(lambda x : x.lower())

all_letters = ''.join(df['name'].tolist())
all_letters = sorted(set(all_letters))
n_letters = len(all_letters)
char_to_index = {char: i+1 for i, char in enumerate(all_letters)}
index_to_char = {i+1: char for i, char in enumerate(all_letters)}


# all_words = " ".join(df['name'].tolist()).split()

# all_words = sorted(set(all_words))
# n_words = len(all_words)
# word_to_index = {word: i for i, word in enumerate(all_words)}
# index_to_word = {i: word for i, word in enumerate(all_words)}


# vocab = {v:k for k,v in enumerate(sorted(df['name'].tolist()))}
# vocab_size = len(word_to_index)
vocab_size = len(char_to_index) + 1

label_mapping = {v:k for k,v in enumerate(sorted(set(df['gender'])))}
df['gender'] = df['gender'].apply(lambda x : label_mapping.get(x))
train,validation = train_test_split(df,test_size=config.TEST_SIZE)

class CustomDataset(Dataset):
    def __init__(self,df,word_to_index):
        self.word_to_index = word_to_index
        self.data = df

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        name =  self.data.iloc[index].get("name")
        
        name_tensors = torch.tensor([char_to_index[char] for char in name])
        padded_seq = self.seq_padding(name_tensors.tolist(),0)
        label = self.data.iloc[index].get("gender")

        return {
            "fname" : torch.tensor(padded_seq),
            "label" : torch.tensor(label)
        }
    
    def seq_padding(self, x, padding_token):
        max_len = config.MAX_LEN
        if len(x) < max_len:
            x = x + [padding_token] * (max_len - len(x))
        elif len(x) > max_len:
            x = x[:max_len]
        return x

train_data = CustomDataset(train,char_to_index)
validation_data = CustomDataset(validation,char_to_index)

batch_size = config.BATCH_SIZE
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
validation_loader = DataLoader(validation_data,batch_size=batch_size,shuffle=False)
    