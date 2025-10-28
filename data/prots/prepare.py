import os
import pickle
import numpy as np
from datasets import load_dataset, Dataset
from config import *

DEBUG = False # I actually really liked this flag
hf_ds = "bayes-group-diffusion/OAS95-aligned-cleaned"

def make_stream_ds():
    return  load_dataset(hf_ds, split="train", streaming=True)

# 64 GB of data
if load_full_ds:
    ds = load_dataset(hf_ds, split="train")
else:
    ds_stream = make_stream_ds()
    rows = list(ds_stream.take(n_rows))
    ds = Dataset.from_list(rows)

if DEBUG:
    print(ds.info)
    print()

df = ds.to_pandas()
if DEBUG:
    print(df.head())
    print()

sequences = df['sequence'].tolist()
init_sequences = df['init_seq'].tolist()

if DEBUG:
    print(sequences[:10])
    print(init_sequences[:10])
    print()

special_tokens = set(['<HEAVY>', '<LIGHT>', '<EOS>'])

def norm_class(x):
    if x is None:
        return ""
    s = str(x).lower()
    if 'mouse' in s:
        special_tokens.add('<MOUSE>')
        return '<MOUSE>'
    special_tokens.add(f'<{s.upper()}>')
    return f'<{s.upper()}>'

def norm_type(x):
    if x is None:
        return ""
    s = str(x).lower()
    if 'heavy' in s:
        return '<HEAVY>'
    if 'light' in s:
        return '<LIGHT>'
    special_tokens.add(f'<{s.upper()}>')
    return f'<{s.upper()}>' 

classes = df['class'].apply(norm_class).tolist()
types = df['type'].apply(norm_type).tolist()

if DEBUG:
    print(classes[:10])
    print(types[:10])
    print()

# get all sequences w.r.t. config 
all_sequences = []
i = 0
for row in ds:
    i+=1
    if DEBUG:
        if i % 100 == 0:
            print("row: ", row)
    prompt = "<EOS>"
    if np.random.rand() < P_CLASS:
        prompt += norm_class(row.get('class'))
    if np.random.rand() < P_TYPE:
        prompt += norm_type(row.get('type'))

    if USE_SEQUENCE and row.get('sequence'):  
        prompt += row.get('sequence','').replace('\n','').strip()
    elif not USE_SEQUENCE and row.get('init_seq'):
        prompt += row.get('init_seq','').replace('\n','').strip()
    
    if prompt:
        all_sequences.append(prompt)
    
    if DEBUG:
        if i % 100 == 0:
            print(prompt)

print(f"Prepared {len(all_sequences)} sequences")

# build vocab
base_chars = list('ACDEFGHIKLMNPQRSTVWY-')
special_tokens = list(special_tokens)
if DEBUG:
    print('classes and types: ', special_tokens)
    print()
vocab = base_chars + special_tokens
stoi = { ch:i for i,ch in enumerate(vocab) }
itos = { i:ch for i,ch in enumerate(vocab) }

# new encoding func 
def encode(seq):
    i = 0
    while i < len(seq):
        if seq[i] == '<':  # parse special token
            j = seq.find('>', i)
            if j != -1:
                token = seq[i:j+1]
                if token in stoi:
                    yield stoi[token]
                i = j+1
                continue
        if seq[i] in stoi:
            yield stoi[seq[i]]
        i += 1

n = len(all_sequences)
n_train = int(n*0.9)

# change: concatenate because seqs have different lens 
train_ids = np.concatenate([list(encode(seq)) for seq in all_sequences[:n_train]]).astype(np.uint16)
val_ids = np.concatenate([list(encode(seq)) for seq in all_sequences[n_train:]]).astype(np.uint16)
print(f"train has {len(train_ids)} tokens")
print(f"val has {len(val_ids)} tokens")

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

meta = {
    'vocab_size': len(vocab),
    'itos': itos,
    'stoi': stoi
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("preparing done")
