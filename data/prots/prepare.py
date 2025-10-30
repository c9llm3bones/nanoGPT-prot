import os
import pickle
import numpy as np
from datasets import load_dataset, Dataset
from config import *
import re

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

special_tokens = set(['<HEAVY>', '<LIGHT>'])

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
all_classes = []
all_types = []
i = 0
rng = np.random.RandomState(seed) # for deterministic sampling
for row in ds:
    i+=1
    if DEBUG:
        if i % 100 == 0:
            print("row: ", row)
    expr = "<EOS>"
    if use_sequence and row.get('sequence'):
        expr += row.get('sequence','').replace('\n','').strip()
    elif not use_sequence and row.get('init_seq'):
        expr += row.get('init_seq','').replace('\n','').strip()

    if not re.match(r"^<EOS>(?:<[^>]+>)*$", expr): # ensure to have AA
        all_sequences.append(expr)
        all_classes.append(norm_class(row.get('class')))
        all_types.append(norm_type(row.get('type')))

    if DEBUG:
        if i % 100 == 0:
            print(expr)

if DEBUG:
    print(all_sequences[:1000])
print(f"Prepared {len(all_sequences)} sequences")

# build vocab
base_chars = list('ACDEFGHIKLMNPQRSTVWY')
if use_sequence:
    base_chars.append('-')
special_tokens = ['<EOS>'] + ['<UNK>'] + sorted(list(special_tokens)) # sort for deterministic encoding
if DEBUG:
    print('classes and types: ', special_tokens)
    print()
vocab = special_tokens + base_chars
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
                yield stoi.get(token, stoi['<UNK>'])
                i = j+1
                continue
            else:
                yield stoi['<UNK>']
                i += 1
                continue
        yield stoi.get(seq[i], stoi['<UNK>'])
        i += 1

all_seq_lists = [list(encode(seq)) for seq in all_sequences] # lists of encoded seqs
all_seq_ids = np.concatenate(all_seq_lists).astype(np.uint16) # cat all seq lists

eos_token = stoi['<EOS>']
# ids of all <EOS> tokens
eos_ids = np.where(all_seq_ids == eos_token)[0]

# maps for types and classes w.r.t. eos_ids, -1 for unknown
seq_class_ids = np.array([stoi.get(c, -1) if c else -1 for c in all_classes], dtype=np.uint16)
seq_type_ids  = np.array([stoi.get(t, -1) if t else -1 for t in all_types ], dtype=np.uint16)

if DEBUG:
  for (type_, class_)  in zip(seq_type_ids, seq_class_ids):
    print(itos[type_], " ", itos[class_])

print("total tokens (all):", len(all_seq_ids) + len(seq_class_ids) + len(seq_type_ids))
if DEBUG:
    print(all_seq_ids[:1000])

n = len(all_seq_ids)
n_train = int(n * 0.9)
train_ids = all_seq_ids[:n_train]
val_ids = all_seq_ids[n_train:]
print(f"train has {len(train_ids)} tokens")
print(f"val   has {len(val_ids)} tokens")

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
seq_class_ids.tofile(os.path.join(os.path.dirname(__file__), 'seq_class_ids.bin'))
seq_type_ids.tofile(os.path.join(os.path.dirname(__file__), 'seq_type_ids.bin'))
eos_ids.tofile(os.path.join(os.path.dirname(__file__), 'eos_ids.bin'))

meta = {
    'vocab_size': len(vocab),
    'itos': itos,
    'stoi': stoi
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("preparing done")