
# prots dataset, character-level with special tokens

Antibody protein sequences, treated on character-level with optional `<class>` and `<type>` tokens.

Sequence format:

expr = <EOS>(<class>?)(<type>?)<seq>
dataset = (expr)*

- `<EOS>` — end-of-sequence token  
- `<class>` — optional class token (e.g., `<MOUSE>`), inserted probabilistically  
- `<type>` — optional type token (`<HEAVY>` or `<LIGHT>`), inserted probabilistically  
- `<seq>` — protein sequence  

After running `prepare.py`:

- `train.bin` — encoded training sequences  
- `val.bin` — encoded validation sequences  
- `eos_ids.bin` — indices of `<EOS>` tokens  
- `seq_class_ids.bin` — class token IDs for each sequence  w.r.t `eos_ids.bin`
- `seq_type_ids.bin` — type token IDs for each sequence  w.r.t `eos_ids.bin`
- `meta.pkl` — vocabulary and mappings (`stoi`, `itos`)  
