
# prots dataset, character-level with special tokens

Protein character-level sequences with  `<EOS>`,  `<class>` and `<type>` tokens.

`data\prots\config.py`  — configuration file for dataset preprocessing

Sequence format:

`expr = <EOS>(<class>?)(<type>?)<seq>`
`dataset = (expr)*`

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
