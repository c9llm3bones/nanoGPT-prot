# nanoGPT adaptation for protein sequences

This project adapts **nanoGPT** for antibody protein sequences. 
The model works on **character-level**, with  special tokens for **eos**, **class** and **type**. 

---

## Preparing the Dataset

Run the dataset preprocessing script:

```bash
python data/prots/prepare.py
```
This will generate all .bin and .pkl files in data/prots/

*check data/prots/readme.md for details*

## Training

```bash
python train.py 
```

That's it! It's performing training loop. 

In train.py constants are changed to perform training on CPU.
For GPU running, please, set it to the original Karpaty's values.