# nanoGPT — Built from Scratch

A character-level GPT implemented from scratch in PyTorch, following [Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) tutorial. Built and trained in Google Colab.

## What this is

This project implements a decoder-only Transformer (GPT) from the ground up — no high-level libraries abstracting away the architecture. It covers:

- Tokenization (character-level)
- Self-attention, implemented from first principles
- Multi-head attention
- Transformer blocks (attention + feedforward + residual connections + layer norm)
- Training loop with train/val loss tracking
- Text generation via autoregressive sampling

## Dataset

Trained on the **Tiny Shakespeare** dataset — ~1MB of Shakespeare's works, tokenized at the character level.

## Results

Final loss after training:
- Train loss: 1.20
- Val loss: 1.49

Sample generated text:
```
step 3500 : train loss 1.2863, val loss 1.5286
step 4000 : train loss 1.2561, val loss 1.5109
step 4500 : train loss 1.2312, val loss 1.4990
step 4999 : train loss 1.2059, val loss 1.4973
```

## How to run

The entire project is a single Colab notebook: [`nano_gpt.ipynb`](./nano_gpt.ipynb).

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Run all cells top to bottom (GPU runtime recommended: `Runtime > Change runtime type > T4 GPU`)
3. The final cells generate sample Shakespeare-style text from the trained model

No local setup required — everything runs in-notebook.

## What I learned

- How self-attention actually computes "what to attend to" via query/key/value projections
- Why residual connections and layer norm are necessary for training deep Transformers to converge
- The difference between a decoder-only architecture (like GPT) and encoder-decoder setups
- How sampling temperature and autoregressive generation work in practice

## Credits

- Tutorial: [Andrej Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Reference implementation: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Dataset: [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

## Next steps

- Scale up context length and embedding dimensions
- Swap in a BPE tokenizer (e.g. `tiktoken`) instead of character-level tokens
- Move from Colab to a local/cloud training script for larger runs
