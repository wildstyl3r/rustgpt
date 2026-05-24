# RustGPT

A simple testbed project to cover various optimization tricks used in modern transformer architectures. Was abandoned for a while, but currently is under active development. Its origin can be traced back to [Let's build GPT:...](https://www.youtube.com/watch?v=kCc8FmEb1nY) video by Andrej Karpathy, and the default startup parameters are near to those featured at some point in the video.

The project is focused primarily on tiny models and training/inference on CPU.

### general stuff
- [x] model weights saving and loading (May 2026)
- [x] enum dispatch based model construction managed from CLI (May 13, 2026)
- [x] RMSnorm (May 13, 2026)
- [ ] BPE tokenizer
### transformer block modifications
- [x] parallel transformer block (May 13, 2026)
- [x] rotary position embedding
- [x] polar position embedding
- [ ] Mixture-of-Experts
### linear attention
- [ ] random feature attention
- [ ] Taylor series based softmax approximation
### quasilinear attention
- [ ] sliced ReLU attention
### generic attention tricks
- [ ] MQA
- [ ] GQA
- [ ] MLA
### inference
- [ ] KV-caching
- [ ] speculative decoding
### optimization
- [ ] Muon optimizer (???)