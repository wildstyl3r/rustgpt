# RustGPT

A simple testbed project to cover various optimization tricks used in modern transformer architectures. Was abandoned for a while, but currently is under active development. Its origin can be traced back to [Let's build GPT:...](https://www.youtube.com/watch?v=kCc8FmEb1nY) video by Andrej Karpathy, and default startup parameters are near to those featured at some point in the video.

Project is focused primarily on tiny models and training/inference on CPU.

- [x] model weights saving and loading (May 2026)
- [x] enum dispatch based model construction managed from CLI (May 13, 2026)
- [x] RMSnorm (May 13, 2026)
- [x] parallel transformer block (May 13, 2026)
- [ ] BPE tokenizer
- [ ] rotary position embedding
- [ ] polar position embedding
- [ ] KV-caching
- [ ] speculative decoding
- [ ] random feature attention
- [ ] MQA
- [ ] GQA
- [ ] MLA
- [ ] Muon optimizer (???)