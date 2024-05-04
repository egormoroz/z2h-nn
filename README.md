# About

Here I try to implement NN stuff in pure rust.

# Broad Roadmap (will add more later)

- [x] MLP to solve XOR
- [ ] MLP for MNIST with 90%+ accuracy
- [ ] CNN for MNIST with 95%+ accuracy
- [ ] RNN, LSTM, all that jazz. Train on TinyShakespeare and get reasonable results
- [ ] Transformer. Train LLaMA-like on TinyShakespeare and get reasonable output.

# TODOs
## Must do
- [ ] Reasonably fast (75%+ time of numpy+MKL, or 100-ish GFLOPS on my machine) SGEMM 1 thread 
(done at nnops/examples/bench.rs, but only for specific matrix sizes)
- [ ] Reasonably fast (50%+ time of numpy+MKL, maybe like 250 GFLOPS??) SGEMM multithread
- [ ] Reasonably fast conv2d (I think with good SGEMM it's should be easy)

## Maaayybeee
- [ ] autograd (I have zero idea how to do it in rust wihout going insane)
- [ ] cuda ops
- [ ] okay-ish cuda SGEMM: 50% of cublas or like 5 TFLOPS on my machine
- [ ] quantization if I get like 10x smarter

