use tch::nn::{self, Module};

impl super::Additive for nn::Embedding {
    fn new(vs: nn::Path, context_window: i64, emb_dim: i64) -> Self {
        nn::embedding(vs, context_window, emb_dim, Default::default())
    }

    fn inject(&self, x: tch::Tensor) -> tch::Tensor {
        let (_b, t, _c) = x.size3().unwrap();
        x + self.forward(&tch::Tensor::arange(t, (tch::Kind::Int, tch::Device::Cpu)))
    }
}

impl super::Additive for super::None {
    fn new(_: nn::Path, _: i64, _: i64) -> Self {}

    fn inject(&self, x: tch::Tensor) -> tch::Tensor {
        x
    }
}
