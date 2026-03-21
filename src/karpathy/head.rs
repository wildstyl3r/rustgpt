use tch::{
    nn::{self, Module},
    IndexOp, Tensor,
};

#[derive(Debug)]
pub struct Head {
    key: nn::Linear,
    query: nn::Linear,
    value: nn::Linear,
    tril: Tensor,
    dropout: f64,
}

impl Head {
    pub fn new(
        path: nn::Path,
        head_size: i64,
        n_embed: i64,
        block_size: i64,
        dropout: f64,
    ) -> Self {
        Head {
            key: nn::linear(
                &path / "head_k",
                n_embed,
                head_size,
                nn::LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            query: nn::linear(
                &path / "head_q",
                n_embed,
                head_size,
                nn::LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            value: nn::linear(
                &path / "head_v",
                n_embed,
                head_size,
                nn::LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            tril: Tensor::ones(
                [block_size, block_size],
                (tch::Kind::Float, tch::Device::Cpu),
            )
            .tril(0),
            dropout,
        }
    }
}

impl Module for Head {
    fn forward(&self, x: &Tensor) -> Tensor {
        let (_b, t, c) = x.size3().unwrap();
        let k = self.key.forward(x);
        let q = self.query.forward(x);
        let wei = (q.matmul(&k.transpose(-2, -1)) * (c as f64).powf(-0.5))
            .masked_fill(&self.tril.i((..t, ..t)).eq(0), f64::NEG_INFINITY)
            .softmax(-1, tch::Kind::Float)
            .dropout(self.dropout, true);
        wei.matmul(&self.value.forward(x))
    }
}
