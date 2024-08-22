use tch::{Tensor, nn::{self, Module}, IndexOp};

#[derive(Debug)]
struct Head {
    key: nn::Linear,
    query: nn::Linear,
    value: nn::Linear,
    tril: Tensor,
    dropout: f64,
}

impl Head {
    fn new(path: nn::Path, head_size: i64, n_embed: i64, block_size: i64, dropout: f64) -> Self {
        Head {
            key: nn::linear(&path / "head_k", n_embed, head_size, nn::LinearConfig{bias: false, ..Default::default()}),
            query: nn::linear(&path / "head_q", n_embed, head_size, nn::LinearConfig{bias: false, ..Default::default()}),
            value: nn::linear(&path / "head_v", n_embed, head_size, nn::LinearConfig{bias: false, ..Default::default()}),
            tril: Tensor::ones([block_size, block_size], (tch::Kind::Float, tch::Device::Cpu)).tril(0),
            dropout
        }
    }

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

#[derive(Debug)]
pub struct MultiHead {
    heads: Vec<Head>,
    projection: nn::Linear,
    dropout: f64,
}

impl MultiHead {
    pub fn new(path: nn::Path, num_heads: usize, n_embed: i64, block_size: i64, dropout: f64) -> Self {
        let head_size = (n_embed as usize / num_heads) as i64;
        MultiHead {
            heads: (0..num_heads).map(|i|
                Head::new(
                    &path / ("multi_head_c".to_string() + &i.to_string()),
                    head_size,
                    n_embed,
                    block_size,
                    dropout
                )
            ).collect(),
            projection: nn::linear(&path / "proj", n_embed, n_embed, Default::default()),
            dropout
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.projection.forward(
            &Tensor::cat(
                &self.heads.iter().map(|h| h.forward(x)).collect::<Vec<_>>(),
                -1
            )
        ).dropout(self.dropout, true)
    }
}