use tch::nn::{self, Module};

pub trait Storage: Module + Send + 'static {
    fn new(path: nn::Path, emb_dim: i64, dropout: f64) -> Self;
}

#[derive(Debug)]
pub struct FeedForward {
    net: nn::Sequential,
}

impl Module for FeedForward {
    fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {
        self.net.forward(xs)
    }
}
impl Storage for FeedForward {
    fn new(path: nn::Path, emb_dim: i64, dropout: f64) -> Self {
        Self {
            net: nn::seq()
                .add(nn::linear(
                    &path / "l1",
                    emb_dim,
                    4 * emb_dim,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(
                    &path / "l2",
                    4 * emb_dim,
                    emb_dim,
                    Default::default(),
                ))
                .add_fn(move |xs| xs.dropout(dropout, true)),
        }
    }
}
