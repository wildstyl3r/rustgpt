use tch::{
    nn::{self, Module},
    Tensor,
};

#[derive(Debug)]
pub struct FeedForward {
    net: nn::Sequential,
}

impl FeedForward {
    pub fn new(path: nn::Path, n_embed: i64, dropout: f64) -> Self {
        FeedForward {
            net: nn::seq()
                .add(nn::linear(
                    &path / "l1",
                    n_embed,
                    4 * n_embed,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(
                    &path / "l2",
                    4 * n_embed,
                    n_embed,
                    Default::default(),
                ))
                .add_fn(move |xs| xs.dropout(dropout, true)),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.net.forward(x)
    }
}
