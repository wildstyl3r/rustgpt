use tch::{
    nn::{self, Module},
    Tensor,
};

use crate::baseline::{ffwd::FeedForward, multihead::MultiHead};

#[derive(Debug)]
pub struct TransformerBlock {
    sa_ln: nn::LayerNorm,
    self_attention: MultiHead,
    ff_ln: nn::LayerNorm,
    ffwd: FeedForward,
}

impl TransformerBlock {
    pub fn new(
        path: nn::Path,
        n_embed: i64,
        num_heads: usize,
        block_size: i64,
        dropout: f64,
    ) -> Self {
        //let head_size = (n_embed as usize / num_heads) as i64;
        TransformerBlock {
            sa_ln: nn::layer_norm(&path / "sa_ln", vec![n_embed], Default::default()),
            self_attention: MultiHead::new(&path / "b_sa", num_heads, n_embed, block_size, dropout),
            ff_ln: nn::layer_norm(&path / "ff_ln", vec![n_embed], Default::default()),
            ffwd: FeedForward::new(&path / "b_ffwd", n_embed, dropout),
        }
    }
}
impl nn::ModuleT for TransformerBlock {
    fn forward_t(&self, x: &tch::Tensor, train: bool) -> Tensor {
        let x = self.self_attention.forward_t(&self.sa_ln.forward(x), train) + x;
        self.ffwd.forward(&self.ff_ln.forward(&x)) + x
    }
}
