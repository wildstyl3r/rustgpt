use tch::{nn, Tensor};

use crate::karpathy::{ffwd::FeedForward, multihead::MultiHead};

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
impl nn::Module for TransformerBlock {
    fn forward(&self, x: &tch::Tensor) -> Tensor {
        let x = self.self_attention.forward(&self.sa_ln.forward(x)) + x;
        self.ffwd.forward(&self.ff_ln.forward(&x)) + x
    }
}
