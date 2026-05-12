pub mod activation;
pub mod attention;
pub mod storage;
use crate::lm::norm::Norm;

use tch::{
    nn::{self, ModuleT},
    Tensor,
};

pub trait TransformerBlock: ModuleT + 'static {
    fn new(
        path: nn::Path,
        emb_dim: i64,
        multihead_dim: i64,
        num_heads: i64,
        context_window: i64,
        dropout: f64,
        causal_mask: Tensor,
    ) -> Self;
}

#[derive(Debug)]
pub struct SequentialBlock<N: Norm, S: attention::SelfAttention, M: storage::Storage> {
    attention_norm: N,
    self_attention: S,
    storage_norm: N,
    storage: M,
}

impl<N: Norm, S: attention::SelfAttention, M: storage::Storage> ModuleT
    for SequentialBlock<N, S, M>
{
    fn forward_t(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor {
        let xs = self
            .self_attention
            .forward_t(&self.attention_norm.forward(xs), train)
            + xs;
        self.storage.forward(&self.storage_norm.forward(&xs)) + xs
    }
}

impl<N: Norm, S: attention::SelfAttention, M: storage::Storage> TransformerBlock
    for SequentialBlock<N, S, M>
{
    fn new(
        path: nn::Path,
        emb_dim: i64,
        multihead_dim: i64,
        num_heads: i64,
        context_window: i64,
        dropout: f64,
        causal_mask: Tensor,
    ) -> Self {
        Self {
            attention_norm: N::new(&path / "sa_norm", emb_dim),
            self_attention: S::new(
                &path / "self_attention",
                emb_dim,
                multihead_dim,
                num_heads,
                dropout,
                causal_mask,
                context_window,
            ),
            storage_norm: N::new(&path / "st_norm", emb_dim),
            storage: M::new(&path / "storage", emb_dim, dropout),
        }
    }
}
