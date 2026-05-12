pub mod block;
pub mod norm;
pub mod position_embedding;
use crate::{interface::LanguageModel, lm::block::TransformerBlock};

use clap::Args;
use serde::{Deserialize, Serialize};
use tch::{
    nn::{self, Module, ModuleT},
    Tensor,
};

#[derive(Args, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    #[arg(long, default_value_t = 0.2)]
    pub dropout: f64,

    #[arg(long, default_value_t = 8)]
    pub context_window: i64,

    #[arg(long, default_value_t = 4)]
    pub n_blocks: i64,

    #[arg(long, default_value_t = 32)]
    pub emb_dim: i64,

    #[arg(long, default_value_t = 32)]
    pub multihead_dim: i64,

    #[arg(long, default_value_t = 4)]
    pub num_heads: i64,
}

#[derive(Debug)]
pub struct Model<A: position_embedding::Additive> {
    token_embedding_table: nn::Embedding,
    additive_pe: A,
    blocks: nn::SequentialT,
    final_ln: nn::LayerNorm,
    language_modeling_head: nn::Linear,
    context_window: i64,
}

impl<A: position_embedding::Additive + std::fmt::Debug> Model<A> {
    pub fn new<B: TransformerBlock>(path: nn::Path, vocab_size: i64, config: &ModelConfig) -> Self {
        let causal_mask = Tensor::ones(
            [config.context_window, config.context_window],
            (tch::Kind::Float, tch::Device::Cpu),
        )
        .tril(0)
        .eq(0);

        Model {
            token_embedding_table: nn::embedding(
                &path / "embedding",
                vocab_size,
                config.emb_dim,
                Default::default(),
            ),
            additive_pe: A::new(&path / "additive_pe", config.context_window, config.emb_dim),
            blocks: {
                (0..config.n_blocks)
                    .fold(nn::seq_t(), |s, i| {
                        s.add(B::new(
                            &path / ("b".to_owned() + &i.to_string()),
                            config.emb_dim,
                            config.multihead_dim,
                            config.num_heads,
                            config.context_window,
                            config.dropout,
                            causal_mask.shallow_clone(),
                        ))
                    })
                    .add(nn::layer_norm(
                        &path / "blocks_ln",
                        vec![config.emb_dim],
                        Default::default(),
                    ))
            },
            final_ln: nn::layer_norm(&path / "ln_f", vec![config.emb_dim], Default::default()),
            language_modeling_head: nn::linear(
                &path / "lm_head",
                config.emb_dim,
                vocab_size,
                Default::default(),
            ),
            context_window: config.context_window,
        }
    }
}

impl<A: position_embedding::Additive + std::fmt::Debug> ModuleT for Model<A> {
    fn forward_t(&self, idx: &Tensor, train: bool) -> Tensor {
        let token_embeddings = self.token_embedding_table.forward(idx);

        let x = self
            .blocks
            .forward_t(&self.additive_pe.inject(token_embeddings), train);

        self.language_modeling_head
            .forward(&self.final_ln.forward(&x))
    }
}

impl<A: position_embedding::Additive> LanguageModel for Model<A> {
    fn get_context_window(&self) -> i64 {
        self.context_window
    }
}
