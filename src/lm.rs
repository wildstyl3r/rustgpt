pub mod block;
pub mod norm;
pub mod position_embedding;
use crate::interface::LanguageModel;

use clap::Args;
use serde::{Deserialize, Serialize};
use tch::{
    nn::{self, Module, ModuleT},
    Tensor,
};

#[derive(Args, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    #[arg(long, value_enum,default_value_t = position_embedding::additive::PositionEmbeddingOptions::Trainable)]
    pub additive_pe: position_embedding::additive::PositionEmbeddingOptions,

    #[arg(long, default_value_t = 0.2)]
    pub dropout: f64,

    #[arg(long, default_value_t = 8)]
    pub context_window: i64,

    #[arg(long, default_value_t = 4)]
    pub n_blocks: i64,

    #[command(flatten)]
    pub block: block::BlockGroup,

    #[arg(long, default_value_t = 32)]
    pub emb_dim: i64,
}

#[derive(Debug)]
pub struct Model {
    token_embedding_table: nn::Embedding,
    additive_pe: position_embedding::additive::PositionEmbedding,
    blocks: nn::SequentialT,

    final_ln: nn::LayerNorm,
    language_modeling_head: nn::Linear,
    context_window: i64,
}

impl Model {
    pub fn new(path: nn::Path, vocab_size: i64, config: &ModelConfig) -> Self {
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
            additive_pe: position_embedding::additive::embedding(
                &config.additive_pe,
                &path / "additive_pe",
                config.emb_dim,
                config.context_window,
            ),
            blocks: {
                (0..config.n_blocks)
                    .fold(nn::seq_t(), |s, i| {
                        s.add(block::block(
                            &path / ("b".to_owned() + &i.to_string()),
                            &config.block.block_option,
                            &config.block.block_config,
                            config.emb_dim,
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

impl ModuleT for Model {
    fn forward_t(&self, idx: &Tensor, train: bool) -> Tensor {
        let token_embeddings = self.token_embedding_table.forward(idx);

        let x = self
            .blocks
            .forward_t(&self.additive_pe.inject(token_embeddings), train);

        self.language_modeling_head
            .forward(&self.final_ln.forward(&x))
    }
}

impl LanguageModel for Model {
    fn get_context_window(&self) -> i64 {
        self.context_window
    }
}
