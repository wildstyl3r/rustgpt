use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use tch::nn;

#[derive(ValueEnum, Debug, Serialize, Deserialize, Clone)]
pub enum PositionEmbeddingOptions {
    RoPE,
    PoPE,
    None,
}

#[derive(Debug)]
pub enum PositionEmbedding {
    // RoPE(),
    // PoPE(),
    None,
}

impl PositionEmbedding {
    pub fn inject(&self, x: tch::Tensor) -> tch::Tensor {
        match self {
            PositionEmbedding::None => x,
        }
    }
}

pub fn embedding(
    options: &PositionEmbeddingOptions,
    path: nn::Path,
    emb_dim: i64,
    context_window: i64,
) -> PositionEmbedding {
    match options {
        PositionEmbeddingOptions::RoPE => todo!(),
        PositionEmbeddingOptions::PoPE => todo!(),
        PositionEmbeddingOptions::None => PositionEmbedding::None,
    }
}
