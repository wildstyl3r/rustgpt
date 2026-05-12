use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use tch::nn::{self, Module};

#[derive(ValueEnum, Debug, Serialize, Deserialize, Clone)]
pub enum PositionEmbeddingOptions {
    Trainable,
    None,
}

#[derive(Debug)]
pub enum PositionEmbedding {
    Trainable(nn::Embedding),
    None,
}

impl PositionEmbedding {
    pub fn inject(&self, x: tch::Tensor) -> tch::Tensor {
        match self {
            PositionEmbedding::Trainable(embedding) => {
                let (_b, t, _c) = x.size3().unwrap();
                x + embedding.forward(&tch::Tensor::arange(t, (tch::Kind::Int, tch::Device::Cpu)))
            }
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
        PositionEmbeddingOptions::Trainable => PositionEmbedding::Trainable(nn::embedding(
            path,
            context_window,
            emb_dim,
            Default::default(),
        )),
        PositionEmbeddingOptions::None => PositionEmbedding::None,
    }
}
