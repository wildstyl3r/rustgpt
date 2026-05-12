use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use tch::nn::{self, Module};

#[derive(ValueEnum, Debug, Serialize, Deserialize, Clone)]
pub enum StorageOptions {
    FeedForward,
}

#[derive(Debug)]
pub enum Storage {
    FeedForward(nn::Sequential),
}

impl Module for Storage {
    fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {
        match self {
            Storage::FeedForward(sequential) => sequential.forward(xs),
        }
    }
}

pub fn storage(path: nn::Path, options: &StorageOptions, emb_dim: i64, dropout: f64) -> Storage {
    match options {
        StorageOptions::FeedForward => Storage::FeedForward(
            nn::seq()
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
        ),
    }
}
