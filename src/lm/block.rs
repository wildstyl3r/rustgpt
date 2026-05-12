pub mod activation;
pub mod attention;
pub mod storage;
use crate::lm::{
    block::{attention::SelfAttention, storage::Storage},
    norm::{self, Norm},
};

use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};
use tch::{
    nn::{self, Module, ModuleT},
    Tensor,
};

#[derive(Args, Debug, Serialize, Deserialize, Clone)]
pub struct BlockConfig {
    #[arg(long, value_enum, default_value_t = norm::NormOptions::LayerNorm)]
    pub norm: norm::NormOptions,

    #[command(flatten)]
    pub self_attention: attention::SelfAttentionGroup,

    #[arg(long, value_enum, default_value_t = storage::StorageOptions::Feedforward)]
    pub storage: storage::StorageOptions,
}

#[derive(Args, Debug, Serialize, Deserialize)]
pub struct BlockGroup {
    #[arg(value_enum, long, default_value_t = BlockOption::Sequential)]
    pub block_option: BlockOption,
    #[command(flatten)]
    pub block_config: BlockConfig,
}

#[derive(ValueEnum, Debug, Serialize, Deserialize, Clone)]
pub enum BlockOption {
    Sequential,
    Parallel,
}

#[derive(Debug)]
pub enum Block {
    Sequential(SequentialBlock),
    // Parallel(todo!()),
}

impl ModuleT for Block {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        match self {
            Block::Sequential(sequential_block) => sequential_block.forward_t(xs, train),
        }
    }
}

pub fn block(
    path: nn::Path,
    block_type: &BlockOption,
    config: &BlockConfig,
    emb_dim: i64,
    context_window: i64,
    dropout: f64,
    causal_mask: Tensor,
) -> Block {
    match block_type {
        BlockOption::Sequential => Block::Sequential(SequentialBlock::new(
            path,
            emb_dim,
            config,
            context_window,
            dropout,
            causal_mask,
        )),
        BlockOption::Parallel => todo!(),
    }
}

#[derive(Debug)]
pub struct SequentialBlock {
    attention_norm: Norm,
    self_attention: SelfAttention,
    storage_norm: Norm,
    storage: Storage,
}

impl ModuleT for SequentialBlock {
    fn forward_t(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor {
        let xs = self
            .self_attention
            .forward_t(&self.attention_norm.forward(xs), train)
            + xs;
        self.storage.forward(&self.storage_norm.forward(&xs)) + xs
    }
}

impl SequentialBlock {
    fn new(
        path: nn::Path,
        emb_dim: i64,
        config: &BlockConfig,
        context_window: i64,
        dropout: f64,
        causal_mask: Tensor,
    ) -> Self {
        Self {
            attention_norm: norm::norm(&path / "sa_norm", &config.norm, emb_dim),
            self_attention: attention::self_attention(
                &path / "self_attention",
                &config.self_attention.attention_option,
                &config.self_attention.attention_config,
                emb_dim,
                dropout,
                causal_mask,
                context_window,
            ),
            storage_norm: norm::norm(&path / "st_norm", &config.norm, emb_dim),
            storage: storage::storage(&path / "storage", &config.storage, emb_dim, dropout),
        }
    }
}
