mod quadratic;
use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};
use tch::nn::{self, ModuleT};

use crate::lm::{
    block::{self, activation::AttentionActivation},
    position_embedding, Result,
};

fn masked_attention(
    q: &tch::Tensor,
    k: &tch::Tensor,
    v: &tch::Tensor,
    mask: &tch::Tensor,
    dropout: f64,
    train: bool,
    activation: &AttentionActivation,
) -> tch::Tensor {
    activation
        .apply(
            &q.matmul(&k.transpose(-1, -2))
                .masked_fill(mask, f64::NEG_INFINITY),
        )
        .dropout(dropout, train)
        .matmul(v)
}

#[derive(Args, Debug, Serialize, Deserialize)]
#[group(multiple = false)]
pub struct SelfAttentionGroup {
    #[arg(long, value_enum, default_value_t = SelfAttentionOptions::MultiHeadSelfAttention)]
    pub attention_option: SelfAttentionOptions,
    #[command(flatten)]
    pub attention_config: AttentionConfig,
}

#[derive(Debug, ValueEnum, Clone, Serialize, Deserialize)]
pub enum SelfAttentionOptions {
    MultiHeadSelfAttention,
}

#[derive(Args, Debug, Serialize, Deserialize)]
pub struct AttentionConfig {
    #[arg(long, value_enum, default_value_t = position_embedding::rotary::PositionEmbeddingOptions::None)]
    pub rotary_pe: position_embedding::rotary::PositionEmbeddingOptions,

    #[arg(skip)]
    #[serde(skip)]
    pub frequencies: Option<(tch::Tensor, tch::Tensor)>,

    #[arg(long, default_value_t = 10_000.)]
    pub frequency_base: f64,

    #[arg(long, value_enum, default_value_t = block::activation::AttentionActivationOptions::Softmax)]
    pub activation: block::activation::AttentionActivationOptions,

    #[arg(long, default_value_t = 32)]
    pub multihead_dim: i64,

    #[arg(long, default_value_t = 4)]
    pub num_heads: i64,

    #[arg(long)]
    pub qk_norm: bool,
}

#[derive(Debug)]
pub enum SelfAttention {
    MultiHead(quadratic::MultiHeadSelfAttention),
}
impl ModuleT for SelfAttention {
    fn forward_t(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor {
        match self {
            SelfAttention::MultiHead(multi_head_self_attention) => {
                multi_head_self_attention.forward_t(xs, train)
            }
        }
    }
}

pub fn self_attention(
    path: nn::Path,
    options: &SelfAttentionOptions,
    config: &AttentionConfig,
    emb_dim: i64,
    dropout: f64,
    causal_mask: tch::Tensor,
) -> Result<SelfAttention> {
    Ok(match options {
        SelfAttentionOptions::MultiHeadSelfAttention => SelfAttention::MultiHead(
            quadratic::MultiHeadSelfAttention::new(path, emb_dim, config, dropout, causal_mask)?,
        ),
    })
}
