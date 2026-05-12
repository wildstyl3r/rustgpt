use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};
use tch::{
    nn::{self, Module, ModuleT},
    IndexOp,
};

use crate::lm::{
    block::{self, activation::AttentionActivation},
    position_embedding,
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

#[derive(Args, Clone, Debug, Serialize, Deserialize)]
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

#[derive(Args, Debug, Serialize, Deserialize, Clone)]
pub struct AttentionConfig {
    #[arg(long, value_enum, default_value_t = position_embedding::rotary::PositionEmbeddingOptions::None)]
    pub rotary_pe: position_embedding::rotary::PositionEmbeddingOptions,

    #[arg(long, value_enum, default_value_t = block::activation::AttentionActivationOptions::Softmax)]
    pub activation: block::activation::AttentionActivationOptions,

    #[arg(long, default_value_t = 32)]
    pub multihead_dim: i64,

    #[arg(long, default_value_t = 4)]
    pub num_heads: i64,
}

#[derive(Debug)]
pub enum SelfAttention {
    MultiHead(MultiHeadSelfAttention),
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
    context_window: i64,
) -> SelfAttention {
    match options {
        SelfAttentionOptions::MultiHeadSelfAttention => {
            SelfAttention::MultiHead(MultiHeadSelfAttention::new(
                path,
                emb_dim,
                config,
                dropout,
                causal_mask,
                context_window,
            ))
        }
    }
}

#[derive(Debug)]
pub struct MultiHeadSelfAttention {
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    output: nn::Linear,

    activation: AttentionActivation,
    rotary_pe: position_embedding::rotary::PositionEmbedding,
    head_dim: i64,
    dropout: f64,
    vec_scale: f64,
    causal_mask: tch::Tensor,
}
impl ModuleT for MultiHeadSelfAttention {
    fn forward_t(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor {
        //[b, t, proj_out_dim] = [b, t, num_heads * head_dim]
        let q = &self.query.forward(xs) * self.vec_scale;
        let k = &self.key.forward(xs) * self.vec_scale;

        let (b, t, mulhead_dim) = q.size3().unwrap();
        let q = self.rotary_pe.inject(
            q.view([b, t, mulhead_dim / self.head_dim, self.head_dim])
                .transpose(-2, -3),
        );
        let k = self.rotary_pe.inject(
            k.view([b, t, mulhead_dim / self.head_dim, self.head_dim])
                .transpose(-2, -3),
        );
        let v = self
            .value
            .forward(xs)
            .view([b, t, mulhead_dim / self.head_dim, self.head_dim])
            .transpose(-2, -3);

        self.output
            .forward(
                &masked_attention(
                    &q,
                    &k,
                    &v,
                    &self.causal_mask.i((..t, ..t)),
                    self.dropout,
                    train,
                    &self.activation,
                )
                .transpose(-2, -3)
                .reshape([b, t, mulhead_dim]),
            )
            .dropout(self.dropout, train)
    }
}

impl MultiHeadSelfAttention {
    fn new(
        path: nn::Path,
        emb_dim: i64,
        config: &AttentionConfig,
        dropout: f64,
        causal_mask: tch::Tensor,
        context_window: i64,
    ) -> Self {
        Self {
            query: nn::linear(
                &path / "w_q",
                emb_dim,
                config.multihead_dim,
                nn::LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            key: nn::linear(
                &path / "w_k",
                emb_dim,
                config.multihead_dim,
                nn::LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            value: nn::linear(
                &path / "w_v",
                emb_dim,
                config.multihead_dim,
                nn::LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            output: nn::linear(
                &path / "w_q",
                config.multihead_dim,
                emb_dim,
                nn::LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            rotary_pe: position_embedding::rotary::embedding(
                &config.rotary_pe,
                &path / "rotary_pe",
                emb_dim,
                context_window,
            ),
            head_dim: config.multihead_dim / config.num_heads,
            dropout,
            vec_scale: (config.multihead_dim as f64).powf(-0.25),
            causal_mask,
            activation: block::activation::attention_activation(&config.activation),
        }
    }
    pub fn projection_weights(&self) -> (tch::Tensor, tch::Tensor, tch::Tensor, tch::Tensor) {
        (
            self.query.ws.shallow_clone(),
            self.key.ws.shallow_clone(),
            self.value.ws.shallow_clone(),
            self.output.ws.shallow_clone(),
        )
    }
}
