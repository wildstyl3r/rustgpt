use tch::{
    nn::{self, Module, ModuleT},
    IndexOp,
};

use crate::lm::{
    block::{self, activation::AttentionActivation},
    position_embedding, Result,
};

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

    polar_bias: Option<tch::Tensor>,
}
impl ModuleT for MultiHeadSelfAttention {
    fn forward_t(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor {
        let q = &self.query.forward(xs) * self.vec_scale;
        let k = &self.key.forward(xs) * self.vec_scale;

        let (b, t, mulhead_dim) = q.size3().unwrap();
        let q = self.rotary_pe.inject(
            q.view([b, t, mulhead_dim / self.head_dim, self.head_dim])
                .transpose(-2, -3),
            &None,
        );
        let k = self.rotary_pe.inject(
            k.view([b, t, mulhead_dim / self.head_dim, self.head_dim])
                .transpose(-2, -3),
            &self.polar_bias,
        );
        let v = self
            .value
            .forward(xs)
            .view([b, t, mulhead_dim / self.head_dim, self.head_dim])
            .transpose(-2, -3);

        self.output
            .forward(
                &super::masked_attention(
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
    pub fn new(
        path: nn::Path,
        emb_dim: i64,
        config: &super::AttentionConfig,
        dropout: f64,
        causal_mask: tch::Tensor,
    ) -> Result<Self> {
        let polar_bias = match config.rotary_pe {
            position_embedding::rotary::PositionEmbeddingOptions::PopeBias => Some(path.randn(
                "polar_bias",
                &[config.num_heads, config.multihead_dim / config.num_heads],
                0.,
                0.75,
            )),
            _ => None,
        };
        Ok(Self {
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
                config
                    .frequencies
                    .as_ref()
                    .map(|(c, s)| (c.shallow_clone(), s.shallow_clone())),
            )?,
            head_dim: config.multihead_dim / config.num_heads,
            dropout,
            vec_scale: (config.multihead_dim as f64).powf(-0.25),
            causal_mask,
            activation: block::activation::attention_activation(&config.activation),
            polar_bias,
        })
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
