use std::marker::PhantomData;

use tch::{
    nn::{self, Module, ModuleT},
    IndexOp,
};

use crate::lm::{
    block::activation::AttentionActivation,
    position_embedding::{self, Rotary},
};

fn masked_attention<A: AttentionActivation>(
    q: &tch::Tensor,
    k: &tch::Tensor,
    v: &tch::Tensor,
    mask: &tch::Tensor,
    dropout: f64,
    train: bool,
) -> tch::Tensor {
    A::apply(
        &q.matmul(&k.transpose(-1, -2))
            .masked_fill(mask, f64::NEG_INFINITY),
    )
    .dropout(dropout, train)
    .matmul(v)
}

pub trait SelfAttention: ModuleT + Send + 'static {
    fn new(
        path: nn::Path,
        emb_dim: i64,
        multihead_dim: i64,
        num_heads: i64,
        dropout: f64,
        causal_mask: tch::Tensor,
        context_window: i64,
    ) -> Self;
    fn projection_weights(&self) -> (tch::Tensor, tch::Tensor, tch::Tensor, tch::Tensor);
}

#[derive(Debug)]
pub struct MultiHeadSelfAttention<R: position_embedding::Rotary, A: AttentionActivation> {
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    output: nn::Linear,

    rotary_pe: R,
    _att_act_marker: PhantomData<A>,
    head_dim: i64,
    dropout: f64,
    vec_scale: f64,
    causal_mask: tch::Tensor,
}
impl<R: position_embedding::Rotary, A: AttentionActivation> ModuleT
    for MultiHeadSelfAttention<R, A>
{
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
                &masked_attention::<A>(
                    &q,
                    &k,
                    &v,
                    &self.causal_mask.i((..t, ..t)),
                    self.dropout,
                    train,
                )
                .transpose(-2, -3)
                .reshape([b, t, mulhead_dim]),
            )
            .dropout(self.dropout, train)
    }
}

impl<R: Rotary, A: AttentionActivation> SelfAttention for MultiHeadSelfAttention<R, A> {
    fn new(
        path: nn::Path,
        emb_dim: i64,
        multihead_dim: i64,
        num_heads: i64,
        dropout: f64,
        causal_mask: tch::Tensor,
        context_window: i64,
    ) -> Self {
        Self {
            query: nn::linear(
                &path / "w_q",
                emb_dim,
                multihead_dim,
                nn::LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            key: nn::linear(
                &path / "w_k",
                emb_dim,
                multihead_dim,
                nn::LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            value: nn::linear(
                &path / "w_v",
                emb_dim,
                multihead_dim,
                nn::LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            output: nn::linear(
                &path / "w_q",
                multihead_dim,
                emb_dim,
                nn::LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            rotary_pe: R::new(&path / "rotary_pe", context_window, emb_dim),
            head_dim: multihead_dim / num_heads,
            dropout,
            vec_scale: (multihead_dim as f64).powf(-0.25),
            causal_mask,
            _att_act_marker: PhantomData,
        }
    }

    fn projection_weights(&self) -> (tch::Tensor, tch::Tensor, tch::Tensor, tch::Tensor) {
        (
            self.query.ws.shallow_clone(),
            self.key.ws.shallow_clone(),
            self.value.ws.shallow_clone(),
            self.output.ws.shallow_clone(),
        )
    }
}
