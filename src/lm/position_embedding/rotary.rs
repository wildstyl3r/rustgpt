use std::{cmp::min, f64::consts::PI};

use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use tch::Tensor;

use crate::lm::{block::attention::AttentionConfig, ModelError, Result};

#[derive(ValueEnum, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum PositionEmbeddingOptions {
    Rope,
    Pope,
    PopeBias,
    None,
}

#[derive(Debug)]
pub enum PositionEmbedding {
    RoPE((Tensor, Tensor)),
    PoPE((Tensor, Tensor)),
    None,
}

impl PositionEmbedding {
    pub fn inject(&self, x: tch::Tensor, polar_bias: &Option<tch::Tensor>) -> tch::Tensor {
        match self {
            PositionEmbedding::None => x,
            PositionEmbedding::RoPE((cos, sin)) => {
                //x: [b, num_head, t, head_dim]
                //f: [t, hdp]
                let (tf, _) = cos.size2().unwrap();
                let mut shape = x.size();
                let tx = shape[2];
                //x: [b, num_head, t, head_dim] -> [b, num_head, t, hdp, 2]
                shape.pop();
                shape.push(-1);
                shape.push(2);
                let x = x.reshape(shape);
                tch::Tensor::stack(
                    &[
                        tch::Tensor::linalg_vecdot(
                            &tch::Tensor::stack(&[cos, &-sin], -1).slice(-3, 0, min(tx, tf), 1),
                            &x,
                            -1,
                        ),
                        tch::Tensor::linalg_vecdot(
                            &tch::Tensor::stack(&[sin, cos], -1).slice(-3, 0, min(tx, tf), 1),
                            &x,
                            -1,
                        ),
                    ],
                    -1,
                )
                .flatten(-2, -1)
            }
            PositionEmbedding::PoPE((cos, sin)) => {
                //x: [b, num_head, t, head_dim]
                //f: [t, head_dim]
                let (_b, _n, tx, _d) = x.size4().unwrap();
                let (tf, _) = cos.size2().unwrap();
                let mut cos = cos.slice(0, 0, min(tx, tf), 1);
                let mut sin = sin.slice(0, 0, min(tx, tf), 1);
                //pb: [num_head, head_dim]
                if let Some(unbounded_bias) = polar_bias.as_ref() {
                    //bd: [num_head, _, head_dim]
                    let bounded_delta = ((unbounded_bias.tanh() - 1) * PI).unsqueeze(1);
                    let cos_delta = bounded_delta.cos();
                    let sin_delta = bounded_delta.sin();
                    (sin, cos) = (
                        &sin * &cos_delta + &cos * &sin_delta,
                        cos * cos_delta - sin * sin_delta,
                    )
                }
                let mu = x.softplus();
                tch::Tensor::cat(&[&(&mu * cos), &(mu * sin)], -1)
            }
        }
    }
}

pub fn embedding(
    options: &PositionEmbeddingOptions,
    freqs: Option<(Tensor, Tensor)>,
) -> Result<PositionEmbedding> {
    Ok(match options {
        PositionEmbeddingOptions::Rope => {
            PositionEmbedding::RoPE(freqs.ok_or(ModelError::InitializationError(
                "rotary embedding init: no precomputed frequencies".to_string(),
            ))?)
        }
        PositionEmbeddingOptions::Pope | PositionEmbeddingOptions::PopeBias => {
            PositionEmbedding::PoPE(freqs.ok_or(ModelError::InitializationError(
                "polar embedding init: no precomputed frequencies".to_string(),
            ))?)
        }
        PositionEmbeddingOptions::None => PositionEmbedding::None,
    })
}

pub fn precompute(att_config: &mut AttentionConfig, context_window: i64) {
    match att_config.rotary_pe {
        PositionEmbeddingOptions::Rope => {
            let d = att_config.multihead_dim / att_config.num_heads;
            let frequencies = Tensor::arange(context_window, (tch::Kind::Float, tch::Device::Cpu))
                .outer(&Tensor::from_slice(
                    &(1..=d / 2)
                        .map(|i| {
                            (att_config.frequency_base.powf(-(2 * (i - 1)) as f64) / (d as f64))
                                as f32
                        })
                        .collect::<Vec<f32>>(),
                ));
            att_config.frequencies = Some((frequencies.cos(), frequencies.sin()))
        }
        PositionEmbeddingOptions::Pope | PositionEmbeddingOptions::PopeBias => {
            let d = att_config.multihead_dim / att_config.num_heads;
            let frequencies = Tensor::arange(context_window, (tch::Kind::Float, tch::Device::Cpu))
                .outer(&Tensor::from_slice(
                    &(1..=d)
                        .map(|i| {
                            (att_config.frequency_base.powf((i - 1) as f64) / (d as f64)) as f32
                        })
                        .collect::<Vec<f32>>(),
                ));
            att_config.frequencies = Some((frequencies.cos(), frequencies.sin()))
        }
        PositionEmbeddingOptions::None => (),
    }
}
