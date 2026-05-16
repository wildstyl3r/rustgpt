use std::cmp::min;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use tch::Tensor;

use crate::lm::{block::attention::AttentionConfig, ModelError, Result};

#[derive(ValueEnum, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum PositionEmbeddingOptions {
    Rope,
    Pope,
    None,
}

#[derive(Debug)]
pub enum PositionEmbedding {
    RoPE((Tensor, Tensor)),
    PoPE((Tensor, Tensor)),
    None,
}

impl PositionEmbedding {
    pub fn inject(&self, x: tch::Tensor) -> tch::Tensor {
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
                let (b, n, tx, d) = x.size4().unwrap();
                let (tf, _) = cos.size2().unwrap();
                let cos = cos.slice(0, 0, min(tx, tf), 1);
                let sin = sin.slice(0, 0, min(tx, tf), 1);
                let mut mu = x.softplus();
                let output = tch::Tensor::empty([b, n, tx, d * 2], (x.kind(), x.device()));
                output.slice(-1, 0, d, 1).copy_(&(&mu * &cos));
                output.slice(-1, d, 2 * d, 1).copy_(&mu.multiply_(&sin));
                output
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
        PositionEmbeddingOptions::Pope => {
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
        PositionEmbeddingOptions::Pope => {
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
