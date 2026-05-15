use std::cmp::min;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use tch::Tensor;

use crate::lm::{ModelError, Result};

#[derive(ValueEnum, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum PositionEmbeddingOptions {
    Rope,
    Pope,
    None,
}

#[derive(Debug)]
pub enum PositionEmbedding {
    RoPE((Tensor, Tensor)),
    // PoPE(),
    None,
}

impl PositionEmbedding {
    pub fn inject(&self, x: tch::Tensor) -> tch::Tensor {
        match self {
            PositionEmbedding::None => x,
            PositionEmbedding::RoPE((cos, sin)) => {
                let (_, _, tx, _) = x.size4().unwrap();
                let (tf, _) = cos.size2().unwrap();
                //[b, num_head, t, head_dim]
                //[b, num_head, t, hdp, 2]
                //[t, hdp] x2-> [t, hdp, 2]
                let mut shape = x.size();
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
        PositionEmbeddingOptions::Pope => todo!(),
        PositionEmbeddingOptions::None => PositionEmbedding::None,
    })
}
