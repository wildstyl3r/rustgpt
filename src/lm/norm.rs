use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use tch::nn::{self, LayerNorm, Module, Path};

#[derive(Debug)]
pub enum Norm {
    Layer(LayerNorm),
}
impl Module for Norm {
    fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {
        match self {
            Norm::Layer(layer_norm) => layer_norm.forward(xs),
        }
    }
}

#[derive(ValueEnum, Debug, Serialize, Deserialize, Clone)]
pub enum NormOptions {
    LayerNorm,
}

pub fn norm(path: Path, norm: &NormOptions, emb_dim: i64) -> Norm {
    match norm {
        NormOptions::LayerNorm => {
            Norm::Layer(nn::layer_norm(path, vec![emb_dim], Default::default()))
        }
    }
}

// impl Module for Box<dyn Norm> {
//     fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {
//         (**self).forward(xs)
//     }
// }
