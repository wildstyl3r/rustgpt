use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[derive(ValueEnum, Debug, Serialize, Deserialize, Clone)]
pub enum AttentionActivationOptions {
    Softmax,
    Relu,
    Renorm,
}

#[derive(Debug)]
pub enum AttentionActivation {
    Softmax,
    ReLU,
    ReNorm,
}

impl AttentionActivation {
    pub fn apply(&self, x: &tch::Tensor) -> tch::Tensor {
        match self {
            AttentionActivation::Softmax => x.softmax(-1, tch::Kind::Float),
            AttentionActivation::ReLU => x.relu(),
            AttentionActivation::ReNorm => {
                let r = x.relu();
                &r / (r.sum_dim_intlist(&[-1][..], true, None) + 1e-8)
            }
        }
    }
}

pub fn attention_activation(options: &AttentionActivationOptions) -> AttentionActivation {
    match options {
        AttentionActivationOptions::Softmax => AttentionActivation::Softmax,
        AttentionActivationOptions::Relu => AttentionActivation::ReLU,
        AttentionActivationOptions::Renorm => AttentionActivation::ReNorm,
    }
}
