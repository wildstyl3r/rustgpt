use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[derive(ValueEnum, Debug, Serialize, Deserialize, Clone)]
pub enum AttentionActivationOptions {
    Softmax,
    ReLU,
}

#[derive(Debug)]
pub enum AttentionActivation {
    Softmax,
    ReLU,
}

impl AttentionActivation {
    pub fn apply(&self, x: &tch::Tensor) -> tch::Tensor {
        match self {
            AttentionActivation::Softmax => x.softmax(-1, tch::Kind::Float),
            AttentionActivation::ReLU => x.relu(),
        }
    }
}

pub fn attention_activation(options: &AttentionActivationOptions) -> AttentionActivation {
    match options {
        AttentionActivationOptions::Softmax => AttentionActivation::Softmax,
        AttentionActivationOptions::ReLU => AttentionActivation::ReLU,
    }
}
