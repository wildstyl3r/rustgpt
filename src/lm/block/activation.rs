use std::fmt::Debug;

pub trait AttentionActivation: Debug + Send + 'static {
    fn apply(xs: &tch::Tensor) -> tch::Tensor;
}

#[derive(Debug)]
pub struct Softmax;
impl AttentionActivation for Softmax {
    fn apply(xs: &tch::Tensor) -> tch::Tensor {
        xs.softmax(-1, tch::Kind::Float)
    }
}

#[derive(Debug)]
pub struct ReLU;
impl AttentionActivation for ReLU {
    fn apply(xs: &tch::Tensor) -> tch::Tensor {
        xs.relu()
    }
}
