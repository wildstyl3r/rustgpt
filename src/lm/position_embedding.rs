pub mod additive;
pub mod rotary;

use std::fmt::Debug;
use tch::{nn, Tensor};

pub trait Additive: Debug + Send {
    fn new(vs: nn::Path, context_window: i64, emb_dim: i64) -> Self;
    fn inject(&self, x: Tensor) -> Tensor;
}

pub trait Rotary: Debug + Send + 'static {
    fn new(vs: nn::Path, context_window: i64, emb_dim: i64) -> Self;
    fn inject(&self, x: Tensor) -> Tensor;
}

pub type None = ();
