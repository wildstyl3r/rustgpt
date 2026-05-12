use tch::nn::{self, LayerNorm, Module, Path};

pub trait Norm: Module + 'static {
    fn new(path: Path, emb_dim: i64) -> Self;
}

impl Norm for LayerNorm {
    fn new(path: Path, emb_dim: i64) -> Self {
        nn::layer_norm(path, vec![emb_dim], Default::default())
    }
}
// impl Norm for RmsNorm {}
