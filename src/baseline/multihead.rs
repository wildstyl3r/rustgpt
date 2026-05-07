use tch::{
    nn::{self, Module, ModuleT},
    Tensor,
};

use crate::baseline::head::Head;

#[derive(Debug)]
pub struct MultiHead {
    heads: Vec<Head>,
    projection: nn::Linear,
    dropout: f64,
}

impl MultiHead {
    pub fn new(
        path: nn::Path,
        num_heads: usize,
        n_embed: i64,
        block_size: i64,
        dropout: f64,
    ) -> Self {
        let head_size = (n_embed as usize / num_heads) as i64;
        MultiHead {
            heads: (0..num_heads)
                .map(|i| {
                    Head::new(
                        &path / ("multi_head_c".to_string() + &i.to_string()),
                        head_size,
                        n_embed,
                        block_size,
                        dropout,
                    )
                })
                .collect(),
            projection: nn::linear(&path / "proj", n_embed, n_embed, Default::default()),
            dropout,
        }
    }
}

impl ModuleT for MultiHead {
    fn forward_t(&self, x: &Tensor, train: bool) -> Tensor {
        self.projection
            .forward(&Tensor::cat(
                &self
                    .heads
                    .iter()
                    .map(|h| h.forward_t(x, train))
                    .collect::<Vec<_>>(),
                -1,
            ))
            .dropout(self.dropout, true)
    }
}
