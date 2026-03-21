use crate::{karpathy::block::TransformerBlock, language_model::LanguageModel};
use tch::{
    nn::{self, Module},
    Tensor,
};

#[derive(Debug)]
pub struct TransformerLanguageModel {
    token_embedding_table: nn::Embedding,
    position_embedding_table: nn::Embedding,
    blocks: nn::Sequential,
    final_ln: nn::LayerNorm,
    language_modeling_head: nn::Linear,
    block_size: i64,
}

impl TransformerLanguageModel {
    pub fn new(
        path: nn::Path,
        vocab_size: i64,
        n_embed: i64,
        block_size: i64,
        n_blocks: i64,
        dropout: f64,
    ) -> Self {
        TransformerLanguageModel {
            token_embedding_table: nn::embedding(
                &path / "embedding",
                vocab_size,
                n_embed,
                Default::default(),
            ),
            position_embedding_table: nn::embedding(
                &path / "pos_embedding",
                block_size,
                n_embed,
                Default::default(),
            ),
            blocks: {
                (0..n_blocks)
                    .fold(nn::seq(), |s, i| {
                        s.add(TransformerBlock::new(
                            &path / ("b".to_owned() + &i.to_string()),
                            n_embed,
                            4,
                            block_size,
                            dropout,
                        ))
                    })
                    .add(nn::layer_norm(
                        &path / "blocks_ln",
                        vec![n_embed],
                        Default::default(),
                    ))
            },
            final_ln: nn::layer_norm(&path / "ln_f", vec![n_embed], Default::default()),
            language_modeling_head: nn::linear(
                &path / "lm_head",
                n_embed,
                vocab_size,
                Default::default(),
            ),
            block_size,
        }
    }
}

impl Module for TransformerLanguageModel {
    fn forward(&self, idx: &Tensor) -> Tensor {
        let (_b, t) = idx.size2().unwrap();
        let token_embeddings = self.token_embedding_table.forward(idx); //[B,T,C]
        let position_embeddings = self
            .position_embedding_table
            .forward(&Tensor::arange(t, (tch::Kind::Int, tch::Device::Cpu))); //[T,C]

        let x = self
            .blocks
            .forward(&(token_embeddings + position_embeddings));
        //[B,T,vocab_size]

        self.language_modeling_head
            .forward(&self.final_ln.forward(&x))
    }
}

impl LanguageModel for TransformerLanguageModel {
    fn get_block_size(&self) -> i64 {
        self.block_size
    }
}
