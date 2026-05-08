use crate::{baseline::block::TransformerBlock, cli::ModelArgs, language_model::LanguageModel};
use tch::{
    nn::{self, Module, ModuleT},
    Tensor,
};

#[derive(Debug)]
pub struct BaselineModel {
    token_embedding_table: nn::Embedding,
    position_embedding_table: nn::Embedding,
    blocks: nn::SequentialT,
    final_ln: nn::LayerNorm,
    language_modeling_head: nn::Linear,
    block_size: i64,
}

impl BaselineModel {
    pub fn new(path: nn::Path, vocab_size: i64, config: &ModelArgs) -> Self {
        BaselineModel {
            token_embedding_table: nn::embedding(
                &path / "embedding",
                vocab_size,
                config.n_embed,
                Default::default(),
            ),
            position_embedding_table: nn::embedding(
                &path / "pos_embedding",
                config.block_size,
                config.n_embed,
                Default::default(),
            ),
            blocks: {
                (0..config.n_blocks)
                    .fold(nn::seq_t(), |s, i| {
                        s.add(TransformerBlock::new(
                            &path / ("b".to_owned() + &i.to_string()),
                            config.n_embed,
                            4,
                            config.block_size,
                            config.dropout,
                        ))
                    })
                    .add(nn::layer_norm(
                        &path / "blocks_ln",
                        vec![config.n_embed],
                        Default::default(),
                    ))
            },
            final_ln: nn::layer_norm(&path / "ln_f", vec![config.n_embed], Default::default()),
            language_modeling_head: nn::linear(
                &path / "lm_head",
                config.n_embed,
                vocab_size,
                Default::default(),
            ),
            block_size: config.block_size,
        }
    }
}

impl ModuleT for BaselineModel {
    fn forward_t(&self, idx: &Tensor, train: bool) -> Tensor {
        let (_b, t) = idx.size2().unwrap();
        let token_embeddings = self.token_embedding_table.forward(idx); //[B,T,C]
        let position_embeddings = self
            .position_embedding_table
            .forward(&Tensor::arange(t, (tch::Kind::Int, tch::Device::Cpu))); //[T,C]

        let x = self
            .blocks
            .forward_t(&(token_embeddings + position_embeddings), train);
        //[B,T,vocab_size]

        self.language_modeling_head
            .forward(&self.final_ln.forward(&x))
    }
}

impl LanguageModel for BaselineModel {
    fn get_block_size(&self) -> i64 {
        self.block_size
    }
}
