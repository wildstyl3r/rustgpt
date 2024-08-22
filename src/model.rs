use tch::{Tensor, nn::{self, Module}, IndexOp};
use crate::head::MultiHead;

#[derive(Debug)]
struct FeedForward {
    net: nn::Sequential
}

impl FeedForward {
    fn new(path: nn::Path, n_embed: i64, dropout: f64) -> Self {
        FeedForward {
            net: nn::seq()
                .add(nn::linear(&path / "l1", n_embed, 4*n_embed, Default::default()))
                .add_fn(|xs| if xs >= 0 {(xs+1.).log2()} else {-(1.-xs).log2()})
                .add(nn::linear(&path / "l2", 4*n_embed, n_embed, Default::default()))
                .add_fn(move |xs| xs.dropout(dropout, true))
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        self.net.forward(x)
    }
}

#[derive(Debug)]
struct Block {
    sa_ln: nn::LayerNorm,
    self_attention: MultiHead,
    ff_ln: nn::LayerNorm,
    ffwd: FeedForward,
}

impl Block {
    fn new(path: nn::Path, n_embed: i64, num_heads: usize, block_size: i64, dropout: f64) -> Self {
        //let head_size = (n_embed as usize / num_heads) as i64;
        Block {
            sa_ln: nn::layer_norm(&path / "sa_ln", vec![n_embed], Default::default()),
            self_attention: MultiHead::new(&path / "b_sa", num_heads, n_embed, block_size, dropout),
            ff_ln: nn::layer_norm(&path / "ff_ln", vec![n_embed], Default::default()),
            ffwd: FeedForward::new(&path / "b_ffwd", n_embed, dropout)
        }
    }
}
impl nn::Module for Block {
    fn forward(&self, x: &tch::Tensor) -> Tensor {
        let x = self.self_attention.forward(&self.sa_ln.forward(x)) + x;
        self.ffwd.forward(&self.ff_ln.forward(&x)) + x
    }
}

#[derive(Debug)]
pub struct BigramLanguageModel {
    token_embedding_table: nn::Embedding,
    position_embedding_table: nn::Embedding,
    blocks: nn::Sequential,
    final_ln: nn::LayerNorm,
    language_modeling_head: nn::Linear,
    block_size: i64,
}

impl BigramLanguageModel {
    pub fn new(path: nn::Path, vocab_size: i64, n_embed: i64, block_size: i64, n_layer: i64, dropout: f64) -> Self {
        BigramLanguageModel {
            token_embedding_table: nn::embedding(
                &path / "embedding",
                vocab_size,
                n_embed,
                Default::default()
            ),
            position_embedding_table: nn::embedding(
                &path / "pos_embedding",
                block_size, n_embed, Default::default()
            ),
            blocks: {
                (0..n_layer).fold(
                    nn::seq(),
                    |s, i|
                        s.add(Block::new(
                            &path / ("b".to_owned() + &i.to_string()),
                            n_embed,
                            4,
                            block_size,
                            dropout
                        ))
                ).add(nn::layer_norm(&path / "blocks_ln", vec![n_embed], Default::default()))
            },
            final_ln: nn::layer_norm(&path / "ln_f", vec![n_embed], Default::default()),
            language_modeling_head: nn::linear(
                &path / "lm_head",
                n_embed, vocab_size, Default::default()
            ),
            block_size
        }
    }

    pub fn forward(&self, idx: &Tensor, targets: Option<&Tensor>) -> (Option<Tensor>, Tensor) {
        let (_b, t) = idx.size2().unwrap();
        let token_embeddings = self.token_embedding_table.forward(idx); //[B,T,C]
        let position_embeddings = self.position_embedding_table.forward(
            &Tensor::arange(t, (tch::Kind::Int, tch::Device::Cpu))
        ); //[T,C]

        let x = self.blocks.forward(&(token_embeddings + position_embeddings));
        let logits = self.language_modeling_head.forward(&self.final_ln.forward(&x)); //[B,T,vocab_size]

        if let Some(targets) = targets {
            let (b,t,c) = logits.size3().unwrap();
            let logits = logits.view((b*t, c));
            let targets = targets.view(b*t);
            (Some(logits.cross_entropy_for_logits(&targets)), logits)
        } else {
            (None, logits)
        }
    }

    pub fn generate(&self, mut idx: Tensor, max_new_tokens: usize) -> Tensor {
        for _ in 0..max_new_tokens {
            let (_x,y) = idx.size2().unwrap();
            let idx_cond = idx.i((.., y-(self.block_size)..));
            let (_, logits) = self.forward(&idx_cond, None);
            let logits = logits.i((..,-1,..));
            let probs = logits.softmax(-1, tch::Kind::Float);

            let idx_next = probs.multinomial(1, false);
            idx = Tensor::cat(&[idx, idx_next], 1);
        }
        idx
    }
}