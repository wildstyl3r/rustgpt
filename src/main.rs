use std::collections::{BTreeMap, BTreeSet};
use tch::{Tensor, nn::{self, Module, OptimizerConfig}, IndexOp};

struct Vocabulary {
    stoi: BTreeMap<String, i64>,
    itos: Vec<String>
}

impl Vocabulary {
    fn new(s: &str) -> Self {
        let mut set = BTreeSet::new();
        for c in s.chars() {
            set.insert(c);
        }
        let mut stoi = BTreeMap::new();
        let mut itos = Vec::new();
        for (i, c) in set.iter().enumerate() {
            itos.push(c.to_string());
            stoi.insert(c.to_string(), i as i64);
        }
        Vocabulary { stoi, itos}
    }

    fn len(&self) -> usize {
        self.stoi.len()
    }
}

fn encode(s: String, vocabulary: Option<&Vocabulary>) -> Vec<i64> {
    match vocabulary {
        Some(v) => s.chars().map(|c| v.stoi[&c.to_string()]).collect(),
        None => s.chars().map(|c| (c as u32) as i64).collect()
    }
}

fn decode(v: Vec<i64>, vocabulary: Option<&Vocabulary>) -> String {
    match vocabulary {
        Some(voc) => v.iter().map(|i| voc.itos[*i as usize].clone()).collect(),
        None => v.iter().filter_map(|i| char::from_u32(*i as u32)).collect()
    }
}


fn get_batch(data: &Tensor, batch_size: i64, block_size: i64) -> (Tensor, Tensor) {
    let ix = Tensor::randint(data.size1().unwrap() - block_size, [batch_size], (tch::Kind::Int, tch::Device::Cpu));

    let x = Tensor::stack(&ix.iter::<i64>().unwrap().map(|i| data.i(i .. i + block_size)).collect::<Vec<_>>(), 0);
    let y = Tensor::stack(&ix.iter::<i64>().unwrap().map(|i| data.i(i + 1 .. i + block_size + 1)).collect::<Vec<_>>(), 0);
    (x, y)
}

fn estimate_loss(eval_iters: i64, train_data: &Tensor, validation_data: &Tensor, batch_size: i64, block_size: i64, m: &BigramLanguageModel) -> [Tensor; 2] {
    tch::no_grad(||{
        let mut losses_train = Tensor::zeros(eval_iters, (tch::Kind::Float, tch::Device::Cpu));
        for k in 0..eval_iters {
            let (x, y) = get_batch(train_data, batch_size, block_size);
            let (loss, _) = m.forward(&x, Some(&y));
            let loss = loss.unwrap();
            losses_train = losses_train.index_put_(&[Some(Tensor::from(k))], &loss, false);
        }

        let mut losses_val = Tensor::zeros(eval_iters, (tch::Kind::Float, tch::Device::Cpu));
        for k in 0..eval_iters {
            let (x, y) = get_batch(validation_data, batch_size, block_size);
            let (loss, _) = m.forward(&x, Some(&y));
            let loss = loss.unwrap();
            losses_val = losses_val.index_put_(&[Some(Tensor::from(k))], &loss, false);
        }
        //[0.0, 0.0]
        [losses_train.mean(tch::Kind::Float), losses_val.mean(tch::Kind::Float)]
    })
}

#[derive(Debug)]
struct Head {
    key: nn::Linear,
    query: nn::Linear,
    value: nn::Linear,
    tril: Tensor,
}

impl Head {
    fn new(path: nn::Path, head_size: i64, n_embed: i64, block_size: i64) -> Self {
        Head {
            key: nn::linear(&path / "head_k", n_embed, head_size, nn::LinearConfig{bias: false, ..Default::default()}),
            query: nn::linear(&path / "head_q", n_embed, head_size, nn::LinearConfig{bias: false, ..Default::default()}),
            value: nn::linear(&path / "head_v", n_embed, head_size, nn::LinearConfig{bias: false, ..Default::default()}),
            tril: Tensor::ones([block_size, block_size], (tch::Kind::Float, tch::Device::Cpu)).tril(0),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let (_b, t, c) = x.size3().unwrap();
        let k = self.key.forward(x);
        let q = self.query.forward(x);
        let wei = (q.matmul(&k.transpose(-2, -1)) / (c as f64).sqrt())//powf(-0.5))
            .masked_fill(&self.tril.i((..t, ..t)).eq(0), f64::NEG_INFINITY)
            .softmax(-1, tch::Kind::Float);
        let v = self.value.forward(x);
        wei.matmul(&v)
    }
}


#[derive(Debug)]
struct BigramLanguageModel {
    token_embedding_table: nn::Embedding,
    position_embedding_table: nn::Embedding,
    self_attention_head: Head,
    language_modeling_head: nn::Linear,
    block_size: i64,
}

impl BigramLanguageModel {
    fn new(path: nn::Path, vocab_size: i64, n_embed: i64, block_size: i64) -> Self {
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
            self_attention_head: Head::new(
                &path / "self_att_head",
                n_embed, n_embed, block_size
            ),
            language_modeling_head: nn::linear(
                &path / "lm_head",
                n_embed, vocab_size, Default::default()
            ),
            block_size
        }
    }

    fn forward(&self, idx: &Tensor, targets: Option<&Tensor>) -> (Option<Tensor>, Tensor) {
        let (_b, t) = idx.size2().unwrap();
        let token_embeddings = self.token_embedding_table.forward(idx); //[B,T,C]
        let position_embeddings = self.position_embedding_table.forward(&Tensor::arange(t, (tch::Kind::Int, tch::Device::Cpu))); //[T,C]

        let x = self.self_attention_head.forward(&(token_embeddings + position_embeddings));

        let logits = self.language_modeling_head.forward(&x); //[B,T,vocab_size]

        if let Some(targets) = targets {
            let (b,t,c) = logits.size3().unwrap();
            let logits = logits.view((b*t, c));
            let targets = targets.view(b*t);
            (Some(logits.cross_entropy_for_logits(&targets)), logits)
        } else {
            (None, logits)
        }
    }

    fn generate(&self, mut idx: Tensor, max_new_tokens: usize) -> Tensor {
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

fn main() {
    tch::manual_seed(1337);
    let batch_size = 32;
    let block_size = 8;
    let max_iters = 5000;
    let eval_interval = 500;
    let learning_rate = 1e-3;
    let eval_iters = 200;
    let n_embed = 32;

    let text = std::fs::read_to_string("input.txt").unwrap();
    let vocabulary = Vocabulary::new(&text);
    let vocab_size = vocabulary.len();

    let data = Tensor::from_slice(&encode(text, Some(&vocabulary)));

    let len = data.size1().expect("Unable to get data tensor size: main");
    let n = (0.9 * len as f32) as i64;
    let train_data = data.i(0..n);
    let validation_data = data.i(n..len-1);


    let vs = tch::nn::VarStore::new(data.device());
    let m = BigramLanguageModel::new(vs.root(), vocab_size.try_into().unwrap(), n_embed, block_size);

    let mut optimizer = nn::AdamW::default().build(&vs, learning_rate).unwrap();

    for i in 0..max_iters {
        if i % eval_interval == 0 {
            let losses = estimate_loss(eval_iters, &train_data, &validation_data, batch_size, block_size, &m);
            println!("step: {}, train loss: {}, val loss: {}", i, losses[0], losses[1]);
        }

        let (xb, yb) = get_batch(&train_data, batch_size, block_size);

        let (loss, _) = m.forward(&xb, Some(&yb));
        optimizer.zero_grad();
        if let Some(l) = loss {
            l.backward();
        } else {
            panic!();
        }
        optimizer.step();
    }


    let idx = Tensor::zeros([1,batch_size], (tch::Kind::Int, tch::Device::Cpu));
    println!("{}", decode(Vec::<i64>::try_from(m.generate(idx, 500).i(0)).unwrap(), Some(&vocabulary)));
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dencoder () {
        let s = String::from("Hello, test!12345");
        assert_eq!(s.clone(), decode(encode(s, None), None))
    }
}