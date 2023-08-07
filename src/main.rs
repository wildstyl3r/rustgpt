use std::collections::{HashMap, HashSet};
use tch::{Tensor, nn::{self, Module, OptimizerConfig}, IndexOp};
use rand::{rngs::StdRng, SeedableRng};

struct Vocabulary {
    stoi: HashMap<String, i64>,
    itos: Vec<String>
}

impl Vocabulary {
    fn new(s: &str) -> Self {
        let mut set = HashSet::new();
        for c in s.chars() {
            set.insert(c);
        }
        let mut stoi = HashMap::new();
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

fn normalize(s: String) -> String {
    todo!();
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
    let mut rng = StdRng::seed_from_u64(23);

    let ixr = rand::seq::index::sample(
        &mut rng,
        (data.size1().expect("tch size err: get_batch") - block_size) as usize,
        batch_size as usize
    );
    let context = Tensor::stack(
        &ixr.iter().map(|i|
            data.i(i as i64 .. i as i64 + block_size)
        ).collect::<Vec<_>>(),
        0
    );
    let target = Tensor::stack(
        &ixr.iter().map(|i|
            data.i(i as i64 + 1 .. i as i64 + block_size + 1)
        ).collect::<Vec<_>>(),
        0
    );
    (context, target)
}

#[derive(Debug)]
struct BigramLanguageModel {
    token_embedding_table: nn::Embedding,
    last_loss: Option<Tensor>,
}

impl BigramLanguageModel {
    fn new(path: nn::Path, vocab_size: i64) -> Self {
        BigramLanguageModel {
            token_embedding_table: nn::embedding(
                path / "embedding",
                vocab_size,
                vocab_size,
                Default::default()
            ),
            last_loss: None
        }
    }

    fn forward(&self, idx: &Tensor, targets: Option<&Tensor>) -> (Option<Tensor>, Tensor) {
        let logits = self.token_embedding_table.forward(idx);
        match targets {
            Some(tar) => {
                let (b,t,c) = logits.size3().unwrap();
                let logits = logits.view((b*t, c));
                let targets = tar.view(b*t);
                (Some(logits.cross_entropy_for_logits(&targets)), logits)
            },
            None => (None, logits)
        }
    }

    fn generate(&self, mut idx: Tensor, max_new_tokens: usize) -> Tensor {
        for i in 0..max_new_tokens {
            let (_, logits) = self.forward(&idx, None);
            let logits = logits.i((..,-1,..));
            let probs = logits.softmax(-1, tch::Kind::Float);

            let idx_next = probs.multinomial(1, false);
            idx = Tensor::cat(&[idx, idx_next], 1);
        }
        idx
    }
}


fn main() {
    let text = std::fs::read_to_string("input.txt").unwrap_or("No text".to_string());
    let vocabulary = Vocabulary::new(&text);
    let vocab_size = vocabulary.len();

    let data = Tensor::from_slice(&encode(text, Some(&vocabulary)));

    let len = data.size1().expect("Unable to get data tensor size: main");
    let n = (0.9 * len as f32) as i64;
    let train_data = data.i(0..n);
    let validation_data = data.i(n..len-1);

    let block_size = 8;
    let batch_size = 32;

    let (xb, yb) = get_batch(&train_data, batch_size, block_size);

    let vs = tch::nn::VarStore::new(data.device());
    let m = BigramLanguageModel::new(vs.root(), vocab_size.try_into().unwrap());

    // for b in 0..batch_size {
    //     for t in 0..block_size {
    //         let context = xb.i((b, ..t+1));
    //         let target = yb.i((b, t));
    //         println!("input: {}, target: {}", context, target);
    //     }
    // }
    let idx = Tensor::zeros([1,1], (tch::Kind::Int, tch::Device::Cpu));

    let mut optimizer = nn::AdamW::default().build(&vs, 10e-3).unwrap();

    for _ in 0..1000 {
        let (xb, yb) = get_batch(&train_data, batch_size, block_size);
        let (loss, _) = m.forward(&xb, Some(&yb));
        optimizer.zero_grad();
        if let Some(l) = loss {
            l.backward();
            println!("{}", l);
        }
        optimizer.step();
    }

    println!("{}", decode(Vec::<i64>::try_from(m.generate(idx, 400).i(0)).unwrap(), Some(&vocabulary)));
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dencoder () {
        let s = String::from("Hello, test!12345");
        assert_eq!(s.clone(), decode(encode(s, None), None))
    }

    #[test]
    fn test_bigram_language_model() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let vocab_size = 100;
        let batch_size = 2;
        let block_size = 5;

        let model = BigramLanguageModel::new(vs.root(), vocab_size);
        let xs = Tensor::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to_kind(tch::Kind::Int);
        let xs = xs.view([batch_size, block_size]);
        println!("xs: {:?}", xs);
        let (b, t) = xs.size2().unwrap();
        assert_eq!(b, batch_size);
        assert_eq!(t, block_size);

        let (_, logits) = model.forward(&xs, None);
        assert_eq!(logits.size(), [batch_size, block_size, vocab_size]);
    }
}