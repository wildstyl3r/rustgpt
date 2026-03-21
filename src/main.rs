use tch::{nn, nn::OptimizerConfig, IndexOp, Tensor};

mod karpathy;
mod language_model;
mod utils;
mod vocabulary;
use crate::karpathy::model::TransformerLanguageModel;
use crate::language_model::LanguageModel;
use crate::utils::{decode, encode, estimate_loss, get_batch};
use crate::vocabulary::Vocabulary;

fn main() {
    tch::manual_seed(1337);
    let batch_size = 32;
    let block_size = 8;
    let max_iters = 15001;
    let eval_interval = 500;
    let learning_rate = 3e-3;
    let eval_iters = 200;
    let n_embed = 32;
    let n_blocks = 4;
    let dropout = 0.2;

    let text = std::fs::read_to_string("input.txt").unwrap();
    let vocabulary = Vocabulary::new(&text);
    let vocab_size = vocabulary.len();

    let data = Tensor::from_slice(&encode(text, Some(&vocabulary)));

    let len = data.size1().expect("Unable to get data tensor size: main");
    let n = (0.9 * len as f32) as i64;
    let train_data = data.i(0..n);
    let validation_data = data.i(n..len - 1);

    let vs = tch::nn::VarStore::new(data.device());
    let m = TransformerLanguageModel::new(
        vs.root(),
        vocab_size.try_into().unwrap(),
        n_embed,
        block_size,
        n_blocks,
        dropout,
    );

    let mut optimizer = nn::AdamW::default().build(&vs, learning_rate).unwrap();

    for i in 0..max_iters {
        if i % eval_interval == 0 {
            let losses = estimate_loss(
                eval_iters,
                &train_data,
                &validation_data,
                batch_size,
                block_size,
                &m,
            );
            println!(
                "step: {}, train loss: {:.4}, val loss: {:.4}",
                i, losses.0, losses.1
            );
        }

        let (xb, yb) = get_batch(&train_data, batch_size, block_size);

        let (loss, _) = m.forward_with_loss(&xb, &yb);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }

    let idx = Tensor::zeros([1, batch_size], (tch::Kind::Int, tch::Device::Cpu));
    println!(
        "[{}]",
        decode(
            Vec::<i64>::try_from(m.generate(idx, 500).i(0)).unwrap(),
            Some(&vocabulary)
        )
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dencoder() {
        let s = String::from("Hello, test!12345");
        assert_eq!(s.clone(), decode(encode(s, None), None))
    }
}
