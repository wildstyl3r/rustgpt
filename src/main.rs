use tch::{nn, nn::OptimizerConfig, IndexOp, Tensor};
mod baseline;
mod language_model;
mod tokenizer;
mod utils;
use crate::baseline::model::TransformerLanguageModel;
use crate::language_model::LanguageModel;
use crate::tokenizer::{Token, Tokenizer};
use crate::utils::{estimate_loss, get_batch};

fn main() {
    tch::manual_seed(1337);
    let batch_size = 32;
    let block_size = 8;
    let max_iters = 1501;
    let eval_interval = 500;
    let learning_rate = 3e-3;
    let eval_iters = 200;
    let n_embed = 32;
    let n_blocks = 4;
    let dropout = 0.2;

    let text = std::fs::read_to_string("input.txt").unwrap();

    let tokenizer = Tokenizer::new(&text);

    let data = Tensor::from_slice(&tokenizer.encode(&text));

    let len = data.size1().expect("Unable to get data tensor size: main");
    let n = (0.9 * len as f32) as i64;
    let train_data = data.i(0..n);
    let validation_data = data.i(n..len - 1);

    let vs = tch::nn::VarStore::new(data.device());
    let model = TransformerLanguageModel::new(
        vs.root(),
        tokenizer.vocabulary.len().try_into().unwrap(),
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
                &model,
            );
            println!(
                "step: {}, train loss: {:.4}, val loss: {:.4}",
                i, losses.0, losses.1
            );
        }

        let (xb, yb) = get_batch(&train_data, batch_size, block_size);

        let (loss, _) = model.forward_with_loss(&xb, &yb, true);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }

    let log_dir = std::path::Path::new("checkpoints").join(format!(
        "run_{}_{}",
        chrono::Local::now().format("%Y%m%d_%H%M"),
        env!("GIT_HASH")
    ));
    std::fs::create_dir_all(&log_dir).unwrap();
    vs.save(log_dir.join("model.safetensors")).unwrap();

    let idx = Tensor::zeros([1, batch_size], (tch::Kind::Int, tch::Device::Cpu));
    println!(
        "[{}]",
        tokenizer.decode(&Vec::<Token>::try_from(model.generate(idx, 500).i(0)).unwrap())
    );
}
