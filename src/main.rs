use std::fs::{self, File};
use std::io::{BufReader, BufWriter};

use clap::Parser;
use tch::{nn, nn::OptimizerConfig, IndexOp, Tensor};

mod baseline;
mod cli;
mod language_model;
mod tokenizer;
mod utils;

use crate::baseline::model::TransformerLanguageModel;
use crate::cli::{Cli, Config, Mode, TrainArgs};
use crate::language_model::LanguageModel;
use crate::tokenizer::{Token, Tokenizer};
use crate::utils::{estimate_loss, get_batch};

fn main() {
    let cli = Cli::parse();
    let git_hash = env!("GIT_HASH");

    let (tokenizer, model) = match cli.mode {
        Mode::Train { config } => {
            let params = match config {
                Config::File { path } => {
                    let content = std::fs::read_to_string(path).expect("Read error");
                    toml::from_str(&content).expect("TOML error")
                }
                Config::Cli(p) => p,
            };
            tch::manual_seed(1337);

            let text = std::fs::read_to_string(&params.input_path).unwrap();

            let tokenizer = Tokenizer::new(&text);

            let data = Tensor::from_slice(&tokenizer.encode(&text));

            let len = data.size1().expect("Unable to get data tensor size: main");
            let n = (0.9 * len as f32) as i64;
            let train_data = data.i(0..n);
            let validation_data = data.i(n..len - 1);

            let vs: nn::VarStore = tch::nn::VarStore::new(data.device());
            let model = TransformerLanguageModel::new(
                vs.root(),
                tokenizer.vocabulary.len().try_into().unwrap(),
                params.model.n_embed,
                params.model.block_size,
                params.model.n_blocks,
                params.model.dropout,
            );

            let mut optimizer = nn::AdamW::default()
                .build(&vs, params.learning_rate)
                .unwrap();

            for i in 0..params.max_iters {
                if i % params.eval_interval == 0 {
                    let losses = estimate_loss(
                        params.eval_iters,
                        &train_data,
                        &validation_data,
                        params.batch_size,
                        params.model.block_size,
                        &model,
                    );
                    println!(
                        "step: {}, train loss: {:.4}, val loss: {:.4}",
                        i, losses.0, losses.1
                    );
                }

                let (xb, yb) = get_batch(&train_data, params.batch_size, params.model.block_size);

                let (loss, _) = model.forward_with_loss(&xb, &yb, true);
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();
            }

            let log_dir = std::path::Path::new("checkpoints").join(format!(
                "run_{}_{}",
                chrono::Local::now().format("%Y%m%d_%H%M"),
                git_hash
            ));
            fs::create_dir_all(&log_dir).unwrap();
            vs.save(log_dir.join("model.safetensors")).unwrap();
            let tk_file = File::create(log_dir.join("tokenizer.json")).unwrap();
            serde_json::to_writer(BufWriter::new(tk_file), &tokenizer).unwrap();
            (tokenizer, model)
        }
        Mode::Eval { checkpoint } => {
            let tk_file =
                File::open(checkpoint.join("tokenizer.json")).expect("tokenizer not found");
            let reader = BufReader::new(tk_file);
            let tokenizer: Tokenizer = serde_json::from_reader(reader).unwrap();

            let content =
                std::fs::read_to_string(checkpoint.join("config.toml")).expect("Read error");
            let params: TrainArgs = toml::from_str(&content).expect("TOML error");

            let mut vs: nn::VarStore = tch::nn::VarStore::new(tch::Device::Cpu);
            let model = TransformerLanguageModel::new(
                vs.root(),
                tokenizer.vocabulary.len().try_into().unwrap(),
                params.model.n_embed,
                params.model.block_size,
                params.model.n_blocks,
                params.model.dropout,
            );
            vs.load(checkpoint.join("model.safetensors")).unwrap();
            (tokenizer, model)
        }
    };

    let idx = Tensor::zeros([1, 1], (tch::Kind::Int, tch::Device::Cpu));
    println!(
        "[{}]",
        tokenizer.decode(&Vec::<Token>::try_from(model.generate(idx, 500).i(0)).unwrap())
    );
}
