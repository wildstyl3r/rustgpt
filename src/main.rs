use std::fs::{self, File};
use std::io::{BufReader, BufWriter};

use anyhow::Result;
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
use crate::utils::{estimate_loss, get_batch, train_val_split};

fn main() -> Result<()> {
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

            let text = std::fs::read_to_string(&params.input_path)?;
            let tokenizer = Tokenizer::new(&text);
            let (train, val) = train_val_split(&Tensor::from_slice(&tokenizer.encode(&text)), 0.9)?;

            let vs: nn::VarStore = tch::nn::VarStore::new(tch::Device::Cpu);
            let model = TransformerLanguageModel::new(
                vs.root(),
                tokenizer.vocabulary.len().try_into()?,
                params.model.n_embed,
                params.model.block_size,
                params.model.n_blocks,
                params.model.dropout,
            );

            let mut optimizer = nn::AdamW::default().build(&vs, params.learning_rate)?;

            for i in 0..params.max_iters {
                if i % params.eval_interval == 0 {
                    let losses = estimate_loss(
                        params.eval_iters,
                        &train,
                        &val,
                        params.batch_size,
                        params.model.block_size,
                        &model,
                    );
                    println!(
                        "step: {}, train loss: {:.4}, val loss: {:.4}",
                        i, losses.0, losses.1
                    );
                }

                let (xb, yb) = get_batch(&train, params.batch_size, params.model.block_size);

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
            fs::create_dir_all(&log_dir)?;
            vs.save(log_dir.join("model.safetensors"))?;
            let tk_file = File::create(log_dir.join("tokenizer.json"))?;
            serde_json::to_writer(BufWriter::new(tk_file), &tokenizer)?;
            (tokenizer, model)
        }
        Mode::Eval { checkpoint } => {
            let tk_file =
                File::open(checkpoint.join("tokenizer.json")).expect("tokenizer not found");
            let reader = BufReader::new(tk_file);
            let tokenizer: Tokenizer = serde_json::from_reader(reader)?;

            let content =
                std::fs::read_to_string(checkpoint.join("config.toml")).expect("Read error");
            let params: TrainArgs = toml::from_str(&content).expect("TOML error");

            let mut vs: nn::VarStore = tch::nn::VarStore::new(tch::Device::Cpu);
            let model = TransformerLanguageModel::new(
                vs.root(),
                tokenizer.vocabulary.len().try_into()?,
                params.model.n_embed,
                params.model.block_size,
                params.model.n_blocks,
                params.model.dropout,
            );
            vs.load(checkpoint.join("model.safetensors"))?;
            (tokenizer, model)
        }
    };

    let idx = Tensor::zeros([1, 1], (tch::Kind::Int, tch::Device::Cpu));
    println!(
        "[{}]",
        tokenizer.decode(&Vec::<Token>::try_from(model.generate(idx, 500).i(0))?)
    );
    Ok(())
}
