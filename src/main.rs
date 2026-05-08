use std::fs::{self};

use anyhow::Result;
use clap::Parser;
use tch::{nn, nn::OptimizerConfig, IndexOp, Tensor};

mod baseline;
mod cli;
mod language_model;
mod tokenizer;
mod utils;

use crate::baseline::model::BaselineModel;
use crate::cli::{Cli, Config, Mode, TrainArgs};
use crate::language_model::LanguageModel;
use crate::tokenizer::{Token, Tokenizer};
use crate::utils::{estimate_loss, get_batch, train_val_split};

fn main() -> Result<()> {
    let cli = Cli::parse();
    let git_hash = env!("GIT_HASH");

    let (tokenizer, model) = match cli.mode {
        Mode::Train { config } => {
            let config = match config {
                Config::File { path } => TrainArgs::load(path)?,
                Config::Cli(config) => config,
            };
            tch::manual_seed(1337);

            let text = std::fs::read_to_string(&config.dataset.input_path)?;
            let tokenizer = Tokenizer::new(&text);
            let (train, val) = train_val_split(
                &Tensor::from_slice(&tokenizer.encode(&text)),
                config.dataset.train_share,
            )?;

            let vs: nn::VarStore = tch::nn::VarStore::new(tch::Device::Cpu);
            let model =
                BaselineModel::new(vs.root(), tokenizer.vocabulary.len() as i64, &config.model);

            let mut optimizer = nn::AdamW::default().build(&vs, config.learning_rate)?;

            for i in 0..config.max_iters {
                if i % config.eval_interval == 0 {
                    let losses = estimate_loss(&config, &train, &val, &model);
                    println!(
                        "step: {}, train loss: {:.4}, val loss: {:.4}",
                        i, losses.0, losses.1
                    );
                }

                let (xb, yb) = get_batch(&train, config.batch_size, config.model.block_size);

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
            tokenizer.save(log_dir.join("tokenizer.json"))?;
            (tokenizer, model)
        }
        Mode::Eval { checkpoint } => {
            let mut vs: nn::VarStore = tch::nn::VarStore::new(tch::Device::Cpu);
            let tokenizer = Tokenizer::load(checkpoint.join("tokenizer.json"))?;
            let model = BaselineModel::new(
                vs.root(),
                tokenizer.vocabulary.len() as i64,
                &TrainArgs::load(checkpoint.join("config.toml"))?.model,
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
