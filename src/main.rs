use std::fs::{self};
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use tch::{nn, nn::OptimizerConfig, IndexOp, Tensor};

mod cli;
mod interface;
mod lm;
mod tokenizer;
mod utils;

use crate::cli::{Cli, ConfigSource, Mode, TrainConfig};
use crate::interface::LanguageModel;
use crate::tokenizer::{Token, Tokenizer};
use crate::utils::{estimate_loss, get_batch, train_val_split};

fn main() -> Result<()> {
    let cli = Cli::parse();
    let git_hash = env!("GIT_HASH");

    let (tokenizer, model) = match cli.mode {
        Mode::Train { config } => {
            let config = match config {
                ConfigSource::File { path } => TrainConfig::load(path)?,
                ConfigSource::Cli(config) => config,
            };
            tch::manual_seed(1337);

            let text = std::fs::read_to_string(&config.dataset.input_path)?;
            let tokenizer = Tokenizer::new(&text);
            let (train, val) = train_val_split(
                &Tensor::from_slice(&tokenizer.encode(&text)),
                config.dataset.train_share,
            )?;

            let vs: nn::VarStore = tch::nn::VarStore::new(tch::Device::Cpu);
            let model = lm::Model::new(vs.root(), tokenizer.vocabulary.len() as i64, &config.model);

            let mut optimizer = nn::AdamW::default().build(&vs, config.learning_rate)?;

            let start = Instant::now();
            for i in 0..config.max_iters {
                if i % config.eval_interval == 0 {
                    let losses = estimate_loss(&config, &train, &val, &model);
                    println!(
                        "step: {}, train loss: {:.4}, val loss: {:.4}",
                        i, losses.0, losses.1
                    );
                }

                let (xb, yb) = get_batch(&train, config.batch_size, config.model.context_window);

                let (loss, _) = model.forward_with_loss(&xb, &yb, true);
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();
            }
            println!("elapsed time: {:?}", start.elapsed());

            let log_dir = std::path::Path::new("checkpoints").join(format!(
                "run_{}_{}",
                chrono::Local::now().format("%Y%m%d_%H%M"),
                git_hash
            ));
            fs::create_dir_all(&log_dir)?;
            fs::write(
                log_dir.join("config.toml"),
                toml::to_string_pretty(&config)?,
            )?;
            vs.save(log_dir.join("model.safetensors"))?;
            tokenizer.save(log_dir.join("tokenizer.json"))?;
            (tokenizer, Box::new(model) as Box<dyn LanguageModel>)
        }
        Mode::Eval { checkpoint, seed } => {
            tch::manual_seed(seed);
            let mut vs: nn::VarStore = tch::nn::VarStore::new(tch::Device::Cpu);
            let tokenizer = Tokenizer::load(checkpoint.join("tokenizer.json"))?;
            let model = lm::Model::new(
                vs.root(),
                tokenizer.vocabulary.len() as i64,
                &TrainConfig::load(checkpoint.join("config.toml"))?.model,
            );
            vs.load(checkpoint.join("model.safetensors"))?;
            (tokenizer, Box::new(model) as Box<dyn LanguageModel>)
        }
    };

    let idx = Tensor::zeros([1, 1], (tch::Kind::Int, tch::Device::Cpu));
    println!(
        "[{}]",
        tokenizer.decode(&Vec::<Token>::try_from(model.generate(idx, 500).i(0))?)
    );
    Ok(())
}
