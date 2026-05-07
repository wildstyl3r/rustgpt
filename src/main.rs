use std::fs;

use clap::builder::Str;
use clap::{Args, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use tch::{nn, nn::OptimizerConfig, IndexOp, Tensor};

mod baseline;
mod language_model;
mod tokenizer;
mod utils;

use crate::baseline::model::TransformerLanguageModel;
use crate::language_model::LanguageModel;
use crate::tokenizer::{Token, Tokenizer};
use crate::utils::{estimate_loss, get_batch};

#[derive(Parser, Debug, Serialize, Deserialize)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    mode: Mode,
}

#[derive(Subcommand, Debug, Serialize, Deserialize)]
enum Mode {
    Train {
        #[command(subcommand)]
        config: TrainConfig,
    },
    Eval {
        #[arg(long)]
        checkpoint: std::path::PathBuf,
    },
}

#[derive(Subcommand, Debug, Serialize, Deserialize)]
pub enum TrainConfig {
    File {
        #[arg(long)]
        path: std::path::PathBuf,
    },
    Cli(TrainArgs),
}

#[derive(Args, Debug, Serialize, Deserialize)]
pub struct TrainArgs {
    #[arg(long, default_value_t = 0.2)]
    pub dropout: f64,

    #[arg(long, default_value_t = 3e-3)]
    pub learning_rate: f64,

    #[arg(long, default_value_t = 15001)]
    pub max_iters: i64,

    #[arg(long, default_value_t = 500)]
    pub eval_interval: i64,

    #[arg(long, default_value_t = 200)]
    pub eval_iters: i64,

    #[arg(long, default_value_t = 32)]
    pub batch_size: i64,

    #[arg(long, default_value_t = 8)]
    pub block_size: i64,

    #[arg(long, default_value_t = 4)]
    pub n_blocks: i64,

    #[arg(long, default_value_t = 32)]
    pub n_embed: i64,

    #[arg(long, default_value = "input.txt")]
    pub input_path: String,
}

fn main() {
    let cli = Cli::parse();
    let git_hash = env!("GIT_HASH");

    let (tokenizer, model, params) = match cli.mode {
        Mode::Train { config } => {
            let params = match config {
                TrainConfig::File { path } => {
                    let content = std::fs::read_to_string(path).expect("Read error");
                    toml::from_str(&content).expect("TOML error")
                }
                TrainConfig::Cli(p) => p,
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
                params.n_embed,
                params.block_size,
                params.n_blocks,
                params.dropout,
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
                        params.block_size,
                        &model,
                    );
                    println!(
                        "step: {}, train loss: {:.4}, val loss: {:.4}",
                        i, losses.0, losses.1
                    );
                }

                let (xb, yb) = get_batch(&train_data, params.batch_size, params.block_size);

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
            (tokenizer, model, params)
        }
        Mode::Eval { checkpoint } => {
            todo!()
        }
    };

    let idx = Tensor::zeros([1, params.batch_size], (tch::Kind::Int, tch::Device::Cpu));
    println!(
        "[{}]",
        tokenizer.decode(&Vec::<Token>::try_from(model.generate(idx, 500).i(0)).unwrap())
    );
}
