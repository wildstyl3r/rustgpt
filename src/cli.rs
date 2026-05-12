use std::{path::PathBuf, result};

use clap::{Args, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use toml::de;

use crate::lm::ModelConfig;

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("toml deserialization error: {0}")]
    Serde(#[from] de::Error),
}

pub type Result<T> = result::Result<T, ConfigError>;

#[derive(Parser, Debug, Serialize, Deserialize)]
#[command(author, version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub mode: Mode,
}

#[derive(Subcommand, Debug, Serialize, Deserialize)]
pub enum Mode {
    Train {
        #[command(subcommand)]
        config: ConfigSource,
    },
    Eval {
        #[arg(long)]
        checkpoint: std::path::PathBuf,
    },
}

#[derive(Subcommand, Debug, Serialize, Deserialize)]
pub enum ConfigSource {
    File {
        #[arg(long)]
        path: std::path::PathBuf,
    },
    Cli(TrainConfig),
}

#[derive(Args, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
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

    #[command(flatten)]
    pub dataset: DatasetConfig,

    #[command(flatten)]
    pub model: ModelConfig,
}

impl TrainConfig {
    pub fn load(path: PathBuf) -> Result<Self> {
        Ok(toml::from_str(&std::fs::read_to_string(path)?)?)
    }
}

#[derive(Args, Debug, Serialize, Deserialize)]
pub struct DatasetConfig {
    #[arg(long, default_value_t = 0.9)]
    pub train_share: f32,

    #[arg(long, default_value = "input.txt")]
    pub input_path: String,
}
