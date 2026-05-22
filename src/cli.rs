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
    #[arg(long, default_value_t = 1337)]
    pub seed: i64,
    #[arg(long, default_value_t = 500)]
    pub max_new_tok: i64,
}

#[derive(Subcommand, Debug, Serialize, Deserialize)]
pub enum Mode {
    /// Train a model with provided configuration
    Train {
        #[command(subcommand)]
        config: ConfigSource,
    },
    /// Load model weights from a checkpoint
    Eval {
        /// Relative path to the checkpoint directory
        checkpoint: std::path::PathBuf,
    },
}

#[derive(Subcommand, Debug, Serialize, Deserialize)]
pub enum ConfigSource {
    /// Load the model training configuration from a specified file
    File {
        /// Relative path to the model configuration in TOML format
        path: std::path::PathBuf,
    },
    /// Set up all the parameters using CLI flags
    Cli(TrainConfig),
}

#[derive(Args, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    #[arg(long, default_value_t = 3e-3)]
    pub learning_rate: f64,

    #[arg(long, default_value_t = 10001)]
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

    #[arg(long, default_value = "shakespeare.txt")]
    pub input_path: String,
}
