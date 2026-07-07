use std::fs::{self};
use std::io::Write;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use tch::{nn, nn::OptimizerConfig, IndexOp, Tensor};

mod cli;
mod interface;
mod lm;
mod lr_schedule;
mod muon;
mod tokenizer;
mod utils;

use crate::cli::{Cli, ConfigSource, Mode, TrainConfig};
use crate::interface::LanguageModel;
use crate::tokenizer::{Token, Tokenizer};
use crate::utils::{estimate_loss, get_batch, param_count, train_val_split, write_summary};

#[derive(serde::Serialize)]
struct LossRecord {
    step: i64,
    train_loss: f64,
    val_loss: f64,
    time_from_start: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let git_hash = env!("GIT_HASH");

    let (tokenizer, model) = match cli.mode {
        Mode::Train { config, tag } => {
            let mut config = match config {
                ConfigSource::File { path } => TrainConfig::load(path)?,
                ConfigSource::Cli(config) => config,
            };

            tch::manual_seed(cli.seed);

            let text = std::fs::read_to_string(&config.dataset.input_path)?;
            let tokenizer = Tokenizer::new(&text);
            let (train, val) = train_val_split(
                &Tensor::from_slice(&tokenizer.encode(&text)),
                config.dataset.train_share,
            )?;

            let model_creation_start = Instant::now();
            let vs: nn::VarStore = tch::nn::VarStore::new(tch::Device::Cpu);
            let model = lm::Model::new(
                vs.root(),
                tokenizer.vocabulary.len() as i64,
                &mut config.model,
            )?;
            println!(
                "model created in {:?}",
                model_creation_start - Instant::now()
            );

            let (mut adam_p, mut muon_p) = (Vec::new(), Vec::new());
            for (name, w) in vs.variables() {
                if w.size().len() == 1 || name.contains("embedding") {
                    adam_p.push(w);
                } else {
                    muon_p.push(w);
                }
            }

            let mut adamw = nn::AdamW::default().beta2(0.95).build_copt(0.)?;
            for w in adam_p {
                adamw.add_parameters(&w, 0)?;
            }
            let mut muon = muon::Muon::new(muon_p, 0., 0.95, 0.1, true, false, 1e-7);

            let log_dir = std::path::Path::new("checkpoints").join(format!(
                "run_{}_{}{}",
                chrono::Local::now().format("%Y%m%d_%H%M"),
                git_hash,
                match tag {
                    Some(tag) => format!("_{}", tag),
                    None => String::new(),
                },
            ));
            fs::create_dir_all(&log_dir)?;

            let mut scheme = fs::File::create(log_dir.join("scheme.txt"))?;

            let (total_params, trainable_params) = param_count(&vs);
            scheme.write_fmt(format_args!(
                "total parameters: {}\ntrainable parameters: {}\n",
                total_params, trainable_params
            ))?;
            write_summary(&vs, &scheme)?;

            println!(
                "total parameters: {}\ntrainable parameters: {}",
                total_params, trainable_params
            );

            fs::write(
                log_dir.join("config.toml"),
                toml::to_string_pretty(&config)?,
            )?;
            let mut wtr = csv::Writer::from_path(
                log_dir.join(format!("cw{0}_losses.csv", config.model.context_window)),
            )?;
            let start = Instant::now();
            for step in 0..config.lr_schedule.max_iters {
                if step % config.eval_interval == 0 {
                    let eval_start = Instant::now();
                    let losses = estimate_loss(&config, &train, &val, &model);
                    let eval_end = Instant::now();
                    let dt = eval_end - start;
                    println!(
                        "step: {}, train loss: {:.4}, val loss: {:.4}, time from start {:?}, {} eval iterations done in {:?}",
                        step, losses.0, losses.1, dt, config.eval_iters, eval_end-eval_start
                    );
                    wtr.serialize(LossRecord {
                        step,
                        train_loss: losses.0,
                        val_loss: losses.1,
                        time_from_start: format!("{:?}", dt),
                    })?;
                    wtr.flush()?;
                }

                let (xb, yb) = get_batch(&train, config.batch_size, config.model.context_window);

                let (loss, _) = model.forward_with_loss(&xb, &yb, true);
                let lr = config.lr_schedule.get_lr(step);
                adamw.set_learning_rate(lr)?;
                muon.set_lr(lr);
                adamw.zero_grad()?;
                muon.zero_grad();
                loss.backward();
                adamw.step()?;
                muon.step();
            }
            wtr.flush()?;
            println!("elapsed time: {:?}", start.elapsed());

            vs.save(log_dir.join("model.safetensors"))?;
            tokenizer.save(log_dir.join("tokenizer.json"))?;
            (tokenizer, Box::new(model) as Box<dyn LanguageModel>)
        }
        Mode::Eval { checkpoint } => {
            let mut vs: nn::VarStore = tch::nn::VarStore::new(tch::Device::Cpu);
            let tokenizer = Tokenizer::load(checkpoint.join("tokenizer.json"))?;
            let model = lm::Model::new(
                vs.root(),
                tokenizer.vocabulary.len() as i64,
                &mut TrainConfig::load(checkpoint.join("config.toml"))?.model,
            )?;
            vs.load(checkpoint.join("model.safetensors"))?;
            (tokenizer, Box::new(model) as Box<dyn LanguageModel>)
        }
    };

    tch::manual_seed(cli.seed);
    println!(
        "[{}]",
        tokenizer.decode(&Vec::<Token>::try_from(
            model
                .generate(
                    Tensor::from_slice(&tokenizer.encode("\n")).unsqueeze(0),
                    cli.max_new_tok
                )
                .i(0)
        )?)
    );
    Ok(())
}
