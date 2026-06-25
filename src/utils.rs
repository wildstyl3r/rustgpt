use std::io::Write;

use crate::{cli::TrainConfig, interface::LanguageModel};
use tch::{nn::VarStore, IndexOp, TchError, Tensor};

pub fn get_batch(data: &Tensor, batch_size: i64, context_window: i64) -> (Tensor, Tensor) {
    let ix = Tensor::randint(
        data.size1().unwrap() - context_window,
        [batch_size],
        (tch::Kind::Int, tch::Device::Cpu),
    );
    (
        Tensor::stack(
            //context
            &ix.iter::<i64>()
                .unwrap()
                .map(|i| data.i(i..i + context_window))
                .collect::<Vec<_>>(),
            0,
        ),
        Tensor::stack(
            //target
            &ix.iter::<i64>()
                .unwrap()
                .map(|i| data.i(i + 1..i + context_window + 1))
                .collect::<Vec<_>>(),
            0,
        ),
    )
}

pub fn train_val_split(data: &Tensor, train_share: f32) -> Result<(Tensor, Tensor), TchError> {
    let len = data.size1()?;
    let n = (train_share * len as f32) as i64;
    Ok((data.i(0..n), data.i(n..len - 1)))
}

pub fn estimate_loss<M: LanguageModel>(
    config: &TrainConfig,
    train_data: &Tensor,
    validation_data: &Tensor,
    m: &M,
) -> (f64, f64) {
    tch::no_grad(|| {
        let losses_train = Tensor::stack(
            &(0..config.eval_iters)
                .map(|_| {
                    let (x, y) =
                        get_batch(train_data, config.batch_size, config.model.context_window);
                    let (loss, _) = m.forward_with_loss(&x, &y, false);
                    loss
                })
                .collect::<Vec<_>>(),
            0,
        )
        .mean(tch::Kind::Float)
        .double_value(&[]);

        let losses_val = Tensor::stack(
            &(0..config.eval_iters)
                .map(|_| {
                    let (x, y) = get_batch(
                        validation_data,
                        config.batch_size,
                        config.model.context_window,
                    );
                    let (loss, _) = m.forward_with_loss(&x, &y, false);
                    loss
                })
                .collect::<Vec<_>>(),
            0,
        )
        .mean(tch::Kind::Float)
        .double_value(&[]);
        (losses_train, losses_val)
    })
}

pub fn param_count(vs: &VarStore) -> (i64, i64) {
    (
        vs.variables()
            .values()
            .map(|t| t.size().iter().product::<i64>())
            .sum(),
        vs.trainable_variables()
            .iter()
            .map(|t| t.size().iter().product::<i64>())
            .sum(),
    )
}

pub fn write_summary(vs: &VarStore, mut wr: impl Write) -> std::io::Result<()> {
    let mut sorted_vars: Vec<_> = vs.variables().into_iter().collect();
    sorted_vars.sort_by(|a, b| a.0.cmp(&b.0));
    wr.write_fmt(format_args!("{}\n", "-".repeat(85)))?;

    for (name, tensor) in sorted_vars {
        // if tensor.requires_grad() {
        let shape = tensor.size();
        wr.write_fmt(format_args!(
            "{:<60} {:?} {}\n",
            name,
            shape,
            if tensor.requires_grad() {
                "(trainable)"
            } else {
                ""
            }
        ))?;
        // }
    }
    wr.write_fmt(format_args!("{}\n", "-".repeat(85)))?;
    Ok(())
}
