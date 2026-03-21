use crate::language_model::LanguageModel;
use crate::vocabulary::Vocabulary;
use tch::{IndexOp, Tensor};

pub fn encode(s: String, vocabulary: Option<&Vocabulary>) -> Vec<i64> {
    match vocabulary {
        Some(v) => s.chars().map(|c| v.stoi[&c.to_string()]).collect(),
        None => s.chars().map(|c| (c as u32) as i64).collect(),
    }
}

pub fn decode(v: Vec<i64>, vocabulary: Option<&Vocabulary>) -> String {
    match vocabulary {
        Some(voc) => v.iter().map(|i| voc.itos[*i as usize].clone()).collect(),
        None => v.iter().filter_map(|i| char::from_u32(*i as u32)).collect(),
    }
}

pub fn get_batch(data: &Tensor, batch_size: i64, block_size: i64) -> (Tensor, Tensor) {
    let ix = Tensor::randint(
        data.size1().unwrap() - block_size,
        [batch_size],
        (tch::Kind::Int, tch::Device::Cpu),
    );
    (
        Tensor::stack(
            //context
            &ix.iter::<i64>()
                .unwrap()
                .map(|i| data.i(i..i + block_size))
                .collect::<Vec<_>>(),
            0,
        ),
        Tensor::stack(
            //target
            &ix.iter::<i64>()
                .unwrap()
                .map(|i| data.i(i + 1..i + block_size + 1))
                .collect::<Vec<_>>(),
            0,
        ),
    )
}

pub fn estimate_loss<M: LanguageModel>(
    eval_iters: i64,
    train_data: &Tensor,
    validation_data: &Tensor,
    batch_size: i64,
    block_size: i64,
    m: &M,
) -> (f64, f64) {
    tch::no_grad(|| {
        let losses_train = Tensor::stack(
            &(0..eval_iters)
                .map(|_| {
                    let (x, y) = get_batch(train_data, batch_size, block_size);
                    let (loss, _) = m.forward_with_loss(&x, &y);
                    loss
                })
                .collect::<Vec<_>>(),
            0,
        )
        .mean(tch::Kind::Float)
        .double_value(&[]);

        let losses_val = Tensor::stack(
            &(0..eval_iters)
                .map(|_| {
                    let (x, y) = get_batch(validation_data, batch_size, block_size);
                    let (loss, _) = m.forward_with_loss(&x, &y);
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
