use crate::vocabulary::Vocabulary;
use tch::{Tensor, IndexOp};
use crate::model::BigramLanguageModel;


pub fn encode(s: String, vocabulary: Option<&Vocabulary>) -> Vec<i64> {
    match vocabulary {
        Some(v) => s.chars().map(|c| v.stoi[&c.to_string()]).collect(),
        None => s.chars().map(|c| (c as u32) as i64).collect()
    }
}

pub fn decode(v: Vec<i64>, vocabulary: Option<&Vocabulary>) -> String {
    match vocabulary {
        Some(voc) => v.iter().map(|i| voc.itos[*i as usize].clone()).collect(),
        None => v.iter().filter_map(|i| char::from_u32(*i as u32)).collect()
    }
}


pub fn get_batch(data: &Tensor, batch_size: i64, block_size: i64) -> (Tensor, Tensor) {
    let ix = Tensor::randint(data.size1().unwrap() - block_size, [batch_size], (tch::Kind::Int, tch::Device::Cpu));
    (
        Tensor::stack( //context
            &ix.iter::<i64>().unwrap().map(|i| data.i(i .. i + block_size)).collect::<Vec<_>>(),
            0
        ),
        Tensor::stack( //target
            &ix.iter::<i64>().unwrap().map(|i| data.i(i + 1 .. i + block_size + 1)).collect::<Vec<_>>(),
            0
        )
    )
}

pub fn estimate_loss(eval_iters: i64, train_data: &Tensor, validation_data: &Tensor, batch_size: i64, block_size: i64, m: &BigramLanguageModel) -> [Tensor; 2] {
    tch::no_grad(||{
        let mut losses_train = Tensor::zeros(eval_iters, (tch::Kind::Float, tch::Device::Cpu));
        for k in 0..eval_iters {
            let (x, y) = get_batch(train_data, batch_size, block_size);
            let (loss, _) = m.forward(&x, Some(&y));
            let loss = loss.unwrap();
            losses_train = losses_train.index_put_(&[Some(Tensor::from(k))], &loss, false);
        }

        let mut losses_val = Tensor::zeros(eval_iters, (tch::Kind::Float, tch::Device::Cpu));
        for k in 0..eval_iters {
            let (x, y) = get_batch(validation_data, batch_size, block_size);
            let (loss, _) = m.forward(&x, Some(&y));
            let loss = loss.unwrap();
            losses_val = losses_val.index_put_(&[Some(Tensor::from(k))], &loss, false);
        }
        //[0.0, 0.0]
        [losses_train.mean(tch::Kind::Float), losses_val.mean(tch::Kind::Float)]
    })
}