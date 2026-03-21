use tch::{nn::Module, IndexOp, Tensor};

pub trait LanguageModel: Module {
    fn forward_with_loss(&self, idx: &Tensor, targets: &Tensor) -> (Tensor, Tensor) {
        let logits = self.forward(idx);
        let (b, t, c) = logits.size3().unwrap();
        let logits = logits.view((b * t, c));
        let targets = targets.view(b * t);
        (logits.cross_entropy_for_logits(&targets), logits)
    }

    fn get_block_size(&self) -> i64;

    fn generate(&self, mut idx: Tensor, max_new_tokens: usize) -> Tensor {
        for _ in 0..max_new_tokens {
            let (_x, y) = idx.size2().unwrap();
            let idx_cond = idx.i((.., y - (self.get_block_size())..));
            let logits = self.forward(&idx_cond);
            let logits = logits.i((.., -1, ..));
            let probs = logits.softmax(-1, tch::Kind::Float);

            let idx_next = probs.multinomial(1, false);
            idx = Tensor::cat(&[idx, idx_next], 1);
        }
        idx
    }
}
