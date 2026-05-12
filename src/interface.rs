use std::cmp::max;

use tch::{nn::ModuleT, IndexOp, Tensor};

pub trait LanguageModel: ModuleT {
    fn forward_with_loss(&self, idx: &Tensor, targets: &Tensor, train: bool) -> (Tensor, Tensor) {
        let logits = self.forward_t(idx, train);
        let (b, t, c) = logits.size3().unwrap();
        let logits = logits.view((b * t, c));
        let targets = targets.view(b * t);
        (logits.cross_entropy_for_logits(&targets), logits)
    }

    fn get_context_window(&self) -> i64;

    fn generate(&self, mut idx: Tensor, max_new_tokens: usize) -> Tensor {
        for _ in 0..max_new_tokens {
            let (_x, y) = idx.size2().unwrap();
            let idx_cond = idx.i((.., max(0, y - (self.get_context_window()))..));
            let logits = self.forward_t(&idx_cond, false);
            let logits = logits.i((.., -1, ..));
            let probs = logits.softmax(-1, tch::Kind::Float);

            let idx_next = probs.multinomial(1, false);
            idx = Tensor::cat(&[idx, idx_next], 1);
        }
        idx
    }
}
