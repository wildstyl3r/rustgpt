pub struct Muon {
    parameters: Vec<tch::Tensor>,
    momenta: Vec<tch::Tensor>,

    lr: f64,
    mu: f64,
    weight_decay: f64,
    nesterov: bool,
    turbo: bool,
    eps: f64,
}

const A: f64 = 3.4445;
const B: f64 = -4.775;
const C: f64 = 2.0315;

fn newton_schulz(mut x: tch::Tensor, turbo: bool, eps: f64) -> tch::Tensor {
    // assert!(g.size().len() == 2);
    let tall = x.size()[0] > x.size()[1];
    if tall {
        x = x.transpose(0, 1);
    }

    let (mut a, mut x, iters) = if turbo {
        // https://arxiv.org/abs/2512.04632  Turbo-Muon: Accelerating Orthogonality-Based Optimization with Pre-Conditioning
        // todo: check the per-step optimized ns coefficients a,b,c
        let a = x.matmul(&x.transpose(0, 1));
        let s = a
            .abs()
            .sum_dim_intlist(&[-1][..], false, None)
            .clamp_min(eps)
            .rsqrt();
        (
            a * s.unsqueeze(-2) * s.unsqueeze(-1),
            x * s.unsqueeze(-1),
            4,
        )
    } else {
        let x = &x / (x.frobenius_norm([-2, -1], true) + eps);
        (x.matmul(&x.transpose(0, 1)), x, 5)
    };

    for i in 0..iters {
        if i != 0 {
            a = x.matmul(&x.transpose(0, 1));
        }
        let b = B * &a + C * &a.matmul(&a);
        x = A * &x + b.matmul(&x);
    }
    if tall {
        x = x.transpose(0, 1);
    }
    x
}

impl Muon {
    pub fn new(
        parameters: Vec<tch::Tensor>,
        lr: f64,
        mu: f64,
        weight_decay: f64,
        nesterov: bool,
        turbo: bool,
        eps: f64,
    ) -> Self {
        Muon {
            momenta: parameters.iter().map(tch::Tensor::zeros_like).collect(),
            parameters,
            lr,
            mu,
            weight_decay,
            nesterov,
            turbo,
            eps,
        }
    }

    pub fn step(&mut self) {
        tch::no_grad(|| {
            for (w, m) in self.parameters.iter_mut().zip(&self.momenta) {
                let m_new = self.mu * m + w.grad();
                let m_prepare = if self.nesterov {
                    self.mu * m_new + w.grad()
                } else {
                    m_new
                };
                let shape = m_prepare.size();
                let m_prepare = m_prepare.view([shape[0], -1]);

                // https://arxiv.org/pdf/2502.16982 Muon is scalable for LLM training
                let rms_scale =
                    0.2 * (std::cmp::max(m_prepare.size()[0], m_prepare.size()[1]) as f64).sqrt();

                let o_new = newton_schulz(m_prepare, self.turbo, self.eps).view(&*shape);
                let _ = w.subtract_(
                    &(self.lr * (rms_scale * o_new + w.multiply_scalar(self.weight_decay))),
                );
            }
        })
    }

    pub fn zero_grad(&mut self) {
        for p in &mut self.parameters {
            p.zero_grad();
        }
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr
    }
}
