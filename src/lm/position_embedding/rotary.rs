impl super::Rotary for super::None {
    fn new(_: tch::nn::Path, _: i64, _: i64) -> Self {}

    fn inject(&self, x: tch::Tensor) -> tch::Tensor {
        x
    }
}
