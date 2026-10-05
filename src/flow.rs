#[derive(Clone)]
pub struct Flow {
    valid: bool,
    vec: Vec<f64>
}

impl Flow {
    pub fn new(valid: bool, vec: Vec<f64>) -> Self {
        Flow {
            valid,
            vec
        }
    }
    pub fn valid(&self) -> bool {
        self.valid
    }
    pub fn vec(&self) -> Vec<f64> {
        self.vec.clone()
    }
}