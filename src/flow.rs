#[derive(Clone)]
pub struct Flow {
    valid: bool,
    vec: Vec<f64>,
    eigen_min: f64,
    eigen_ratio: f64,
    eigen_vec_min: Vec<f64>,
    eigen_vec_max: Vec<f64>
}

impl Flow {
    pub fn new(valid: bool, vec: Vec<f64>, eigen_min: f64, eigen_ratio: f64, eigen_vec_min: Vec<f64>, eigen_vec_max: Vec<f64>) -> Self {
        Flow {
            valid,
            vec,
            eigen_min,
            eigen_ratio,
            eigen_vec_min,
            eigen_vec_max
        }
    }
    pub fn valid(&self) -> bool {
        self.valid
    }
    pub fn vec(&self) -> Vec<f64> {
        self.vec.clone()
    }
    pub fn eigen_min(&self) -> f64 {
        self.eigen_min
    }
    pub fn eigen_ratio(&self) -> f64 {
        self.eigen_ratio
    }
    pub fn eigen_vec_min(&self) -> Vec<f64> {
        self.eigen_vec_min.clone()
    }
    pub fn eigen_vec_max(&self) -> Vec<f64> {
        self.eigen_vec_max.clone()
    }
}