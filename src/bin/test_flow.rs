use vo_system::{decode_img, calc_flow_vec};
use std::env;
fn main() {
    let args: Vec<String> = env::args().collect();
    let img_num: i32 = args[1].parse().unwrap();
    let x: u32 = args[2].parse().unwrap();
    let y: u32 = args[3].parse().unwrap();

    // RUN THIS FILE FROM VO SYSTEM NOT SRC DIRECTORY IN TERMINAL
    let path = format!("data/frame{}.png", img_num);
    let img = decode_img(&path);

    // calc_flow_vec()
}