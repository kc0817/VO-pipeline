use vo_system::{get_img, get_gradients, calc_flow_vec, draw_arrow};
use imageproc::drawing::{draw_hollow_rect_mut};
use imageproc::rect::Rect;
use image::Luma;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;
fn main() {
    // arg list: img_num    compress_num    OPTIONAL: window_x    window_Y
    let args: Vec<String> = env::args().collect();
    let img_num: i32 = args[1].parse().unwrap();
    let compress_num: u32 = args[2].parse().unwrap();
    let select_one_window = args.len() > 3;
    let mut x= 0;
    let mut y = 0;
    if select_one_window {
        x = args[3].parse().unwrap();
        y = args[4].parse().unwrap();
    }
    let n = 15;
    let flow_scale = 80.;
    let thick1: u32 = 10;
    let thick2: u32 = 4;
    // // RUN THIS FILE FROM VO SYSTEM NOT SRC DIRECTORY IN TERMINAL
    let path = format!("flow_fields/compressed{}x{}.png", compress_num, img_num);
    println!("path: {path}");
    let path2 = format!("flow_fields/compressed{}x{}.png", compress_num, img_num+1);
    let img0 = get_img(&path).into_luma8();
    let img1 = get_img(&path2).into_luma8();

    let (w, h) = img0.dimensions();
    println!("img dimensions: {w}, {h}");

    let (Ix, Iy, It) = get_gradients(&img0, &img1);

    let f = calc_flow_vec(&Ix, &Iy, &It, n, y, x, true);

    let mut drawing = img0.clone();
    let rect = Rect::at((x*n) as i32, (y*n) as i32).of_size(n, n);
    draw_hollow_rect_mut(&mut drawing, rect, Luma([255u8]));
    
    println!("f valid: {}", f.valid());
    let start = ((x * n + n/2) as i32, (y * n + n/2) as i32);
    if f.valid() {
        let end = ((start.0 as f64 + f.vec()[0] * flow_scale) as i32, (start.1 as f64 + f.vec()[1] * flow_scale) as i32);
        draw_arrow(&mut drawing, start, end, thick1, thick2, Luma([255]));
    }
    // drawing eigenvectors
    let black = Luma([0]);
    let eigen_scale = 1.;
    let end1 = (start.0 + (f.eigen_vec_min()[0] * f.eigen_min() * eigen_scale) as i32,
                            start.1 + (f.eigen_vec_min()[1] * f.eigen_min() * eigen_scale) as i32);
    draw_arrow(&mut drawing, start, end1, 2, 2, black);
    let end2 = (start.0 + (f.eigen_vec_max()[0] * f.eigen_min() * f.eigen_ratio() * eigen_scale) as i32,
                            start.1 + (f.eigen_vec_max()[1] * f.eigen_min() * f.eigen_ratio() * eigen_scale) as i32);
    draw_arrow(&mut drawing, start, end2, 4, 4, black);

    
    drawing.save(format!("window_analysis/rectat{}-{}compress{}x_imgnum{}.png", x, y, compress_num, img_num)).expect("Failed to save drawing");

    
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open("window_analysis/eigen_logs.txt")
        .expect("Failed to open log file");
    writeln!(file, "img {} compression {} | window size {} window x {} window y {} | vx {} vy {} | eigen min {} eigen ratio {}",
            img_num, compress_num, n, x, y, f.vec()[0], f.vec()[1], f.eigen_min(), f.eigen_ratio())
            .expect("Failed to write to log file");
}