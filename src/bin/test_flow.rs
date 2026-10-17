use vo_system::{get_lambda_thresholds, get_img, get_compressed_img, get_shifted_img, get_gradients, calculate_flow_field, draw_arrow};
use imageproc::drawing::{draw_hollow_rect_mut};
use imageproc::rect::Rect;
use image::Luma;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::cmp;
fn main() {
    // RUN THIS FILE FROM VO SYSTEM NOT SRC DIRECTORY IN TERMINAL
    // arg list: img_num    compress_num    OPTIONAL: window_x    window_Y
    let args: Vec<String> = env::args().collect();
    let img_num: i32 = args[1].parse().unwrap();
    let compress_num: u32 = args[2].parse().unwrap();
    let cur_compress: u32 = args[3].parse().unwrap(); // if the image is already compressed to some level
    let testing: u32 = args[4].parse().unwrap();
    let shiftx: i32 = args[5].parse().unwrap();
    let shifty: i32 = args[6].parse().unwrap();
    let n = 15;

    let path = format!("data/frame{}.png", img_num);
    let path2 = format!("data/frame{}.png", img_num+1);
    let img0 = get_compressed_img(get_img(&path).into_luma8(), compress_num - cur_compress);
    let img1 = get_shifted_img(
        &get_compressed_img(get_img(&path2).into_luma8(), compress_num - cur_compress), 
        shiftx, shifty);

    let (w, h) = img0.dimensions();
    println!("img dimensions: {w}, {h}");

    let start_x= cmp::max(0, shiftx) as u32;
    let start_y = cmp::max(0, shifty) as u32;
    let end_x = (w as i32 + cmp::min(0, shiftx)) as u32;
    let end_y = (h as i32 + cmp::min(0, shifty)) as u32;
    let flow_scale = 20.;
    let thick1: u32 = 8;
    let thick2: u32 = 2;

    let (Ix, Iy, It) = get_gradients(&img0, &img1);

    let mut drawing = img0.clone();

    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open("window_analysis/eigen_logs.txt")
        .expect("Failed to open log file");
    
    writeln!(file, "")
            .expect("Failed to write to log file");

    let thresholds = get_lambda_thresholds(compress_num);
    let (min_lambda, max_lambda_ratio, max_residual) = if testing == 1 { get_lambda_thresholds(0) } else { (thresholds.0, thresholds.1, thresholds.2) };

    writeln!(file, "im{}_{}x-----",
                    img_num, compress_num)
                    .expect("Failed to write to log file");
    let (field, avg, std) = calculate_flow_field(&Ix, &Iy, &It, n, min_lambda, max_lambda_ratio, max_residual, start_x, start_y, end_x, end_y);
    let white = Luma([255]);
    let black = Luma([0]);
    for i in 0..field.len() {
        for j in 0..field[i].len() {
            let f = &field[i][j];

            if (i + j) % 2 == 0 {
                let rect = Rect::at((j as u32 *n + start_x) as i32, (i as u32 *n + start_y) as i32).of_size(n, n);
                draw_hollow_rect_mut(&mut drawing, rect, white);
            }
            let start = ((j as u32 * n + n/2 + start_x) as i32, (i as u32 * n + n/2 + start_y) as i32);
            if f.valid() {
                let end = ((start.0 as f64 + f.vec()[0] * flow_scale) as i32, (start.1 as f64 + f.vec()[1] * flow_scale) as i32);
                let testing_valid = testing == 0 || f.eigen_min() > thresholds.0 && f.eigen_ratio() < thresholds.1;
                draw_arrow(&mut drawing, start, end, thick1, thick2, if testing_valid { white } else { black });
            }

            writeln!(file, "    im{}_{}x | x{} y{} | vx {:.3} vy {:.3} | eigen min {:.3} eigen ratio {:.3} | residual {:.3}",
                    img_num, compress_num, j, i, f.vec()[0], f.vec()[1], f.eigen_min(), f.eigen_ratio(), f.residual())
                    .expect("Failed to write to log file");
        }
    }
    writeln!(file, "    AVG x {:.4}  y {:.4} | STD x {:.4} y {:.4}",
                    avg[0], avg[1], std[0], std[1])
                    .expect("Failed to write to log file");

            
    drawing.save(format!("window_analysis/im{}_{}x_labeled.png", img_num, compress_num)).expect("Failed to save drawing");
    img0.save(format!("window_analysis/im{}_{}x_raw.png", img_num, compress_num)).expect("Failed to save drawing");
    img1.save(format!("window_analysis/im{}_{}x_raw.png", img_num+1, compress_num)).expect("Failed to save drawing");
}