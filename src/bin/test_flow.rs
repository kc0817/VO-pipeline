use vo_system::{get_img, get_compressed_img, get_gradients, calc_flow_vec, draw_arrow};
use imageproc::drawing::{draw_hollow_rect_mut};
use imageproc::rect::Rect;
use image::Luma;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;
fn main() {
    // RUN THIS FILE FROM VO SYSTEM NOT SRC DIRECTORY IN TERMINAL
    // arg list: img_num    compress_num    OPTIONAL: window_x    window_Y
    let args: Vec<String> = env::args().collect();
    let img_num: i32 = args[1].parse().unwrap();
    let compress_num: u32 = args[2].parse().unwrap();
    let testing: bool = args[3].parse().unwrap();
    let n = 15;

    let path = format!("data/frame{}.png", img_num);
    let path2 = format!("data/frame{}.png", img_num+1);
    let img0 = get_compressed_img(get_img(&path).into_luma8(), compress_num);
    let img1 = get_compressed_img(get_img(&path2).into_luma8(), compress_num);

    let (w, h) = img0.dimensions();
    println!("img dimensions: {w}, {h}");

    let select_one_window = args.len() > 4;
    let mut start_x= 0;
    let mut start_y = 0;
    let mut end_x = w / n;
    let mut end_y = h / n;
    if select_one_window {
        start_x = args[3].parse().unwrap();
        start_y = args[4].parse().unwrap();
        end_x = start_x + 1;
        end_y = start_y + 1;
    }
    let flow_scale = 20.;
    let thick1: u32 = 8;
    let thick2: u32 = 2;
    let black = Luma([0]);
    let white = Luma([255]);

    let (Ix, Iy, It) = get_gradients(&img0, &img1);

    let mut drawing = img0.clone();

    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open("window_analysis/eigen_logs.txt")
        .expect("Failed to open log file");
    
    writeln!(file, "")
            .expect("Failed to write to log file");

    for y in start_y..end_y {
        for x in start_x..end_x {
            let f = calc_flow_vec(&Ix, &Iy, &It, n, y, x, testing);
            
            if (y * w/n + x) % 2 == 0 {
                let rect = Rect::at((x*n) as i32, (y*n) as i32).of_size(n, n);
                draw_hollow_rect_mut(&mut drawing, rect, white);
            }
            let start = ((x * n + n/2) as i32, (y * n + n/2) as i32);
            if f.valid() {
                let end = ((start.0 as f64 + f.vec()[0] * flow_scale) as i32, (start.1 as f64 + f.vec()[1] * flow_scale) as i32);
                draw_arrow(&mut drawing, start, end, thick1, thick2, Luma([255]));
            }
            // drawing eigenvectors
            if select_one_window {
                let eigen_scale = 10.;
                let end1 = (start.0 + (f.eigen_vec_min()[0] * eigen_scale) as i32,
                                        start.1 + (f.eigen_vec_min()[1] * eigen_scale) as i32);
                draw_arrow(&mut drawing, start, end1, 1, 1, white);
                let end2 = (start.0 + (f.eigen_vec_max()[0] * f.eigen_ratio() * eigen_scale) as i32,
                                        start.1 + (f.eigen_vec_max()[1] * f.eigen_ratio() * eigen_scale) as i32);
                draw_arrow(&mut drawing, start, end2, 3, 3, black);
            }
            
            writeln!(file, "    img{} compress{} | size{}  x{} y{} | vx {} vy {} | eigen min {} eigen ratio {}",
                    img_num, compress_num, n, x, y, f.vec()[0], f.vec()[1], f.eigen_min(), f.eigen_ratio())
                    .expect("Failed to write to log file");
        }
    }
    
            
    drawing.save(format!("window_analysis/imgnum{}_{}x_labeled.png", img_num, compress_num)).expect("Failed to save drawing");
    img0.save(format!("window_analysis/imgnum{}_{}x_raw.png", img_num, compress_num)).expect("Failed to save drawing");
    img1.save(format!("window_analysis/imgnum{}_{}x_raw.png", img_num+1, compress_num)).expect("Failed to save drawing");
}