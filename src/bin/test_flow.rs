use vo_system::{get_img, get_gradients, calc_flow_vec};
use imageproc::drawing::{draw_hollow_rect_mut, draw_antialiased_polygon_mut};
use imageproc::pixelops::interpolate;
use imageproc::rect::Rect;
use imageproc::point::Point;
use image::Luma;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;
fn main() {
    let args: Vec<String> = env::args().collect();
    let img_num: i32 = args[1].parse().unwrap();
    let compress_num: u32 = args[2].parse().unwrap();
    let x: u32 = args[3].parse().unwrap();
    let y: u32 = args[4].parse().unwrap();
    let n = 15;
    let flow_scale = 80.;
    let thick1: u32 = 8;
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

    let f = calc_flow_vec(&Ix, &Iy, &It, n, y, x);

    let mut drawing = img0.clone();
    let rect = Rect::at((x*n) as i32, (y*n) as i32).of_size(n, n);
    draw_hollow_rect_mut(&mut drawing, rect, Luma([255u8]));
    
    println!("f valid: {}", f.valid());
    if f.valid() {
        let start = ((x * n + n/2) as i32, (y * n + n/2) as i32);
        let end = ((start.0 as f64 + f.vec()[0] * flow_scale) as i32, (start.1 as f64 + f.vec()[1] * flow_scale) as i32);
        let Dx = end.0 - start.0;
        let Dy = end.1 - start.1;
        let mag: f64 = ((Dx * Dx + Dy * Dy) as f64).sqrt();
        let dx = Dx as f64 / mag;
        let dy = Dy as f64 / mag;

        let xoff_start = (-dy * thick1 as f64 / 2.) as i32;
        let yoff_start = (dx * thick1 as f64 / 2.) as i32;

        let xoff_end = (-dy * thick2 as f64 / 2.) as i32;
        let yoff_end = (dx * thick2 as f64 / 2.) as i32;
        let points = [Point::new(start.0 + xoff_start, start.1 + yoff_start), Point::new(end.0 + xoff_end, end.1 + yoff_end), 
                                    Point::new(end.0 - xoff_end, end.1 - yoff_end), Point::new(start.0 - xoff_start, start.1 - yoff_start)];
        draw_antialiased_polygon_mut(&mut drawing, &points, Luma([255]), interpolate);
    }
    
    drawing.save(format!("flow_fields/rectat{}-{}compress{}x_imgnum{}.png", x, y, compress_num, img_num)).expect("Failed to save drawing");

    
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open("flow_fields/eigen_logs.txt")
        .expect("Failed to open log file");
    writeln!(file, "img {} compression {} | window size {} window x {} window y {} | vx {} vy {} | eigen min {} eigen ratio {}",
            img_num, compress_num, n, x, y, f.vec()[0], f.vec()[1], f.eigen_min(), f.eigen_ratio())
            .expect("Failed to write to log file");
}