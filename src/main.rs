use image::{DynamicImage, GrayImage, ImageReader, Rgba};
use image::GenericImageView; 
use imageproc::gradients::{horizontal_sobel, vertical_sobel};
use image::Luma;
use imageproc::map::map_pixels2;
use imageproc::point::Point;
use nalgebra::{DMatrix};
use imageproc::drawing;
use imageproc::pixelops::interpolate;
use std::env;
pub mod flow;
use flow::Flow;
use std::fs::OpenOptions;
use std::io::Write;
use chrono::Utc;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {

    let args: Vec<String> = env::args().collect();

    const n: u32 = 15; // neighborhood size
    let draw_freq = 10;

    let mut img_num = 243;
    if args.len() > 1 {
        img_num = args[1].trim().parse().unwrap();
    }
    println!("Processing img {img_num}");
    // first need to acquire image
    let img0_path = format!("data/frame{}.png", img_num);
    let img1_path = format!("data/frame{}.png", img_num+1);

    if let Ok(img0) = decode_img(&img0_path) {
        let (w, h) = img0.dimensions();
        println!("Image dimensions: {w} x {h} pixels");

        if let Ok(img1) = decode_img(&img1_path) {
            println!("both images read successfully; starting time benchmark");
            let start_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("wtf")
                .as_millis();

            let flow_field = calculate_flow_field(&img0, &img1, img_num, n, draw_freq);

            let mut vx_avg: f64 = 0.;
            let mut vy_avg: f64 = 0.;
            let mut num_vecs: f64 = 0.;
            for row in flow_field.iter() {
                for flow in row.iter() {
                    if flow.valid() {
                        num_vecs += 1.;
                        vx_avg += flow.vec()[0];
                        vy_avg += flow.vec()[1];
                    }
                }
            }
            vx_avg /= num_vecs;
            vy_avg /= num_vecs;
            println!("vx: {vx_avg}, vy: {vy_avg}");

            let mut vx_std: f64 = 0.;
            let mut vy_std: f64 = 0.;
            for row in flow_field.iter() {
                for flow in row.iter() {
                    if flow.valid() {
                        vx_std += (flow.vec()[0] - vx_avg).powf(2.);
                        vy_std += (flow.vec()[1] - vy_avg).powf(2.);
                    }
                }
            }
            vx_std = (vx_std / num_vecs).sqrt();
            vy_std = (vy_std / num_vecs).sqrt();

            let end_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("wtf2")
                .as_millis();
            let time_elapsed = (end_ms - start_ms) as f64 / 1000.;

            let date = Utc::now().format("%Y-%m-%d %H:%M").to_string();

            let mut file = OpenOptions::new()
                .write(true)
                .append(true)
                .open("flow_fields/logs.txt")
                .expect("Failed to open log file");
            writeln!(file, "{} | img {} | time(sec) {:.4} | vx_avg {:.4} | vy_avg {:.4} | vx_std {:.4} | vy_std {:.4}",
                    date, img_num, time_elapsed, vx_avg, vy_avg, vx_std, vy_std)
                    .expect("Failed to write to log file");

            println!("done with img {img_num}");
            // check flow field vectors for potential errors and remove outliers           
            // then idk
        }
    }

}

fn decode_img(file_path: &str) -> Result<DynamicImage, image::ImageError> {
    ImageReader::open(file_path)?
        .decode()
}
fn draw_line(img: &mut DynamicImage, start: (i32, i32), end: (i32, i32), thick1: u32, thick2: u32, color:Rgba<u8>) {
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
    drawing::draw_antialiased_polygon_mut(img, &points, color, interpolate);
}

fn calculate_flow_field(img0: &DynamicImage, img1: &DynamicImage, img_num: u32, n: u32, draw_freq: u32) -> Vec<Vec<Flow>> {
    let (w, h) = img0.dimensions();

    // calculating gradients
    let gray_img: GrayImage = img0.to_luma8();
    let next_gray_img: GrayImage = img1.to_luma8();
    let Ix = horizontal_sobel(&gray_img);
    let Iy = vertical_sobel(&gray_img);
    let It = map_pixels2(&gray_img, &next_gray_img, |p, q| Luma([q[0] as i16 - p[0] as i16]));    
    println!("Ix, Iy, and It computed");

    let mut flow_field = vec![vec![Flow::new(false, vec![0., 0.]); (w/n) as usize]; (h/n) as usize];
    let mut drawing = img0.clone();
    let t: f64 = 80.;
    let color = Rgba([255, 255, 255, 100]);
    let thick_start = 16;
    let thick_end = 4;
    let mut i = 0;

    // looping over each window
    let total_rows = h/n;
    while i < total_rows {
        println!("{i} / {total_rows} rows done");
        let mut j = 0;
        while j < w/n {
            let nrows: usize = (n*n) as usize;
            let mut A: DMatrix<f64> = DMatrix::zeros(nrows, 2);
            let mut b: DMatrix<f64> = DMatrix::zeros(nrows, 1);

            // filling out A and b matrices
            let mut y = 0;
            while y < n {
                let mut x = 0;
                while x < n {
                    let px = j * n + x;
                    let py = i * n + y;
                    let col = (y * n + x) as usize;
                    A[(col, 0)] = Ix.get_pixel(px, py)[0] as f64 / 8.;
                    A[(col, 1)] = Iy.get_pixel(px, py)[0] as f64 / 8.;
                    b[col] = -It.get_pixel(px, py)[0] as f64;

                    x += 1;
                }
                y += 1;
            }
        
            let ATA = A.transpose() * &A;
            if let Some(inv) = ATA.try_inverse() {
                let x_star: DMatrix<f64> = inv * A.transpose() * b;
                let flow = Flow::new(true, vec![x_star[(0, 0)], x_star[(1, 0)]]);
                flow_field[i as usize][j as usize] = flow.clone();

                // drawing flow field on image
                if (i * w/n + j) % draw_freq == 0 {
                    let start = ((j * n + n/2) as i32, (i * n + n/2) as i32);
                    let end = (start.0 + (flow.vec()[0] * t) as i32, start.1 + (flow.vec()[1] * t) as i32);

                    draw_line(&mut drawing, start, end, thick_start, thick_end, color);

                    // if j % save_num == 0 {
                    //     drawing.save(&format!("flow_fields/output{}_{}_{}.png", img_num, i, j)).expect(&format!("Failed to save drawing at i={}, j={}", i, j));
                    // }
                }
            } else {
                flow_field[i as usize][j as usize] = Flow::new(false, vec![0., 0.]);
            }
            j += 1;
        }
        i += 1;
    }
    if draw_freq == 0 {
        drawing.save(format!("flow_fields/output{}.png", img_num)).expect("Failed to save drawing");
        println!("drawing saved");
    }
    flow_field
}