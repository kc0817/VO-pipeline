pub mod flow;

use image::{DynamicImage, GrayImage, ImageBuffer, open};
use imageproc::gradients::{horizontal_sobel, vertical_sobel};
use image::Luma;
use imageproc::map::map_pixels2;
use imageproc::point::Point;
use nalgebra::{matrix, SMatrix};
use imageproc::drawing;
use imageproc::pixelops::interpolate;
use flow::Flow;
use std::cmp;
use std::io::Write;
use std::process::{Command, Stdio};

pub const WINDOW_SIZE: u32 = 15;
pub const MIN_LAMBDA_4X: f64 = 20.;
pub const MAX_LAMBDA_RATIO_4X: f64 = 3.;
pub const MIN_LAMBDA_5X: f64 = 11.;
pub const MAX_LAMBDA_RATIO_5X: f64 = 4.;
pub const MIN_LAMBDA_6X: f64 = 7.;
pub const MAX_LAMBDA_RATIO_6X: f64 = 3.;

pub fn get_lambda_thresholds(compression: u32) -> (f64, f64) {
    match compression {
        4 => { (MIN_LAMBDA_4X, MAX_LAMBDA_RATIO_4X) },
        5 => { (MIN_LAMBDA_5X, MAX_LAMBDA_RATIO_5X) },
        6 => { (MIN_LAMBDA_6X, MAX_LAMBDA_RATIO_6X) },
        _ => { (5., 100.) }
    }
}


pub fn get_img(file_path: &str) -> DynamicImage {
    let img = open(file_path).expect("Failed to open image");
    img
}
pub fn get_compressed_img(img: GrayImage, num_compressions: u32) -> GrayImage {
    let mut compressed = img;
    for _ in 0..num_compressions {
        let (w, h) = compressed.dimensions();
        compressed = GrayImage::from_fn(w/2, h/2, |x, y| {
            Luma([apply_kernel(&compressed, (x*2) as i32, (y*2) as i32, w as i32, h as i32)])
        });
    }
    compressed
}
pub fn get_shifted_img(img: &GrayImage, dx: i32, dy: i32) -> GrayImage {
    let (w, h) = img.dimensions();
    GrayImage::from_fn(w, h, |x, y| {
        let new_x = cmp::max(0, cmp::min(w as i32 -1, x as i32 -dx)) as u32;
        let new_y = cmp::max(0, cmp::min(h as i32 -1, y as i32 -dy)) as u32;
        img.get_pixel(new_x, new_y).clone()
    })
}
fn apply_kernel(img: &GrayImage, x: i32, y: i32, w: i32, h: i32) -> u8 {
    let l = cmp::max(0, x-1) as u32;
    let r = cmp::min(w-1, x+1) as u32;
    let u = cmp::max(0, y-1) as u32;
    let d = cmp::min(h-1, y+1) as u32;
    (0.25 * img.get_pixel(x as u32, y as u32)[0] as f64
    + 0.125 * (img.get_pixel(l, y as u32)[0] as f64 + img.get_pixel(r, y as u32)[0] as f64 + img.get_pixel(x as u32, u)[0] as f64 + img.get_pixel(x as u32, d)[0] as f64)
    + 0.0625 * (img.get_pixel(l, u)[0] as f64 + img.get_pixel(l, d)[0] as f64 + img.get_pixel(r, u)[0] as f64 + img.get_pixel(r, d)[0] as f64)
    ).round() as u8
}

pub fn get_grayscale(img: &DynamicImage) -> GrayImage {
    img.to_luma8()
}
pub fn get_gradients(gray_img: &GrayImage, next_gray_img: &GrayImage) -> (ImageBuffer<Luma<i16>, Vec<i16>>, ImageBuffer<Luma<i16>, Vec<i16>>, ImageBuffer<Luma<i16>, Vec<i16>>) {
    // println!("grayscale images computed");
    let Ix = horizontal_sobel(&gray_img);
    let Iy = vertical_sobel(&gray_img);
    let It = map_pixels2(&gray_img, &next_gray_img, |p, q| Luma([q[0] as i16 - p[0] as i16]));    
    // println!("Ix, Iy, and It computed");
    (Ix, Iy, It)
}

// NOTE: x_pixel_end and y_pixel_end are NON INCLUSIVE
pub fn calculate_flow_field(Ix: &ImageBuffer<Luma<i16>, Vec<i16>>, 
    Iy: &ImageBuffer<Luma<i16>, Vec<i16>>, 
    It: &ImageBuffer<Luma<i16>, Vec<i16>>, 
    n: u32,
    min_lambda: f64, max_eigen_ratio: f64,
    x_pixel_start: u32, y_pixel_start: u32, x_pixel_end: u32, y_pixel_end: u32
    ) -> (Vec<Vec<Flow>>, Vec<f64>, Vec<f64>) {
    // let (w, h) = Ix.dimensions();
    let w = x_pixel_end - x_pixel_start;
    let h = y_pixel_end - y_pixel_start;

    let mut flow_field = vec![vec![Flow::new(false, vec![0., 0.], 0., 0., vec![0., 0.], vec![0., 0.]); (w/n) as usize]; (h/n) as usize];
    let mut i = 0;
    let mut avg: Vec<f64> = vec![0., 0.];
    let mut std: Vec<f64> = vec![0., 0.];
    let mut num_good_flow: f64 = 0.;
    // looping over each window
    while i < h/n {
        let mut j = 0;
        while j < w/n {
            let f = calc_flow_vec(&Ix, &Iy, &It, n, 
                x_pixel_start + j * n, y_pixel_start + i * n, 
                min_lambda, max_eigen_ratio);
            if f.valid() {
                avg[0] += f.vec()[0];
                avg[1] += f.vec()[1];
                num_good_flow += 1.;
            }
            flow_field[i as usize][j as usize] = f;
            j += 1;
        }
        i += 1;
    }
    if num_good_flow != 0. {
        avg[0] /= num_good_flow;
        avg[1] /= num_good_flow;
    }

    for row in flow_field.iter() {
        for flow in row.iter() {
            if flow.valid() {
                std[0] += (flow.vec()[0] - avg[0]).powf(2.);
                std[1] += (flow.vec()[1] - avg[1]).powf(2.);
            }
        }
    }
    if num_good_flow != 0. {
        std[0] = (std[0] / num_good_flow).sqrt();
        std[1] = (std[1] / num_good_flow).sqrt();
    }
    // println!("flow field calculated");
    (flow_field, avg, std)
}

pub fn calc_flow_vec(Ix: &ImageBuffer<Luma<i16>, Vec<i16>>, 
        Iy: &ImageBuffer<Luma<i16>, Vec<i16>>,
        It: &ImageBuffer<Luma<i16>, Vec<i16>>,
        n: u32, 
        x_pixel_start: u32, y_pixel_start: u32,
        min_lambda: f64, max_lambda_ratio: f64) -> Flow {
    let mut ATA: SMatrix<f64, 2, 2> = SMatrix::zeros();
    let mut ATb: SMatrix<f64, 2, 1> = SMatrix::zeros();

    // filling out A and b matrices
    let mut y = 0;
    while y < n {
        let mut x = 0;
        while x < n {
            let px = x_pixel_start + x;
            let py = y_pixel_start + y;

            let ix = Ix.get_pixel(px, py)[0] as f64 / 8.;
            let iy = Iy.get_pixel(px, py)[0] as f64 / 8.;
            ATA[(0, 0)] += ix * ix;
            ATA[(1, 0)] += ix * iy;
            ATA[(0, 1)] += iy * ix;
            ATA[(1, 1)] += iy * iy;

            let b = -It.get_pixel(px, py)[0] as f64;
            ATb[(0, 0)] += b * ix;
            ATb[(1, 0)] += b * iy;

            x += 1;
        }
        y += 1;
    }

    // the off diagonal entries are equal so b = c
    let det = ATA[(0, 0)] * ATA[(1, 1)] - ATA[(0, 1)].powf(2.); 
    
    // ATA is well conditioned if lambda1 is not that much greater than lambda2
    // (aka the discriminant is not too big)
    let h = (ATA[(0, 0)] + ATA[(1, 1)]) * 0.5; // (a + d)/2
    let discrim = ((ATA[(0, 0)] - ATA[(1, 1)]) * 0.5).powf(2.) + ATA[(0, 1)].powf(2.);
    let discrim_sqrt = discrim.sqrt();
    let eigen_min = h - discrim_sqrt;
    let scaled_eigen_min = eigen_min / (n * n) as f64;
    let eigen_ratio = if eigen_min.abs() < 0.00001 { 0. } else { (h + discrim_sqrt) / eigen_min };

    let eigen_vec_min: Vec<f64> = if ATA[(0, 1)].abs() < 0.00001 {
        if ATA[(0, 0)] > ATA[(1, 1)] { vec![0., 1.] }
        else { vec![1., 0.] }
    } else {
        let k_min = (ATA[(0, 0)] - eigen_min) / ATA[(0, 1)];
        let eigen_vec_min_mag = (1. + k_min * k_min).sqrt();
        vec![1. / eigen_vec_min_mag, -k_min / eigen_vec_min_mag]
    };
    let eigen_vec_max: Vec<f64> = vec![-eigen_vec_min[1], eigen_vec_min[0]];

    if scaled_eigen_min > min_lambda && eigen_ratio < max_lambda_ratio {
        let inv: SMatrix<f64, 2, 2> = matrix![
            ATA[(1, 1)] / det, -ATA[(1, 0)] / det;
            -ATA[(0, 1)] / det, ATA[(0, 0)] / det
        ];

        let x_star: SMatrix<f64, 2, 1> = inv * ATb;
        Flow::new(true, vec![x_star[(0, 0)], x_star[(1, 0)]], scaled_eigen_min, eigen_ratio, eigen_vec_min, eigen_vec_max)
    } else {
        Flow::new(false, vec![0., 0.], scaled_eigen_min, eigen_ratio, eigen_vec_min, eigen_vec_max)
    }
}


pub fn visualize_flow_field(img: &GrayImage, flow_field: &Vec<Vec<Flow>>, n: usize, draw_freq: usize, flow_scale: f64) -> GrayImage {
    let mut i: usize = 0;
    let mut drawing = img.clone();

    // drawing parameters
    let thick_start = 8;
    let thick_end = 2;
    let shade = Luma([255]);
    while i < flow_field.len() {
        let mut j: usize = 0;
        while j < flow_field[i].len() {
            let f = &flow_field[i][j];
            if f.valid() {
                // drawing flow field on image
                let count = i * flow_field[0].len() + j;
                if count % draw_freq == 0 {
                    let start = ((j * n + n/2) as i32, (i * n + n/2) as i32);
                    let end = (start.0 + (f.vec()[0] * flow_scale) as i32, start.1 + (f.vec()[1] * flow_scale) as i32);

                    draw_arrow(&mut drawing, start, end, thick_start, thick_end, shade);
                }
            }
            j += 1;
        }
        i += 1;
    };
    drawing
}

pub fn draw_arrow(img: &mut GrayImage, start: (i32, i32), end: (i32, i32), thick1: u32, thick2: u32, shade:Luma<u8>) {
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
    drawing::draw_antialiased_polygon_mut(img, &points, shade, interpolate);
}

pub fn make_video(imgs: Vec<GrayImage>, w: u32, h: u32, fps: u32, save_path: &str) {
    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-y",
            "-f", "rawvideo", "-pix_fmt", "gray",
            "-s", &format!("{w}x{h}"),
            "-r", &fps.to_string(),
            "-i", "-",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            &save_path
        ])
        .stdin(Stdio::piped())
        .spawn()
        .expect("Failed to start ffmpeg");
    let stdin = ffmpeg.stdin.as_mut().expect("Failed to open ffmpeg stdin");
    for img in &imgs {
        stdin.write_all(img.as_raw()).expect("Failed to write frame to ffmpeg");
    }
    drop(ffmpeg.stdin.take());
    let status = ffmpeg.wait().expect("ffmpeg didn't run");
    if status.success() {
        println!("Video saved to {save_path}");
    }
}