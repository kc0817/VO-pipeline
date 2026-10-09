pub mod flow;

use image::{DynamicImage, GrayImage, ImageReader, Rgba, ImageBuffer};
use image::GenericImageView; 
use imageproc::gradients::{horizontal_sobel, vertical_sobel};
use image::Luma;
use imageproc::map::map_pixels2;
use imageproc::point::Point;
use nalgebra::{matrix, SMatrix};
use imageproc::drawing;
use imageproc::pixelops::interpolate;
use flow::Flow;

pub fn decode_img(file_path: &str) -> Result<DynamicImage, image::ImageError> {
    ImageReader::open(file_path)?
        .decode()
}
pub fn draw_line(img: &mut DynamicImage, start: (i32, i32), end: (i32, i32), thick1: u32, thick2: u32, color:Rgba<u8>) {
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

pub fn get_gradients(img0: &DynamicImage, img1: &DynamicImage) -> (ImageBuffer<Luma<i16>, Vec<i16>>, ImageBuffer<Luma<i16>, Vec<i16>>, ImageBuffer<Luma<i16>, Vec<i16>>) {
    // calculating gradients
    let gray_img: GrayImage = img0.to_luma8();
    let next_gray_img: GrayImage = img1.to_luma8();
    println!("grayscale images computed");
    let Ix = horizontal_sobel(&gray_img);
    let Iy = vertical_sobel(&gray_img);
    let It = map_pixels2(&gray_img, &next_gray_img, |p, q| Luma([q[0] as i16 - p[0] as i16]));    
    println!("Ix, Iy, and It computed");
    (Ix, Iy, It)
}
pub fn calculate_flow_field(Ix: &ImageBuffer<Luma<i16>, Vec<i16>>, Iy: &ImageBuffer<Luma<i16>, Vec<i16>>, It: &ImageBuffer<Luma<i16>, Vec<i16>>, n: u32) -> (Vec<Vec<Flow>>, Vec<f64>, Vec<f64>) {
    let (w, h) = Ix.dimensions();

    let mut flow_field = vec![vec![Flow::new(false, vec![0., 0.]); (w/n) as usize]; (h/n) as usize];
    let mut i = 0;
    let mut avg: Vec<f64> = vec![0., 0.];
    let mut std: Vec<f64> = vec![0., 0.];
    let mut num_good_flow: f64 = 0.;
    // looping over each window
    let total_rows = h/n;
    while i < total_rows {
        let mut j = 0;
        while j < w/n {
            let f = calc_flow_vec(&Ix, &Iy, &It, n, i, j);
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
    avg[0] /= num_good_flow;
    avg[1] /= num_good_flow;

    for row in flow_field.iter() {
        for flow in row.iter() {
            if flow.valid() {
                std[0] += (flow.vec()[0] - avg[0]).powf(2.);
                std[1] += (flow.vec()[1] - avg[1]).powf(2.);
            }
        }
    }
    std[0] = (std[0] / num_good_flow).sqrt();
    std[1] = (std[1] / num_good_flow).sqrt();

    println!("flow field calculated");
    (flow_field, avg, std)
}

pub fn visualize_flow_field(img: &DynamicImage, flow_field: &Vec<Vec<Flow>>, n: usize, draw_freq: usize, flow_scale: f64) -> DynamicImage {
    let mut i: usize = 0;
    let mut drawing = img.clone();

    // drawing parameters
    let thick_start = 14;
    let thick_end = 4;
    let color = Rgba([255, 255, 255, 255]); // white, with 100% opacity

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

                    draw_line(&mut drawing, start, end, thick_start, thick_end, color);
                }
            }
            j += 1;
        }
        i += 1;
    };
    drawing
}

pub fn calc_flow_vec(Ix: &ImageBuffer<Luma<i16>, Vec<i16>>, 
        Iy: &ImageBuffer<Luma<i16>, Vec<i16>>,
        It: &ImageBuffer<Luma<i16>, Vec<i16>>,
        n: u32, i: u32, j: u32) -> Flow {
    let mut ATA: SMatrix<f64, 2, 2> = SMatrix::zeros();
    let mut ATb: SMatrix<f64, 2, 1> = SMatrix::zeros();

    // filling out A and b matrices
    let mut y = 0;
    while y < n {
        let mut x = 0;
        while x < n {
            let px = j * n + x;
            let py = i * n + y;

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

    // prob need to tune these values
    let epsilon: f64 = 0.;
    let epsilon2: f64 = 500.;
    
    // ATA is well conditioned if lambda1 is not that much greater than lambda2
    // (aka the discriminant is not too big)
    let h = ((ATA[(0, 0)] + ATA[(1, 1)]) * 0.5);
    let discrim = ((ATA[(0, 0)] - ATA[(1, 1)]) * 0.5).powf(2.) + ATA[(0, 1)].powf(2.);
    let discrim_sqrt = discrim.sqrt();
    let lambda_min = h - discrim_sqrt;
    let lambda_max = h + discrim_sqrt;

    // only used in test_flow.rs
    // println!("lambda_min: {}", lambda_min);
    // println!("lambda_max: {}", lambda_max);

    if lambda_min / (n*n) as f64 > epsilon && lambda_max / lambda_min < epsilon2 {
        let inv: SMatrix<f64, 2, 2> = matrix![
            ATA[(1, 1)] / det, -ATA[(1, 0)] / det;
            -ATA[(0, 1)] / det, ATA[(0, 0)] / det
        ];

        let x_star: SMatrix<f64, 2, 1> = inv * ATb;
        let flow = Flow::new(true, vec![x_star[(0, 0)], x_star[(1, 0)]]);
        flow
    } else {
        Flow::new(false, vec![0., 0.])
    }
}


