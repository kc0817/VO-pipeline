use image::{DynamicImage, GrayImage, ImageReader, Rgb, Rgba};
use image::GenericImageView; 
use imageproc::gradients::{horizontal_sobel, vertical_sobel};
use image::Luma;
use imageproc::map::map_pixels2;
use imageproc::point::Point;
use nalgebra::{SMatrix};
use imageproc::drawing;
use imageproc::pixelops::interpolate;
use std::env;
pub mod flow;
use flow::Flow;

fn main() {

    let args: Vec<String> = env::args().collect();

    const n: u32 = 15; // neighborhood size
    let draw = true;
    let debug_num = 10;
    let save_num = 200;

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
            
            let gray_img: GrayImage = img0.to_luma8();
            let next_gray_img: GrayImage = img1.to_luma8();
            println!("both images read successfully");

            let Ix = horizontal_sobel(&gray_img);
            let Iy = vertical_sobel(&gray_img);
            let It = map_pixels2(&gray_img, &next_gray_img, |p, q| Luma([q[0] as i16 - p[0] as i16]));
            
            println!("Ix, Iy, and It computed");
            let num_flow_vecs = h/n * w/n;
            println!("Now computing {num_flow_vecs} flow vectors");

        //     // lens distortion? perspective projection? maybe?? idk


        //     // lets try lucas kanade method
            let mut flow_field = vec![vec![Flow::new(false, vec![0., 0.]); (w/n) as usize]; (h/n) as usize];
            let mut drawing = img0;
            let t: f64 = 80.;
            let color = Rgba([255, 255, 255, 100]);
            let thick_start = 16;
            let thick_end = 4;
            let mut i = 0;

            // looping over each window
            while i < h/n {
                let mut j = 0;
                while j < w/n {

                    if i % debug_num == 0 && j % debug_num == 0 {
                        println!("Processing window {i}, {j}");
                    }

                    const nrows: usize = (n*n) as usize;
                    let mut A: SMatrix<f64, nrows, 2> = SMatrix::zeros();
                    let mut b: SMatrix<f64, nrows, 1> = SMatrix::zeros();

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
                
                    let ATA = A.transpose() * A;
                    if let Some(inv) = ATA.try_inverse() {
                        let x_star: nalgebra::Matrix<f64, nalgebra::Const<2>, nalgebra::Const<1>, nalgebra::ArrayStorage<f64, 2, 1>> = inv * A.transpose() * b;
                        let flow = Flow::new(true, vec![x_star[(0, 0)], x_star[(1, 0)]]);
                        flow_field[i as usize][j as usize] = flow.clone();

                        // drawing flow field on image
                        if draw && i % debug_num == 0 && j % debug_num == 0 {
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
            if draw {
                drawing.save(format!("flow_fields/output{}.png", img_num)).expect("Failed to save drawing");
                println!("drawing saved");
            }

            let mut avg_vx: f64 = 0.;
            let mut avg_vy: f64 = 0.;
            let mut num_vecs: f64 = 0.;
            for row in flow_field.iter() {
                for flow in row.iter() {
                    if flow.valid() {
                        num_vecs += 1.;
                        avg_vx += flow.vec()[0];
                        avg_vy += flow.vec()[1];
                    }
                }
            }
            avg_vx /= num_vecs;
            avg_vy /= num_vecs;
            println!("vx: {avg_vx}, vy: {avg_vy}");


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