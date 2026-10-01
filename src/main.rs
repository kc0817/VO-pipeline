use image::{DynamicImage, GrayImage, ImageReader, Rgba};
use image::GenericImageView; 
use imageproc::gradients::{horizontal_sobel, vertical_sobel};
use image::Luma;
use imageproc::map::map_pixels2;
use nalgebra::{SMatrix};
use imageproc::drawing::draw_antialiased_line_segment;
use imageproc::pixelops::interpolate;
fn main() {
    const n: u32 = 15; // neighborhood size
    println!("n={n}");
    let draw = true;
    let debug_num = 10;
    let save_num = 100;

    // first need to acquire image
    let img0_path = "data/frame0.png";
    let img1_path = "data/frame1.png";
    if let Ok(img0) = decode_img(img0_path) {
        let (w, h) = img0.dimensions();
        println!("Image dimensions: {w} x {h} pixels");

        if let Ok(img1) = decode_img(img1_path) {
            
            let gray_img: GrayImage = img0.to_luma8();
            let next_gray_img: GrayImage = img1.to_luma8();
            println!("both images read successfully");

            let Ix = horizontal_sobel(&gray_img);
            let Iy = vertical_sobel(&gray_img);
            let It = map_pixels2(&gray_img, &next_gray_img, |p, q| Luma([q[0] as i16 - p[0] as i16]));
            
            println!("Ix, Iy, and It computed");

        //     // lens distortion? perspective projection? maybe?? idk


        //     // lets try lucas kanade method
            let mut flow_field = vec![vec![vec![0.; 2]; (w/n) as usize]; (h/n) as usize];
            let mut drawing = img0;
            let t: f64 = n as f64 * 0.8;
            let color = Rgba([0, 0, 255, 100]);
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
                            A[(col, 0)] = Ix.get_pixel(px, py)[0] as f64;
                            A[(col, 1)] = Iy.get_pixel(px, py)[0] as f64;
                            b[col] = It.get_pixel(px, py)[0] as f64;

                            x += 1;
                        }
                        y += 1;
                    }
                
                    let ATA = A.transpose() * A;
                    if let Some(inv) = ATA.try_inverse() {
                        let x_star = inv * A.transpose() * b;
                        let flow = vec![x_star[(0, 0)], x_star[(1, 0)]];
                        flow_field[i as usize][j as usize] = flow.clone();

                        // drawing flow field on image
                        if draw && i % debug_num == 0 && j % debug_num == 0 {
                            let start = ((j * n + n/2) as i32, (i * n + n/2) as i32);
                            let end = (start.0 + (flow[0] * t) as i32, start.1 + (flow[1] * t) as i32);
                            drawing = image::DynamicImage::ImageRgba8(draw_antialiased_line_segment(&drawing, start, end, color, interpolate));
                            if j % save_num == 0 {
                                drawing.save("flow_fields/output0.png").expect(&format!("Failed to save drawing at i={}, j={}", i, j));
                            }
                        }
                    } else {
                        flow_field[i as usize][j as usize] = vec![0., 0.];
                        println!("    not invertible");
                    }
                    j += 1;
                }
                i += 1;
            }
            if draw {
                drawing.save("flow_fields/output0.png").expect("Failed to save drawing");
                println!("drawing saved");
            }
            println!("done");
            // check flow field vectors for potential errors and remove outliers           
            // then idk
        }
    }

}

fn decode_img(file_path: &str) -> Result<DynamicImage, image::ImageError> {
    ImageReader::open(file_path)?
        .decode()
}