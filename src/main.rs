use std::env;
use image::{GrayImage, Luma};
use vo_system::{get_img, get_compressed_img, get_lambda_thresholds, MAX_ITERATIONS, get_shifted_img, 
    get_gradients, calculate_flow_field, draw_arrow, make_video};
use std::cmp;


fn main() {

    let args: Vec<String> = env::args().collect();
    let window_size = 15;
    let start_img: u32 = args[1].trim().parse().unwrap();
    let end_img: u32 = args[2].trim().parse().unwrap();
    let compress_fine: u32 = args[3].trim().parse().unwrap();
    let compress_coarse: u32 = args[4].trim().parse().unwrap();
    let cur_compression: u32 = args[5].trim().parse().unwrap(); // if images are already stored in compressed format

    let should_video: u32 = if args.len() > 6 { args[6].trim().parse().unwrap() } else { 1 };

    if compress_fine < cur_compression {
        println!("compress fine of {compress_fine} is not >= cur compression of {cur_compression}");
        return;
    }
    let num_levels = (compress_coarse - compress_fine) as usize + 1;
    let mut imgs: Vec<Vec<GrayImage>> = Vec::new();

    for i in start_img..(end_img+1) {
        let path = format!("data/frame{i}.png");
        let fine = get_compressed_img(get_img(&path).to_luma8(), compress_fine-cur_compression);
        imgs.push(vec![fine; num_levels]);
        for j in 1..num_levels {
            imgs[(i-start_img) as usize][j as usize] = get_compressed_img(imgs[(i-start_img) as usize][(j-1) as usize].clone(), 1);
        }
    }

    println!("Image compressions finished; starting flow calculations");
    let mut flow_guesses: Vec<Vec<f64>> = Vec::new();

    // iterating over each image
    for i in 0..(imgs.len()-1) {
        let img0s = &imgs[i];
        let img1s = &imgs[i+1];
        let mut flow_guess: Vec<f64> = vec![0., 0.];
        // iterating over each compression from coarse to fine
        for j in 0..num_levels {
            let index = num_levels -1 -j;
            flow_guess[0] = flow_guess[0].round();
            flow_guess[1] = flow_guess[1].round();
            let mut dx = -flow_guess[0] as i32;
            let mut dy = -flow_guess[1] as i32;
            let mut shifted_img = get_shifted_img(
                &img1s[index], 
                dx, 
                dy);
            // println!("on level {j} flow guess = {:.2}, {:.2}", flow_guess[0], flow_guess[1]);
            let (min_lambda, max_lambda_ratio) = get_lambda_thresholds(compress_fine + index as u32);
            let (w, h) = img0s[index].dimensions();
            
            for k in 0..MAX_ITERATIONS {
                let (Ix, Iy, It) = get_gradients(&img0s[index], &shifted_img);
                let (_, avg, _) = calculate_flow_field(&Ix, &Iy, &It, window_size, 
                    min_lambda, max_lambda_ratio,
                    cmp::max(0, dx) as u32, cmp::max(0, dy) as u32,
                    (cmp::min(dx, 0) + w as i32) as u32, (cmp::min(dy, 0) + h as i32) as u32);
                
                let rx = avg[0].round();
                let ry = avg[1].round();
                let delta = 0.0001;
                if (rx.abs() < delta && ry.abs() < delta) || k == MAX_ITERATIONS-1 {
                    flow_guess[0] += avg[0];
                    flow_guess[1] += avg[1];
                    // println!("  adding {:.2}, {:.2} to flow guess", avg[0], avg[1]);
                    break;
                }
                else {
                    flow_guess[0] += rx;
                    flow_guess[1] += ry;
                    dx = -flow_guess[0] as i32;
                    dy = -flow_guess[1] as i32;
                    shifted_img = get_shifted_img(&img1s[index], dx, dy);
                    // println!("  adding {:.2}, {:.2} to flow guess", rx, ry);
                }
            }
            flow_guess[0] *= 2.;
            flow_guess[1] *= 2.;
            
        }
        flow_guess[0] /= 2.;
        flow_guess[1] /= 2.;
        // println!("flow guess: {:.2}, {:.2}", flow_guess[0], flow_guess[1]);
        // keeping this commented so video arrows are to scale
        // let remaining_scale = 2_f64.powf(compress_fine as f64);
        // flow_guess[0] *= remaining_scale;
        // flow_guess[1] *= remaining_scale;

        flow_guesses.push(flow_guess);
    }

    // visualizing images
    let mut drawings: Vec<GrayImage> = Vec::new();
    let (w, h) = imgs[0][0].dimensions();
    let center = (w as i32/2, h as i32/2);
    // let tl = (center.0/2, center.1/2);
    // let tr = (center.0 * 3/2, center.1/2);
    // let bl = (center.0/2, center.1 * 3/2);
    // let br = (center.0 * 3/2, center.1 * 3/2);
    let thick1 = 8;
    let thick2 = 2;
    let white: Luma<u8> = Luma([255]);
    for i in 0..(imgs.len()-1) {
        let mut drawing = imgs[i][0].clone();
        let end = (center.0 + flow_guesses[i][0] as i32, center.1 + flow_guesses[i][1] as i32);
        draw_arrow(&mut drawing, center, end, thick1, thick2, white);
        drawings.push(drawing);
    }

    if should_video == 1 {
        let fps = 30;
        let save_path = format!("videos/iterative_pyramid{}x-{}x_{}-{}.mp4", compress_coarse, compress_fine, start_img, end_img);
        make_video(drawings, w, h, fps, &save_path);
    }
}

