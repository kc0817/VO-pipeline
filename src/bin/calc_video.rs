use std::env;
use image::{GrayImage, Luma};
use vo_system::{get_img, get_grayscale, get_compressed_img, get_gradients, calculate_flow_field, draw_arrow};
use std::io::Write;
use std::process::{Command, Stdio};
fn main() {
    // arg list: start_img, end_img, compress, window_size
    let args: Vec<String> = env::args().collect();
    let start_img: u32 = args[1].trim().parse().unwrap();
    let end_img: u32 = args[2].trim().parse().unwrap();
    let compress_num: u32 = args[3].trim().parse().unwrap();
    let window_size: u32 = if args.len() > 4 { args[4].trim().parse().unwrap() } else { 15 };

    println!("Running from frame {start_img} to {end_img} with {compress_num}x compression and window size of {window_size}");

    // visualization properties
    let arrow_shade = Luma([255]);
    let arrow_scale = 15.;
    let thick1 = 10;
    let thick2 = 4;
    let center;

    // loading all images
    let mut compressed_imgs: Vec<GrayImage> = Vec::new();
    for i in start_img..end_img {
        let path = format!("data/frame{}.png", i);
        let compressed = get_compressed_img(get_grayscale(&get_img(&path)), compress_num);
        compressed_imgs.push(compressed);
    }
    let (w, h) = compressed_imgs[0].dimensions();
    center = ((w/2) as i32, (h/2) as i32);

    println!("Image compressions finished; starting flow calculations");

    // calculating and visualizing flow for each image
    let mut visualized_imgs: Vec<GrayImage> = Vec::new();
    for i in 0..(compressed_imgs.len()-1) {
        let (hori_grad, vert_grad, intensity_diff) = get_gradients(&compressed_imgs[i as usize], &compressed_imgs[(i+1) as usize]);
        let (_, avg_flow, _) = calculate_flow_field(&hori_grad, &vert_grad, &intensity_diff, window_size, false);
        let mut drawing = compressed_imgs[i as usize].clone();
        let end = (center.0 + (avg_flow[0] * arrow_scale) as i32, center.1 + (avg_flow[1] * arrow_scale) as i32);
        draw_arrow(&mut drawing, center, end, thick1, thick2, arrow_shade);
        visualized_imgs.push(drawing);
    }

    println!("Flow calculations and visualizations finished; saving drawings");

    // saving visualizations as video
    let fps = 60;
    let save_path = format!("videos/frame{}-{}compress{}xfps{}.mp4", start_img, end_img, compress_num, fps);
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
    for img in &visualized_imgs {
        stdin.write_all(img.as_raw()).expect("Failed to write frame to ffmpeg");
    }
    drop(ffmpeg.stdin.take());
    let status = ffmpeg.wait().expect("ffmpeg didn't run");
    if status.success() {
        println!("Video saved to {save_path}");
    }
}