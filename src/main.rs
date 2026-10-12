use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use chrono::Utc;
use std::time::{SystemTime, UNIX_EPOCH};
use vo_system::{get_img, get_grayscale, get_compressed_img, get_gradients, calculate_flow_field, visualize_flow_field};


fn main() {

    let args: Vec<String> = env::args().collect();

    const n: u32 = 15; // neighborhood size
    let flow_scale = 80.; // scaling each flow vector when visualizing on image

    let mut img_num = 243;
    let mut compress_num = 1;
    let mut draw_freq: i32 = 0;
    if args.len() > 1 {
        img_num = args[1].trim().parse().unwrap();
        if args.len() > 2 {
            compress_num = args[2].trim().parse().unwrap();
            if args.len() > 3 {
                draw_freq = args[3].trim().parse().unwrap();
            }
        }
    }
    println!("Processing img {img_num}, compress num {compress_num} draw_freq {draw_freq}");
    // first need to acquire image
    let img0_path = format!("data/frame{}.png", img_num);
    let img1_path = format!("data/frame{}.png", img_num+1);

    let mut img0 = get_grayscale(&get_img(&img0_path));
    let mut img1 = get_grayscale(&get_img(&img1_path));
    for _ in 0..compress_num {
        img0 = get_compressed_img(img0);
        img1 = get_compressed_img(img1);
    }
    let (w, h) = img0.dimensions();
    println!("Image dimensions: {w} x {h} pixels");

    println!("both images read successfully; starting time benchmark");
    let start_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("wtf")
        .as_millis();

    let (Ix, Iy, It) = get_gradients(&img0, &img1);
    let (flow_field, avg, std) = calculate_flow_field(&Ix, &Iy, &It, n);
    
    if draw_freq != 0 {
        let drawing;
        let path;
        if draw_freq < 0 {
            drawing = img0.clone();
            path = format!("flow_fields/compressed{}x{}.png", compress_num, img_num);
        }
        else {
            drawing = visualize_flow_field(&img0, &flow_field, n as usize, draw_freq as usize, flow_scale);
            path = format!("flow_fields/labeled{}x{}.png", compress_num, img_num);
        }
        drawing.save(&path).expect("Failed to save drawing");
        println!("drawing saved at {path}");
    }

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
            date, img_num, time_elapsed, avg[0], avg[1], std[0], std[1])
            .expect("Failed to write to log file");

    println!("done with img {img_num}");
    // check flow field vectors for potential errors and remove outliers           
    // then idk
}

