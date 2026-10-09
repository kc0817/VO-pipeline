use image::GenericImageView; 
use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use chrono::Utc;
use std::time::{SystemTime, UNIX_EPOCH};
use vo_system::{decode_img, get_gradients, calculate_flow_field, visualize_flow_field};


fn main() {

    let args: Vec<String> = env::args().collect();

    const n: u32 = 15; // neighborhood size
    let flow_scale = 80.; // scaling each flow vector when visualizing on image

    let mut img_num = 243;
    let mut draw_freq = 0;
    if args.len() > 1 {
        img_num = args[1].trim().parse().unwrap();
        if args.len() > 2 {
            draw_freq = args[2].trim().parse().unwrap();
        }
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

            let (Ix, Iy, It) = get_gradients(&img0, &img1);
            let (flow_field, avg, std) = calculate_flow_field(&Ix, &Iy, &It, n);
            
            if draw_freq != 0 {
                let drawing = visualize_flow_field(&img0, &flow_field, n as usize, draw_freq, flow_scale);
                drawing.save(format!("flow_fields/output{}.png", img_num)).expect("Failed to save drawing");
                println!("drawing saved");
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
    }

}

