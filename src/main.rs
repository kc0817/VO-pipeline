use image::{DynamicImage, ImageReader};
fn main() {
    println!("Hello, world!");

    // first need to acquire image
    let file_path = "";
    if let Ok(img) = decode_img(file_path) {
    // apply any lens distortion if needed but prob not? I assume A8Mini does that
    // maybe i have to remove perspective projection

    // lets try lucas kanade method

    // check flow field vectors for potential errors and remove outliers

    
    // then idk
    }

}

fn decode_img(file_path: &str) -> Result<DynamicImage, image::ImageError> {
    ImageReader::open(file_path)?
        .decode()
}