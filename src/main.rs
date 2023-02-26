#![allow(unused)]

use image::{ImageBuffer, Rgb};
use imageproc::drawing::draw_text_mut;
use num_integer::div_ceil;
use palette::{IntoColor, Pixel, RgbHue};
use rusttype::Scale;
use unicode_segmentation::UnicodeSegmentation;

static FONT: &[u8] = include_bytes!("../res/CallingCode-Regular.ttf");

const LEDEND_SAMPLE_WIDTH: u32 = 20;
const MARGIN_RIGHT: u32 = 5;
const MARGIN_LEFT: u32 = 5;
const MARGIN_TOP: u32 = 5;
const INTRABAND_GAP: u32 = 5;

struct Series {
    /// from 0 to 1.0.
    samples: Vec<f32>,
    name: String,
}

fn get_colour(mut x: f32, mut hue: RgbHue) -> Rgb<u8> {
    if !x.is_finite() {
        return Rgb::from([96, 96, 96]);
    }
    x = x.clamp(0.0, 1.0);
    if x > 0.5 {
        hue += RgbHue::from_degrees(20.0 * x - 10.0);
    }
    let q = palette::Hsl::from_components((hue, 1.0, 0.1 + 0.8 * x));
    let q = q.into_color();
    Rgb(palette::Srgb::from_linear(q).into_format().into_raw())
}

fn main() -> anyhow::Result<()> {
    let width = 1920;
    let block_width = 5;
    let block_height = 5;

    let font = rusttype::Font::try_from_bytes(FONT).unwrap();

    let mut dataset = vec![
        Series {
            samples: vec![0.2, 0.4, 0.6, f32::NAN],
            name: "Qqq".to_owned(),
        },
        Series {
            samples: vec![0.4, 0.6, 0.8, 0.0],
            name: "Hello, world".to_owned(),
        },
        Series {
            samples: vec![1.0, 0.0, 0.5, f32::NAN],
            name: "a B C".to_owned(),
        },
    ];

    let main_hues: Vec<RgbHue> = dataset
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let x = 360.0 * (i as f32) / (dataset.len() as f32);
            RgbHue::from_degrees(x)
        })
        .collect();
    let n = dataset.iter().map(|x| x.samples.len()).max().unwrap_or(0);

    let rows = div_ceil(n as u32 * block_width, width - MARGIN_LEFT - MARGIN_RIGHT);

    // simulate drawing legend to get its height
    let cursor_y = legend(&dataset, &main_hues, width, None, &font);

    let band_height = (dataset.len() as u32) * block_height + INTRABAND_GAP;
    let image_height = cursor_y + band_height * rows;

    let mut img = ImageBuffer::<Rgb<u8>, _>::new(width, image_height);

    img.fill(128);

    // Draw legend at the top
    let mut cursor_y = legend(&dataset, &main_hues, width, Some(&mut img), &font);
    let mut cursor_x = MARGIN_LEFT;

    for i in 0..n {
        for ((j, series), hue) in dataset.iter().enumerate().zip(main_hues.iter()) {
            let c = get_colour(series.samples.get(i).copied().unwrap_or(f32::NAN), *hue);
            for u in (0..block_height) {
                for v in (0..block_width) {
                    img.put_pixel(cursor_x + v, cursor_y + (j as u32) * block_height + u, c);
                }
            }
        }

        cursor_x += block_width;
        if cursor_x > width - MARGIN_RIGHT {
            cursor_y += band_height;
            cursor_x = MARGIN_LEFT;
        }
    }

    img.save("output.png")?;
    Ok(())
}

/// Returns final cursor_y position
fn legend(
    dataset: &Vec<Series>,
    main_hues: &Vec<RgbHue>,
    width: u32,
    mut img: Option<&mut ImageBuffer<Rgb<u8>, Vec<u8>>>,
    font: &rusttype::Font,
) -> u32 {
    let mut cursor_x = MARGIN_LEFT;
    let mut cursor_y = MARGIN_TOP;
    let mut empty = true;
    for (series, main_hue) in dataset.iter().zip(main_hues.iter()) {
        // FIXME: unchecked arithmetic
        let remaining = width - MARGIN_RIGHT - cursor_x - LEDEND_SAMPLE_WIDTH - 2 - 12;
        let text_width = series.name.graphemes(true).count() as u32 * 7;
        if text_width > remaining {
            cursor_y += 20;
            cursor_x = MARGIN_LEFT;
        }

        if let Some(img_) = img {
            for i in 0..LEDEND_SAMPLE_WIDTH {
                let x = i as f32 / (LEDEND_SAMPLE_WIDTH - 1) as f32;
                let c = get_colour(x, *main_hue);
                for j in 0..14 {
                    img_.put_pixel(cursor_x + i, cursor_y + j, c);
                }
            }

            let c = get_colour(0.8, *main_hue);

            draw_text_mut(
                img_,
                c,
                (cursor_x + LEDEND_SAMPLE_WIDTH + 2) as i32,
                cursor_y as i32,
                Scale::uniform(14.0),
                &font,
                &series.name,
            );
            img = Some(img_);
        }

        cursor_x += LEDEND_SAMPLE_WIDTH + 2 + text_width + 12;
        empty = false;
    }
    if !empty {
        cursor_y += 20;
    }
    cursor_y
}
