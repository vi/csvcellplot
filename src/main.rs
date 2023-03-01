use std::path::PathBuf;

use image::{ImageBuffer, Rgb};
use imageproc::drawing::draw_text_mut;
use lerp::Lerp;
use num_integer::div_ceil;
use palette::{IntoColor, Pixel, RgbHue};
use rusttype::Scale;
use unicode_segmentation::UnicodeSegmentation;

static FONT: &[u8] = include_bytes!("../res/CallingCode-Regular.ttf");

const LEDEND_SAMPLE_WIDTH: u32 = 20;
const MARGIN_RIGHT: u32 = 5;
const MARGIN_LEFT: u32 = 5;
const MARGIN_TOP: u32 = 5;

/// read csv data from stdin and write png to file
#[derive(argh::FromArgs)]
struct Opts {
    /// name of output file to write png image to
    #[argh(positional)]
    output_file: PathBuf,

    /// width of the image to write, default 1920
    #[argh(option, short = 'W', default = "1920")]
    image_width: u32,

    /// input file to read CSV from, instead of stdin
    #[argh(option, short = 'i')]
    input_csv: Option<PathBuf>,

    /// width of a cell, in pixels
    #[argh(option, short = 'w', default = "8")]
    cell_width: u32,

    /// height of a cell, in pixels
    #[argh(option, short = 'h', default = "8")]
    cell_height: u32,

    /// do not run data though filter (interpolation), assume they are already from 0 to 1.
    #[argh(switch, short = 'n')]
    no_fiter: bool,

    /// do not hide trivial series
    #[argh(switch, short = 'H')]
    no_hide: bool,

    /// output additionla csv with filtered (interpolated) data
    #[argh(option)]
    debug_filterted_csv: Option<PathBuf>,
}

struct Series {
    /// from 0 to 1.0.
    samples: Vec<f64>,
    name: String,
    hidden: bool,
}

/// x - datum from 0.0 to 1.0, pix_i - number of pixel in this cell, for gradient
#[inline]
fn get_colour(mut x: f64, mut hue: RgbHue, pix_i: u32, pix_n: u32) -> Rgb<u8> {
    if !x.is_finite() {
        if pix_i % 2 == 0 {
            return Rgb::from([96, 96, 96]);
        } else {
            return Rgb::from([140, 140, 140]);
        }
    }
    x = x.clamp(0.0, 1.0);
    if x > 0.5 {
        hue += RgbHue::from_degrees(20.0 * x as f32 - 10.0);
    }
    let mut lightness = 0.1 + 0.7 * x;
    if pix_n > 1 {
        lightness += 0.1 - 0.2 * (pix_i as f64) / (pix_n - 1) as f64;
    }
    let q = palette::Hsl::from_components((hue, 1.0, lightness as f32));
    let q = q.into_color();
    Rgb(palette::Srgb::from_linear(q).into_format().into_raw())
}

fn main() -> anyhow::Result<()> {
    let opts: Opts = argh::from_env();

    let width = opts.image_width;
    let block_width = opts.cell_width;
    let block_height = opts.cell_height;

    let font = rusttype::Font::try_from_bytes(FONT).unwrap();

    let mut dataset = Vec::<Series>::with_capacity(4);

    {
        let input: Box<dyn std::io::Read>;
        if let Some(input_file_path) = opts.input_csv {
            input = Box::new(std::fs::File::open(input_file_path)?);
        } else {
            input = Box::new(std::io::stdin());
        }
        //let input = std::io::BufReader::with_capacity(1024*256, input);

        let mut csv = csv::Reader::from_reader(input);

        for h in csv.headers()? {
            dataset.push(Series {
                samples: Vec::with_capacity(4096),
                name: h.to_owned(),
                hidden: false,
            })
        }

        for r in csv.records() {
            let r = r?;
            for (i, s) in r.iter().enumerate() {
                let x = s.parse().unwrap_or(f64::NAN);
                dataset[i].samples.push(x);
            }
        }
    }

    let n = dataset.iter().map(|x| x.samples.len()).max().unwrap_or(0);

    if !opts.no_fiter {
        for serie in dataset.iter_mut() {
            let mut sorted = Vec::with_capacity(serie.samples.len());
            for x in &serie.samples {
                if x.is_finite() && !opts.no_fiter {
                    sorted.push(x);
                }
            }
            let mut dummy = false;
            let n = sorted.len();
            if n < 2 {
                dummy = true;
            } else {
                sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

                let allowed_duplicates = 0; // (n / MAX_INTERP).max(1);
                let mut duplicate = f64::NAN;
                let mut duplicate_n = 0usize;
                sorted.retain(|p| {
                    if duplicate == **p {
                        duplicate_n += 1;
                        duplicate_n < allowed_duplicates
                    } else {
                        duplicate = **p;
                        duplicate_n = 0;
                        true
                    }
                });
                let n = sorted.len();

                if n < 2 {
                    dummy = true;
                } else {
                    const MAX_INTERP: usize = 32;
                    let interpolation_n = (n - 1).min(MAX_INTERP);
                    #[derive(Debug)]
                    struct InterpolationPoint {
                        start_sample: f64,
                        //stop_sample: f64,
                        inv_stop_sample_minus_start_sample: f64,
                        start_outrange: f64,
                        stop_outrange: f64,
                    }
                    let mut interpolation_points = Vec::with_capacity(interpolation_n);
                    let epsilon: f64 =
                        (0.00000001f64).max((sorted[n - 1] - sorted[0]) / 1000_000.0);
                    for i in 0..interpolation_n {
                        let mut start_outrange = (i as f64) / (interpolation_n as f64);
                        let mut stop_outrange = ((i + 1) as f64) / (interpolation_n as f64);
                        let start_sample_index = i as f64 * (n - 1) as f64 / interpolation_n as f64;
                        let stop_sample_index =
                            (i + 1) as f64 * (n - 1) as f64 / interpolation_n as f64;

                        let start_sample_index_i = (start_sample_index.floor() as usize).min(n - 1);
                        let start_sample_index_j = (start_sample_index_i + 1).min(n - 1);
                        let start_sample_index_t = start_sample_index.fract();
                        let mut start_sample = sorted[start_sample_index_i]
                            .lerp(*sorted[start_sample_index_j], start_sample_index_t);
                        let stop_sample_index_i = (stop_sample_index.floor() as usize).min(n - 1);
                        let stop_sample_index_j = (stop_sample_index_i + 1).min(n - 1);
                        let stop_sample_index_t = stop_sample_index.fract();
                        let mut stop_sample = sorted[stop_sample_index_i]
                            .lerp(*sorted[stop_sample_index_j], stop_sample_index_t);

                        if i == 0 {
                            start_sample = *sorted[0];
                        }
                        if i == interpolation_n - 1 {
                            stop_sample = *sorted[n - 1];
                        }

                        if stop_sample - start_sample < epsilon {
                            start_outrange = 0.5 * start_outrange + 0.5 * stop_outrange;
                            stop_outrange = start_outrange;
                            stop_sample = start_sample + 1.0;
                        }

                        //dbg!(epsilon,start_sample_index,stop_sample_index,n,stop_sample_index_i,stop_sample_index_j,stop_sample,sorted[stop_sample_index_i],sorted[stop_sample_index_j]);
                        let inv_stop_sample_minus_start_sample = 1.0 / (stop_sample - start_sample);

                        interpolation_points.push(InterpolationPoint {
                            start_outrange,
                            start_sample,
                            stop_outrange,
                            inv_stop_sample_minus_start_sample,
                        });
                    }

                    //dbg!(&interpolation_points);

                    for x in &mut serie.samples {
                        if x.is_finite() {
                            let ret = interpolation_points
                                .binary_search_by(|cand| cand.start_sample.partial_cmp(x).unwrap());
                            let mut index = match ret {
                                Ok(t) => t,
                                Err(t) => t.saturating_sub(1),
                            };
                            if index >= interpolation_n {
                                index = interpolation_n - 1;
                            }

                            let InterpolationPoint {
                                start_outrange,
                                start_sample,
                                stop_outrange,
                                inv_stop_sample_minus_start_sample,
                            } = interpolation_points[index];

                            let t = (*x - start_sample) * inv_stop_sample_minus_start_sample;
                            let y = start_outrange.lerp(stop_outrange, t);
                            //dbg!(*x, index, t, y);
                            *x = y;
                        }
                    }
                }
            }
            if dummy {
                serie.hidden = true;
                for x in &mut serie.samples {
                    if x.is_finite() {
                        *x = 0.5;
                    }
                }
            }
        }
    }

    if ! opts.no_hide {
        dataset.retain(|x|!x.hidden);
    }

    if let Some(dbgout) = opts.debug_filterted_csv {
        let mut csvout = csv::Writer::from_path(dbgout)?;
        for serie in &dataset {
            csvout.write_field(&serie.name)?;
        }
        csvout.write_record(None::<&[u8]>)?;

        for i in 0..n {
            for serie in &dataset {
                match serie.samples.get(i) {
                    Some(x) if x.is_finite() => {
                        csvout.write_field(format!("{x}"))?;
                    }
                    _ => csvout.write_field("")?,
                }
            }
            csvout.write_record(None::<&[u8]>)?;
        }

        csvout.flush()?;
        drop(csvout);
        eprintln!("Finished writing debug csv");
    }

    let mut hue_step = 360.0 / (dataset.len() as f32);
    if hue_step < 55.0 {
        hue_step = 55.0
    };

    let main_hues: Vec<RgbHue> = dataset
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let x = hue_step * (i as f32);
            RgbHue::from_degrees(x)
        })
        .collect();
    let n = dataset.iter().map(|x| x.samples.len()).max().unwrap_or(0);

    let rows = div_ceil(n as u32 * block_width, width - MARGIN_LEFT - MARGIN_RIGHT);

    // simulate drawing legend to get its height
    let cursor_y = legend(&dataset, &main_hues, width, None, &font);

    let mut intraband_gap = 0u32;

    if dataset.len() > 1 {
        intraband_gap = (dataset.len() as u32 + 3) / 4;
        intraband_gap *= block_height;
    }

    let band_height = (dataset.len() as u32) * block_height + intraband_gap;
    let image_height = cursor_y + band_height * rows;

    let mut img = ImageBuffer::<Rgb<u8>, _>::new(width, image_height);

    img.fill(128);

    // Draw legend at the top
    let mut cursor_y = legend(&dataset, &main_hues, width, Some(&mut img), &font);
    let mut cursor_x = MARGIN_LEFT;

    for i in 0..n {
        for ((j, series), hue) in dataset.iter().enumerate().zip(main_hues.iter()) {
            let pix_n = block_height * block_height;
            let mut pix_i = 0;
            for v in 0..block_width {
                for u in 0..block_height {
                    let c = get_colour(
                        series.samples.get(i).copied().unwrap_or(f64::NAN),
                        *hue,
                        pix_i,
                        pix_n,
                    );
                    img.put_pixel(cursor_x + v, cursor_y + (j as u32) * block_height + u, c);
                    pix_i += 1;
                }
            }
        }

        cursor_x += block_width;
        if cursor_x > width - MARGIN_RIGHT {
            cursor_y += band_height;
            cursor_x = MARGIN_LEFT;
        }
    }

    img.save(opts.output_file)?;
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
        let fixed_width = MARGIN_RIGHT + LEDEND_SAMPLE_WIDTH + 2 + 12;
        let text_width = series.name.graphemes(true).count() as u32 * 7;
        if cursor_x + fixed_width + text_width > width {
            cursor_y += 20;
            cursor_x = MARGIN_LEFT;
        }

        if let Some(img_) = img {
            for i in 0..LEDEND_SAMPLE_WIDTH {
                let x = i as f64 / (LEDEND_SAMPLE_WIDTH - 1) as f64;
                let c = get_colour(x, *main_hue, 0, 1);
                for j in 0..14 {
                    img_.put_pixel(cursor_x + i, cursor_y + j, c);
                }
            }

            let c = get_colour(0.8, *main_hue, 0, 1);

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
