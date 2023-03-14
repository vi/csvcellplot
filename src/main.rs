use std::{collections::HashMap, path::{PathBuf, Path}};

use anyhow::Context;
use image::{ImageBuffer, Rgb};
use imageproc::drawing::{draw_text_mut, text_size};
use lerp::Lerp;
use num_integer::div_ceil;
use palette::{Hsl, IntoColor, Pixel, RgbHue, Srgb};
use rusttype::Scale;
use slab::Slab;

static FONT: &[u8] = include_bytes!("../res/SometypeMono-Regular.ttf");

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

    /// width of the image to write, default 1920 or automatic if -R is present
    #[argh(option, short = 'W')]
    image_width: Option<u32>,

    /// input file to read CSV from, instead of stdin
    #[argh(option, short = 'i')]
    input_csv: Option<PathBuf>,

    /// width of a cell, in pixels
    #[argh(option, short = 'w', default = "24")]
    cell_width: u32,

    /// height of a cell, in pixels
    #[argh(option, short = 'h', default = "24")]
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

    /// explicitly specify column colours, like `column1=red,column2=FF00FF`
    /// colours may also contain a number of modifier postfix characters:
    /// `+`, `-` - shift hue
    /// `/` - desaturate.
    /// `@` - hue drift
    /// `_`,`.` - decrease,increase min lightness
    /// `^`, `~` - decrease, increase max lightness
    /// `%`, `&` - decrease, increase gradientness
    #[argh(option, short = 'c', default = "Default::default()")]
    colour_overrides: ColourOverrides,

    /// use this saturation value for colours not specified explicitly. Defaults to 1.0.
    #[argh(option, short = 'S', default = "1.0")]
    default_saturation: f32,

    /// defaults to 0.2
    #[argh(option, short = 'x', default = "0.2")]
    default_min_lightness: f32,

    /// defaults to 0.75
    #[argh(option, short = 'X', default = "0.75")]
    default_max_lightness: f32,

    /// defaults to 0.0
    #[argh(option, short = 'G', default = "0.0")]
    default_gradientness: f32,

    /// shift hue toghether with lightness. defaults to 0.0
    #[argh(option, short = 'D', default = "0.0")]
    default_hue_drift: f32,

    /// maximum number of cell of one data series in a row
    #[argh(option, short = 'R')]
    max_cells_in_row: Option<usize>,

    /// shotrcut for -x 0 -X 1 -S 0
    #[argh(switch, short = 'g')]
    grayscale: bool,

    /// font file (ttf) to render legend text. Default is embedded font Dharma Type Sometype Mono
    #[argh(option)]
    legend_font: Option<PathBuf>,

    /// font scale to render legend text. Default is 14.
    /// Setting it to 0 prevents rendering legend.
    #[argh(option, default = "24.0")]
    legend_font_scale: f32,

    /// use CIE L*C*h° instead of HSL, also automatically lower the `-S` unless explicitly specified
    #[argh(switch, short = 'L')]
    lch: bool,

    /// maximum hue angle when auto-assigning hues.
    /// Auto-assigned hues will loop around when this angle is surpassed
    #[argh(option, default = "32.0")]
    max_hue_angle: f32,

    /// maximum number of ranked points to bring the values range to 0..1.
    /// Use value 1 for linear interpolation, use high value for ranked
    #[argh(option, default = "32")]
    max_interpolation_points: usize,

    /// partition data by values of this column and produce multiple output images.
    /// Normalisation is still performned together.
    #[argh(option, short='p')]
    partition: Option<String>,
}

#[derive(Clone, Copy)]
struct Style {
    hue: RgbHue<f32>,
    saturation: f32,
    min_lightness: f32,
    max_lightness: f32,
    gradientness: f32,
    hue_drift: f32,
}

#[derive(Default)]
struct ColourOverrides(HashMap<String, Style>);

impl std::str::FromStr for ColourOverrides {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut ret = HashMap::new();
        for chunk in s.split(',') {
            let Some((before_equals, mut after_equals)) = chunk.split_once('=') else {
                anyhow::bail!("Each colour override chunk should contain `=`")
            };
            let mut hue_steps = 0;
            let mut desaturates = 0;
            let mut min_lightness = 0.2;
            let mut max_lightness = 0.75;
            let mut gradientness = 0.0;
            let mut hue_drift = 0.0;
            loop {
                let Some(last_char) = after_equals.as_bytes().last() else { break };
                match *last_char {
                    b'+' => hue_steps += 1,
                    b'-' => hue_steps -= 1,
                    b'/' => desaturates += 1,
                    b'@' => hue_drift += 10.0,
                    b'_' => min_lightness -= 0.05,
                    b'.' => min_lightness += 0.05,
                    b'^' => max_lightness -= 0.05,
                    b'~' => max_lightness += 0.05,
                    b'%' => gradientness -= 0.05,
                    b'&' => gradientness += 0.05,
                    _ => break,
                }
                after_equals = &after_equals[0..(after_equals.len() - 1)];
            }
            let colour = match palette::named::from_str(after_equals) {
                Some(x) => x,
                _ => match Srgb::from_str(after_equals) {
                    Ok(x) => x,
                    _ => anyhow::bail!(
                        "colour should be name like `red` or hex spec like `#fff` or `004455`"
                    ),
                },
            };

            let colour = colour.into_format::<f32>();
            let hsl: Hsl = colour.into_color();
            let mut hue = hsl.hue.to_degrees();
            let mut saturation = hsl.saturation;

            hue += 5.0 * hue_steps as f32;
            for _ in 0..desaturates {
                saturation *= 0.8;
            }

            let style = Style {
                hue: RgbHue::from_degrees(hue),
                saturation,
                min_lightness,
                max_lightness,
                gradientness,
                hue_drift,
            };

            ret.insert(before_equals.to_owned(), style);
        }
        Ok(ColourOverrides(ret))
    }
}

struct Series {
    /// from 0 to 1.0.
    samples: Vec<f64>,
    name: String,
    hidden: bool,
}

#[derive(Default)]
struct Partitions {
    name2idx: HashMap<String, usize>,
    idx2name: Slab<String>,
    samples: Vec<usize>,
}

/// x - datum from 0.0 to 1.0, pix_i - number of pixel in this cell, for gradient
#[inline]
fn get_colour(mut x: f64, style: &Style, pix_i: u32, pix_n: u32, lch: bool) -> Rgb<u8> {
    if !x.is_finite() {
        if pix_i % 2 == 0 {
            return Rgb::from([96, 96, 96]);
        } else {
            return Rgb::from([140, 140, 140]);
        }
    }
    x = x.clamp(0.0, 1.0);
    let x = x as f32;
    let mut hue = style.hue;
    hue += x * style.hue_drift;
    let mut lightness = style.min_lightness + x * (style.max_lightness - style.min_lightness);
    if pix_n > 1 {
        let gradient_pos = (pix_i as f32) / (pix_n - 1) as f32;
        lightness += style.gradientness * (2.0 * gradient_pos - 1.0);
    }
    if !lch {
        let q = palette::Hsl::from_components((hue, style.saturation, lightness));
        let q = q.into_color();
        Rgb(palette::Srgb::from_linear(q).into_format().into_raw())
    } else {
        let q = palette::Lch::from_components((
            25.0 + 75.0 * lightness,
            100.0 * style.saturation,
            hue.to_degrees() + 30.0,
        ));
        let q = q.into_color();
        Rgb(palette::Srgb::from_linear(q).into_format().into_raw())
    }
}

fn main() -> anyhow::Result<()> {
    let mut opts: Opts = argh::from_env();

    if opts.grayscale {
        opts.default_min_lightness = 0.0;
        opts.default_max_lightness = 1.0;
        opts.default_saturation = 0.0;
    }

    let width = match opts.image_width {
        Some(x) => x,
        None => match opts.max_cells_in_row {
            None => 400,
            Some(y) => (MARGIN_LEFT + MARGIN_RIGHT + opts.cell_width * y as u32).max(64),
        },
    };

    if width < MARGIN_LEFT + MARGIN_RIGHT + LEDEND_SAMPLE_WIDTH {
        anyhow::bail!("Image width too small")
    }

    let block_width = opts.cell_width;
    let block_height = opts.cell_height;

    let font = if let Some(ref fontpath) = opts.legend_font {
        rusttype::Font::try_from_vec(std::fs::read(fontpath)?)
            .context("Invalid font file content")?
    } else {
        rusttype::Font::try_from_bytes(FONT).unwrap()
    };

    let mut dataset = Vec::<Series>::with_capacity(4);
    let mut partitions = Partitions::default();

    load_dataset(&opts, &mut dataset, &mut partitions)?;

    let n = dataset.iter().map(|x| x.samples.len()).max().unwrap_or(0);

    if !opts.no_fiter {
        interpolate(&mut dataset, &opts);
    }

    if !opts.no_hide {
        dataset.retain(|x| !x.hidden);
    }

    if let Some(ref dbgout) = opts.debug_filterted_csv {
        write_interpolated_csv(dbgout, &dataset, n, &partitions)?;
    }

    if opts.output_file.as_os_str() == "-" {
        return Ok(());
    }

    let mut hue_step = 360.0 / (dataset.len() as f32);
    if hue_step < opts.max_hue_angle {
        hue_step = opts.max_hue_angle
    };

    let styles: Vec<Style> = dataset
        .iter()
        .enumerate()
        .map(|(i, series)| {
            if let Some(r#override) = opts.colour_overrides.0.get(&series.name) {
                *r#override
            } else {
                let x = hue_step * (i as f32);
                Style {
                    hue: RgbHue::from_degrees(x),
                    saturation: opts.default_saturation,
                    min_lightness: opts.default_min_lightness,
                    max_lightness: opts.default_max_lightness,
                    gradientness: opts.default_gradientness,
                    hue_drift: opts.default_hue_drift,
                }
            }
        })
        .collect();

    if opts.partition.is_none() {
        prepare_and_write_pic(
            block_width,
            width,
            &opts,
            dataset,
            &styles,
            &font,
            block_height,
            &opts.output_file,
            None,
        )?;
    } else {
        for (part_index, part_name) in partitions.idx2name.iter() {
            let filtered_dataset = Vec::from_iter(dataset.iter().map(|s|{
                Series {
                    hidden: s.hidden,
                    name: s.name.to_owned(),
                    samples: Vec::from_iter(s.samples.iter().enumerate().flat_map(|(index, value)|{
                        if partitions.samples.get(index) == Some(&part_index) {
                            Some(*value)
                        } else {
                            None
                        }
                    })),
                }
            }));
            let mut outfile = opts.output_file.clone();
            if ! outfile.set_extension(format!("{:04}.png", part_index)) {
                anyhow::bail!("Failed to set filename suffic for partitioned pictures");
            }

            prepare_and_write_pic(
                block_width,
                width,
                &opts,
                filtered_dataset,
                &styles,
                &font,
                block_height,
                &outfile,
                Some(part_name),
            )?;
        }
    }
    Ok(())
}

fn prepare_and_write_pic(
    block_width: u32,
    width: u32,
    opts: &Opts,
    dataset: Vec<Series>,
    styles: &Vec<Style>,
    font: &rusttype::Font,
    block_height: u32,
    output_file: &Path,
    partition_name: Option<&str>,
) -> Result<(), anyhow::Error> {
    let n = dataset.iter().map(|x| x.samples.len()).max().unwrap_or(0);
    let mut rows = div_ceil(n as u32 * block_width, width - MARGIN_LEFT - MARGIN_RIGHT);
    if let Some(maxrowlen) = opts.max_cells_in_row {
        let rows2: u32 = div_ceil(n, maxrowlen).try_into()?;
        rows = rows.max(rows2);
    }
    let cursor_y = legend(
        &dataset,
        &styles,
        width,
        None,
        &font,
        opts.legend_font_scale,
        opts.lch,
        partition_name,
    );
    let mut intraband_gap = 0u32;
    if dataset.len() > 1 {
        intraband_gap = (dataset.len() as u32 + 3) / 4;
        intraband_gap *= block_height;
    }
    let band_height = (dataset.len() as u32) * block_height + intraband_gap;
    let image_height = cursor_y + band_height * rows;
    let mut img = ImageBuffer::<Rgb<u8>, _>::new(width, image_height);
    img.fill(128);
    let cursor_y = legend(
        &dataset,
        &styles,
        width,
        Some(&mut img),
        &font,
        opts.legend_font_scale,
        opts.lch,
        partition_name,
    );
    draw_cells(
        n,
        dataset,
        styles,
        block_height,
        block_width,
        cursor_y,
        width,
        image_height,
        &mut img,
        &opts,
        band_height,
    );
    img.save(output_file)?;
    Ok(())
}

fn draw_cells(
    n: usize,
    dataset: Vec<Series>,
    styles: &Vec<Style>,
    block_height: u32,
    block_width: u32,
    mut cursor_y: u32,
    image_width: u32,
    image_height: u32,
    img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    opts: &Opts,
    band_height: u32,
) {
    let mut cursor_x = MARGIN_LEFT;
    let mut cell_i_in_row = 0usize;
    for i in 0..n {
        for ((j, series), style) in dataset.iter().enumerate().zip(styles.iter()) {
            let pix_n = block_height * block_height;
            let mut pix_i = 0;
            for v in 0..block_width {
                for u in 0..block_height {
                    let c = get_colour(
                        series.samples.get(i).copied().unwrap_or(f64::NAN),
                        style,
                        pix_i,
                        pix_n,
                        opts.lch,
                    );
                    let x = cursor_x + v;
                    let y = cursor_y + (j as u32) * block_height + u;
                    if x < image_width && y < image_height {
                        img.put_pixel(x, y, c);
                    }
                    pix_i += 1;
                }
            }
        }

        cursor_x += block_width;
        cell_i_in_row += 1;

        let mut begin_new_row = false;
        if let Some(maxrowlen) = opts.max_cells_in_row {
            if cell_i_in_row >= maxrowlen {
                begin_new_row = true;
            }
        }
        if cursor_x + block_width + MARGIN_RIGHT > image_width {
            begin_new_row = true;
        }
        if begin_new_row {
            cell_i_in_row = 0;
            cursor_y += band_height;
            cursor_x = MARGIN_LEFT;
        }
    }
}

fn write_interpolated_csv(
    dbgout: &PathBuf,
    dataset: &Vec<Series>,
    n: usize,
    partitions: &Partitions,
) -> Result<(), anyhow::Error> {
    let mut csvout = csv::Writer::from_path(dbgout)?;
    if !partitions.idx2name.is_empty() {
        csvout.write_field("partition")?;
    }
    for serie in dataset {
        csvout.write_field(&serie.name)?;
    }
    csvout.write_record(None::<&[u8]>)?;
    for i in 0..n {
        if !partitions.idx2name.is_empty() {
            match partitions
                .samples
                .get(i)
                .map(|q| partitions.idx2name.get(*q))
            {
                Some(Some(s)) => csvout.write_field(s)?,
                _ => csvout.write_field("???")?,
            }
        }
        for serie in dataset {
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
    Ok(())
}

fn interpolate(dataset: &mut Vec<Series>, opts: &Opts) {
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
                let interpolation_n = (n - 1).min(opts.max_interpolation_points);
                #[derive(Debug)]
                struct InterpolationPoint {
                    start_sample: f64,
                    //stop_sample: f64,
                    inv_stop_sample_minus_start_sample: f64,
                    start_outrange: f64,
                    stop_outrange: f64,
                }
                let mut interpolation_points = Vec::with_capacity(interpolation_n);
                let epsilon: f64 = (0.00000001f64).max((sorted[n - 1] - sorted[0]) / 1000_000.0);
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

fn load_dataset(
    opts: &Opts,
    dataset: &mut Vec<Series>,
    partitions: &mut Partitions,
) -> Result<(), anyhow::Error> {
    let input: Box<dyn std::io::Read>;
    if let Some(input_file_path) = &opts.input_csv {
        input = Box::new(std::fs::File::open(input_file_path)?);
    } else {
        input = Box::new(std::io::stdin());
    }
    let mut csv = csv::Reader::from_reader(input);
    let mut partition_column_index = None;
    for (i, h) in csv.headers()?.into_iter().enumerate() {
        if let Some(ref partcol) = opts.partition {
            if h == partcol {
                if partition_column_index.is_some() {
                    anyhow::bail!("Multiple partition columns found");
                }
                partition_column_index = Some(i);
                continue;
            }
        }
        dataset.push(Series {
            samples: Vec::with_capacity(4096),
            name: h.to_owned(),
            hidden: false,
        })
    }
    if opts.partition.is_some() && partition_column_index.is_none() {
        anyhow::bail!("partition column not found");
    }
    Ok(for r in csv.records() {
        let r = r?;
        let mut dataseries_index = 0;
        for (i, s) in r.iter().enumerate() {
            if Some(i) == partition_column_index {
                let partition_index = if let Some(q) = partitions.name2idx.get(s) {
                    *q
                } else {
                    let newidx = partitions.idx2name.insert(s.to_owned());
                    partitions.name2idx.insert(s.to_owned(), newidx);
                    newidx
                };
                partitions.samples.push(partition_index);
                continue;
            }
            let x = s.parse().unwrap_or(f64::NAN);
            dataset[dataseries_index].samples.push(x);
            dataseries_index+=1;
        }
    })
}

/// Returns final cursor_y position
fn legend(
    dataset: &Vec<Series>,
    styles: &Vec<Style>,
    width: u32,
    mut img: Option<&mut ImageBuffer<Rgb<u8>, Vec<u8>>>,
    font: &rusttype::Font,
    font_scale: f32,
    lch: bool,
    partition_name: Option<&str>,
) -> u32 {
    if font_scale < 0.1 {
        return MARGIN_TOP;
    }
    let mut cursor_x = MARGIN_LEFT;
    let mut cursor_y = MARGIN_TOP;
    let mut empty = true;
    let mut current_legend_row_height = 20u32;
    if let Some(partname) = partition_name {
        let (_tw, text_height) = text_size(Scale::uniform(font_scale), &font, partname);

        if let Some(img_) = img {
            draw_text_mut(
                img_,
                [255,255,255].into(),
                cursor_x as i32,
                cursor_y as i32,
                Scale::uniform(font_scale),
                &font,
                partname,
            );
            img = Some(img_);
        }

        cursor_y += 4;
        cursor_y += text_height as u32;
    }
    for (series, style) in dataset.iter().zip(styles.iter()) {
        let fixed_width = MARGIN_RIGHT + LEDEND_SAMPLE_WIDTH + 2 + 12;

        //let text_width = series.name.graphemes(true).count() as u32 * 7;

        let (text_width, text_height) = text_size(Scale::uniform(font_scale), &font, &series.name);
        let (text_width, text_height) = (text_width as u32, text_height as u32);

        if cursor_x + fixed_width + text_width > width {
            cursor_y += current_legend_row_height;
            cursor_x = MARGIN_LEFT;
            current_legend_row_height = 20;
        }

        if current_legend_row_height < text_height + 3 {
            current_legend_row_height = text_height + 3;
        }

        if let Some(img_) = img {
            for i in 0..LEDEND_SAMPLE_WIDTH {
                let x = i as f64 / (LEDEND_SAMPLE_WIDTH - 1) as f64;
                let c = get_colour(x, style, 0, 1, lch);
                for j in 0..text_height {
                    img_.put_pixel(cursor_x + i, cursor_y + j, c);
                }
            }

            let mut style = style.clone();
            style.gradientness = 0.0;
            style.min_lightness = 0.0;
            style.max_lightness = 1.0;
            let c = get_colour(0.75, &style, 0, 1, lch);

            draw_text_mut(
                img_,
                c,
                (cursor_x + LEDEND_SAMPLE_WIDTH + 2) as i32,
                cursor_y as i32,
                Scale::uniform(font_scale),
                &font,
                &series.name,
            );
            img = Some(img_);
        }

        cursor_x += LEDEND_SAMPLE_WIDTH + 2 + text_width + 12;
        empty = false;
    }
    if !empty {
        cursor_y += current_legend_row_height;
    }
    cursor_y
}
