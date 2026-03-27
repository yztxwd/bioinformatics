use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

#[derive(Debug, Default, Clone, Copy)]
struct MethRecord {
    methylated: u64,
    depth: u64,
}

fn parse_value_field(field: &str) -> Result<(u64, u64), String> {
    let parts: Vec<&str>  = field.trim().split(',').collect();
    if parts.len() < 2 {
        return Err(format!("Invalid value field: {}", field));
    }

    let methylated = parts[0]
        .parse::<u64>()
        .map_err(|e| format!("Failed to parse methylated count '{}': {}", parts[0], e))?;

    let depth = parts[1]
        .parse::<u64>()
        .map_err(|e| format!("Failed to parse depth '{}': {}", parts[1], e))?;

    Ok((methylated, depth))
}

fn is_header_or_empty(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return true;
    }

    // Skip header like: chr loc sample_name
    // or anything whose 2nd column is not an integer coordinates
    let cols: Vec<&str> = trimmed.split_whitespace().collect();
    if cols.len() < 3 {
        return true;
    }

    cols[1].parse::<u64>().is_err()
}

fn process_file<P: AsRef<Path>> (
    path: P,
    merged: &mut BTreeMap<(String, u64), MethRecord>,
) -> Result<(), String> {
    let file = File::open(&path)
        .map_err(|e| format!("Failed to open file '{}': {}", path.as_ref().display(), e))?;
    let reader = BufReader::new(file);

    for (line_no, line_res) in reader.lines().enumerate() {
        let line = line_res.map_err(|e| {
            format !(
                "Failed reading {} at line {}: {}",
                path.as_ref().display(),
                line_no + 1,
                e
            )
        })?;

        if is_header_or_empty(&line) {
            continue;
        }

        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 3 {
            return Err(format!(
                    "Malformed line in {} at line {}: {}",
                    path.as_ref().display(),
                    line_no + 1,
                    line
                ));
        }

        let chr = cols[0].to_string();
        let loc = cols[1].parse::<u64>().map_err(|e| {
            format!(
                "Failed parsing coordinate in {} at line {}: {} ({})",
                path.as_ref().display(),
                line_no + 1,
                cols[1],
                e
            )
        })?;

        let (methylated, depth) = parse_value_field(cols[2]).map_err(|e| {
            format!(
                "{} in {} at line {}",
                e,
                path.as_ref().display(),
                line_no + 1
            )
        })?;

        let entry = merged.entry((chr, loc)).or_default();
        entry.methylated += methylated;
        entry.depth += depth;
    }

    Ok(())
}

fn write_output<P: AsRef<Path>>(
    path: P,
    merged: &BTreeMap<(String, u64), MethRecord>,
) -> Result<(), String> {
    let file = File::create(&path)
        .map_err(|e| format!("Failed to create output file '{}': {}", path.as_ref().display(), e))?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "chr\tloc\tmerged_mCG")
        .map_err(|e| format!("Failed writing header: {}", e))?;

    for ((chr, loc), rec) in merged {
        let percent = if rec.depth > 0 {
            (rec.methylated as f64 / rec.depth as f64) * 100.0
        } else {
            0.0
        };

        writeln!(
            writer,
            "{}\t{}\t{},{},{:.2}",
            chr, loc, rec.methylated, rec.depth, percent
        )
        .map_err(|e| format!("Failed writing record for {}:{}: {}", chr, loc, e))?;
    }

    writer
        .flush()
        .map_err(|e| format!("Failed to flush output: {}", e))?;

    Ok(())
}

fn print_usage(program: &str) {
    eprintln!("
Merge multiple methylation tab files by summing methylated and depth counts for each coordinate.\n
Usage:
    {} <output_file> <input_file1> [<input_file2> ...]", program);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 || args[1] == "-h" || args[1] == "--help" {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    let output_path = &args[1];
    let input_files = &args[2..];

    let mut merged: BTreeMap<(String, u64), MethRecord> = BTreeMap::new();

    for input in input_files {
        eprintln!("Processing file: {}", input);
        if let Err(e) = process_file(input, &mut merged) {
            return Err(io::Error::new(io::ErrorKind::Other, e).into());
        }
    }

    if let Err(e) = write_output(output_path, &merged) {
        return Err(io::Error::new(io::ErrorKind::Other, e).into());
    }

    eprintln!(
        "Done. Wrote {} merged coordinates to {}",
        merged.len(),
        output_path
    );

    Ok(())
}
