use cmake::Config;
use std::path::{Path, PathBuf};

fn main() {
    let whispercpp_path: &Path = Path::new("./third-party/whisper.cpp");
    let dst: PathBuf = Config::new(whispercpp_path)
        .very_verbose(true)
        .build();

    println!("cargo:rustc-link-search=native={}", dst.display());
}
