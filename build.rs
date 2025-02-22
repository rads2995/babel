use cmake::Config;
use std::path::{Path, PathBuf};

fn main() {
    let whispercpp_path: &Path = Path::new("./third-party/whisper.cpp");
    let llamacpp_path: &Path = Path::new("./third-party/llama.cpp");

    let whispercpp_cmake: PathBuf = Config::new(whispercpp_path)
        .define("GGML_VULKAN", "1")
        .define("BUILD_SHARED_LIBS", "OFF")
        .build_arg("-j6")
        .build();

    let llamacpp_cmake: PathBuf = Config::new(llamacpp_path)
        .define("GGML_VULKAN", "1")
        .define("BUILD_SHARED_LIBS", "OFF")
        .build_arg("-j6")
        .build();

    println!(
        "cargo:rustc-link-search=native={}",
        whispercpp_cmake.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=whisper");

    println!(
        "cargo:rustc-link-search=native={}",
        llamacpp_cmake.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=llama");
}
