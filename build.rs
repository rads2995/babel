use cmake::Config;
use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let build_libs: bool = env::var("BUILD_LIBS")
        .map(|v: String| v == "1")
        .unwrap_or(false);

    if build_libs {
        let whispercpp_path: &Path = Path::new("./third-party/whisper.cpp");
        let llamacpp_path: &Path = Path::new("./third-party/llama.cpp");

        let whispercpp_cmake: PathBuf = Config::new(whispercpp_path)
            .define("GGML_VULKAN", "1")
            .define("BUILD_SHARED_LIBS", "ON")
            .build_arg("-j6")
            .build();

        let llamacpp_cmake: PathBuf = Config::new(llamacpp_path)
            .define("GGML_VULKAN", "1")
            .define("BUILD_SHARED_LIBS", "ON")
            .build_arg("-j6")
            .build();

        println!(
            "cargo:rustc-link-search=native={}",
            whispercpp_cmake.join("lib").display()
        );
        println!("cargo:rustc-link-lib=dylib=whisper");

        println!(
            "cargo:rustc-link-search=native={}",
            llamacpp_cmake.join("lib").display()
        );
        println!("cargo:rustc-link-lib=dylib=llama");
    }
}
