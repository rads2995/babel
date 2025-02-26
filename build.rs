use cmake::Config;
use std::fs;
use std::path::PathBuf;

fn main() {
    let whispercpp_path: PathBuf = fs::canonicalize("./third-party/whisper.cpp").unwrap();
    let llamacpp_path: PathBuf = fs::canonicalize("./third-party/llama.cpp").unwrap();

    let whispercpp_cmake: PathBuf = Config::new(&whispercpp_path)
        .define("GGML_VULKAN", "1")
        .define("BUILD_SHARED_LIBS", "ON")
        .generator("Ninja")
        .build_target("all")
        .out_dir(&whispercpp_path)
        .build();

    let llamacpp_cmake: PathBuf = Config::new(&llamacpp_path)
        .define("GGML_HIP", "ON")
        .define("AMDGPU_TARGETS", "gfx1010")
        .define("BUILD_SHARED_LIBS", "ON")
        .generator("Ninja")
        .build_target("all")
        .out_dir(&llamacpp_path)
        .build();

    println!(
        "cargo:rustc-link-search=native={}",
        whispercpp_cmake.join("build/src").display()
    );
    println!("cargo:rustc-link-lib=dylib=whisper");

    println!(
        "cargo:rustc-link-search=native={}",
        llamacpp_cmake.join("build/bin").display()
    );
    println!("cargo:rustc-link-lib=dylib=llama");
}
