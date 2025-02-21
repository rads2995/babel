# babel

## Introduction

## How to Build
```bash
cargo build --release
```

## How to Test

Test to be added.

## How to Run
```bash
cargo run
```

## Results

```
whispewhisper_init_from_file_with_params_no_state: loading model from 'ggml-large-v3-turbo.bin'
whisper_init_with_params_no_state: use gpu    = 1
whisper_init_with_params_no_state: flash attn = 0
whisper_init_with_params_no_state: gpu_device = 0
whisper_init_with_params_no_state: dtw        = 0
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 5700 (RADV NAVI10) (radv) | uma: 0 | fp16: 1 | warp size: 64 | matrix cores: none
whisper_init_with_params_no_state: devices    = 2
whisper_init_with_params_no_state: backends   = 2
whisper_model_load: loading model
whisper_model_load: n_vocab       = 51866
whisper_model_load: n_audio_ctx   = 1500
whisper_model_load: n_audio_state = 1280
whisper_model_load: n_audio_head  = 20
whisper_model_load: n_audio_layer = 32
whisper_model_load: n_text_ctx    = 448
whisper_model_load: n_text_state  = 1280
whisper_model_load: n_text_head   = 20
whisper_model_load: n_text_layer  = 4
whisper_model_load: n_mels        = 128
whisper_model_load: ftype         = 1
whisper_model_load: qntvr         = 0
whisper_model_load: type          = 5 (large v3)
whisper_model_load: adding 1609 extra tokens
whisper_model_load: n_langs       = 100
whisper_model_load:  Vulkan0 total size =  1623.92 MB
whisper_model_load: model size    = 1623.92 MB
whisper_backend_init_gpu: using Vulkan0 backend
whisper_init_state: kv self size  =   10.49 MB
whisper_init_state: kv cross size =   31.46 MB
whisper_init_state: kv pad  size  =    7.86 MB
whisper_init_state: compute buffer (conv)   =   37.67 MB
whisper_init_state: compute buffer (encode) =  212.29 MB
whisper_init_state: compute buffer (cross)  =    9.25 MB
whisper_init_state: compute buffer (decode) =  100.03 MB

system_info: n_threads = 4 / 12 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | COREML = 0 | OPENVINO = 0 |

main: processing 'third-party/whisper.cpp/samples/jfk.wav' (176000 samples, 11.0 sec), 4 threads, 1 processors, 5 beams + best of 5, lang = en, task = transcribe, timestamps = 1 ...


[00:00:00.300 --> 00:00:09.360]   And so, my fellow Americans, ask not what your country can do for you, ask what you can
[00:00:09.360 --> 00:00:11.000]   do for your country.


whisper_print_timings:     load time =   670.71 ms
whisper_print_timings:     fallbacks =   0 p /   0 h
whisper_print_timings:      mel time =    26.09 ms
whisper_print_timings:   sample time =    70.92 ms /   146 runs (    0.49 ms per run)
whisper_print_timings:   encode time =   738.31 ms /     1 runs (  738.31 ms per run)
whisper_print_timings:   decode time =     0.00 ms /     1 runs (    0.00 ms per run)
whisper_print_timings:   batchd time =   199.58 ms /   144 runs (    1.39 ms per run)
whisper_print_timings:   prompt time =     0.00 ms /     1 runs (    0.00 ms per run)
whisper_print_timings:    total time =  1783.41 ms
```
