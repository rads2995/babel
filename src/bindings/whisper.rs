#![allow(non_camel_case_types, unused)]

use super::ggml::{ggml_abort_callback, ggml_log_callback};

#[repr(C)]
pub(crate) struct whisper_model_loader {
    pub(crate) context: *mut core::ffi::c_void,
    pub(crate) read: core::option::Option<
        unsafe extern "C" fn(
            ctx: *mut core::ffi::c_void,
            output: *mut core::ffi::c_void,
            read_size: usize,
        ) -> usize,
    >,
    pub(crate) eof: core::option::Option<unsafe extern "C" fn(ctx: *mut core::ffi::c_void) -> bool>,
    pub(crate) close: core::option::Option<unsafe extern "C" fn(ctx: *mut core::ffi::c_void)>,
}

#[repr(C)]
pub struct whisper_context {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}
#[repr(C)]
pub struct whisper_state {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}
pub type whisper_pos = i32;
pub type whisper_token = i32;
pub type whisper_seq_id = i32;

pub type whisper_alignment_heads_preset = core::ffi::c_uint;

#[repr(C)]
pub struct whisper_ahead {
    pub n_text_layer: core::ffi::c_int,
    pub n_head: core::ffi::c_int,
}

#[repr(C)]
pub struct whisper_aheads {
    pub n_heads: usize,
    pub heads: *const whisper_ahead,
}

#[repr(C)]
pub struct whisper_context_params {
    pub use_gpu: bool,
    pub flash_attn: bool,
    pub gpu_device: core::ffi::c_int,
    pub dtw_token_timestamps: bool,
    pub dtw_aheads_preset: whisper_alignment_heads_preset,
    pub dtw_n_top: core::ffi::c_int,
    pub dtw_aheads: whisper_aheads,
    pub dtw_mem_size: usize,
}

#[repr(C)]
pub struct whisper_token_data {
    pub id: whisper_token,
    pub tid: whisper_token,
    pub p: f32,
    pub plog: f32,
    pub pt: f32,
    pub ptsum: f32,
    pub t0: i64,
    pub t1: i64,
    pub t_dtw: i64,
    pub vlen: f32,
}

pub type whisper_gretype = core::ffi::c_uint;

#[repr(C)]
pub struct whisper_grammar_element {
    pub type_: whisper_gretype,
    pub value: u32,
}

unsafe extern "C" {
    pub fn whisper_init_from_file_with_params(
        path_model: *const core::ffi::c_char,
        params: whisper_context_params,
    ) -> *mut whisper_context;
}
unsafe extern "C" {
    pub fn whisper_init_from_buffer_with_params(
        buffer: *mut core::ffi::c_void,
        buffer_size: usize,
        params: whisper_context_params,
    ) -> *mut whisper_context;
}
unsafe extern "C" {
    pub fn whisper_init_with_params(
        loader: *mut whisper_model_loader,
        params: whisper_context_params,
    ) -> *mut whisper_context;
}
unsafe extern "C" {
    pub fn whisper_init_from_file_with_params_no_state(
        path_model: *const core::ffi::c_char,
        params: whisper_context_params,
    ) -> *mut whisper_context;
}
unsafe extern "C" {
    pub fn whisper_init_from_buffer_with_params_no_state(
        buffer: *mut core::ffi::c_void,
        buffer_size: usize,
        params: whisper_context_params,
    ) -> *mut whisper_context;
}
unsafe extern "C" {
    pub fn whisper_init_with_params_no_state(
        loader: *mut whisper_model_loader,
        params: whisper_context_params,
    ) -> *mut whisper_context;
}
unsafe extern "C" {
    pub fn whisper_init_from_file(path_model: *const core::ffi::c_char) -> *mut whisper_context;
}
unsafe extern "C" {
    pub fn whisper_init_from_buffer(
        buffer: *mut core::ffi::c_void,
        buffer_size: usize,
    ) -> *mut whisper_context;
}
unsafe extern "C" {
    pub fn whisper_init(loader: *mut whisper_model_loader) -> *mut whisper_context;
}
unsafe extern "C" {
    pub fn whisper_init_from_file_no_state(
        path_model: *const core::ffi::c_char,
    ) -> *mut whisper_context;
}
unsafe extern "C" {
    pub fn whisper_init_from_buffer_no_state(
        buffer: *mut core::ffi::c_void,
        buffer_size: usize,
    ) -> *mut whisper_context;
}
unsafe extern "C" {
    pub fn whisper_init_no_state(loader: *mut whisper_model_loader) -> *mut whisper_context;
}
unsafe extern "C" {
    pub fn whisper_init_state(ctx: *mut whisper_context) -> *mut whisper_state;
}
unsafe extern "C" {
    pub fn whisper_ctx_init_openvino_encoder_with_state(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        model_path: *const core::ffi::c_char,
        device: *const core::ffi::c_char,
        cache_dir: *const core::ffi::c_char,
    ) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_ctx_init_openvino_encoder(
        ctx: *mut whisper_context,
        model_path: *const core::ffi::c_char,
        device: *const core::ffi::c_char,
        cache_dir: *const core::ffi::c_char,
    ) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_free(ctx: *mut whisper_context);
}
unsafe extern "C" {
    pub fn whisper_free_state(state: *mut whisper_state);
}
unsafe extern "C" {
    pub fn whisper_free_params(params: *mut whisper_full_params);
}
unsafe extern "C" {
    pub fn whisper_free_context_params(params: *mut whisper_context_params);
}
unsafe extern "C" {
    pub fn whisper_pcm_to_mel(
        ctx: *mut whisper_context,
        samples: *const f32,
        n_samples: core::ffi::c_int,
        n_threads: core::ffi::c_int,
    ) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_pcm_to_mel_with_state(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        samples: *const f32,
        n_samples: core::ffi::c_int,
        n_threads: core::ffi::c_int,
    ) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_set_mel(
        ctx: *mut whisper_context,
        data: *const f32,
        n_len: core::ffi::c_int,
        n_mel: core::ffi::c_int,
    ) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_set_mel_with_state(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        data: *const f32,
        n_len: core::ffi::c_int,
        n_mel: core::ffi::c_int,
    ) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_encode(
        ctx: *mut whisper_context,
        offset: core::ffi::c_int,
        n_threads: core::ffi::c_int,
    ) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_encode_with_state(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        offset: core::ffi::c_int,
        n_threads: core::ffi::c_int,
    ) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_decode(
        ctx: *mut whisper_context,
        tokens: *const whisper_token,
        n_tokens: core::ffi::c_int,
        n_past: core::ffi::c_int,
        n_threads: core::ffi::c_int,
    ) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_decode_with_state(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        tokens: *const whisper_token,
        n_tokens: core::ffi::c_int,
        n_past: core::ffi::c_int,
        n_threads: core::ffi::c_int,
    ) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_tokenize(
        ctx: *mut whisper_context,
        text: *const core::ffi::c_char,
        tokens: *mut whisper_token,
        n_max_tokens: core::ffi::c_int,
    ) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_token_count(
        ctx: *mut whisper_context,
        text: *const core::ffi::c_char,
    ) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_lang_max_id() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_lang_id(lang: *const core::ffi::c_char) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_lang_str(id: core::ffi::c_int) -> *const core::ffi::c_char;
}
unsafe extern "C" {
    pub fn whisper_lang_str_full(id: core::ffi::c_int) -> *const core::ffi::c_char;
}
unsafe extern "C" {
    pub fn whisper_lang_auto_detect(
        ctx: *mut whisper_context,
        offset_ms: core::ffi::c_int,
        n_threads: core::ffi::c_int,
        lang_probs: *mut f32,
    ) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_lang_auto_detect_with_state(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        offset_ms: core::ffi::c_int,
        n_threads: core::ffi::c_int,
        lang_probs: *mut f32,
    ) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_n_len(ctx: *mut whisper_context) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_n_len_from_state(state: *mut whisper_state) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_n_vocab(ctx: *mut whisper_context) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_n_text_ctx(ctx: *mut whisper_context) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_n_audio_ctx(ctx: *mut whisper_context) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_is_multilingual(ctx: *mut whisper_context) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_n_vocab(ctx: *mut whisper_context) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_n_audio_ctx(ctx: *mut whisper_context) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_n_audio_state(ctx: *mut whisper_context) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_n_audio_head(ctx: *mut whisper_context) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_n_audio_layer(ctx: *mut whisper_context) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_n_text_ctx(ctx: *mut whisper_context) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_n_text_state(ctx: *mut whisper_context) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_n_text_head(ctx: *mut whisper_context) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_n_text_layer(ctx: *mut whisper_context) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_n_mels(ctx: *mut whisper_context) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_ftype(ctx: *mut whisper_context) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_type(ctx: *mut whisper_context) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_get_logits(ctx: *mut whisper_context) -> *mut f32;
}
unsafe extern "C" {
    pub fn whisper_get_logits_from_state(state: *mut whisper_state) -> *mut f32;
}
unsafe extern "C" {
    pub fn whisper_token_to_str(
        ctx: *mut whisper_context,
        token: whisper_token,
    ) -> *const core::ffi::c_char;
}
unsafe extern "C" {
    pub fn whisper_model_type_readable(ctx: *mut whisper_context) -> *const core::ffi::c_char;
}
unsafe extern "C" {
    pub fn whisper_token_eot(ctx: *mut whisper_context) -> whisper_token;
}
unsafe extern "C" {
    pub fn whisper_token_sot(ctx: *mut whisper_context) -> whisper_token;
}
unsafe extern "C" {
    pub fn whisper_token_solm(ctx: *mut whisper_context) -> whisper_token;
}
unsafe extern "C" {
    pub fn whisper_token_prev(ctx: *mut whisper_context) -> whisper_token;
}
unsafe extern "C" {
    pub fn whisper_token_nosp(ctx: *mut whisper_context) -> whisper_token;
}
unsafe extern "C" {
    pub fn whisper_token_not(ctx: *mut whisper_context) -> whisper_token;
}
unsafe extern "C" {
    pub fn whisper_token_beg(ctx: *mut whisper_context) -> whisper_token;
}
unsafe extern "C" {
    pub fn whisper_token_lang(
        ctx: *mut whisper_context,
        lang_id: core::ffi::c_int,
    ) -> whisper_token;
}
unsafe extern "C" {
    pub fn whisper_token_translate(ctx: *mut whisper_context) -> whisper_token;
}
unsafe extern "C" {
    pub fn whisper_token_transcribe(ctx: *mut whisper_context) -> whisper_token;
}
#[repr(C)]
pub struct whisper_timings {
    pub sample_ms: f32,
    pub encode_ms: f32,
    pub decode_ms: f32,
    pub batchd_ms: f32,
    pub prompt_ms: f32,
}

unsafe extern "C" {
    pub fn whisper_get_timings(ctx: *mut whisper_context) -> *mut whisper_timings;
}
unsafe extern "C" {
    pub fn whisper_print_timings(ctx: *mut whisper_context);
}
unsafe extern "C" {
    pub fn whisper_reset_timings(ctx: *mut whisper_context);
}
unsafe extern "C" {
    pub fn whisper_print_system_info() -> *const core::ffi::c_char;
}

pub type whisper_sampling_strategy = core::ffi::c_uint;
pub type whisper_new_segment_callback = core::option::Option<
    unsafe extern "C" fn(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        n_new: core::ffi::c_int,
        user_data: *mut core::ffi::c_void,
    ),
>;

pub type whisper_progress_callback = core::option::Option<
    unsafe extern "C" fn(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        progress: core::ffi::c_int,
        user_data: *mut core::ffi::c_void,
    ),
>;
pub type whisper_encoder_begin_callback = core::option::Option<
    unsafe extern "C" fn(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        user_data: *mut core::ffi::c_void,
    ) -> bool,
>;
pub type whisper_logits_filter_callback = core::option::Option<
    unsafe extern "C" fn(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        tokens: *const whisper_token_data,
        n_tokens: core::ffi::c_int,
        logits: *mut f32,
        user_data: *mut core::ffi::c_void,
    ),
>;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct whisper_full_params {
    pub strategy: whisper_sampling_strategy,
    pub n_threads: core::ffi::c_int,
    pub n_max_text_ctx: core::ffi::c_int,
    pub offset_ms: core::ffi::c_int,
    pub duration_ms: core::ffi::c_int,
    pub translate: bool,
    pub no_context: bool,
    pub no_timestamps: bool,
    pub single_segment: bool,
    pub print_special: bool,
    pub print_progress: bool,
    pub print_realtime: bool,
    pub print_timestamps: bool,
    pub token_timestamps: bool,
    pub thold_pt: f32,
    pub thold_ptsum: f32,
    pub max_len: core::ffi::c_int,
    pub split_on_word: bool,
    pub max_tokens: core::ffi::c_int,
    pub debug_mode: bool,
    pub audio_ctx: core::ffi::c_int,
    pub tdrz_enable: bool,
    pub suppress_regex: *const core::ffi::c_char,
    pub initial_prompt: *const core::ffi::c_char,
    pub prompt_tokens: *const whisper_token,
    pub prompt_n_tokens: core::ffi::c_int,
    pub language: *const core::ffi::c_char,
    pub detect_language: bool,
    pub suppress_blank: bool,
    pub suppress_nst: bool,
    pub temperature: f32,
    pub max_initial_ts: f32,
    pub length_penalty: f32,
    pub temperature_inc: f32,
    pub entropy_thold: f32,
    pub logprob_thold: f32,
    pub no_speech_thold: f32,
    pub greedy: whisper_full_params__bindgen_ty_1,
    pub beam_search: whisper_full_params__bindgen_ty_2,
    pub new_segment_callback: whisper_new_segment_callback,
    pub new_segment_callback_user_data: *mut core::ffi::c_void,
    pub progress_callback: whisper_progress_callback,
    pub progress_callback_user_data: *mut core::ffi::c_void,
    pub encoder_begin_callback: whisper_encoder_begin_callback,
    pub encoder_begin_callback_user_data: *mut core::ffi::c_void,
    pub abort_callback: ggml_abort_callback,
    pub abort_callback_user_data: *mut core::ffi::c_void,
    pub logits_filter_callback: whisper_logits_filter_callback,
    pub logits_filter_callback_user_data: *mut core::ffi::c_void,
    pub grammar_rules: *mut *const whisper_grammar_element,
    pub n_grammar_rules: usize,
    pub i_start_rule: usize,
    pub grammar_penalty: f32,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct whisper_full_params__bindgen_ty_1 {
    pub best_of: core::ffi::c_int,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct whisper_full_params__bindgen_ty_2 {
    pub beam_size: core::ffi::c_int,
    pub patience: f32,
}

unsafe extern "C" {
    pub fn whisper_context_default_params_by_ref() -> *mut whisper_context_params;
}
unsafe extern "C" {
    pub fn whisper_context_default_params() -> whisper_context_params;
}
unsafe extern "C" {
    pub fn whisper_full_default_params_by_ref(
        strategy: whisper_sampling_strategy,
    ) -> *mut whisper_full_params;
}
unsafe extern "C" {
    pub fn whisper_full_default_params(strategy: whisper_sampling_strategy) -> whisper_full_params;
}
unsafe extern "C" {
    pub fn whisper_full(
        ctx: *mut whisper_context,
        params: whisper_full_params,
        samples: *const f32,
        n_samples: core::ffi::c_int,
    ) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_full_with_state(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        params: whisper_full_params,
        samples: *const f32,
        n_samples: core::ffi::c_int,
    ) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_full_parallel(
        ctx: *mut whisper_context,
        params: whisper_full_params,
        samples: *const f32,
        n_samples: core::ffi::c_int,
        n_processors: core::ffi::c_int,
    ) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_full_n_segments(ctx: *mut whisper_context) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_full_n_segments_from_state(state: *mut whisper_state) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_full_lang_id(ctx: *mut whisper_context) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_full_lang_id_from_state(state: *mut whisper_state) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_full_get_segment_t0(
        ctx: *mut whisper_context,
        i_segment: core::ffi::c_int,
    ) -> i64;
}
unsafe extern "C" {
    pub fn whisper_full_get_segment_t0_from_state(
        state: *mut whisper_state,
        i_segment: core::ffi::c_int,
    ) -> i64;
}
unsafe extern "C" {
    pub fn whisper_full_get_segment_t1(
        ctx: *mut whisper_context,
        i_segment: core::ffi::c_int,
    ) -> i64;
}
unsafe extern "C" {
    pub fn whisper_full_get_segment_t1_from_state(
        state: *mut whisper_state,
        i_segment: core::ffi::c_int,
    ) -> i64;
}
unsafe extern "C" {
    pub fn whisper_full_get_segment_speaker_turn_next(
        ctx: *mut whisper_context,
        i_segment: core::ffi::c_int,
    ) -> bool;
}
unsafe extern "C" {
    pub fn whisper_full_get_segment_speaker_turn_next_from_state(
        state: *mut whisper_state,
        i_segment: core::ffi::c_int,
    ) -> bool;
}
unsafe extern "C" {
    pub fn whisper_full_get_segment_text(
        ctx: *mut whisper_context,
        i_segment: core::ffi::c_int,
    ) -> *const core::ffi::c_char;
}
unsafe extern "C" {
    pub fn whisper_full_get_segment_text_from_state(
        state: *mut whisper_state,
        i_segment: core::ffi::c_int,
    ) -> *const core::ffi::c_char;
}
unsafe extern "C" {
    pub fn whisper_full_n_tokens(
        ctx: *mut whisper_context,
        i_segment: core::ffi::c_int,
    ) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_full_n_tokens_from_state(
        state: *mut whisper_state,
        i_segment: core::ffi::c_int,
    ) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_full_get_token_text(
        ctx: *mut whisper_context,
        i_segment: core::ffi::c_int,
        i_token: core::ffi::c_int,
    ) -> *const core::ffi::c_char;
}
unsafe extern "C" {
    pub fn whisper_full_get_token_text_from_state(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        i_segment: core::ffi::c_int,
        i_token: core::ffi::c_int,
    ) -> *const core::ffi::c_char;
}
unsafe extern "C" {
    pub fn whisper_full_get_token_id(
        ctx: *mut whisper_context,
        i_segment: core::ffi::c_int,
        i_token: core::ffi::c_int,
    ) -> whisper_token;
}
unsafe extern "C" {
    pub fn whisper_full_get_token_id_from_state(
        state: *mut whisper_state,
        i_segment: core::ffi::c_int,
        i_token: core::ffi::c_int,
    ) -> whisper_token;
}
unsafe extern "C" {
    pub fn whisper_full_get_token_data(
        ctx: *mut whisper_context,
        i_segment: core::ffi::c_int,
        i_token: core::ffi::c_int,
    ) -> whisper_token_data;
}
unsafe extern "C" {
    pub fn whisper_full_get_token_data_from_state(
        state: *mut whisper_state,
        i_segment: core::ffi::c_int,
        i_token: core::ffi::c_int,
    ) -> whisper_token_data;
}
unsafe extern "C" {
    pub fn whisper_full_get_token_p(
        ctx: *mut whisper_context,
        i_segment: core::ffi::c_int,
        i_token: core::ffi::c_int,
    ) -> f32;
}
unsafe extern "C" {
    pub fn whisper_full_get_token_p_from_state(
        state: *mut whisper_state,
        i_segment: core::ffi::c_int,
        i_token: core::ffi::c_int,
    ) -> f32;
}
unsafe extern "C" {
    pub fn whisper_bench_memcpy(n_threads: core::ffi::c_int) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_bench_memcpy_str(n_threads: core::ffi::c_int) -> *const core::ffi::c_char;
}
unsafe extern "C" {
    pub fn whisper_bench_ggml_mul_mat(n_threads: core::ffi::c_int) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn whisper_bench_ggml_mul_mat_str(n_threads: core::ffi::c_int) -> *const core::ffi::c_char;
}
unsafe extern "C" {
    pub fn whisper_log_set(log_callback: ggml_log_callback, user_data: *mut core::ffi::c_void);
}
unsafe extern "C" {
    pub fn whisper_full_get_segment_no_speech_prob(
        ctx: *mut whisper_context,
        i_segment: core::ffi::c_int,
    ) -> f32;
}
unsafe extern "C" {
    pub fn whisper_full_get_segment_no_speech_prob_from_state(
        state: *mut whisper_state,
        i_segment: core::ffi::c_int,
    ) -> f32;
}
