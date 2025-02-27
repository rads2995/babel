#![allow(non_camel_case_types, unused, clippy::upper_case_acronyms)]

use crate::bindings::ggml::{ggml_abort_callback, ggml_log_callback, ggml_threadpool_t, ggml_type};

use crate::bindings::ggml_cpu::ggml_numa_strategy;

use crate::bindings::ggml_backend::{ggml_backend_dev_t, ggml_backend_sched_eval_callback};

#[repr(C)]
pub(crate) struct llama_vocab {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub(crate) struct llama_model {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub(crate) struct llama_context {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub(crate) struct llama_sampler {
    pub(crate) iface: *const llama_sampler_i,
    pub(crate) ctx: llama_sampler_context_t,
}

#[repr(C)]
pub(crate) struct llama_token_data {
    pub(crate) id: llama_token,
    pub(crate) logit: f32,
    pub(crate) p: f32,
}

#[repr(C)]
pub(crate) struct llama_token_data_array {
    pub(crate) data: *mut llama_token_data,
    pub(crate) size: usize,
    pub(crate) selected: i64,
    pub(crate) sorted: bool,
}

#[repr(C)]
pub(crate) struct llama_batch {
    pub(crate) n_tokens: i32,
    pub(crate) token: *mut llama_token,
    pub(crate) embd: *mut f32,
    pub(crate) pos: *mut llama_pos,
    pub(crate) n_seq_id: *mut i32,
    pub(crate) seq_id: *mut *mut llama_seq_id,
    pub(crate) logits: *mut i8,
}

#[repr(C)]
pub(crate) struct llama_model_kv_override {
    pub(crate) tag: llama_model_kv_override_type,
    pub(crate) key: [core::ffi::c_char; 128usize],
    pub(crate) __bindgen_anon_1: llama_model_kv_override__bindgen_ty_1,
}

#[repr(C)]
pub(crate) union llama_model_kv_override__bindgen_ty_1 {
    pub(crate) val_i64: i64,
    pub(crate) val_f64: f64,
    pub(crate) val_bool: bool,
    pub(crate) val_str: [core::ffi::c_char; 128usize],
}

#[repr(C)]
pub(crate) struct llama_model_params {
    pub(crate) devices: *mut ggml_backend_dev_t,
    pub(crate) n_gpu_layers: i32,
    pub(crate) split_mode: llama_split_mode,
    pub(crate) main_gpu: i32,
    pub(crate) tensor_split: *const f32,
    pub(crate) progress_callback: llama_progress_callback,
    pub(crate) progress_callback_user_data: *mut core::ffi::c_void,
    pub(crate) kv_overrides: *const llama_model_kv_override,
    pub(crate) vocab_only: bool,
    pub(crate) use_mmap: bool,
    pub(crate) use_mlock: bool,
    pub(crate) check_tensors: bool,
}

#[repr(C)]
pub(crate) struct llama_context_params {
    pub(crate) n_ctx: u32,
    pub(crate) n_batch: u32,
    pub(crate) n_ubatch: u32,
    pub(crate) n_seq_max: u32,
    pub(crate) n_threads: i32,
    pub(crate) n_threads_batch: i32,
    pub(crate) rope_scaling_type: llama_rope_scaling_type,
    pub(crate) pooling_type: llama_pooling_type,
    pub(crate) attention_type: llama_attention_type,
    pub(crate) rope_freq_base: f32,
    pub(crate) rope_freq_scale: f32,
    pub(crate) yarn_ext_factor: f32,
    pub(crate) yarn_attn_factor: f32,
    pub(crate) yarn_beta_fast: f32,
    pub(crate) yarn_beta_slow: f32,
    pub(crate) yarn_orig_ctx: u32,
    pub(crate) defrag_thold: f32,
    pub(crate) cb_eval: ggml_backend_sched_eval_callback,
    pub(crate) cb_eval_user_data: *mut core::ffi::c_void,
    pub(crate) type_k: ggml_type,
    pub(crate) type_v: ggml_type,
    pub(crate) logits_all: bool,
    pub(crate) embeddings: bool,
    pub(crate) offload_kqv: bool,
    pub(crate) flash_attn: bool,
    pub(crate) no_perf: bool,
    pub(crate) abort_callback: ggml_abort_callback,
    pub(crate) abort_callback_data: *mut core::ffi::c_void,
}

#[repr(C)]
pub(crate) struct llama_model_quantize_params {
    pub(crate) nthread: i32,
    pub(crate) ftype: llama_ftype,
    pub(crate) output_tensor_type: ggml_type,
    pub(crate) token_embedding_type: ggml_type,
    pub(crate) allow_requantize: bool,
    pub(crate) quantize_output_tensor: bool,
    pub(crate) only_copy: bool,
    pub(crate) pure_: bool,
    pub(crate) keep_split: bool,
    pub(crate) imatrix: *mut core::ffi::c_void,
    pub(crate) kv_overrides: *mut core::ffi::c_void,
}

#[repr(C)]
pub(crate) struct llama_logit_bias {
    pub(crate) token: llama_token,
    pub(crate) bias: f32,
}

#[repr(C)]
pub(crate) struct llama_sampler_chain_params {
    pub(crate) no_perf: bool,
}

#[repr(C)]
pub(crate) struct llama_chat_message {
    pub(crate) role: *const core::ffi::c_char,
    pub(crate) content: *const core::ffi::c_char,
}

#[repr(C)]
pub(crate) struct llama_adapter_lora {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub(crate) struct llama_kv_cache_view_cell {
    pub(crate) pos: llama_pos,
}

#[repr(C)]
pub(crate) struct llama_kv_cache_view {
    pub(crate) n_cells: i32,
    pub(crate) n_seq_max: i32,
    pub(crate) token_count: i32,
    pub(crate) used_cells: i32,
    pub(crate) max_contiguous: i32,
    pub(crate) max_contiguous_idx: i32,
    pub(crate) cells: *mut llama_kv_cache_view_cell,
    pub(crate) cells_sequences: *mut llama_seq_id,
}

#[repr(C)]
pub(crate) struct llama_sampler_i {
    pub(crate) name: ::std::option::Option<
        unsafe extern "C" fn(smpl: *const llama_sampler) -> *const core::ffi::c_char,
    >,
    pub(crate) accept:
        ::std::option::Option<unsafe extern "C" fn(smpl: *mut llama_sampler, token: llama_token)>,
    pub(crate) apply: ::std::option::Option<
        unsafe extern "C" fn(smpl: *mut llama_sampler, cur_p: *mut llama_token_data_array),
    >,
    pub(crate) reset: ::std::option::Option<unsafe extern "C" fn(smpl: *mut llama_sampler)>,
    pub(crate) clone: ::std::option::Option<
        unsafe extern "C" fn(smpl: *const llama_sampler) -> *mut llama_sampler,
    >,
    pub(crate) free: ::std::option::Option<unsafe extern "C" fn(smpl: *mut llama_sampler)>,
}

#[repr(C)]
pub(crate) struct llama_perf_context_data {
    pub(crate) t_start_ms: f64,
    pub(crate) t_load_ms: f64,
    pub(crate) t_p_eval_ms: f64,
    pub(crate) t_eval_ms: f64,
    pub(crate) n_p_eval: i32,
    pub(crate) n_eval: i32,
}

#[repr(C)]
pub(crate) struct llama_perf_sampler_data {
    pub(crate) t_sample_ms: f64,
    pub(crate) n_sample: i32,
}

pub(crate) type llama_pos = i32;
pub(crate) type llama_token = i32;
pub(crate) type llama_seq_id = i32;
pub(crate) type llama_vocab_type = core::ffi::c_uint;
pub(crate) type llama_vocab_pre_type = core::ffi::c_uint;
pub(crate) type llama_rope_type = core::ffi::c_int;
pub(crate) type llama_token_type = core::ffi::c_uint;
pub(crate) type llama_token_attr = core::ffi::c_uint;
pub(crate) type llama_ftype = core::ffi::c_uint;
pub(crate) type llama_rope_scaling_type = core::ffi::c_int;
pub(crate) type llama_pooling_type = core::ffi::c_int;
pub(crate) type llama_attention_type = core::ffi::c_int;
pub(crate) type llama_split_mode = core::ffi::c_uint;
pub(crate) type llama_model_kv_override_type = core::ffi::c_uint;
pub(crate) type llama_sampler_context_t = *mut core::ffi::c_void;

pub(crate) type llama_progress_callback = ::std::option::Option<
    unsafe extern "C" fn(progress: f32, user_data: *mut core::ffi::c_void) -> bool,
>;

unsafe extern "C" {

    pub(crate) unsafe fn llama_model_default_params() -> llama_model_params;

    pub(crate) unsafe fn llama_context_default_params() -> llama_context_params;

    pub(crate) unsafe fn llama_sampler_chain_default_params() -> llama_sampler_chain_params;

    pub(crate) unsafe fn llama_model_quantize_default_params() -> llama_model_quantize_params;

    pub(crate) unsafe fn llama_backend_init();

    pub(crate) unsafe fn llama_backend_free();

    pub(crate) unsafe fn llama_numa_init(numa: ggml_numa_strategy);

    pub(crate) unsafe fn llama_attach_threadpool(
        ctx: *mut llama_context,
        threadpool: ggml_threadpool_t,
        threadpool_batch: ggml_threadpool_t,
    );

    pub(crate) unsafe fn llama_detach_threadpool(ctx: *mut llama_context);

    pub(crate) unsafe fn llama_load_model_from_file(
        path_model: *const core::ffi::c_char,
        params: llama_model_params,
    ) -> *mut llama_model;

    pub(crate) unsafe fn llama_model_load_from_file(
        path_model: *const core::ffi::c_char,
        params: llama_model_params,
    ) -> *mut llama_model;

    pub(crate) unsafe fn llama_model_load_from_splits(
        paths: *mut *const core::ffi::c_char,
        n_paths: usize,
        params: llama_model_params,
    ) -> *mut llama_model;

    pub(crate) unsafe fn llama_free_model(model: *mut llama_model);

    pub(crate) unsafe fn llama_model_free(model: *mut llama_model);

    pub(crate) unsafe fn llama_init_from_model(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;

    pub(crate) unsafe fn llama_new_context_with_model(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;

    pub(crate) unsafe fn llama_free(ctx: *mut llama_context);

    pub(crate) unsafe fn llama_time_us() -> i64;

    pub(crate) unsafe fn llama_max_devices() -> usize;

    pub(crate) unsafe fn llama_supports_mmap() -> bool;

    pub(crate) unsafe fn llama_supports_mlock() -> bool;

    pub(crate) unsafe fn llama_supports_gpu_offload() -> bool;

    pub(crate) unsafe fn llama_supports_rpc() -> bool;

    pub(crate) unsafe fn llama_n_ctx(ctx: *const llama_context) -> u32;

    pub(crate) unsafe fn llama_n_batch(ctx: *const llama_context) -> u32;

    pub(crate) unsafe fn llama_n_ubatch(ctx: *const llama_context) -> u32;

    pub(crate) unsafe fn llama_n_seq_max(ctx: *const llama_context) -> u32;

    pub(crate) unsafe fn llama_n_ctx_train(model: *const llama_model) -> i32;

    pub(crate) unsafe fn llama_n_embd(model: *const llama_model) -> i32;

    pub(crate) unsafe fn llama_n_layer(model: *const llama_model) -> i32;

    pub(crate) unsafe fn llama_n_head(model: *const llama_model) -> i32;

    pub(crate) unsafe fn llama_n_vocab(vocab: *const llama_vocab) -> i32;

    pub(crate) unsafe fn llama_get_model(ctx: *const llama_context) -> *const llama_model;

    pub(crate) unsafe fn llama_pooling_type(ctx: *const llama_context) -> llama_pooling_type;

    pub(crate) unsafe fn llama_model_get_vocab(model: *const llama_model) -> *const llama_vocab;

    pub(crate) unsafe fn llama_model_rope_type(model: *const llama_model) -> llama_rope_type;

    pub(crate) unsafe fn llama_model_n_ctx_train(model: *const llama_model) -> i32;

    pub(crate) unsafe fn llama_model_n_embd(model: *const llama_model) -> i32;

    pub(crate) unsafe fn llama_model_n_layer(model: *const llama_model) -> i32;

    pub(crate) unsafe fn llama_model_n_head(model: *const llama_model) -> i32;

    pub(crate) unsafe fn llama_model_rope_freq_scale_train(model: *const llama_model) -> f32;

    pub(crate) unsafe fn llama_vocab_type(vocab: *const llama_vocab) -> llama_vocab_type;

    pub(crate) unsafe fn llama_vocab_n_tokens(vocab: *const llama_vocab) -> i32;

    pub(crate) unsafe fn llama_model_meta_val_str(
        model: *const llama_model,
        key: *const core::ffi::c_char,
        buf: *mut core::ffi::c_char,
        buf_size: usize,
    ) -> i32;

    pub(crate) unsafe fn llama_model_meta_count(model: *const llama_model) -> i32;

    pub(crate) unsafe fn llama_model_meta_key_by_index(
        model: *const llama_model,
        i: i32,
        buf: *mut core::ffi::c_char,
        buf_size: usize,
    ) -> i32;

    pub(crate) unsafe fn llama_model_meta_val_str_by_index(
        model: *const llama_model,
        i: i32,
        buf: *mut core::ffi::c_char,
        buf_size: usize,
    ) -> i32;

    pub(crate) unsafe fn llama_model_desc(
        model: *const llama_model,
        buf: *mut core::ffi::c_char,
        buf_size: usize,
    ) -> i32;

    pub(crate) unsafe fn llama_model_size(model: *const llama_model) -> u64;

    pub(crate) unsafe fn llama_model_chat_template(
        model: *const llama_model,
        name: *const core::ffi::c_char,
    ) -> *const core::ffi::c_char;

    pub(crate) unsafe fn llama_model_n_params(model: *const llama_model) -> u64;

    pub(crate) unsafe fn llama_model_has_encoder(model: *const llama_model) -> bool;

    pub(crate) unsafe fn llama_model_has_decoder(model: *const llama_model) -> bool;

    pub(crate) unsafe fn llama_model_decoder_start_token(model: *const llama_model) -> llama_token;

    pub(crate) unsafe fn llama_model_is_recurrent(model: *const llama_model) -> bool;

    pub(crate) unsafe fn llama_model_quantize(
        fname_inp: *const core::ffi::c_char,
        fname_out: *const core::ffi::c_char,
        params: *const llama_model_quantize_params,
    ) -> u32;

    pub(crate) unsafe fn llama_adapter_lora_init(
        model: *mut llama_model,
        path_lora: *const core::ffi::c_char,
    ) -> *mut llama_adapter_lora;

    pub(crate) unsafe fn llama_adapter_lora_free(adapter: *mut llama_adapter_lora);

    pub(crate) unsafe fn llama_set_adapter_lora(
        ctx: *mut llama_context,
        adapter: *mut llama_adapter_lora,
        scale: f32,
    ) -> i32;

    pub(crate) unsafe fn llama_rm_adapter_lora(
        ctx: *mut llama_context,
        adapter: *mut llama_adapter_lora,
    ) -> i32;

    pub(crate) unsafe fn llama_clear_adapter_lora(ctx: *mut llama_context);

    pub(crate) unsafe fn llama_apply_adapter_cvec(
        ctx: *mut llama_context,
        data: *const f32,
        len: usize,
        n_embd: i32,
        il_start: i32,
        il_end: i32,
    ) -> i32;

    pub(crate) unsafe fn llama_kv_cache_view_init(
        ctx: *const llama_context,
        n_seq_max: i32,
    ) -> llama_kv_cache_view;

    pub(crate) unsafe fn llama_kv_cache_view_free(view: *mut llama_kv_cache_view);

    pub(crate) unsafe fn llama_kv_cache_view_update(
        ctx: *const llama_context,
        view: *mut llama_kv_cache_view,
    );

    pub(crate) unsafe fn llama_get_kv_cache_token_count(ctx: *const llama_context) -> i32;

    pub(crate) unsafe fn llama_get_kv_cache_used_cells(ctx: *const llama_context) -> i32;

    pub(crate) unsafe fn llama_kv_cache_clear(ctx: *mut llama_context);

    pub(crate) unsafe fn llama_kv_cache_seq_rm(
        ctx: *mut llama_context,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
    ) -> bool;

    pub(crate) unsafe fn llama_kv_cache_seq_cp(
        ctx: *mut llama_context,
        seq_id_src: llama_seq_id,
        seq_id_dst: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
    );

    pub(crate) unsafe fn llama_kv_cache_seq_keep(ctx: *mut llama_context, seq_id: llama_seq_id);

    pub(crate) unsafe fn llama_kv_cache_seq_add(
        ctx: *mut llama_context,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
        delta: llama_pos,
    );

    pub(crate) unsafe fn llama_kv_cache_seq_div(
        ctx: *mut llama_context,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
        d: core::ffi::c_int,
    );

    pub(crate) unsafe fn llama_kv_cache_seq_pos_max(
        ctx: *mut llama_context,
        seq_id: llama_seq_id,
    ) -> llama_pos;

    pub(crate) unsafe fn llama_kv_cache_defrag(ctx: *mut llama_context);

    pub(crate) unsafe fn llama_kv_cache_update(ctx: *mut llama_context);

    pub(crate) unsafe fn llama_kv_cache_can_shift(ctx: *mut llama_context) -> bool;

    pub(crate) unsafe fn llama_state_get_size(ctx: *mut llama_context) -> usize;

    pub(crate) unsafe fn llama_get_state_size(ctx: *mut llama_context) -> usize;

    pub(crate) unsafe fn llama_state_get_data(
        ctx: *mut llama_context,
        dst: *mut u8,
        size: usize,
    ) -> usize;

    pub(crate) unsafe fn llama_copy_state_data(ctx: *mut llama_context, dst: *mut u8) -> usize;

    pub(crate) unsafe fn llama_state_set_data(
        ctx: *mut llama_context,
        src: *const u8,
        size: usize,
    ) -> usize;

    pub(crate) unsafe fn llama_set_state_data(ctx: *mut llama_context, src: *const u8) -> usize;

    pub(crate) unsafe fn llama_state_load_file(
        ctx: *mut llama_context,
        path_session: *const core::ffi::c_char,
        tokens_out: *mut llama_token,
        n_token_capacity: usize,
        n_token_count_out: *mut usize,
    ) -> bool;

    pub(crate) unsafe fn llama_load_session_file(
        ctx: *mut llama_context,
        path_session: *const core::ffi::c_char,
        tokens_out: *mut llama_token,
        n_token_capacity: usize,
        n_token_count_out: *mut usize,
    ) -> bool;

    pub(crate) unsafe fn llama_state_save_file(
        ctx: *mut llama_context,
        path_session: *const core::ffi::c_char,
        tokens: *const llama_token,
        n_token_count: usize,
    ) -> bool;

    pub(crate) unsafe fn llama_save_session_file(
        ctx: *mut llama_context,
        path_session: *const core::ffi::c_char,
        tokens: *const llama_token,
        n_token_count: usize,
    ) -> bool;

    pub(crate) unsafe fn llama_state_seq_get_size(
        ctx: *mut llama_context,
        seq_id: llama_seq_id,
    ) -> usize;

    pub(crate) unsafe fn llama_state_seq_get_data(
        ctx: *mut llama_context,
        dst: *mut u8,
        size: usize,
        seq_id: llama_seq_id,
    ) -> usize;

    pub(crate) unsafe fn llama_state_seq_set_data(
        ctx: *mut llama_context,
        src: *const u8,
        size: usize,
        dest_seq_id: llama_seq_id,
    ) -> usize;

    pub(crate) unsafe fn llama_state_seq_save_file(
        ctx: *mut llama_context,
        filepath: *const core::ffi::c_char,
        seq_id: llama_seq_id,
        tokens: *const llama_token,
        n_token_count: usize,
    ) -> usize;

    pub(crate) unsafe fn llama_state_seq_load_file(
        ctx: *mut llama_context,
        filepath: *const core::ffi::c_char,
        dest_seq_id: llama_seq_id,
        tokens_out: *mut llama_token,
        n_token_capacity: usize,
        n_token_count_out: *mut usize,
    ) -> usize;

    pub(crate) unsafe fn llama_batch_get_one(
        tokens: *mut llama_token,
        n_tokens: i32,
    ) -> llama_batch;

    pub(crate) unsafe fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> llama_batch;

    pub(crate) unsafe fn llama_batch_free(batch: llama_batch);

    pub(crate) unsafe fn llama_encode(ctx: *mut llama_context, batch: llama_batch) -> i32;

    pub(crate) unsafe fn llama_decode(ctx: *mut llama_context, batch: llama_batch) -> i32;

    pub(crate) unsafe fn llama_set_n_threads(
        ctx: *mut llama_context,
        n_threads: i32,
        n_threads_batch: i32,
    );

    pub(crate) unsafe fn llama_n_threads(ctx: *mut llama_context) -> i32;

    pub(crate) unsafe fn llama_n_threads_batch(ctx: *mut llama_context) -> i32;

    pub(crate) unsafe fn llama_set_embeddings(ctx: *mut llama_context, embeddings: bool);

    pub(crate) unsafe fn llama_set_causal_attn(ctx: *mut llama_context, causal_attn: bool);

    pub(crate) unsafe fn llama_set_abort_callback(
        ctx: *mut llama_context,
        abort_callback: ggml_abort_callback,
        abort_callback_data: *mut core::ffi::c_void,
    );

    pub(crate) unsafe fn llama_synchronize(ctx: *mut llama_context);

    pub(crate) unsafe fn llama_get_logits(ctx: *mut llama_context) -> *mut f32;

    pub(crate) unsafe fn llama_get_logits_ith(ctx: *mut llama_context, i: i32) -> *mut f32;

    pub(crate) unsafe fn llama_get_embeddings(ctx: *mut llama_context) -> *mut f32;

    pub(crate) unsafe fn llama_get_embeddings_ith(ctx: *mut llama_context, i: i32) -> *mut f32;

    pub(crate) unsafe fn llama_get_embeddings_seq(
        ctx: *mut llama_context,
        seq_id: llama_seq_id,
    ) -> *mut f32;

    pub(crate) unsafe fn llama_vocab_get_text(
        vocab: *const llama_vocab,
        token: llama_token,
    ) -> *const core::ffi::c_char;

    pub(crate) unsafe fn llama_vocab_get_score(
        vocab: *const llama_vocab,
        token: llama_token,
    ) -> f32;

    pub(crate) unsafe fn llama_vocab_get_attr(
        vocab: *const llama_vocab,
        token: llama_token,
    ) -> llama_token_attr;

    pub(crate) unsafe fn llama_vocab_is_eog(vocab: *const llama_vocab, token: llama_token) -> bool;

    pub(crate) unsafe fn llama_vocab_is_control(
        vocab: *const llama_vocab,
        token: llama_token,
    ) -> bool;

    pub(crate) unsafe fn llama_vocab_bos(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_vocab_eos(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_vocab_eot(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_vocab_sep(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_vocab_nl(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_vocab_pad(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_vocab_get_add_bos(vocab: *const llama_vocab) -> bool;

    pub(crate) unsafe fn llama_vocab_get_add_eos(vocab: *const llama_vocab) -> bool;

    pub(crate) unsafe fn llama_vocab_fim_pre(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_vocab_fim_suf(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_vocab_fim_mid(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_vocab_fim_pad(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_vocab_fim_rep(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_vocab_fim_sep(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_token_get_text(
        vocab: *const llama_vocab,
        token: llama_token,
    ) -> *const core::ffi::c_char;

    pub(crate) unsafe fn llama_token_get_score(
        vocab: *const llama_vocab,
        token: llama_token,
    ) -> f32;

    pub(crate) unsafe fn llama_token_get_attr(
        vocab: *const llama_vocab,
        token: llama_token,
    ) -> llama_token_attr;

    pub(crate) unsafe fn llama_token_is_eog(vocab: *const llama_vocab, token: llama_token) -> bool;

    pub(crate) unsafe fn llama_token_is_control(
        vocab: *const llama_vocab,
        token: llama_token,
    ) -> bool;

    pub(crate) unsafe fn llama_token_bos(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_token_eos(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_token_eot(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_token_cls(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_token_sep(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_token_nl(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_token_pad(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_add_bos_token(vocab: *const llama_vocab) -> bool;

    pub(crate) unsafe fn llama_add_eos_token(vocab: *const llama_vocab) -> bool;

    pub(crate) unsafe fn llama_token_fim_pre(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_token_fim_suf(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_token_fim_mid(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_token_fim_pad(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_token_fim_rep(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_token_fim_sep(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_vocab_cls(vocab: *const llama_vocab) -> llama_token;

    pub(crate) unsafe fn llama_tokenize(
        vocab: *const llama_vocab,
        text: *const core::ffi::c_char,
        text_len: i32,
        tokens: *mut llama_token,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;

    pub(crate) unsafe fn llama_token_to_piece(
        vocab: *const llama_vocab,
        token: llama_token,
        buf: *mut core::ffi::c_char,
        length: i32,
        lstrip: i32,
        special: bool,
    ) -> i32;

    pub(crate) unsafe fn llama_detokenize(
        vocab: *const llama_vocab,
        tokens: *const llama_token,
        n_tokens: i32,
        text: *mut core::ffi::c_char,
        text_len_max: i32,
        remove_special: bool,
        unparse_special: bool,
    ) -> i32;

    pub(crate) unsafe fn llama_chat_apply_template(
        tmpl: *const core::ffi::c_char,
        chat: *const llama_chat_message,
        n_msg: usize,
        add_ass: bool,
        buf: *mut core::ffi::c_char,
        length: i32,
    ) -> i32;

    pub(crate) unsafe fn llama_chat_builtin_templates(
        output: *mut *const core::ffi::c_char,
        len: usize,
    ) -> i32;

    pub(crate) unsafe fn llama_sampler_init(
        iface: *const llama_sampler_i,
        ctx: llama_sampler_context_t,
    ) -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_name(smpl: *const llama_sampler)
    -> *const core::ffi::c_char;

    pub(crate) unsafe fn llama_sampler_accept(smpl: *mut llama_sampler, token: llama_token);

    pub(crate) unsafe fn llama_sampler_apply(
        smpl: *mut llama_sampler,
        cur_p: *mut llama_token_data_array,
    );

    pub(crate) unsafe fn llama_sampler_reset(smpl: *mut llama_sampler);

    pub(crate) unsafe fn llama_sampler_clone(smpl: *const llama_sampler) -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_free(smpl: *mut llama_sampler);

    pub(crate) unsafe fn llama_sampler_chain_init(
        params: llama_sampler_chain_params,
    ) -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_chain_add(
        chain: *mut llama_sampler,
        smpl: *mut llama_sampler,
    );

    pub(crate) unsafe fn llama_sampler_chain_get(
        chain: *const llama_sampler,
        i: i32,
    ) -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_chain_n(chain: *const llama_sampler) -> core::ffi::c_int;

    pub(crate) unsafe fn llama_sampler_chain_remove(
        chain: *mut llama_sampler,
        i: i32,
    ) -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_init_greedy() -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_init_dist(seed: u32) -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_init_softmax() -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_init_top_k(k: i32) -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_init_top_p(p: f32, min_keep: usize) -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_init_min_p(p: f32, min_keep: usize) -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_init_typical(p: f32, min_keep: usize) -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_init_temp(t: f32) -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_init_temp_ext(
        t: f32,
        delta: f32,
        exponent: f32,
    ) -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_init_xtc(
        p: f32,
        t: f32,
        min_keep: usize,
        seed: u32,
    ) -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_init_top_n_sigma(n: f32) -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_init_mirostat(
        n_vocab: i32,
        seed: u32,
        tau: f32,
        eta: f32,
        m: i32,
    ) -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_init_mirostat_v2(
        seed: u32,
        tau: f32,
        eta: f32,
    ) -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_init_grammar(
        vocab: *const llama_vocab,
        grammar_str: *const core::ffi::c_char,
        grammar_root: *const core::ffi::c_char,
    ) -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_init_grammar_lazy(
        vocab: *const llama_vocab,
        grammar_str: *const core::ffi::c_char,
        grammar_root: *const core::ffi::c_char,
        trigger_words: *mut *const core::ffi::c_char,
        num_trigger_words: usize,
        trigger_tokens: *const llama_token,
        num_trigger_tokens: usize,
    ) -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_init_penalties(
        penalty_last_n: i32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    ) -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_init_dry(
        vocab: *const llama_vocab,
        n_ctx_train: i32,
        dry_multiplier: f32,
        dry_base: f32,
        dry_allowed_length: i32,
        dry_penalty_last_n: i32,
        seq_breakers: *mut *const core::ffi::c_char,
        num_breakers: usize,
    ) -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_init_logit_bias(
        n_vocab: i32,
        n_logit_bias: i32,
        logit_bias: *const llama_logit_bias,
    ) -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_init_infill(vocab: *const llama_vocab)
    -> *mut llama_sampler;

    pub(crate) unsafe fn llama_sampler_get_seed(smpl: *const llama_sampler) -> u32;

    pub(crate) unsafe fn llama_sampler_sample(
        smpl: *mut llama_sampler,
        ctx: *mut llama_context,
        idx: i32,
    ) -> llama_token;

    pub(crate) unsafe fn llama_split_path(
        split_path: *mut core::ffi::c_char,
        maxlen: usize,
        path_prefix: *const core::ffi::c_char,
        split_no: core::ffi::c_int,
        split_count: core::ffi::c_int,
    ) -> core::ffi::c_int;

    pub(crate) unsafe fn llama_split_prefix(
        split_prefix: *mut core::ffi::c_char,
        maxlen: usize,
        split_path: *const core::ffi::c_char,
        split_no: core::ffi::c_int,
        split_count: core::ffi::c_int,
    ) -> core::ffi::c_int;

    pub(crate) unsafe fn llama_print_system_info() -> *const core::ffi::c_char;

    pub(crate) unsafe fn llama_log_set(
        log_callback: ggml_log_callback,
        user_data: *mut core::ffi::c_void,
    );

    pub(crate) unsafe fn llama_perf_context(ctx: *const llama_context) -> llama_perf_context_data;

    pub(crate) unsafe fn llama_perf_context_print(ctx: *const llama_context);

    pub(crate) unsafe fn llama_perf_context_reset(ctx: *mut llama_context);

    pub(crate) unsafe fn llama_perf_sampler(chain: *const llama_sampler)
    -> llama_perf_sampler_data;

    pub(crate) unsafe fn llama_perf_sampler_print(chain: *const llama_sampler);

    pub(crate) unsafe fn llama_perf_sampler_reset(chain: *mut llama_sampler);
}
