#![allow(non_camel_case_types, unused, clippy::upper_case_acronyms)]

use super::ggml::{
    ggml_abort_callback, ggml_cgraph, ggml_context, ggml_status, ggml_tensor, ggml_threadpool,
    ggml_threadpool_params, ggml_threadpool_t, ggml_type, ggml_type_traits_cpu,
};

use super::ggml_backend::{ggml_backend_reg_t, ggml_backend_t};

#[repr(C)]
pub(crate) struct ggml_cplan {
    pub(crate) work_size: usize,
    pub(crate) work_data: *mut u8,
    pub(crate) n_threads: core::ffi::c_int,
    pub(crate) threadpool: *mut ggml_threadpool,
    pub(crate) abort_callback: ggml_abort_callback,
    pub(crate) abort_callback_data: *mut core::ffi::c_void,
}

pub(crate) type ggml_numa_strategy = core::ffi::c_uint;

unsafe extern "C" {

    pub(crate) unsafe fn ggml_numa_init(numa: ggml_numa_strategy);

    pub(crate) unsafe fn ggml_is_numa() -> bool;

    pub(crate) unsafe fn ggml_new_i32(ctx: *mut ggml_context, value: i32) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_new_f32(ctx: *mut ggml_context, value: f32) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_set_i32(tensor: *mut ggml_tensor, value: i32) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_set_f32(tensor: *mut ggml_tensor, value: f32) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_get_i32_1d(tensor: *const ggml_tensor, i: core::ffi::c_int) -> i32;

    pub(crate) unsafe fn ggml_set_i32_1d(
        tensor: *const ggml_tensor,
        i: core::ffi::c_int,
        value: i32,
    );

    pub(crate) unsafe fn ggml_get_i32_nd(
        tensor: *const ggml_tensor,
        i0: core::ffi::c_int,
        i1: core::ffi::c_int,
        i2: core::ffi::c_int,
        i3: core::ffi::c_int,
    ) -> i32;

    pub(crate) unsafe fn ggml_set_i32_nd(
        tensor: *const ggml_tensor,
        i0: core::ffi::c_int,
        i1: core::ffi::c_int,
        i2: core::ffi::c_int,
        i3: core::ffi::c_int,
        value: i32,
    );

    pub(crate) unsafe fn ggml_get_f32_1d(tensor: *const ggml_tensor, i: core::ffi::c_int) -> f32;

    pub(crate) unsafe fn ggml_set_f32_1d(
        tensor: *const ggml_tensor,
        i: core::ffi::c_int,
        value: f32,
    );

    pub(crate) unsafe fn ggml_get_f32_nd(
        tensor: *const ggml_tensor,
        i0: core::ffi::c_int,
        i1: core::ffi::c_int,
        i2: core::ffi::c_int,
        i3: core::ffi::c_int,
    ) -> f32;

    pub(crate) unsafe fn ggml_set_f32_nd(
        tensor: *const ggml_tensor,
        i0: core::ffi::c_int,
        i1: core::ffi::c_int,
        i2: core::ffi::c_int,
        i3: core::ffi::c_int,
        value: f32,
    );

    pub(crate) unsafe fn ggml_threadpool_new(
        params: *mut ggml_threadpool_params,
    ) -> *mut ggml_threadpool;

    pub(crate) unsafe fn ggml_threadpool_free(threadpool: *mut ggml_threadpool);

    pub(crate) unsafe fn ggml_threadpool_get_n_threads(
        threadpool: *mut ggml_threadpool,
    ) -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_threadpool_pause(threadpool: *mut ggml_threadpool);

    pub(crate) unsafe fn ggml_threadpool_resume(threadpool: *mut ggml_threadpool);

    pub(crate) unsafe fn ggml_graph_plan(
        cgraph: *const ggml_cgraph,
        n_threads: core::ffi::c_int,
        threadpool: *mut ggml_threadpool,
    ) -> ggml_cplan;

    pub(crate) unsafe fn ggml_graph_compute(
        cgraph: *mut ggml_cgraph,
        cplan: *mut ggml_cplan,
    ) -> ggml_status;

    pub(crate) unsafe fn ggml_graph_compute_with_ctx(
        ctx: *mut ggml_context,
        cgraph: *mut ggml_cgraph,
        n_threads: core::ffi::c_int,
    ) -> ggml_status;

    pub(crate) unsafe fn ggml_cpu_has_sse3() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_has_ssse3() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_has_avx() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_has_avx_vnni() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_has_avx2() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_has_f16c() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_has_fma() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_has_avx512() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_has_avx512_vbmi() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_has_avx512_vnni() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_has_avx512_bf16() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_has_amx_int8() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_has_neon() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_has_arm_fma() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_has_fp16_va() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_has_dotprod() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_has_matmul_int8() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_has_sve() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_get_sve_cnt() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_has_riscv_v() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_has_vsx() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_has_wasm_simd() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_cpu_has_llamafile() -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_get_type_traits_cpu(type_: ggml_type) -> *const ggml_type_traits_cpu;

    pub(crate) unsafe fn ggml_cpu_init();

    pub(crate) unsafe fn ggml_backend_cpu_init() -> ggml_backend_t;

    pub(crate) unsafe fn ggml_backend_is_cpu(backend: ggml_backend_t) -> bool;

    pub(crate) unsafe fn ggml_backend_cpu_set_n_threads(
        backend_cpu: ggml_backend_t,
        n_threads: core::ffi::c_int,
    );

    pub(crate) unsafe fn ggml_backend_cpu_set_threadpool(
        backend_cpu: ggml_backend_t,
        threadpool: ggml_threadpool_t,
    );

    pub(crate) unsafe fn ggml_backend_cpu_set_abort_callback(
        backend_cpu: ggml_backend_t,
        abort_callback: ggml_abort_callback,
        abort_callback_data: *mut core::ffi::c_void,
    );

    pub(crate) unsafe fn ggml_backend_cpu_reg() -> ggml_backend_reg_t;
}
