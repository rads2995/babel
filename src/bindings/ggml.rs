#![allow(non_camel_case_types, unused, clippy::upper_case_acronyms)]

use super::ggml_alloc::{
    ggml_backend, ggml_backend_buffer, ggml_backend_buffer_type, ggml_gallocr,
};

#[repr(C)]
pub(crate) struct FILE {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub(crate) struct ggml_object {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub(crate) struct ggml_context {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub(crate) struct ggml_cgraph {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub(crate) struct ggml_bf16_t {
    pub(crate) bits: u16,
}

#[repr(C)]
pub(crate) struct ggml_init_params {
    pub(crate) mem_size: usize,
    pub(crate) mem_buffer: *mut core::ffi::c_void,
    pub(crate) no_alloc: bool,
}

#[repr(C)]
pub(crate) struct ggml_tensor {
    pub(crate) type_: ggml_type,
    pub(crate) buffer: *mut ggml_backend_buffer,
    pub(crate) ne: [i64; 4usize],
    pub(crate) nb: [usize; 4usize],
    pub(crate) op: ggml_op,
    pub(crate) op_params: [i32; 16usize],
    pub(crate) flags: i32,
    pub(crate) src: [*mut ggml_tensor; 10usize],
    pub(crate) view_src: *mut ggml_tensor,
    pub(crate) view_offs: usize,
    pub(crate) data: *mut core::ffi::c_void,
    pub(crate) name: [core::ffi::c_char; 64usize],
    pub(crate) extra: *mut core::ffi::c_void,
    pub(crate) padding: [core::ffi::c_char; 8usize],
}

#[repr(C)]
pub(crate) struct ggml_type_traits {
    pub(crate) type_name: *const core::ffi::c_char,
    pub(crate) blck_size: i64,
    pub(crate) blck_size_interleave: i64,
    pub(crate) type_size: usize,
    pub(crate) is_quantized: bool,
    pub(crate) to_float: ggml_to_float_t,
    pub(crate) from_float_ref: ggml_from_float_t,
}

#[repr(C)]
pub(crate) struct ggml_threadpool_params {
    pub(crate) cpumask: [bool; 512usize],
    pub(crate) n_threads: core::ffi::c_int,
    pub(crate) prio: ggml_sched_priority,
    pub(crate) poll: u32,
    pub(crate) strict_cpu: bool,
    pub(crate) paused: bool,
}

#[repr(C)]
pub(crate) struct ggml_threadpool {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub(crate) struct ggml_backend_event {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub(crate) struct ggml_backend_reg {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub(crate) struct ggml_backend_device {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub(crate) struct ggml_backend_sched {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub(crate) struct ggml_type_traits_cpu {
    pub(crate) from_float: ggml_from_float_t,
    pub(crate) vec_dot: ggml_vec_dot_t,
    pub(crate) vec_dot_type: ggml_type,
    pub(crate) nrows: i64,
}

pub(crate) type ggml_status = core::ffi::c_int;
pub(crate) type ggml_fp16_t = u16;
pub(crate) type ggml_type = core::ffi::c_uint;
pub(crate) type ggml_prec = core::ffi::c_uint;
pub(crate) type ggml_ftype = core::ffi::c_int;
pub(crate) type ggml_op = core::ffi::c_uint;
pub(crate) type ggml_unary_op = core::ffi::c_uint;
pub(crate) type ggml_object_type = core::ffi::c_uint;
pub(crate) type ggml_log_level = core::ffi::c_uint;
pub(crate) type ggml_tensor_flag = core::ffi::c_uint;
pub(crate) type ggml_guid = [u8; 16usize];
pub(crate) type ggml_guid_t = *mut ggml_guid;
pub(crate) type ggml_op_pool = core::ffi::c_uint;
pub(crate) type ggml_sched_priority = core::ffi::c_uint;
pub(crate) type ggml_threadpool_t = *mut ggml_threadpool;
pub(crate) type ggml_sort_order = core::ffi::c_uint;
pub(crate) type ggml_gallocr_t = *mut ggml_gallocr;

pub(crate) type ggml_abort_callback =
    core::option::Option<unsafe extern "C" fn(data: *mut core::ffi::c_void) -> bool>;

pub(crate) type ggml_unary_op_f32_t = core::option::Option<
    unsafe extern "C" fn(arg1: core::ffi::c_int, arg2: *mut f32, arg3: *const f32),
>;

pub(crate) type ggml_binary_op_f32_t = core::option::Option<
    unsafe extern "C" fn(
        arg1: core::ffi::c_int,
        arg2: *mut f32,
        arg3: *const f32,
        arg4: *const f32,
    ),
>;

pub(crate) type ggml_custom1_op_f32_t =
    core::option::Option<unsafe extern "C" fn(arg1: *mut ggml_tensor, arg2: *const ggml_tensor)>;

pub(crate) type ggml_custom2_op_f32_t = core::option::Option<
    unsafe extern "C" fn(
        arg1: *mut ggml_tensor,
        arg2: *const ggml_tensor,
        arg3: *const ggml_tensor,
    ),
>;

pub(crate) type ggml_custom3_op_f32_t = core::option::Option<
    unsafe extern "C" fn(
        arg1: *mut ggml_tensor,
        arg2: *const ggml_tensor,
        arg3: *const ggml_tensor,
        arg4: *const ggml_tensor,
    ),
>;

pub(crate) type ggml_log_callback = core::option::Option<
    unsafe extern "C" fn(
        level: ggml_log_level,
        text: *const core::ffi::c_char,
        user_data: *mut core::ffi::c_void,
    ),
>;

pub(crate) type ggml_custom1_op_t = core::option::Option<
    unsafe extern "C" fn(
        dst: *mut ggml_tensor,
        a: *const ggml_tensor,
        ith: core::ffi::c_int,
        nth: core::ffi::c_int,
        userdata: *mut core::ffi::c_void,
    ),
>;

pub(crate) type ggml_custom2_op_t = core::option::Option<
    unsafe extern "C" fn(
        dst: *mut ggml_tensor,
        a: *const ggml_tensor,
        b: *const ggml_tensor,
        ith: core::ffi::c_int,
        nth: core::ffi::c_int,
        userdata: *mut core::ffi::c_void,
    ),
>;

pub(crate) type ggml_custom3_op_t = core::option::Option<
    unsafe extern "C" fn(
        dst: *mut ggml_tensor,
        a: *const ggml_tensor,
        b: *const ggml_tensor,
        c: *const ggml_tensor,
        ith: core::ffi::c_int,
        nth: core::ffi::c_int,
        userdata: *mut core::ffi::c_void,
    ),
>;

pub(crate) type ggml_to_float_t =
    core::option::Option<unsafe extern "C" fn(x: *const core::ffi::c_void, y: *mut f32, k: i64)>;

pub(crate) type ggml_from_float_t =
    core::option::Option<unsafe extern "C" fn(x: *const f32, y: *mut core::ffi::c_void, k: i64)>;

pub(crate) type ggml_backend_eval_callback = core::option::Option<
    unsafe extern "C" fn(
        node_index: core::ffi::c_int,
        t1: *mut ggml_tensor,
        t2: *mut ggml_tensor,
        user_data: *mut core::ffi::c_void,
    ) -> bool,
>;

pub(crate) type ggml_vec_dot_t = core::option::Option<
    unsafe extern "C" fn(
        n: core::ffi::c_int,
        s: *mut f32,
        bs: usize,
        x: *const core::ffi::c_void,
        bx: usize,
        y: *const core::ffi::c_void,
        by: usize,
        nrc: core::ffi::c_int,
    ),
>;

unsafe extern "C" {

    pub(crate) unsafe fn ggml_abort(
        file: *const core::ffi::c_char,
        line: core::ffi::c_int,
        fmt: *const core::ffi::c_char,
        ...
    );

    pub(crate) unsafe fn ggml_status_to_string(status: ggml_status) -> *const core::ffi::c_char;

    pub(crate) unsafe fn ggml_fp16_to_fp32(arg1: ggml_fp16_t) -> f32;

    pub(crate) unsafe fn ggml_fp32_to_fp16(arg1: f32) -> ggml_fp16_t;

    pub(crate) unsafe fn ggml_fp16_to_fp32_row(arg1: *const ggml_fp16_t, arg2: *mut f32, arg3: i64);

    pub(crate) unsafe fn ggml_fp32_to_fp16_row(arg1: *const f32, arg2: *mut ggml_fp16_t, arg3: i64);

    pub(crate) unsafe fn ggml_fp32_to_bf16(arg1: f32) -> ggml_bf16_t;

    pub(crate) unsafe fn ggml_bf16_to_fp32(arg1: ggml_bf16_t) -> f32;

    pub(crate) unsafe fn ggml_bf16_to_fp32_row(arg1: *const ggml_bf16_t, arg2: *mut f32, arg3: i64);

    pub(crate) unsafe fn ggml_fp32_to_bf16_row_ref(
        arg1: *const f32,
        arg2: *mut ggml_bf16_t,
        arg3: i64,
    );

    pub(crate) unsafe fn ggml_fp32_to_bf16_row(arg1: *const f32, arg2: *mut ggml_bf16_t, arg3: i64);

    pub(crate) unsafe fn ggml_guid_matches(guid_a: ggml_guid_t, guid_b: ggml_guid_t) -> bool;

    pub(crate) unsafe fn ggml_time_init();

    pub(crate) unsafe fn ggml_time_ms() -> i64;

    pub(crate) unsafe fn ggml_time_us() -> i64;

    pub(crate) unsafe fn ggml_cycles() -> i64;

    pub(crate) unsafe fn ggml_cycles_per_ms() -> i64;

    pub(crate) unsafe fn ggml_fopen(
        fname: *const core::ffi::c_char,
        mode: *const core::ffi::c_char,
    ) -> *mut FILE;

    pub(crate) unsafe fn ggml_print_object(obj: *const ggml_object);

    pub(crate) unsafe fn ggml_print_objects(ctx: *const ggml_context);

    pub(crate) unsafe fn ggml_nelements(tensor: *const ggml_tensor) -> i64;

    pub(crate) unsafe fn ggml_nrows(tensor: *const ggml_tensor) -> i64;

    pub(crate) unsafe fn ggml_nbytes(tensor: *const ggml_tensor) -> usize;

    pub(crate) unsafe fn ggml_nbytes_pad(tensor: *const ggml_tensor) -> usize;

    pub(crate) unsafe fn ggml_blck_size(type_: ggml_type) -> i64;

    pub(crate) unsafe fn ggml_type_size(type_: ggml_type) -> usize;

    pub(crate) unsafe fn ggml_row_size(type_: ggml_type, ne: i64) -> usize;

    pub(crate) unsafe fn ggml_type_sizef(type_: ggml_type) -> f64;

    pub(crate) unsafe fn ggml_type_name(type_: ggml_type) -> *const core::ffi::c_char;

    pub(crate) unsafe fn ggml_op_name(op: ggml_op) -> *const core::ffi::c_char;

    pub(crate) unsafe fn ggml_op_symbol(op: ggml_op) -> *const core::ffi::c_char;

    pub(crate) unsafe fn ggml_unary_op_name(op: ggml_unary_op) -> *const core::ffi::c_char;

    pub(crate) unsafe fn ggml_op_desc(t: *const ggml_tensor) -> *const core::ffi::c_char;

    pub(crate) unsafe fn ggml_element_size(tensor: *const ggml_tensor) -> usize;

    pub(crate) unsafe fn ggml_is_quantized(type_: ggml_type) -> bool;

    pub(crate) unsafe fn ggml_ftype_to_ggml_type(ftype: ggml_ftype) -> ggml_type;

    pub(crate) unsafe fn ggml_is_transposed(tensor: *const ggml_tensor) -> bool;

    pub(crate) unsafe fn ggml_is_permuted(tensor: *const ggml_tensor) -> bool;

    pub(crate) unsafe fn ggml_is_empty(tensor: *const ggml_tensor) -> bool;

    pub(crate) unsafe fn ggml_is_scalar(tensor: *const ggml_tensor) -> bool;

    pub(crate) unsafe fn ggml_is_vector(tensor: *const ggml_tensor) -> bool;

    pub(crate) unsafe fn ggml_is_matrix(tensor: *const ggml_tensor) -> bool;

    pub(crate) unsafe fn ggml_is_3d(tensor: *const ggml_tensor) -> bool;

    pub(crate) unsafe fn ggml_n_dims(tensor: *const ggml_tensor) -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_is_contiguous(tensor: *const ggml_tensor) -> bool;

    pub(crate) unsafe fn ggml_is_contiguous_0(tensor: *const ggml_tensor) -> bool;

    pub(crate) unsafe fn ggml_is_contiguous_1(tensor: *const ggml_tensor) -> bool;

    pub(crate) unsafe fn ggml_is_contiguous_2(tensor: *const ggml_tensor) -> bool;

    pub(crate) unsafe fn ggml_are_same_shape(
        t0: *const ggml_tensor,
        t1: *const ggml_tensor,
    ) -> bool;

    pub(crate) unsafe fn ggml_are_same_stride(
        t0: *const ggml_tensor,
        t1: *const ggml_tensor,
    ) -> bool;

    pub(crate) unsafe fn ggml_can_repeat(t0: *const ggml_tensor, t1: *const ggml_tensor) -> bool;

    pub(crate) unsafe fn ggml_tensor_overhead() -> usize;

    pub(crate) unsafe fn ggml_validate_row_data(
        type_: ggml_type,
        data: *const core::ffi::c_void,
        nbytes: usize,
    ) -> bool;

    pub(crate) unsafe fn ggml_init(params: ggml_init_params) -> *mut ggml_context;

    pub(crate) unsafe fn ggml_reset(ctx: *mut ggml_context);

    pub(crate) unsafe fn ggml_free(ctx: *mut ggml_context);

    pub(crate) unsafe fn ggml_used_mem(ctx: *const ggml_context) -> usize;

    pub(crate) unsafe fn ggml_get_no_alloc(ctx: *mut ggml_context) -> bool;

    pub(crate) unsafe fn ggml_set_no_alloc(ctx: *mut ggml_context, no_alloc: bool);

    pub(crate) unsafe fn ggml_get_mem_buffer(ctx: *const ggml_context) -> *mut core::ffi::c_void;

    pub(crate) unsafe fn ggml_get_mem_size(ctx: *const ggml_context) -> usize;

    pub(crate) unsafe fn ggml_get_max_tensor_size(ctx: *const ggml_context) -> usize;

    pub(crate) unsafe fn ggml_new_tensor(
        ctx: *mut ggml_context,
        type_: ggml_type,
        n_dims: core::ffi::c_int,
        ne: *const i64,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_new_tensor_1d(
        ctx: *mut ggml_context,
        type_: ggml_type,
        ne0: i64,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_new_tensor_2d(
        ctx: *mut ggml_context,
        type_: ggml_type,
        ne0: i64,
        ne1: i64,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_new_tensor_3d(
        ctx: *mut ggml_context,
        type_: ggml_type,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_new_tensor_4d(
        ctx: *mut ggml_context,
        type_: ggml_type,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_new_buffer(
        ctx: *mut ggml_context,
        nbytes: usize,
    ) -> *mut core::ffi::c_void;

    pub(crate) unsafe fn ggml_dup_tensor(
        ctx: *mut ggml_context,
        src: *const ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_view_tensor(
        ctx: *mut ggml_context,
        src: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_get_first_tensor(ctx: *const ggml_context) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_get_next_tensor(
        ctx: *const ggml_context,
        tensor: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_get_tensor(
        ctx: *mut ggml_context,
        name: *const core::ffi::c_char,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_unravel_index(
        tensor: *const ggml_tensor,
        i: i64,
        i0: *mut i64,
        i1: *mut i64,
        i2: *mut i64,
        i3: *mut i64,
    );

    pub(crate) unsafe fn ggml_get_unary_op(tensor: *const ggml_tensor) -> ggml_unary_op;

    pub(crate) unsafe fn ggml_get_data(tensor: *const ggml_tensor) -> *mut core::ffi::c_void;

    pub(crate) unsafe fn ggml_get_data_f32(tensor: *const ggml_tensor) -> *mut f32;

    pub(crate) unsafe fn ggml_get_name(tensor: *const ggml_tensor) -> *const core::ffi::c_char;

    pub(crate) unsafe fn ggml_set_name(
        tensor: *mut ggml_tensor,
        name: *const core::ffi::c_char,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_format_name(
        tensor: *mut ggml_tensor,
        fmt: *const core::ffi::c_char,
        ...
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_set_input(tensor: *mut ggml_tensor);

    pub(crate) unsafe fn ggml_set_output(tensor: *mut ggml_tensor);

    pub(crate) unsafe fn ggml_set_param(ctx: *mut ggml_context, tensor: *mut ggml_tensor);

    pub(crate) unsafe fn ggml_set_loss(tensor: *mut ggml_tensor);

    pub(crate) unsafe fn ggml_dup(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_dup_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_add(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_add_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_add_cast(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        type_: ggml_type,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_add1(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_add1_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_acc(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        nb1: usize,
        nb2: usize,
        nb3: usize,
        offset: usize,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_acc_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        nb1: usize,
        nb2: usize,
        nb3: usize,
        offset: usize,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_sub(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_sub_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_mul(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_mul_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_div(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_div_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_sqr(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_sqr_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_sqrt(ctx: *mut ggml_context, a: *mut ggml_tensor)
    -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_sqrt_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_log(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_log_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_sin(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_sin_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_cos(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_cos_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_sum(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_sum_rows(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_mean(ctx: *mut ggml_context, a: *mut ggml_tensor)
    -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_argmax(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_count_equal(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_repeat(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_repeat_back(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_concat(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        dim: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_abs(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_abs_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_sgn(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_sgn_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_neg(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_neg_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_step(ctx: *mut ggml_context, a: *mut ggml_tensor)
    -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_step_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_tanh(ctx: *mut ggml_context, a: *mut ggml_tensor)
    -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_tanh_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_elu(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_elu_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_relu(ctx: *mut ggml_context, a: *mut ggml_tensor)
    -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_leaky_relu(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        negative_slope: f32,
        inplace: bool,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_relu_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_sigmoid(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_sigmoid_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_gelu(ctx: *mut ggml_context, a: *mut ggml_tensor)
    -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_gelu_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_gelu_quick(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_gelu_quick_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_silu(ctx: *mut ggml_context, a: *mut ggml_tensor)
    -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_silu_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_silu_back(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_hardswish(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_hardsigmoid(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_exp(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_exp_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_norm(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        eps: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_norm_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        eps: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_rms_norm(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        eps: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_rms_norm_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        eps: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_group_norm(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        n_groups: core::ffi::c_int,
        eps: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_group_norm_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        n_groups: core::ffi::c_int,
        eps: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_rms_norm_back(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        eps: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_mul_mat(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_mul_mat_set_prec(a: *mut ggml_tensor, prec: ggml_prec);

    pub(crate) unsafe fn ggml_mul_mat_id(
        ctx: *mut ggml_context,
        as_: *mut ggml_tensor,
        b: *mut ggml_tensor,
        ids: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_out_prod(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_scale(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        s: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_scale_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        s: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_set(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        nb1: usize,
        nb2: usize,
        nb3: usize,
        offset: usize,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_set_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        nb1: usize,
        nb2: usize,
        nb3: usize,
        offset: usize,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_set_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        offset: usize,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_set_1d_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        offset: usize,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_set_2d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        nb1: usize,
        offset: usize,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_set_2d_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        nb1: usize,
        offset: usize,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_cpy(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_cast(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        type_: ggml_type,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_cont(ctx: *mut ggml_context, a: *mut ggml_tensor)
    -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_cont_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_cont_2d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_cont_3d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_cont_4d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_reshape(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_reshape_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_reshape_2d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_reshape_3d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_reshape_4d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_view_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        offset: usize,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_view_2d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        nb1: usize,
        offset: usize,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_view_3d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        nb1: usize,
        nb2: usize,
        offset: usize,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_view_4d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        nb1: usize,
        nb2: usize,
        nb3: usize,
        offset: usize,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_permute(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        axis0: core::ffi::c_int,
        axis1: core::ffi::c_int,
        axis2: core::ffi::c_int,
        axis3: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_transpose(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_get_rows(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_get_rows_back(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_diag(ctx: *mut ggml_context, a: *mut ggml_tensor)
    -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_diag_mask_inf(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        n_past: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_diag_mask_inf_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        n_past: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_diag_mask_zero(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        n_past: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_diag_mask_zero_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        n_past: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_soft_max(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_soft_max_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_soft_max_ext(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        mask: *mut ggml_tensor,
        scale: f32,
        max_bias: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_soft_max_ext_back(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        scale: f32,
        max_bias: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_soft_max_ext_back_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        scale: f32,
        max_bias: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_rope(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        n_dims: core::ffi::c_int,
        mode: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_rope_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        n_dims: core::ffi::c_int,
        mode: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_rope_ext(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
        n_dims: core::ffi::c_int,
        mode: core::ffi::c_int,
        n_ctx_orig: core::ffi::c_int,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_rope_multi(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
        n_dims: core::ffi::c_int,
        sections: *mut core::ffi::c_int,
        mode: core::ffi::c_int,
        n_ctx_orig: core::ffi::c_int,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_rope_ext_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
        n_dims: core::ffi::c_int,
        mode: core::ffi::c_int,
        n_ctx_orig: core::ffi::c_int,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_rope_custom(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        n_dims: core::ffi::c_int,
        mode: core::ffi::c_int,
        n_ctx_orig: core::ffi::c_int,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_rope_custom_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        n_dims: core::ffi::c_int,
        mode: core::ffi::c_int,
        n_ctx_orig: core::ffi::c_int,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_rope_yarn_corr_dims(
        n_dims: core::ffi::c_int,
        n_ctx_orig: core::ffi::c_int,
        freq_base: f32,
        beta_fast: f32,
        beta_slow: f32,
        dims: *mut f32,
    );

    pub(crate) unsafe fn ggml_rope_ext_back(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
        n_dims: core::ffi::c_int,
        mode: core::ffi::c_int,
        n_ctx_orig: core::ffi::c_int,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_rope_multi_back(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
        n_dims: core::ffi::c_int,
        sections: *mut core::ffi::c_int,
        mode: core::ffi::c_int,
        n_ctx_orig: core::ffi::c_int,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_clamp(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        min: f32,
        max: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_im2col(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        s0: core::ffi::c_int,
        s1: core::ffi::c_int,
        p0: core::ffi::c_int,
        p1: core::ffi::c_int,
        d0: core::ffi::c_int,
        d1: core::ffi::c_int,
        is_2D: bool,
        dst_type: ggml_type,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_im2col_back(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        ne: *mut i64,
        s0: core::ffi::c_int,
        s1: core::ffi::c_int,
        p0: core::ffi::c_int,
        p1: core::ffi::c_int,
        d0: core::ffi::c_int,
        d1: core::ffi::c_int,
        is_2D: bool,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_conv_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        s0: core::ffi::c_int,
        p0: core::ffi::c_int,
        d0: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_conv_1d_ph(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        s: core::ffi::c_int,
        d: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_conv_1d_dw(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        s0: core::ffi::c_int,
        p0: core::ffi::c_int,
        d0: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_conv_1d_dw_ph(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        s0: core::ffi::c_int,
        d0: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_conv_transpose_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        s0: core::ffi::c_int,
        p0: core::ffi::c_int,
        d0: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_conv_2d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        s0: core::ffi::c_int,
        s1: core::ffi::c_int,
        p0: core::ffi::c_int,
        p1: core::ffi::c_int,
        d0: core::ffi::c_int,
        d1: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_conv_2d_sk_p0(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_conv_2d_s1_ph(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_conv_2d_dw(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        s0: core::ffi::c_int,
        s1: core::ffi::c_int,
        p0: core::ffi::c_int,
        p1: core::ffi::c_int,
        d0: core::ffi::c_int,
        d1: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_conv_transpose_2d_p0(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        stride: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_pool_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        op: ggml_op_pool,
        k0: core::ffi::c_int,
        s0: core::ffi::c_int,
        p0: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_pool_2d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        op: ggml_op_pool,
        k0: core::ffi::c_int,
        k1: core::ffi::c_int,
        s0: core::ffi::c_int,
        s1: core::ffi::c_int,
        p0: f32,
        p1: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_pool_2d_back(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        af: *mut ggml_tensor,
        op: ggml_op_pool,
        k0: core::ffi::c_int,
        k1: core::ffi::c_int,
        s0: core::ffi::c_int,
        s1: core::ffi::c_int,
        p0: f32,
        p1: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_upscale(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        scale_factor: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_upscale_ext(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: core::ffi::c_int,
        ne1: core::ffi::c_int,
        ne2: core::ffi::c_int,
        ne3: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_pad(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        p0: core::ffi::c_int,
        p1: core::ffi::c_int,
        p2: core::ffi::c_int,
        p3: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_pad_reflect_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        p0: core::ffi::c_int,
        p1: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_timestep_embedding(
        ctx: *mut ggml_context,
        timesteps: *mut ggml_tensor,
        dim: core::ffi::c_int,
        max_period: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_argsort(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        order: ggml_sort_order,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_arange(
        ctx: *mut ggml_context,
        start: f32,
        stop: f32,
        step: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_top_k(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        k: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_flash_attn_ext(
        ctx: *mut ggml_context,
        q: *mut ggml_tensor,
        k: *mut ggml_tensor,
        v: *mut ggml_tensor,
        mask: *mut ggml_tensor,
        scale: f32,
        max_bias: f32,
        logit_softcap: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_flash_attn_ext_set_prec(a: *mut ggml_tensor, prec: ggml_prec);

    pub(crate) unsafe fn ggml_flash_attn_ext_get_prec(a: *const ggml_tensor) -> ggml_prec;

    pub(crate) unsafe fn ggml_flash_attn_back(
        ctx: *mut ggml_context,
        q: *mut ggml_tensor,
        k: *mut ggml_tensor,
        v: *mut ggml_tensor,
        d: *mut ggml_tensor,
        masked: bool,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_ssm_conv(
        ctx: *mut ggml_context,
        sx: *mut ggml_tensor,
        c: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_ssm_scan(
        ctx: *mut ggml_context,
        s: *mut ggml_tensor,
        x: *mut ggml_tensor,
        dt: *mut ggml_tensor,
        A: *mut ggml_tensor,
        B: *mut ggml_tensor,
        C: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_win_part(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        w: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_win_unpart(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        w0: core::ffi::c_int,
        h0: core::ffi::c_int,
        w: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_unary(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        op: ggml_unary_op,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_unary_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        op: ggml_unary_op,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_get_rel_pos(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        qh: core::ffi::c_int,
        kh: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_add_rel_pos(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        pw: *mut ggml_tensor,
        ph: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_add_rel_pos_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        pw: *mut ggml_tensor,
        ph: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_rwkv_wkv6(
        ctx: *mut ggml_context,
        k: *mut ggml_tensor,
        v: *mut ggml_tensor,
        r: *mut ggml_tensor,
        tf: *mut ggml_tensor,
        td: *mut ggml_tensor,
        state: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_gated_linear_attn(
        ctx: *mut ggml_context,
        k: *mut ggml_tensor,
        v: *mut ggml_tensor,
        q: *mut ggml_tensor,
        g: *mut ggml_tensor,
        state: *mut ggml_tensor,
        scale: f32,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_map_unary_f32(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        fun: ggml_unary_op_f32_t,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_map_unary_inplace_f32(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        fun: ggml_unary_op_f32_t,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_map_binary_f32(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        fun: ggml_binary_op_f32_t,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_map_binary_inplace_f32(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        fun: ggml_binary_op_f32_t,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_map_custom1_f32(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        fun: ggml_custom1_op_f32_t,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_map_custom1_inplace_f32(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        fun: ggml_custom1_op_f32_t,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_map_custom2_f32(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        fun: ggml_custom2_op_f32_t,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_map_custom2_inplace_f32(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        fun: ggml_custom2_op_f32_t,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_map_custom3_f32(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
        fun: ggml_custom3_op_f32_t,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_map_custom3_inplace_f32(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
        fun: ggml_custom3_op_f32_t,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_map_custom1(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        fun: ggml_custom1_op_t,
        n_tasks: core::ffi::c_int,
        userdata: *mut core::ffi::c_void,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_map_custom1_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        fun: ggml_custom1_op_t,
        n_tasks: core::ffi::c_int,
        userdata: *mut core::ffi::c_void,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_map_custom2(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        fun: ggml_custom2_op_t,
        n_tasks: core::ffi::c_int,
        userdata: *mut core::ffi::c_void,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_map_custom2_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        fun: ggml_custom2_op_t,
        n_tasks: core::ffi::c_int,
        userdata: *mut core::ffi::c_void,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_map_custom3(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
        fun: ggml_custom3_op_t,
        n_tasks: core::ffi::c_int,
        userdata: *mut core::ffi::c_void,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_map_custom3_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
        fun: ggml_custom3_op_t,
        n_tasks: core::ffi::c_int,
        userdata: *mut core::ffi::c_void,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_cross_entropy_loss(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_cross_entropy_loss_back(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_opt_step_adamw(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        grad: *mut ggml_tensor,
        m: *mut ggml_tensor,
        v: *mut ggml_tensor,
        adamw_params: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_build_forward_expand(
        cgraph: *mut ggml_cgraph,
        tensor: *mut ggml_tensor,
    );

    pub(crate) unsafe fn ggml_build_backward_expand(
        ctx_static: *mut ggml_context,
        ctx_compute: *mut ggml_context,
        cgraph: *mut ggml_cgraph,
        accumulate: bool,
    );

    pub(crate) unsafe fn ggml_new_graph(ctx: *mut ggml_context) -> *mut ggml_cgraph;

    pub(crate) unsafe fn ggml_new_graph_custom(
        ctx: *mut ggml_context,
        size: usize,
        grads: bool,
    ) -> *mut ggml_cgraph;

    pub(crate) unsafe fn ggml_graph_dup(
        ctx: *mut ggml_context,
        cgraph: *mut ggml_cgraph,
    ) -> *mut ggml_cgraph;

    pub(crate) unsafe fn ggml_graph_cpy(src: *mut ggml_cgraph, dst: *mut ggml_cgraph);

    pub(crate) unsafe fn ggml_graph_reset(cgraph: *mut ggml_cgraph);

    pub(crate) unsafe fn ggml_graph_clear(cgraph: *mut ggml_cgraph);

    pub(crate) unsafe fn ggml_graph_size(cgraph: *mut ggml_cgraph) -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_graph_node(
        cgraph: *mut ggml_cgraph,
        i: core::ffi::c_int,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_graph_nodes(cgraph: *mut ggml_cgraph) -> *mut *mut ggml_tensor;

    pub(crate) unsafe fn ggml_graph_n_nodes(cgraph: *mut ggml_cgraph) -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_graph_add_node(cgraph: *mut ggml_cgraph, tensor: *mut ggml_tensor);

    pub(crate) unsafe fn ggml_graph_overhead() -> usize;

    pub(crate) unsafe fn ggml_graph_overhead_custom(size: usize, grads: bool) -> usize;

    pub(crate) unsafe fn ggml_graph_get_tensor(
        cgraph: *const ggml_cgraph,
        name: *const core::ffi::c_char,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_graph_get_grad(
        cgraph: *const ggml_cgraph,
        node: *const ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_graph_get_grad_acc(
        cgraph: *const ggml_cgraph,
        node: *const ggml_tensor,
    ) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_graph_export(
        cgraph: *const ggml_cgraph,
        fname: *const core::ffi::c_char,
    );

    pub(crate) unsafe fn ggml_graph_import(
        fname: *const core::ffi::c_char,
        ctx_data: *mut *mut ggml_context,
        ctx_eval: *mut *mut ggml_context,
    ) -> *mut ggml_cgraph;

    pub(crate) unsafe fn ggml_graph_print(cgraph: *const ggml_cgraph);

    pub(crate) unsafe fn ggml_graph_dump_dot(
        gb: *const ggml_cgraph,
        gf: *const ggml_cgraph,
        filename: *const core::ffi::c_char,
    );

    pub(crate) unsafe fn ggml_log_set(
        log_callback: ggml_log_callback,
        user_data: *mut core::ffi::c_void,
    );

    pub(crate) unsafe fn ggml_set_zero(tensor: *mut ggml_tensor) -> *mut ggml_tensor;

    pub(crate) unsafe fn ggml_quantize_init(type_: ggml_type);

    pub(crate) unsafe fn ggml_quantize_free();

    pub(crate) unsafe fn ggml_quantize_requires_imatrix(type_: ggml_type) -> bool;

    pub(crate) unsafe fn ggml_quantize_chunk(
        type_: ggml_type,
        src: *const f32,
        dst: *mut core::ffi::c_void,
        start: i64,
        nrows: i64,
        n_per_row: i64,
        imatrix: *const f32,
    ) -> usize;

    pub(crate) unsafe fn ggml_get_type_traits(type_: ggml_type) -> *const ggml_type_traits;

    pub(crate) unsafe fn ggml_threadpool_params_default(
        n_threads: core::ffi::c_int,
    ) -> ggml_threadpool_params;

    pub(crate) unsafe fn ggml_threadpool_params_init(
        p: *mut ggml_threadpool_params,
        n_threads: core::ffi::c_int,
    );

    pub(crate) unsafe fn ggml_threadpool_params_match(
        p0: *const ggml_threadpool_params,
        p1: *const ggml_threadpool_params,
    ) -> bool;
}
