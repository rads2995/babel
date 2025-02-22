#[repr(C)]
pub(crate) struct ggml_backend_buffer {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

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
pub struct ggml_backend_buffer_type {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub struct ggml_backend {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub struct ggml_tallocr {
    pub buffer: ggml_backend_buffer_t,
    pub base: *mut core::ffi::c_void,
    pub alignment: usize,
    pub offset: usize,
}

#[repr(C)]
pub struct ggml_backend_event {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub struct ggml_backend_reg {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub struct ggml_backend_device {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
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
pub(crate) type ggml_backend_buffer_type_t = *mut ggml_backend_buffer_type;
pub(crate) type ggml_backend_buffer_t = *mut ggml_backend_buffer;
pub(crate) type ggml_backend_t = *mut ggml_backend;
pub(crate) type ggml_gallocr_t = *mut ggml_gallocr;
pub(crate) type ggml_backend_event_t = *mut ggml_backend_event;
pub(crate) type ggml_backend_graph_plan_t = *mut core::ffi::c_void;
pub(crate) type ggml_backend_reg_t = *mut ggml_backend_reg;
pub(crate) type ggml_backend_buffer_usage = core::ffi::c_uint;

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
pub type ggml_custom1_op_t = core::option::Option<
    unsafe extern "C" fn(
        dst: *mut ggml_tensor,
        a: *const ggml_tensor,
        ith: core::ffi::c_int,
        nth: core::ffi::c_int,
        userdata: *mut core::ffi::c_void,
    ),
>;
pub type ggml_custom2_op_t = core::option::Option<
    unsafe extern "C" fn(
        dst: *mut ggml_tensor,
        a: *const ggml_tensor,
        b: *const ggml_tensor,
        ith: core::ffi::c_int,
        nth: core::ffi::c_int,
        userdata: *mut core::ffi::c_void,
    ),
>;
pub type ggml_custom3_op_t = core::option::Option<
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
pub type ggml_to_float_t =
    core::option::Option<unsafe extern "C" fn(x: *const core::ffi::c_void, y: *mut f32, k: i64)>;
pub type ggml_from_float_t =
    core::option::Option<unsafe extern "C" fn(x: *const f32, y: *mut core::ffi::c_void, k: i64)>;
pub type ggml_backend_dev_t = *mut ggml_backend_device;
unsafe extern "C" {
    pub fn ggml_backend_buft_name(buft: ggml_backend_buffer_type_t) -> *const core::ffi::c_char;
}

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
}

unsafe extern "C" {
    pub fn ggml_op_symbol(op: ggml_op) -> *const core::ffi::c_char;
}
unsafe extern "C" {
    pub fn ggml_unary_op_name(op: ggml_unary_op) -> *const core::ffi::c_char;
}
unsafe extern "C" {
    pub fn ggml_op_desc(t: *const ggml_tensor) -> *const core::ffi::c_char;
}
unsafe extern "C" {
    pub fn ggml_element_size(tensor: *const ggml_tensor) -> usize;
}
unsafe extern "C" {
    pub fn ggml_is_quantized(type_: ggml_type) -> bool;
}
unsafe extern "C" {
    pub fn ggml_ftype_to_ggml_type(ftype: ggml_ftype) -> ggml_type;
}
unsafe extern "C" {
    pub fn ggml_is_transposed(tensor: *const ggml_tensor) -> bool;
}
unsafe extern "C" {
    pub fn ggml_is_permuted(tensor: *const ggml_tensor) -> bool;
}
unsafe extern "C" {
    pub fn ggml_is_empty(tensor: *const ggml_tensor) -> bool;
}
unsafe extern "C" {
    pub fn ggml_is_scalar(tensor: *const ggml_tensor) -> bool;
}
unsafe extern "C" {
    pub fn ggml_is_vector(tensor: *const ggml_tensor) -> bool;
}
unsafe extern "C" {
    pub fn ggml_is_matrix(tensor: *const ggml_tensor) -> bool;
}
unsafe extern "C" {
    pub fn ggml_is_3d(tensor: *const ggml_tensor) -> bool;
}
unsafe extern "C" {
    pub fn ggml_n_dims(tensor: *const ggml_tensor) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_is_contiguous(tensor: *const ggml_tensor) -> bool;
}
unsafe extern "C" {
    pub fn ggml_is_contiguous_0(tensor: *const ggml_tensor) -> bool;
}
unsafe extern "C" {
    pub fn ggml_is_contiguous_1(tensor: *const ggml_tensor) -> bool;
}
unsafe extern "C" {
    pub fn ggml_is_contiguous_2(tensor: *const ggml_tensor) -> bool;
}
unsafe extern "C" {
    pub fn ggml_are_same_shape(t0: *const ggml_tensor, t1: *const ggml_tensor) -> bool;
}
unsafe extern "C" {
    pub fn ggml_are_same_stride(t0: *const ggml_tensor, t1: *const ggml_tensor) -> bool;
}
unsafe extern "C" {
    pub fn ggml_can_repeat(t0: *const ggml_tensor, t1: *const ggml_tensor) -> bool;
}
unsafe extern "C" {
    pub fn ggml_tensor_overhead() -> usize;
}
unsafe extern "C" {
    pub fn ggml_validate_row_data(
        type_: ggml_type,
        data: *const core::ffi::c_void,
        nbytes: usize,
    ) -> bool;
}
unsafe extern "C" {
    pub fn ggml_init(params: ggml_init_params) -> *mut ggml_context;
}
unsafe extern "C" {
    pub fn ggml_reset(ctx: *mut ggml_context);
}
unsafe extern "C" {
    pub fn ggml_free(ctx: *mut ggml_context);
}
unsafe extern "C" {
    pub fn ggml_used_mem(ctx: *const ggml_context) -> usize;
}
unsafe extern "C" {
    pub fn ggml_get_no_alloc(ctx: *mut ggml_context) -> bool;
}
unsafe extern "C" {
    pub fn ggml_set_no_alloc(ctx: *mut ggml_context, no_alloc: bool);
}
unsafe extern "C" {
    pub fn ggml_get_mem_buffer(ctx: *const ggml_context) -> *mut core::ffi::c_void;
}
unsafe extern "C" {
    pub fn ggml_get_mem_size(ctx: *const ggml_context) -> usize;
}
unsafe extern "C" {
    pub fn ggml_get_max_tensor_size(ctx: *const ggml_context) -> usize;
}
unsafe extern "C" {
    pub fn ggml_new_tensor(
        ctx: *mut ggml_context,
        type_: ggml_type,
        n_dims: core::ffi::c_int,
        ne: *const i64,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_new_tensor_1d(
        ctx: *mut ggml_context,
        type_: ggml_type,
        ne0: i64,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_new_tensor_2d(
        ctx: *mut ggml_context,
        type_: ggml_type,
        ne0: i64,
        ne1: i64,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_new_tensor_3d(
        ctx: *mut ggml_context,
        type_: ggml_type,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_new_tensor_4d(
        ctx: *mut ggml_context,
        type_: ggml_type,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_new_buffer(ctx: *mut ggml_context, nbytes: usize) -> *mut core::ffi::c_void;
}
unsafe extern "C" {
    pub fn ggml_dup_tensor(ctx: *mut ggml_context, src: *const ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_view_tensor(ctx: *mut ggml_context, src: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_get_first_tensor(ctx: *const ggml_context) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_get_next_tensor(
        ctx: *const ggml_context,
        tensor: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_get_tensor(
        ctx: *mut ggml_context,
        name: *const core::ffi::c_char,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_unravel_index(
        tensor: *const ggml_tensor,
        i: i64,
        i0: *mut i64,
        i1: *mut i64,
        i2: *mut i64,
        i3: *mut i64,
    );
}
unsafe extern "C" {
    pub fn ggml_get_unary_op(tensor: *const ggml_tensor) -> ggml_unary_op;
}
unsafe extern "C" {
    pub fn ggml_get_data(tensor: *const ggml_tensor) -> *mut core::ffi::c_void;
}
unsafe extern "C" {
    pub fn ggml_get_data_f32(tensor: *const ggml_tensor) -> *mut f32;
}
unsafe extern "C" {
    pub fn ggml_get_name(tensor: *const ggml_tensor) -> *const core::ffi::c_char;
}
unsafe extern "C" {
    pub fn ggml_set_name(
        tensor: *mut ggml_tensor,
        name: *const core::ffi::c_char,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_format_name(
        tensor: *mut ggml_tensor,
        fmt: *const core::ffi::c_char,
        ...
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_set_input(tensor: *mut ggml_tensor);
}
unsafe extern "C" {
    pub fn ggml_set_output(tensor: *mut ggml_tensor);
}
unsafe extern "C" {
    pub fn ggml_set_param(ctx: *mut ggml_context, tensor: *mut ggml_tensor);
}
unsafe extern "C" {
    pub fn ggml_set_loss(tensor: *mut ggml_tensor);
}
unsafe extern "C" {
    pub fn ggml_dup(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_dup_inplace(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_add(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_add_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_add_cast(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        type_: ggml_type,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_add1(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_add1_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_acc(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        nb1: usize,
        nb2: usize,
        nb3: usize,
        offset: usize,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_acc_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        nb1: usize,
        nb2: usize,
        nb3: usize,
        offset: usize,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_sub(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_sub_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_mul(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_mul_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_div(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_div_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_sqr(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_sqr_inplace(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_sqrt(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_sqrt_inplace(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_log(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_log_inplace(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_sin(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_sin_inplace(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_cos(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_cos_inplace(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_sum(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_sum_rows(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_mean(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_argmax(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_count_equal(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_repeat(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_repeat_back(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_concat(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        dim: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_abs(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_abs_inplace(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_sgn(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_sgn_inplace(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_neg(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_neg_inplace(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_step(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_step_inplace(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_tanh(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_tanh_inplace(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_elu(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_elu_inplace(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_relu(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_leaky_relu(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        negative_slope: f32,
        inplace: bool,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_relu_inplace(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_sigmoid(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_sigmoid_inplace(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_gelu(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_gelu_inplace(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_gelu_quick(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_gelu_quick_inplace(ctx: *mut ggml_context, a: *mut ggml_tensor)
    -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_silu(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_silu_inplace(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_silu_back(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_hardswish(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_hardsigmoid(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_exp(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_exp_inplace(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_norm(ctx: *mut ggml_context, a: *mut ggml_tensor, eps: f32) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_norm_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        eps: f32,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_rms_norm(ctx: *mut ggml_context, a: *mut ggml_tensor, eps: f32)
    -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_rms_norm_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        eps: f32,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_group_norm(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        n_groups: core::ffi::c_int,
        eps: f32,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_group_norm_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        n_groups: core::ffi::c_int,
        eps: f32,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_rms_norm_back(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        eps: f32,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_mul_mat(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_mul_mat_set_prec(a: *mut ggml_tensor, prec: ggml_prec);
}
unsafe extern "C" {
    pub fn ggml_mul_mat_id(
        ctx: *mut ggml_context,
        as_: *mut ggml_tensor,
        b: *mut ggml_tensor,
        ids: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_out_prod(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_scale(ctx: *mut ggml_context, a: *mut ggml_tensor, s: f32) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_scale_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        s: f32,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_set(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        nb1: usize,
        nb2: usize,
        nb3: usize,
        offset: usize,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_set_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        nb1: usize,
        nb2: usize,
        nb3: usize,
        offset: usize,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_set_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        offset: usize,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_set_1d_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        offset: usize,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_set_2d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        nb1: usize,
        offset: usize,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_set_2d_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        nb1: usize,
        offset: usize,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_cpy(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_cast(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        type_: ggml_type,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_cont(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_cont_1d(ctx: *mut ggml_context, a: *mut ggml_tensor, ne0: i64) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_cont_2d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_cont_3d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_cont_4d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_reshape(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_reshape_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_reshape_2d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_reshape_3d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_reshape_4d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_view_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        offset: usize,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_view_2d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        nb1: usize,
        offset: usize,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_view_3d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        nb1: usize,
        nb2: usize,
        offset: usize,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_view_4d(
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
}
unsafe extern "C" {
    pub fn ggml_permute(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        axis0: core::ffi::c_int,
        axis1: core::ffi::c_int,
        axis2: core::ffi::c_int,
        axis3: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_transpose(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_get_rows(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_get_rows_back(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_diag(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_diag_mask_inf(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        n_past: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_diag_mask_inf_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        n_past: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_diag_mask_zero(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        n_past: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_diag_mask_zero_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        n_past: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_soft_max(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_soft_max_inplace(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_soft_max_ext(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        mask: *mut ggml_tensor,
        scale: f32,
        max_bias: f32,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_soft_max_ext_back(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        scale: f32,
        max_bias: f32,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_soft_max_ext_back_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        scale: f32,
        max_bias: f32,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_rope(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        n_dims: core::ffi::c_int,
        mode: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_rope_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        n_dims: core::ffi::c_int,
        mode: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_rope_ext(
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
}
unsafe extern "C" {
    pub fn ggml_rope_multi(
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
}
unsafe extern "C" {
    pub fn ggml_rope_ext_inplace(
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
}
unsafe extern "C" {
    pub fn ggml_rope_custom(
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
}
unsafe extern "C" {
    pub fn ggml_rope_custom_inplace(
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
}
unsafe extern "C" {
    pub fn ggml_rope_yarn_corr_dims(
        n_dims: core::ffi::c_int,
        n_ctx_orig: core::ffi::c_int,
        freq_base: f32,
        beta_fast: f32,
        beta_slow: f32,
        dims: *mut f32,
    );
}
unsafe extern "C" {
    pub fn ggml_rope_ext_back(
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
}
unsafe extern "C" {
    pub fn ggml_rope_multi_back(
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
}
unsafe extern "C" {
    pub fn ggml_clamp(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        min: f32,
        max: f32,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_im2col(
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
}
unsafe extern "C" {
    pub fn ggml_im2col_back(
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
}
unsafe extern "C" {
    pub fn ggml_conv_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        s0: core::ffi::c_int,
        p0: core::ffi::c_int,
        d0: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_conv_1d_ph(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        s: core::ffi::c_int,
        d: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_conv_1d_dw(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        s0: core::ffi::c_int,
        p0: core::ffi::c_int,
        d0: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_conv_1d_dw_ph(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        s0: core::ffi::c_int,
        d0: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_conv_transpose_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        s0: core::ffi::c_int,
        p0: core::ffi::c_int,
        d0: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_conv_2d(
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
}
unsafe extern "C" {
    pub fn ggml_conv_2d_sk_p0(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_conv_2d_s1_ph(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_conv_2d_dw(
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
}
unsafe extern "C" {
    pub fn ggml_conv_transpose_2d_p0(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        stride: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}

unsafe extern "C" {
    pub fn ggml_pool_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        op: ggml_op_pool,
        k0: core::ffi::c_int,
        s0: core::ffi::c_int,
        p0: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_pool_2d(
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
}
unsafe extern "C" {
    pub fn ggml_pool_2d_back(
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
}
unsafe extern "C" {
    pub fn ggml_upscale(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        scale_factor: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_upscale_ext(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: core::ffi::c_int,
        ne1: core::ffi::c_int,
        ne2: core::ffi::c_int,
        ne3: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_pad(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        p0: core::ffi::c_int,
        p1: core::ffi::c_int,
        p2: core::ffi::c_int,
        p3: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_pad_reflect_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        p0: core::ffi::c_int,
        p1: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_timestep_embedding(
        ctx: *mut ggml_context,
        timesteps: *mut ggml_tensor,
        dim: core::ffi::c_int,
        max_period: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}

unsafe extern "C" {
    pub fn ggml_argsort(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        order: ggml_sort_order,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_arange(
        ctx: *mut ggml_context,
        start: f32,
        stop: f32,
        step: f32,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_top_k(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        k: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_flash_attn_ext(
        ctx: *mut ggml_context,
        q: *mut ggml_tensor,
        k: *mut ggml_tensor,
        v: *mut ggml_tensor,
        mask: *mut ggml_tensor,
        scale: f32,
        max_bias: f32,
        logit_softcap: f32,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_flash_attn_ext_set_prec(a: *mut ggml_tensor, prec: ggml_prec);
}
unsafe extern "C" {
    pub fn ggml_flash_attn_ext_get_prec(a: *const ggml_tensor) -> ggml_prec;
}
unsafe extern "C" {
    pub fn ggml_flash_attn_back(
        ctx: *mut ggml_context,
        q: *mut ggml_tensor,
        k: *mut ggml_tensor,
        v: *mut ggml_tensor,
        d: *mut ggml_tensor,
        masked: bool,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_ssm_conv(
        ctx: *mut ggml_context,
        sx: *mut ggml_tensor,
        c: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_ssm_scan(
        ctx: *mut ggml_context,
        s: *mut ggml_tensor,
        x: *mut ggml_tensor,
        dt: *mut ggml_tensor,
        A: *mut ggml_tensor,
        B: *mut ggml_tensor,
        C: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_win_part(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        w: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_win_unpart(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        w0: core::ffi::c_int,
        h0: core::ffi::c_int,
        w: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_unary(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        op: ggml_unary_op,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_unary_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        op: ggml_unary_op,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_get_rel_pos(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        qh: core::ffi::c_int,
        kh: core::ffi::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_add_rel_pos(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        pw: *mut ggml_tensor,
        ph: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_add_rel_pos_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        pw: *mut ggml_tensor,
        ph: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_rwkv_wkv6(
        ctx: *mut ggml_context,
        k: *mut ggml_tensor,
        v: *mut ggml_tensor,
        r: *mut ggml_tensor,
        tf: *mut ggml_tensor,
        td: *mut ggml_tensor,
        state: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_gated_linear_attn(
        ctx: *mut ggml_context,
        k: *mut ggml_tensor,
        v: *mut ggml_tensor,
        q: *mut ggml_tensor,
        g: *mut ggml_tensor,
        state: *mut ggml_tensor,
        scale: f32,
    ) -> *mut ggml_tensor;
}

unsafe extern "C" {
    pub fn ggml_map_unary_f32(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        fun: ggml_unary_op_f32_t,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_map_unary_inplace_f32(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        fun: ggml_unary_op_f32_t,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_map_binary_f32(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        fun: ggml_binary_op_f32_t,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_map_binary_inplace_f32(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        fun: ggml_binary_op_f32_t,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_map_custom1_f32(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        fun: ggml_custom1_op_f32_t,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_map_custom1_inplace_f32(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        fun: ggml_custom1_op_f32_t,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_map_custom2_f32(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        fun: ggml_custom2_op_f32_t,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_map_custom2_inplace_f32(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        fun: ggml_custom2_op_f32_t,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_map_custom3_f32(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
        fun: ggml_custom3_op_f32_t,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_map_custom3_inplace_f32(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
        fun: ggml_custom3_op_f32_t,
    ) -> *mut ggml_tensor;
}

unsafe extern "C" {
    pub fn ggml_map_custom1(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        fun: ggml_custom1_op_t,
        n_tasks: core::ffi::c_int,
        userdata: *mut core::ffi::c_void,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_map_custom1_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        fun: ggml_custom1_op_t,
        n_tasks: core::ffi::c_int,
        userdata: *mut core::ffi::c_void,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_map_custom2(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        fun: ggml_custom2_op_t,
        n_tasks: core::ffi::c_int,
        userdata: *mut core::ffi::c_void,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_map_custom2_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        fun: ggml_custom2_op_t,
        n_tasks: core::ffi::c_int,
        userdata: *mut core::ffi::c_void,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_map_custom3(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
        fun: ggml_custom3_op_t,
        n_tasks: core::ffi::c_int,
        userdata: *mut core::ffi::c_void,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_map_custom3_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
        fun: ggml_custom3_op_t,
        n_tasks: core::ffi::c_int,
        userdata: *mut core::ffi::c_void,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_cross_entropy_loss(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_cross_entropy_loss_back(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_opt_step_adamw(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        grad: *mut ggml_tensor,
        m: *mut ggml_tensor,
        v: *mut ggml_tensor,
        adamw_params: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_build_forward_expand(cgraph: *mut ggml_cgraph, tensor: *mut ggml_tensor);
}
unsafe extern "C" {
    pub fn ggml_build_backward_expand(
        ctx_static: *mut ggml_context,
        ctx_compute: *mut ggml_context,
        cgraph: *mut ggml_cgraph,
        accumulate: bool,
    );
}
unsafe extern "C" {
    pub fn ggml_new_graph(ctx: *mut ggml_context) -> *mut ggml_cgraph;
}
unsafe extern "C" {
    pub fn ggml_new_graph_custom(
        ctx: *mut ggml_context,
        size: usize,
        grads: bool,
    ) -> *mut ggml_cgraph;
}
unsafe extern "C" {
    pub fn ggml_graph_dup(ctx: *mut ggml_context, cgraph: *mut ggml_cgraph) -> *mut ggml_cgraph;
}
unsafe extern "C" {
    pub fn ggml_graph_cpy(src: *mut ggml_cgraph, dst: *mut ggml_cgraph);
}
unsafe extern "C" {
    pub fn ggml_graph_reset(cgraph: *mut ggml_cgraph);
}
unsafe extern "C" {
    pub fn ggml_graph_clear(cgraph: *mut ggml_cgraph);
}
unsafe extern "C" {
    pub fn ggml_graph_size(cgraph: *mut ggml_cgraph) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_graph_node(cgraph: *mut ggml_cgraph, i: core::ffi::c_int) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_graph_nodes(cgraph: *mut ggml_cgraph) -> *mut *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_graph_n_nodes(cgraph: *mut ggml_cgraph) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_graph_add_node(cgraph: *mut ggml_cgraph, tensor: *mut ggml_tensor);
}
unsafe extern "C" {
    pub fn ggml_graph_overhead() -> usize;
}
unsafe extern "C" {
    pub fn ggml_graph_overhead_custom(size: usize, grads: bool) -> usize;
}
unsafe extern "C" {
    pub fn ggml_graph_get_tensor(
        cgraph: *const ggml_cgraph,
        name: *const core::ffi::c_char,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_graph_get_grad(
        cgraph: *const ggml_cgraph,
        node: *const ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_graph_get_grad_acc(
        cgraph: *const ggml_cgraph,
        node: *const ggml_tensor,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_graph_export(cgraph: *const ggml_cgraph, fname: *const core::ffi::c_char);
}
unsafe extern "C" {
    pub fn ggml_graph_import(
        fname: *const core::ffi::c_char,
        ctx_data: *mut *mut ggml_context,
        ctx_eval: *mut *mut ggml_context,
    ) -> *mut ggml_cgraph;
}
unsafe extern "C" {
    pub fn ggml_graph_print(cgraph: *const ggml_cgraph);
}
unsafe extern "C" {
    pub fn ggml_graph_dump_dot(
        gb: *const ggml_cgraph,
        gf: *const ggml_cgraph,
        filename: *const core::ffi::c_char,
    );
}

unsafe extern "C" {
    pub fn ggml_log_set(log_callback: ggml_log_callback, user_data: *mut core::ffi::c_void);
}
unsafe extern "C" {
    pub fn ggml_set_zero(tensor: *mut ggml_tensor) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_quantize_init(type_: ggml_type);
}
unsafe extern "C" {
    pub fn ggml_quantize_free();
}
unsafe extern "C" {
    pub fn ggml_quantize_requires_imatrix(type_: ggml_type) -> bool;
}
unsafe extern "C" {
    pub fn ggml_quantize_chunk(
        type_: ggml_type,
        src: *const f32,
        dst: *mut core::ffi::c_void,
        start: i64,
        nrows: i64,
        n_per_row: i64,
        imatrix: *const f32,
    ) -> usize;
}

unsafe extern "C" {
    pub fn ggml_get_type_traits(type_: ggml_type) -> *const ggml_type_traits;
}

unsafe extern "C" {
    pub fn ggml_threadpool_params_default(n_threads: core::ffi::c_int) -> ggml_threadpool_params;
}
unsafe extern "C" {
    pub fn ggml_threadpool_params_init(p: *mut ggml_threadpool_params, n_threads: core::ffi::c_int);
}
unsafe extern "C" {
    pub fn ggml_threadpool_params_match(
        p0: *const ggml_threadpool_params,
        p1: *const ggml_threadpool_params,
    ) -> bool;
}

unsafe extern "C" {
    pub fn ggml_tallocr_new(buffer: ggml_backend_buffer_t) -> ggml_tallocr;
}
unsafe extern "C" {
    pub fn ggml_tallocr_alloc(talloc: *mut ggml_tallocr, tensor: *mut ggml_tensor);
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_gallocr {
    _unused: [u8; 0],
}

unsafe extern "C" {
    pub fn ggml_gallocr_new(buft: ggml_backend_buffer_type_t) -> ggml_gallocr_t;
}
unsafe extern "C" {
    pub fn ggml_gallocr_new_n(
        bufts: *mut ggml_backend_buffer_type_t,
        n_bufs: core::ffi::c_int,
    ) -> ggml_gallocr_t;
}
unsafe extern "C" {
    pub fn ggml_gallocr_free(galloc: ggml_gallocr_t);
}
unsafe extern "C" {
    pub fn ggml_gallocr_reserve(galloc: ggml_gallocr_t, graph: *mut ggml_cgraph) -> bool;
}
unsafe extern "C" {
    pub fn ggml_gallocr_reserve_n(
        galloc: ggml_gallocr_t,
        graph: *mut ggml_cgraph,
        node_buffer_ids: *const core::ffi::c_int,
        leaf_buffer_ids: *const core::ffi::c_int,
    ) -> bool;
}
unsafe extern "C" {
    pub fn ggml_gallocr_alloc_graph(galloc: ggml_gallocr_t, graph: *mut ggml_cgraph) -> bool;
}
unsafe extern "C" {
    pub fn ggml_gallocr_get_buffer_size(
        galloc: ggml_gallocr_t,
        buffer_id: core::ffi::c_int,
    ) -> usize;
}
unsafe extern "C" {
    pub fn ggml_backend_alloc_ctx_tensors_from_buft(
        ctx: *mut ggml_context,
        buft: ggml_backend_buffer_type_t,
    ) -> *mut ggml_backend_buffer;
}
unsafe extern "C" {
    pub fn ggml_backend_alloc_ctx_tensors(
        ctx: *mut ggml_context,
        backend: ggml_backend_t,
    ) -> *mut ggml_backend_buffer;
}

unsafe extern "C" {
    pub fn ggml_backend_buft_alloc_buffer(
        buft: ggml_backend_buffer_type_t,
        size: usize,
    ) -> ggml_backend_buffer_t;
}
unsafe extern "C" {
    pub fn ggml_backend_buft_get_alignment(buft: ggml_backend_buffer_type_t) -> usize;
}
unsafe extern "C" {
    pub fn ggml_backend_buft_get_max_size(buft: ggml_backend_buffer_type_t) -> usize;
}
unsafe extern "C" {
    pub fn ggml_backend_buft_get_alloc_size(
        buft: ggml_backend_buffer_type_t,
        tensor: *mut ggml_tensor,
    ) -> usize;
}
unsafe extern "C" {
    pub fn ggml_backend_buft_is_host(buft: ggml_backend_buffer_type_t) -> bool;
}
unsafe extern "C" {
    pub fn ggml_backend_buft_get_device(buft: ggml_backend_buffer_type_t) -> ggml_backend_dev_t;
}

unsafe extern "C" {
    pub fn ggml_backend_buffer_name(buffer: ggml_backend_buffer_t) -> *const core::ffi::c_char;
}
unsafe extern "C" {
    pub fn ggml_backend_buffer_free(buffer: ggml_backend_buffer_t);
}
unsafe extern "C" {
    pub fn ggml_backend_buffer_get_base(buffer: ggml_backend_buffer_t) -> *mut core::ffi::c_void;
}
unsafe extern "C" {
    pub fn ggml_backend_buffer_get_size(buffer: ggml_backend_buffer_t) -> usize;
}
unsafe extern "C" {
    pub fn ggml_backend_buffer_init_tensor(buffer: ggml_backend_buffer_t, tensor: *mut ggml_tensor);
}
unsafe extern "C" {
    pub fn ggml_backend_buffer_get_alignment(buffer: ggml_backend_buffer_t) -> usize;
}
unsafe extern "C" {
    pub fn ggml_backend_buffer_get_max_size(buffer: ggml_backend_buffer_t) -> usize;
}
unsafe extern "C" {
    pub fn ggml_backend_buffer_get_alloc_size(
        buffer: ggml_backend_buffer_t,
        tensor: *mut ggml_tensor,
    ) -> usize;
}
unsafe extern "C" {
    pub fn ggml_backend_buffer_clear(buffer: ggml_backend_buffer_t, value: u8);
}
unsafe extern "C" {
    pub fn ggml_backend_buffer_is_host(buffer: ggml_backend_buffer_t) -> bool;
}
unsafe extern "C" {
    pub fn ggml_backend_buffer_set_usage(
        buffer: ggml_backend_buffer_t,
        usage: ggml_backend_buffer_usage,
    );
}
unsafe extern "C" {
    pub fn ggml_backend_buffer_get_usage(
        buffer: ggml_backend_buffer_t,
    ) -> ggml_backend_buffer_usage;
}
unsafe extern "C" {
    pub fn ggml_backend_buffer_get_type(
        buffer: ggml_backend_buffer_t,
    ) -> ggml_backend_buffer_type_t;
}
unsafe extern "C" {
    pub fn ggml_backend_buffer_reset(buffer: ggml_backend_buffer_t);
}
unsafe extern "C" {
    pub fn ggml_backend_tensor_copy(src: *mut ggml_tensor, dst: *mut ggml_tensor);
}
unsafe extern "C" {
    pub fn ggml_backend_guid(backend: ggml_backend_t) -> ggml_guid_t;
}
unsafe extern "C" {
    pub fn ggml_backend_name(backend: ggml_backend_t) -> *const core::ffi::c_char;
}
unsafe extern "C" {
    pub fn ggml_backend_free(backend: ggml_backend_t);
}
unsafe extern "C" {
    pub fn ggml_backend_get_default_buffer_type(
        backend: ggml_backend_t,
    ) -> ggml_backend_buffer_type_t;
}
unsafe extern "C" {
    pub fn ggml_backend_alloc_buffer(backend: ggml_backend_t, size: usize)
    -> ggml_backend_buffer_t;
}
unsafe extern "C" {
    pub fn ggml_backend_get_alignment(backend: ggml_backend_t) -> usize;
}
unsafe extern "C" {
    pub fn ggml_backend_get_max_size(backend: ggml_backend_t) -> usize;
}
unsafe extern "C" {
    pub fn ggml_backend_tensor_set_async(
        backend: ggml_backend_t,
        tensor: *mut ggml_tensor,
        data: *const core::ffi::c_void,
        offset: usize,
        size: usize,
    );
}
unsafe extern "C" {
    pub fn ggml_backend_tensor_get_async(
        backend: ggml_backend_t,
        tensor: *const ggml_tensor,
        data: *mut core::ffi::c_void,
        offset: usize,
        size: usize,
    );
}
unsafe extern "C" {
    pub fn ggml_backend_tensor_set(
        tensor: *mut ggml_tensor,
        data: *const core::ffi::c_void,
        offset: usize,
        size: usize,
    );
}
unsafe extern "C" {
    pub fn ggml_backend_tensor_get(
        tensor: *const ggml_tensor,
        data: *mut core::ffi::c_void,
        offset: usize,
        size: usize,
    );
}
unsafe extern "C" {
    pub fn ggml_backend_tensor_memset(
        tensor: *mut ggml_tensor,
        value: u8,
        offset: usize,
        size: usize,
    );
}
unsafe extern "C" {
    pub fn ggml_backend_synchronize(backend: ggml_backend_t);
}
unsafe extern "C" {
    pub fn ggml_backend_graph_plan_create(
        backend: ggml_backend_t,
        cgraph: *mut ggml_cgraph,
    ) -> ggml_backend_graph_plan_t;
}
unsafe extern "C" {
    pub fn ggml_backend_graph_plan_free(backend: ggml_backend_t, plan: ggml_backend_graph_plan_t);
}
unsafe extern "C" {
    pub fn ggml_backend_graph_plan_compute(
        backend: ggml_backend_t,
        plan: ggml_backend_graph_plan_t,
    ) -> ggml_status;
}
unsafe extern "C" {
    pub fn ggml_backend_graph_compute(
        backend: ggml_backend_t,
        cgraph: *mut ggml_cgraph,
    ) -> ggml_status;
}
unsafe extern "C" {
    pub fn ggml_backend_graph_compute_async(
        backend: ggml_backend_t,
        cgraph: *mut ggml_cgraph,
    ) -> ggml_status;
}
unsafe extern "C" {
    pub fn ggml_backend_supports_op(backend: ggml_backend_t, op: *const ggml_tensor) -> bool;
}
unsafe extern "C" {
    pub fn ggml_backend_supports_buft(
        backend: ggml_backend_t,
        buft: ggml_backend_buffer_type_t,
    ) -> bool;
}
unsafe extern "C" {
    pub fn ggml_backend_offload_op(backend: ggml_backend_t, op: *const ggml_tensor) -> bool;
}
unsafe extern "C" {
    pub fn ggml_backend_tensor_copy_async(
        backend_src: ggml_backend_t,
        backend_dst: ggml_backend_t,
        src: *mut ggml_tensor,
        dst: *mut ggml_tensor,
    );
}
unsafe extern "C" {
    pub fn ggml_backend_get_device(backend: ggml_backend_t) -> ggml_backend_dev_t;
}
unsafe extern "C" {
    pub fn ggml_backend_event_new(device: ggml_backend_dev_t) -> ggml_backend_event_t;
}
unsafe extern "C" {
    pub fn ggml_backend_event_free(event: ggml_backend_event_t);
}
unsafe extern "C" {
    pub fn ggml_backend_event_record(event: ggml_backend_event_t, backend: ggml_backend_t);
}
unsafe extern "C" {
    pub fn ggml_backend_event_synchronize(event: ggml_backend_event_t);
}
unsafe extern "C" {
    pub fn ggml_backend_event_wait(backend: ggml_backend_t, event: ggml_backend_event_t);
}
pub const ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU: ggml_backend_dev_type = 0;
pub const ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU: ggml_backend_dev_type = 1;
pub const ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_ACCEL: ggml_backend_dev_type = 2;
pub type ggml_backend_dev_type = core::ffi::c_uint;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_backend_dev_caps {
    pub async_: bool,
    pub host_buffer: bool,
    pub buffer_from_host_ptr: bool,
    pub events: bool,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of ggml_backend_dev_caps"][::std::mem::size_of::<ggml_backend_dev_caps>() - 4usize];
    ["Alignment of ggml_backend_dev_caps"]
        [::std::mem::align_of::<ggml_backend_dev_caps>() - 1usize];
    ["Offset of field: ggml_backend_dev_caps::async_"]
        [::std::mem::offset_of!(ggml_backend_dev_caps, async_) - 0usize];
    ["Offset of field: ggml_backend_dev_caps::host_buffer"]
        [::std::mem::offset_of!(ggml_backend_dev_caps, host_buffer) - 1usize];
    ["Offset of field: ggml_backend_dev_caps::buffer_from_host_ptr"]
        [::std::mem::offset_of!(ggml_backend_dev_caps, buffer_from_host_ptr) - 2usize];
    ["Offset of field: ggml_backend_dev_caps::events"]
        [::std::mem::offset_of!(ggml_backend_dev_caps, events) - 3usize];
};
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_backend_dev_props {
    pub name: *const core::ffi::c_char,
    pub description: *const core::ffi::c_char,
    pub memory_free: usize,
    pub memory_total: usize,
    pub type_: ggml_backend_dev_type,
    pub caps: ggml_backend_dev_caps,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of ggml_backend_dev_props"][::std::mem::size_of::<ggml_backend_dev_props>() - 40usize];
    ["Alignment of ggml_backend_dev_props"]
        [::std::mem::align_of::<ggml_backend_dev_props>() - 8usize];
    ["Offset of field: ggml_backend_dev_props::name"]
        [::std::mem::offset_of!(ggml_backend_dev_props, name) - 0usize];
    ["Offset of field: ggml_backend_dev_props::description"]
        [::std::mem::offset_of!(ggml_backend_dev_props, description) - 8usize];
    ["Offset of field: ggml_backend_dev_props::memory_free"]
        [::std::mem::offset_of!(ggml_backend_dev_props, memory_free) - 16usize];
    ["Offset of field: ggml_backend_dev_props::memory_total"]
        [::std::mem::offset_of!(ggml_backend_dev_props, memory_total) - 24usize];
    ["Offset of field: ggml_backend_dev_props::type_"]
        [::std::mem::offset_of!(ggml_backend_dev_props, type_) - 32usize];
    ["Offset of field: ggml_backend_dev_props::caps"]
        [::std::mem::offset_of!(ggml_backend_dev_props, caps) - 36usize];
};
unsafe extern "C" {
    pub fn ggml_backend_dev_name(device: ggml_backend_dev_t) -> *const core::ffi::c_char;
}
unsafe extern "C" {
    pub fn ggml_backend_dev_description(device: ggml_backend_dev_t) -> *const core::ffi::c_char;
}
unsafe extern "C" {
    pub fn ggml_backend_dev_memory(device: ggml_backend_dev_t, free: *mut usize, total: *mut usize);
}
unsafe extern "C" {
    pub fn ggml_backend_dev_type(device: ggml_backend_dev_t) -> ggml_backend_dev_type;
}
unsafe extern "C" {
    pub fn ggml_backend_dev_get_props(
        device: ggml_backend_dev_t,
        props: *mut ggml_backend_dev_props,
    );
}
unsafe extern "C" {
    pub fn ggml_backend_dev_backend_reg(device: ggml_backend_dev_t) -> ggml_backend_reg_t;
}
unsafe extern "C" {
    pub fn ggml_backend_dev_init(
        device: ggml_backend_dev_t,
        params: *const core::ffi::c_char,
    ) -> ggml_backend_t;
}
unsafe extern "C" {
    pub fn ggml_backend_dev_buffer_type(device: ggml_backend_dev_t) -> ggml_backend_buffer_type_t;
}
unsafe extern "C" {
    pub fn ggml_backend_dev_host_buffer_type(
        device: ggml_backend_dev_t,
    ) -> ggml_backend_buffer_type_t;
}
unsafe extern "C" {
    pub fn ggml_backend_dev_buffer_from_host_ptr(
        device: ggml_backend_dev_t,
        ptr: *mut core::ffi::c_void,
        size: usize,
        max_tensor_size: usize,
    ) -> ggml_backend_buffer_t;
}
unsafe extern "C" {
    pub fn ggml_backend_dev_supports_op(device: ggml_backend_dev_t, op: *const ggml_tensor)
    -> bool;
}
unsafe extern "C" {
    pub fn ggml_backend_dev_supports_buft(
        device: ggml_backend_dev_t,
        buft: ggml_backend_buffer_type_t,
    ) -> bool;
}
unsafe extern "C" {
    pub fn ggml_backend_dev_offload_op(device: ggml_backend_dev_t, op: *const ggml_tensor) -> bool;
}
unsafe extern "C" {
    pub fn ggml_backend_reg_name(reg: ggml_backend_reg_t) -> *const core::ffi::c_char;
}
unsafe extern "C" {
    pub fn ggml_backend_reg_dev_count(reg: ggml_backend_reg_t) -> usize;
}
unsafe extern "C" {
    pub fn ggml_backend_reg_dev_get(reg: ggml_backend_reg_t, index: usize) -> ggml_backend_dev_t;
}
unsafe extern "C" {
    pub fn ggml_backend_reg_get_proc_address(
        reg: ggml_backend_reg_t,
        name: *const core::ffi::c_char,
    ) -> *mut core::ffi::c_void;
}
pub type ggml_backend_split_buffer_type_t = core::option::Option<
    unsafe extern "C" fn(
        main_device: core::ffi::c_int,
        tensor_split: *const f32,
    ) -> ggml_backend_buffer_type_t,
>;
pub type ggml_backend_set_n_threads_t = core::option::Option<
    unsafe extern "C" fn(backend: ggml_backend_t, n_threads: core::ffi::c_int),
>;
pub type ggml_backend_dev_get_extra_bufts_t = core::option::Option<
    unsafe extern "C" fn(device: ggml_backend_dev_t) -> *mut ggml_backend_buffer_type_t,
>;
pub type ggml_backend_set_abort_callback_t = core::option::Option<
    unsafe extern "C" fn(
        backend: ggml_backend_t,
        abort_callback: ggml_abort_callback,
        abort_callback_data: *mut core::ffi::c_void,
    ),
>;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_backend_feature {
    pub name: *const core::ffi::c_char,
    pub value: *const core::ffi::c_char,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of ggml_backend_feature"][::std::mem::size_of::<ggml_backend_feature>() - 16usize];
    ["Alignment of ggml_backend_feature"][::std::mem::align_of::<ggml_backend_feature>() - 8usize];
    ["Offset of field: ggml_backend_feature::name"]
        [::std::mem::offset_of!(ggml_backend_feature, name) - 0usize];
    ["Offset of field: ggml_backend_feature::value"]
        [::std::mem::offset_of!(ggml_backend_feature, value) - 8usize];
};
pub type ggml_backend_get_features_t = core::option::Option<
    unsafe extern "C" fn(reg: ggml_backend_reg_t) -> *mut ggml_backend_feature,
>;
unsafe extern "C" {
    pub fn ggml_backend_device_register(device: ggml_backend_dev_t);
}
unsafe extern "C" {
    pub fn ggml_backend_reg_count() -> usize;
}
unsafe extern "C" {
    pub fn ggml_backend_reg_get(index: usize) -> ggml_backend_reg_t;
}
unsafe extern "C" {
    pub fn ggml_backend_reg_by_name(name: *const core::ffi::c_char) -> ggml_backend_reg_t;
}
unsafe extern "C" {
    pub fn ggml_backend_dev_count() -> usize;
}
unsafe extern "C" {
    pub fn ggml_backend_dev_get(index: usize) -> ggml_backend_dev_t;
}
unsafe extern "C" {
    pub fn ggml_backend_dev_by_name(name: *const core::ffi::c_char) -> ggml_backend_dev_t;
}
unsafe extern "C" {
    pub fn ggml_backend_dev_by_type(type_: ggml_backend_dev_type) -> ggml_backend_dev_t;
}
unsafe extern "C" {
    pub fn ggml_backend_init_by_name(
        name: *const core::ffi::c_char,
        params: *const core::ffi::c_char,
    ) -> ggml_backend_t;
}
unsafe extern "C" {
    pub fn ggml_backend_init_by_type(
        type_: ggml_backend_dev_type,
        params: *const core::ffi::c_char,
    ) -> ggml_backend_t;
}
unsafe extern "C" {
    pub fn ggml_backend_init_best() -> ggml_backend_t;
}
unsafe extern "C" {
    pub fn ggml_backend_load(path: *const core::ffi::c_char) -> ggml_backend_reg_t;
}
unsafe extern "C" {
    pub fn ggml_backend_unload(reg: ggml_backend_reg_t);
}
unsafe extern "C" {
    pub fn ggml_backend_load_all();
}
unsafe extern "C" {
    pub fn ggml_backend_load_all_from_path(dir_path: *const core::ffi::c_char);
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_backend_sched {
    _unused: [u8; 0],
}
pub type ggml_backend_sched_t = *mut ggml_backend_sched;
pub type ggml_backend_sched_eval_callback = core::option::Option<
    unsafe extern "C" fn(t: *mut ggml_tensor, ask: bool, user_data: *mut core::ffi::c_void) -> bool,
>;
unsafe extern "C" {
    pub fn ggml_backend_sched_new(
        backends: *mut ggml_backend_t,
        bufts: *mut ggml_backend_buffer_type_t,
        n_backends: core::ffi::c_int,
        graph_size: usize,
        parallel: bool,
    ) -> ggml_backend_sched_t;
}
unsafe extern "C" {
    pub fn ggml_backend_sched_free(sched: ggml_backend_sched_t);
}
unsafe extern "C" {
    pub fn ggml_backend_sched_reserve(
        sched: ggml_backend_sched_t,
        measure_graph: *mut ggml_cgraph,
    ) -> bool;
}
unsafe extern "C" {
    pub fn ggml_backend_sched_get_n_backends(sched: ggml_backend_sched_t) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_backend_sched_get_backend(
        sched: ggml_backend_sched_t,
        i: core::ffi::c_int,
    ) -> ggml_backend_t;
}
unsafe extern "C" {
    pub fn ggml_backend_sched_get_n_splits(sched: ggml_backend_sched_t) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_backend_sched_get_n_copies(sched: ggml_backend_sched_t) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_backend_sched_get_buffer_size(
        sched: ggml_backend_sched_t,
        backend: ggml_backend_t,
    ) -> usize;
}
unsafe extern "C" {
    pub fn ggml_backend_sched_set_tensor_backend(
        sched: ggml_backend_sched_t,
        node: *mut ggml_tensor,
        backend: ggml_backend_t,
    );
}
unsafe extern "C" {
    pub fn ggml_backend_sched_get_tensor_backend(
        sched: ggml_backend_sched_t,
        node: *mut ggml_tensor,
    ) -> ggml_backend_t;
}
unsafe extern "C" {
    pub fn ggml_backend_sched_alloc_graph(
        sched: ggml_backend_sched_t,
        graph: *mut ggml_cgraph,
    ) -> bool;
}
unsafe extern "C" {
    pub fn ggml_backend_sched_graph_compute(
        sched: ggml_backend_sched_t,
        graph: *mut ggml_cgraph,
    ) -> ggml_status;
}
unsafe extern "C" {
    pub fn ggml_backend_sched_graph_compute_async(
        sched: ggml_backend_sched_t,
        graph: *mut ggml_cgraph,
    ) -> ggml_status;
}
unsafe extern "C" {
    pub fn ggml_backend_sched_synchronize(sched: ggml_backend_sched_t);
}
unsafe extern "C" {
    pub fn ggml_backend_sched_reset(sched: ggml_backend_sched_t);
}
unsafe extern "C" {
    pub fn ggml_backend_sched_set_eval_callback(
        sched: ggml_backend_sched_t,
        callback: ggml_backend_sched_eval_callback,
        user_data: *mut core::ffi::c_void,
    );
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_backend_graph_copy {
    pub buffer: ggml_backend_buffer_t,
    pub ctx_allocated: *mut ggml_context,
    pub ctx_unallocated: *mut ggml_context,
    pub graph: *mut ggml_cgraph,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of ggml_backend_graph_copy"][::std::mem::size_of::<ggml_backend_graph_copy>() - 32usize];
    ["Alignment of ggml_backend_graph_copy"]
        [::std::mem::align_of::<ggml_backend_graph_copy>() - 8usize];
    ["Offset of field: ggml_backend_graph_copy::buffer"]
        [::std::mem::offset_of!(ggml_backend_graph_copy, buffer) - 0usize];
    ["Offset of field: ggml_backend_graph_copy::ctx_allocated"]
        [::std::mem::offset_of!(ggml_backend_graph_copy, ctx_allocated) - 8usize];
    ["Offset of field: ggml_backend_graph_copy::ctx_unallocated"]
        [::std::mem::offset_of!(ggml_backend_graph_copy, ctx_unallocated) - 16usize];
    ["Offset of field: ggml_backend_graph_copy::graph"]
        [::std::mem::offset_of!(ggml_backend_graph_copy, graph) - 24usize];
};
unsafe extern "C" {
    pub fn ggml_backend_graph_copy(
        backend: ggml_backend_t,
        graph: *mut ggml_cgraph,
    ) -> ggml_backend_graph_copy;
}
unsafe extern "C" {
    pub fn ggml_backend_graph_copy_free(copy: ggml_backend_graph_copy);
}
pub type ggml_backend_eval_callback = core::option::Option<
    unsafe extern "C" fn(
        node_index: core::ffi::c_int,
        t1: *mut ggml_tensor,
        t2: *mut ggml_tensor,
        user_data: *mut core::ffi::c_void,
    ) -> bool,
>;
unsafe extern "C" {
    pub fn ggml_backend_compare_graph_backend(
        backend1: ggml_backend_t,
        backend2: ggml_backend_t,
        graph: *mut ggml_cgraph,
        callback: ggml_backend_eval_callback,
        user_data: *mut core::ffi::c_void,
    ) -> bool;
}
unsafe extern "C" {
    pub fn ggml_backend_tensor_alloc(
        buffer: ggml_backend_buffer_t,
        tensor: *mut ggml_tensor,
        addr: *mut core::ffi::c_void,
    );
}
unsafe extern "C" {
    pub fn ggml_backend_view_init(tensor: *mut ggml_tensor);
}
unsafe extern "C" {
    pub fn ggml_backend_cpu_buffer_from_ptr(
        ptr: *mut core::ffi::c_void,
        size: usize,
    ) -> ggml_backend_buffer_t;
}
unsafe extern "C" {
    pub fn ggml_backend_cpu_buffer_type() -> ggml_backend_buffer_type_t;
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_cplan {
    pub work_size: usize,
    pub work_data: *mut u8,
    pub n_threads: core::ffi::c_int,
    pub threadpool: *mut ggml_threadpool,
    pub abort_callback: ggml_abort_callback,
    pub abort_callback_data: *mut core::ffi::c_void,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of ggml_cplan"][::std::mem::size_of::<ggml_cplan>() - 48usize];
    ["Alignment of ggml_cplan"][::std::mem::align_of::<ggml_cplan>() - 8usize];
    ["Offset of field: ggml_cplan::work_size"]
        [::std::mem::offset_of!(ggml_cplan, work_size) - 0usize];
    ["Offset of field: ggml_cplan::work_data"]
        [::std::mem::offset_of!(ggml_cplan, work_data) - 8usize];
    ["Offset of field: ggml_cplan::n_threads"]
        [::std::mem::offset_of!(ggml_cplan, n_threads) - 16usize];
    ["Offset of field: ggml_cplan::threadpool"]
        [::std::mem::offset_of!(ggml_cplan, threadpool) - 24usize];
    ["Offset of field: ggml_cplan::abort_callback"]
        [::std::mem::offset_of!(ggml_cplan, abort_callback) - 32usize];
    ["Offset of field: ggml_cplan::abort_callback_data"]
        [::std::mem::offset_of!(ggml_cplan, abort_callback_data) - 40usize];
};
pub const ggml_numa_strategy_GGML_NUMA_STRATEGY_DISABLED: ggml_numa_strategy = 0;
pub const ggml_numa_strategy_GGML_NUMA_STRATEGY_DISTRIBUTE: ggml_numa_strategy = 1;
pub const ggml_numa_strategy_GGML_NUMA_STRATEGY_ISOLATE: ggml_numa_strategy = 2;
pub const ggml_numa_strategy_GGML_NUMA_STRATEGY_NUMACTL: ggml_numa_strategy = 3;
pub const ggml_numa_strategy_GGML_NUMA_STRATEGY_MIRROR: ggml_numa_strategy = 4;
pub const ggml_numa_strategy_GGML_NUMA_STRATEGY_COUNT: ggml_numa_strategy = 5;
pub type ggml_numa_strategy = core::ffi::c_uint;
unsafe extern "C" {
    pub fn ggml_numa_init(numa: ggml_numa_strategy);
}
unsafe extern "C" {
    pub fn ggml_is_numa() -> bool;
}
unsafe extern "C" {
    pub fn ggml_new_i32(ctx: *mut ggml_context, value: i32) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_new_f32(ctx: *mut ggml_context, value: f32) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_set_i32(tensor: *mut ggml_tensor, value: i32) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_set_f32(tensor: *mut ggml_tensor, value: f32) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_get_i32_1d(tensor: *const ggml_tensor, i: core::ffi::c_int) -> i32;
}
unsafe extern "C" {
    pub fn ggml_set_i32_1d(tensor: *const ggml_tensor, i: core::ffi::c_int, value: i32);
}
unsafe extern "C" {
    pub fn ggml_get_i32_nd(
        tensor: *const ggml_tensor,
        i0: core::ffi::c_int,
        i1: core::ffi::c_int,
        i2: core::ffi::c_int,
        i3: core::ffi::c_int,
    ) -> i32;
}
unsafe extern "C" {
    pub fn ggml_set_i32_nd(
        tensor: *const ggml_tensor,
        i0: core::ffi::c_int,
        i1: core::ffi::c_int,
        i2: core::ffi::c_int,
        i3: core::ffi::c_int,
        value: i32,
    );
}
unsafe extern "C" {
    pub fn ggml_get_f32_1d(tensor: *const ggml_tensor, i: core::ffi::c_int) -> f32;
}
unsafe extern "C" {
    pub fn ggml_set_f32_1d(tensor: *const ggml_tensor, i: core::ffi::c_int, value: f32);
}
unsafe extern "C" {
    pub fn ggml_get_f32_nd(
        tensor: *const ggml_tensor,
        i0: core::ffi::c_int,
        i1: core::ffi::c_int,
        i2: core::ffi::c_int,
        i3: core::ffi::c_int,
    ) -> f32;
}
unsafe extern "C" {
    pub fn ggml_set_f32_nd(
        tensor: *const ggml_tensor,
        i0: core::ffi::c_int,
        i1: core::ffi::c_int,
        i2: core::ffi::c_int,
        i3: core::ffi::c_int,
        value: f32,
    );
}
unsafe extern "C" {
    pub fn ggml_threadpool_new(params: *mut ggml_threadpool_params) -> *mut ggml_threadpool;
}
unsafe extern "C" {
    pub fn ggml_threadpool_free(threadpool: *mut ggml_threadpool);
}
unsafe extern "C" {
    pub fn ggml_threadpool_get_n_threads(threadpool: *mut ggml_threadpool) -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_threadpool_pause(threadpool: *mut ggml_threadpool);
}
unsafe extern "C" {
    pub fn ggml_threadpool_resume(threadpool: *mut ggml_threadpool);
}
unsafe extern "C" {
    pub fn ggml_graph_plan(
        cgraph: *const ggml_cgraph,
        n_threads: core::ffi::c_int,
        threadpool: *mut ggml_threadpool,
    ) -> ggml_cplan;
}
unsafe extern "C" {
    pub fn ggml_graph_compute(cgraph: *mut ggml_cgraph, cplan: *mut ggml_cplan) -> ggml_status;
}
unsafe extern "C" {
    pub fn ggml_graph_compute_with_ctx(
        ctx: *mut ggml_context,
        cgraph: *mut ggml_cgraph,
        n_threads: core::ffi::c_int,
    ) -> ggml_status;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_sse3() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_ssse3() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_avx() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_avx_vnni() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_avx2() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_f16c() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_fma() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_avx512() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_avx512_vbmi() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_avx512_vnni() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_avx512_bf16() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_amx_int8() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_neon() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_arm_fma() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_fp16_va() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_dotprod() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_matmul_int8() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_sve() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_get_sve_cnt() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_riscv_v() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_vsx() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_wasm_simd() -> core::ffi::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_llamafile() -> core::ffi::c_int;
}
pub type ggml_vec_dot_t = core::option::Option<
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
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_type_traits_cpu {
    pub from_float: ggml_from_float_t,
    pub vec_dot: ggml_vec_dot_t,
    pub vec_dot_type: ggml_type,
    pub nrows: i64,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of ggml_type_traits_cpu"][::std::mem::size_of::<ggml_type_traits_cpu>() - 32usize];
    ["Alignment of ggml_type_traits_cpu"][::std::mem::align_of::<ggml_type_traits_cpu>() - 8usize];
    ["Offset of field: ggml_type_traits_cpu::from_float"]
        [::std::mem::offset_of!(ggml_type_traits_cpu, from_float) - 0usize];
    ["Offset of field: ggml_type_traits_cpu::vec_dot"]
        [::std::mem::offset_of!(ggml_type_traits_cpu, vec_dot) - 8usize];
    ["Offset of field: ggml_type_traits_cpu::vec_dot_type"]
        [::std::mem::offset_of!(ggml_type_traits_cpu, vec_dot_type) - 16usize];
    ["Offset of field: ggml_type_traits_cpu::nrows"]
        [::std::mem::offset_of!(ggml_type_traits_cpu, nrows) - 24usize];
};
unsafe extern "C" {
    pub fn ggml_get_type_traits_cpu(type_: ggml_type) -> *const ggml_type_traits_cpu;
}
unsafe extern "C" {
    pub fn ggml_cpu_init();
}
unsafe extern "C" {
    pub fn ggml_backend_cpu_init() -> ggml_backend_t;
}
unsafe extern "C" {
    pub fn ggml_backend_is_cpu(backend: ggml_backend_t) -> bool;
}
unsafe extern "C" {
    pub fn ggml_backend_cpu_set_n_threads(backend_cpu: ggml_backend_t, n_threads: core::ffi::c_int);
}
unsafe extern "C" {
    pub fn ggml_backend_cpu_set_threadpool(
        backend_cpu: ggml_backend_t,
        threadpool: ggml_threadpool_t,
    );
}
unsafe extern "C" {
    pub fn ggml_backend_cpu_set_abort_callback(
        backend_cpu: ggml_backend_t,
        abort_callback: ggml_abort_callback,
        abort_callback_data: *mut core::ffi::c_void,
    );
}
unsafe extern "C" {
    pub fn ggml_backend_cpu_reg() -> ggml_backend_reg_t;
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct whisper_context {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct whisper_state {
    _unused: [u8; 0],
}
pub type whisper_pos = i32;
pub type whisper_token = i32;
pub type whisper_seq_id = i32;
pub const whisper_alignment_heads_preset_WHISPER_AHEADS_NONE: whisper_alignment_heads_preset = 0;
pub const whisper_alignment_heads_preset_WHISPER_AHEADS_N_TOP_MOST: whisper_alignment_heads_preset =
    1;
pub const whisper_alignment_heads_preset_WHISPER_AHEADS_CUSTOM: whisper_alignment_heads_preset = 2;
pub const whisper_alignment_heads_preset_WHISPER_AHEADS_TINY_EN: whisper_alignment_heads_preset = 3;
pub const whisper_alignment_heads_preset_WHISPER_AHEADS_TINY: whisper_alignment_heads_preset = 4;
pub const whisper_alignment_heads_preset_WHISPER_AHEADS_BASE_EN: whisper_alignment_heads_preset = 5;
pub const whisper_alignment_heads_preset_WHISPER_AHEADS_BASE: whisper_alignment_heads_preset = 6;
pub const whisper_alignment_heads_preset_WHISPER_AHEADS_SMALL_EN: whisper_alignment_heads_preset =
    7;
pub const whisper_alignment_heads_preset_WHISPER_AHEADS_SMALL: whisper_alignment_heads_preset = 8;
pub const whisper_alignment_heads_preset_WHISPER_AHEADS_MEDIUM_EN: whisper_alignment_heads_preset =
    9;
pub const whisper_alignment_heads_preset_WHISPER_AHEADS_MEDIUM: whisper_alignment_heads_preset = 10;
pub const whisper_alignment_heads_preset_WHISPER_AHEADS_LARGE_V1: whisper_alignment_heads_preset =
    11;
pub const whisper_alignment_heads_preset_WHISPER_AHEADS_LARGE_V2: whisper_alignment_heads_preset =
    12;
pub const whisper_alignment_heads_preset_WHISPER_AHEADS_LARGE_V3: whisper_alignment_heads_preset =
    13;
pub const whisper_alignment_heads_preset_WHISPER_AHEADS_LARGE_V3_TURBO:
    whisper_alignment_heads_preset = 14;
pub type whisper_alignment_heads_preset = core::ffi::c_uint;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct whisper_ahead {
    pub n_text_layer: core::ffi::c_int,
    pub n_head: core::ffi::c_int,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of whisper_ahead"][::std::mem::size_of::<whisper_ahead>() - 8usize];
    ["Alignment of whisper_ahead"][::std::mem::align_of::<whisper_ahead>() - 4usize];
    ["Offset of field: whisper_ahead::n_text_layer"]
        [::std::mem::offset_of!(whisper_ahead, n_text_layer) - 0usize];
    ["Offset of field: whisper_ahead::n_head"]
        [::std::mem::offset_of!(whisper_ahead, n_head) - 4usize];
};
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct whisper_aheads {
    pub n_heads: usize,
    pub heads: *const whisper_ahead,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of whisper_aheads"][::std::mem::size_of::<whisper_aheads>() - 16usize];
    ["Alignment of whisper_aheads"][::std::mem::align_of::<whisper_aheads>() - 8usize];
    ["Offset of field: whisper_aheads::n_heads"]
        [::std::mem::offset_of!(whisper_aheads, n_heads) - 0usize];
    ["Offset of field: whisper_aheads::heads"]
        [::std::mem::offset_of!(whisper_aheads, heads) - 8usize];
};
#[repr(C)]
#[derive(Debug, Copy, Clone)]
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
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of whisper_context_params"][::std::mem::size_of::<whisper_context_params>() - 48usize];
    ["Alignment of whisper_context_params"]
        [::std::mem::align_of::<whisper_context_params>() - 8usize];
    ["Offset of field: whisper_context_params::use_gpu"]
        [::std::mem::offset_of!(whisper_context_params, use_gpu) - 0usize];
    ["Offset of field: whisper_context_params::flash_attn"]
        [::std::mem::offset_of!(whisper_context_params, flash_attn) - 1usize];
    ["Offset of field: whisper_context_params::gpu_device"]
        [::std::mem::offset_of!(whisper_context_params, gpu_device) - 4usize];
    ["Offset of field: whisper_context_params::dtw_token_timestamps"]
        [::std::mem::offset_of!(whisper_context_params, dtw_token_timestamps) - 8usize];
    ["Offset of field: whisper_context_params::dtw_aheads_preset"]
        [::std::mem::offset_of!(whisper_context_params, dtw_aheads_preset) - 12usize];
    ["Offset of field: whisper_context_params::dtw_n_top"]
        [::std::mem::offset_of!(whisper_context_params, dtw_n_top) - 16usize];
    ["Offset of field: whisper_context_params::dtw_aheads"]
        [::std::mem::offset_of!(whisper_context_params, dtw_aheads) - 24usize];
    ["Offset of field: whisper_context_params::dtw_mem_size"]
        [::std::mem::offset_of!(whisper_context_params, dtw_mem_size) - 40usize];
};
#[repr(C)]
#[derive(Debug, Copy, Clone)]
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

pub const whisper_gretype_WHISPER_GRETYPE_END: whisper_gretype = 0;
pub const whisper_gretype_WHISPER_GRETYPE_ALT: whisper_gretype = 1;
pub const whisper_gretype_WHISPER_GRETYPE_RULE_REF: whisper_gretype = 2;
pub const whisper_gretype_WHISPER_GRETYPE_CHAR: whisper_gretype = 3;
pub const whisper_gretype_WHISPER_GRETYPE_CHAR_NOT: whisper_gretype = 4;
pub const whisper_gretype_WHISPER_GRETYPE_CHAR_RNG_UPPER: whisper_gretype = 5;
pub const whisper_gretype_WHISPER_GRETYPE_CHAR_ALT: whisper_gretype = 6;
pub type whisper_gretype = core::ffi::c_uint;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct whisper_grammar_element {
    pub type_: whisper_gretype,
    pub value: u32,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of whisper_grammar_element"][::std::mem::size_of::<whisper_grammar_element>() - 8usize];
    ["Alignment of whisper_grammar_element"]
        [::std::mem::align_of::<whisper_grammar_element>() - 4usize];
    ["Offset of field: whisper_grammar_element::type_"]
        [::std::mem::offset_of!(whisper_grammar_element, type_) - 0usize];
    ["Offset of field: whisper_grammar_element::value"]
        [::std::mem::offset_of!(whisper_grammar_element, value) - 4usize];
};
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
#[derive(Debug, Copy, Clone)]
pub struct whisper_timings {
    pub sample_ms: f32,
    pub encode_ms: f32,
    pub decode_ms: f32,
    pub batchd_ms: f32,
    pub prompt_ms: f32,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of whisper_timings"][::std::mem::size_of::<whisper_timings>() - 20usize];
    ["Alignment of whisper_timings"][::std::mem::align_of::<whisper_timings>() - 4usize];
    ["Offset of field: whisper_timings::sample_ms"]
        [::std::mem::offset_of!(whisper_timings, sample_ms) - 0usize];
    ["Offset of field: whisper_timings::encode_ms"]
        [::std::mem::offset_of!(whisper_timings, encode_ms) - 4usize];
    ["Offset of field: whisper_timings::decode_ms"]
        [::std::mem::offset_of!(whisper_timings, decode_ms) - 8usize];
    ["Offset of field: whisper_timings::batchd_ms"]
        [::std::mem::offset_of!(whisper_timings, batchd_ms) - 12usize];
    ["Offset of field: whisper_timings::prompt_ms"]
        [::std::mem::offset_of!(whisper_timings, prompt_ms) - 16usize];
};
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
pub const whisper_sampling_strategy_WHISPER_SAMPLING_GREEDY: whisper_sampling_strategy = 0;
pub const whisper_sampling_strategy_WHISPER_SAMPLING_BEAM_SEARCH: whisper_sampling_strategy = 1;
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
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of whisper_full_params__bindgen_ty_1"]
        [::std::mem::size_of::<whisper_full_params__bindgen_ty_1>() - 4usize];
    ["Alignment of whisper_full_params__bindgen_ty_1"]
        [::std::mem::align_of::<whisper_full_params__bindgen_ty_1>() - 4usize];
    ["Offset of field: whisper_full_params__bindgen_ty_1::best_of"]
        [::std::mem::offset_of!(whisper_full_params__bindgen_ty_1, best_of) - 0usize];
};
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct whisper_full_params__bindgen_ty_2 {
    pub beam_size: core::ffi::c_int,
    pub patience: f32,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of whisper_full_params__bindgen_ty_2"]
        [::std::mem::size_of::<whisper_full_params__bindgen_ty_2>() - 8usize];
    ["Alignment of whisper_full_params__bindgen_ty_2"]
        [::std::mem::align_of::<whisper_full_params__bindgen_ty_2>() - 4usize];
    ["Offset of field: whisper_full_params__bindgen_ty_2::beam_size"]
        [::std::mem::offset_of!(whisper_full_params__bindgen_ty_2, beam_size) - 0usize];
    ["Offset of field: whisper_full_params__bindgen_ty_2::patience"]
        [::std::mem::offset_of!(whisper_full_params__bindgen_ty_2, patience) - 4usize];
};
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of whisper_full_params"][::std::mem::size_of::<whisper_full_params>() - 264usize];
    ["Alignment of whisper_full_params"][::std::mem::align_of::<whisper_full_params>() - 8usize];
    ["Offset of field: whisper_full_params::strategy"]
        [::std::mem::offset_of!(whisper_full_params, strategy) - 0usize];
    ["Offset of field: whisper_full_params::n_threads"]
        [::std::mem::offset_of!(whisper_full_params, n_threads) - 4usize];
    ["Offset of field: whisper_full_params::n_max_text_ctx"]
        [::std::mem::offset_of!(whisper_full_params, n_max_text_ctx) - 8usize];
    ["Offset of field: whisper_full_params::offset_ms"]
        [::std::mem::offset_of!(whisper_full_params, offset_ms) - 12usize];
    ["Offset of field: whisper_full_params::duration_ms"]
        [::std::mem::offset_of!(whisper_full_params, duration_ms) - 16usize];
    ["Offset of field: whisper_full_params::translate"]
        [::std::mem::offset_of!(whisper_full_params, translate) - 20usize];
    ["Offset of field: whisper_full_params::no_context"]
        [::std::mem::offset_of!(whisper_full_params, no_context) - 21usize];
    ["Offset of field: whisper_full_params::no_timestamps"]
        [::std::mem::offset_of!(whisper_full_params, no_timestamps) - 22usize];
    ["Offset of field: whisper_full_params::single_segment"]
        [::std::mem::offset_of!(whisper_full_params, single_segment) - 23usize];
    ["Offset of field: whisper_full_params::print_special"]
        [::std::mem::offset_of!(whisper_full_params, print_special) - 24usize];
    ["Offset of field: whisper_full_params::print_progress"]
        [::std::mem::offset_of!(whisper_full_params, print_progress) - 25usize];
    ["Offset of field: whisper_full_params::print_realtime"]
        [::std::mem::offset_of!(whisper_full_params, print_realtime) - 26usize];
    ["Offset of field: whisper_full_params::print_timestamps"]
        [::std::mem::offset_of!(whisper_full_params, print_timestamps) - 27usize];
    ["Offset of field: whisper_full_params::token_timestamps"]
        [::std::mem::offset_of!(whisper_full_params, token_timestamps) - 28usize];
    ["Offset of field: whisper_full_params::thold_pt"]
        [::std::mem::offset_of!(whisper_full_params, thold_pt) - 32usize];
    ["Offset of field: whisper_full_params::thold_ptsum"]
        [::std::mem::offset_of!(whisper_full_params, thold_ptsum) - 36usize];
    ["Offset of field: whisper_full_params::max_len"]
        [::std::mem::offset_of!(whisper_full_params, max_len) - 40usize];
    ["Offset of field: whisper_full_params::split_on_word"]
        [::std::mem::offset_of!(whisper_full_params, split_on_word) - 44usize];
    ["Offset of field: whisper_full_params::max_tokens"]
        [::std::mem::offset_of!(whisper_full_params, max_tokens) - 48usize];
    ["Offset of field: whisper_full_params::debug_mode"]
        [::std::mem::offset_of!(whisper_full_params, debug_mode) - 52usize];
    ["Offset of field: whisper_full_params::audio_ctx"]
        [::std::mem::offset_of!(whisper_full_params, audio_ctx) - 56usize];
    ["Offset of field: whisper_full_params::tdrz_enable"]
        [::std::mem::offset_of!(whisper_full_params, tdrz_enable) - 60usize];
    ["Offset of field: whisper_full_params::suppress_regex"]
        [::std::mem::offset_of!(whisper_full_params, suppress_regex) - 64usize];
    ["Offset of field: whisper_full_params::initial_prompt"]
        [::std::mem::offset_of!(whisper_full_params, initial_prompt) - 72usize];
    ["Offset of field: whisper_full_params::prompt_tokens"]
        [::std::mem::offset_of!(whisper_full_params, prompt_tokens) - 80usize];
    ["Offset of field: whisper_full_params::prompt_n_tokens"]
        [::std::mem::offset_of!(whisper_full_params, prompt_n_tokens) - 88usize];
    ["Offset of field: whisper_full_params::language"]
        [::std::mem::offset_of!(whisper_full_params, language) - 96usize];
    ["Offset of field: whisper_full_params::detect_language"]
        [::std::mem::offset_of!(whisper_full_params, detect_language) - 104usize];
    ["Offset of field: whisper_full_params::suppress_blank"]
        [::std::mem::offset_of!(whisper_full_params, suppress_blank) - 105usize];
    ["Offset of field: whisper_full_params::suppress_nst"]
        [::std::mem::offset_of!(whisper_full_params, suppress_nst) - 106usize];
    ["Offset of field: whisper_full_params::temperature"]
        [::std::mem::offset_of!(whisper_full_params, temperature) - 108usize];
    ["Offset of field: whisper_full_params::max_initial_ts"]
        [::std::mem::offset_of!(whisper_full_params, max_initial_ts) - 112usize];
    ["Offset of field: whisper_full_params::length_penalty"]
        [::std::mem::offset_of!(whisper_full_params, length_penalty) - 116usize];
    ["Offset of field: whisper_full_params::temperature_inc"]
        [::std::mem::offset_of!(whisper_full_params, temperature_inc) - 120usize];
    ["Offset of field: whisper_full_params::entropy_thold"]
        [::std::mem::offset_of!(whisper_full_params, entropy_thold) - 124usize];
    ["Offset of field: whisper_full_params::logprob_thold"]
        [::std::mem::offset_of!(whisper_full_params, logprob_thold) - 128usize];
    ["Offset of field: whisper_full_params::no_speech_thold"]
        [::std::mem::offset_of!(whisper_full_params, no_speech_thold) - 132usize];
    ["Offset of field: whisper_full_params::greedy"]
        [::std::mem::offset_of!(whisper_full_params, greedy) - 136usize];
    ["Offset of field: whisper_full_params::beam_search"]
        [::std::mem::offset_of!(whisper_full_params, beam_search) - 140usize];
    ["Offset of field: whisper_full_params::new_segment_callback"]
        [::std::mem::offset_of!(whisper_full_params, new_segment_callback) - 152usize];
    ["Offset of field: whisper_full_params::new_segment_callback_user_data"]
        [::std::mem::offset_of!(whisper_full_params, new_segment_callback_user_data) - 160usize];
    ["Offset of field: whisper_full_params::progress_callback"]
        [::std::mem::offset_of!(whisper_full_params, progress_callback) - 168usize];
    ["Offset of field: whisper_full_params::progress_callback_user_data"]
        [::std::mem::offset_of!(whisper_full_params, progress_callback_user_data) - 176usize];
    ["Offset of field: whisper_full_params::encoder_begin_callback"]
        [::std::mem::offset_of!(whisper_full_params, encoder_begin_callback) - 184usize];
    ["Offset of field: whisper_full_params::encoder_begin_callback_user_data"]
        [::std::mem::offset_of!(whisper_full_params, encoder_begin_callback_user_data) - 192usize];
    ["Offset of field: whisper_full_params::abort_callback"]
        [::std::mem::offset_of!(whisper_full_params, abort_callback) - 200usize];
    ["Offset of field: whisper_full_params::abort_callback_user_data"]
        [::std::mem::offset_of!(whisper_full_params, abort_callback_user_data) - 208usize];
    ["Offset of field: whisper_full_params::logits_filter_callback"]
        [::std::mem::offset_of!(whisper_full_params, logits_filter_callback) - 216usize];
    ["Offset of field: whisper_full_params::logits_filter_callback_user_data"]
        [::std::mem::offset_of!(whisper_full_params, logits_filter_callback_user_data) - 224usize];
    ["Offset of field: whisper_full_params::grammar_rules"]
        [::std::mem::offset_of!(whisper_full_params, grammar_rules) - 232usize];
    ["Offset of field: whisper_full_params::n_grammar_rules"]
        [::std::mem::offset_of!(whisper_full_params, n_grammar_rules) - 240usize];
    ["Offset of field: whisper_full_params::i_start_rule"]
        [::std::mem::offset_of!(whisper_full_params, i_start_rule) - 248usize];
    ["Offset of field: whisper_full_params::grammar_penalty"]
        [::std::mem::offset_of!(whisper_full_params, grammar_penalty) - 256usize];
};
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
