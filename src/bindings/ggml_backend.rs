#![allow(non_camel_case_types, unused, clippy::upper_case_acronyms)]

use super::ggml::{
    ggml_abort_callback, ggml_backend_device, ggml_backend_eval_callback, ggml_backend_event,
    ggml_backend_reg, ggml_backend_sched, ggml_cgraph, ggml_context, ggml_guid_t, ggml_status,
    ggml_tensor,
};

use super::ggml_alloc::{ggml_backend, ggml_backend_buffer, ggml_backend_buffer_type};

#[repr(C)]
pub(crate) struct ggml_backend_dev_caps {
    pub(crate) async_: bool,
    pub(crate) host_buffer: bool,
    pub(crate) buffer_from_host_ptr: bool,
    pub(crate) events: bool,
}

#[repr(C)]
pub(crate) struct ggml_backend_dev_props {
    pub(crate) name: *const core::ffi::c_char,
    pub(crate) description: *const core::ffi::c_char,
    pub(crate) memory_free: usize,
    pub(crate) memory_total: usize,
    pub(crate) type_: ggml_backend_dev_type,
    pub(crate) caps: ggml_backend_dev_caps,
}

#[repr(C)]
pub(crate) struct ggml_backend_feature {
    pub(crate) name: *const core::ffi::c_char,
    pub(crate) value: *const core::ffi::c_char,
}

#[repr(C)]
pub(crate) struct ggml_backend_graph_copy {
    pub(crate) buffer: ggml_backend_buffer_t,
    pub(crate) ctx_allocated: *mut ggml_context,
    pub(crate) ctx_unallocated: *mut ggml_context,
    pub(crate) graph: *mut ggml_cgraph,
}

pub(crate) type ggml_backend_buffer_type_t = *mut ggml_backend_buffer_type;
pub(crate) type ggml_backend_buffer_t = *mut ggml_backend_buffer;
pub(crate) type ggml_backend_event_t = *mut ggml_backend_event;
pub(crate) type ggml_backend_t = *mut ggml_backend;
pub(crate) type ggml_backend_graph_plan_t = *mut core::ffi::c_void;
pub(crate) type ggml_backend_reg_t = *mut ggml_backend_reg;
pub(crate) type ggml_backend_dev_t = *mut ggml_backend_device;
pub(crate) type ggml_backend_buffer_usage = core::ffi::c_uint;
pub(crate) type ggml_backend_dev_type = core::ffi::c_uint;
pub(crate) type ggml_backend_sched_t = *mut ggml_backend_sched;

pub(crate) type ggml_backend_split_buffer_type_t = core::option::Option<
    unsafe extern "C" fn(
        main_device: core::ffi::c_int,
        tensor_split: *const f32,
    ) -> ggml_backend_buffer_type_t,
>;

pub(crate) type ggml_backend_set_n_threads_t = core::option::Option<
    unsafe extern "C" fn(backend: ggml_backend_t, n_threads: core::ffi::c_int),
>;

pub(crate) type ggml_backend_dev_get_extra_bufts_t = core::option::Option<
    unsafe extern "C" fn(device: ggml_backend_dev_t) -> *mut ggml_backend_buffer_type_t,
>;

pub(crate) type ggml_backend_set_abort_callback_t = core::option::Option<
    unsafe extern "C" fn(
        backend: ggml_backend_t,
        abort_callback: ggml_abort_callback,
        abort_callback_data: *mut core::ffi::c_void,
    ),
>;

pub(crate) type ggml_backend_get_features_t = core::option::Option<
    unsafe extern "C" fn(reg: ggml_backend_reg_t) -> *mut ggml_backend_feature,
>;

pub(crate) type ggml_backend_sched_eval_callback = core::option::Option<
    unsafe extern "C" fn(t: *mut ggml_tensor, ask: bool, user_data: *mut core::ffi::c_void) -> bool,
>;

unsafe extern "C" {
    pub(crate) unsafe fn ggml_backend_buft_name(
        buft: ggml_backend_buffer_type_t,
    ) -> *const core::ffi::c_char;

    pub(crate) unsafe fn ggml_backend_buft_alloc_buffer(
        buft: ggml_backend_buffer_type_t,
        size: usize,
    ) -> ggml_backend_buffer_t;

    pub(crate) unsafe fn ggml_backend_buft_get_alignment(buft: ggml_backend_buffer_type_t)
    -> usize;

    pub(crate) unsafe fn ggml_backend_buft_get_max_size(buft: ggml_backend_buffer_type_t) -> usize;

    pub(crate) unsafe fn ggml_backend_buft_get_alloc_size(
        buft: ggml_backend_buffer_type_t,
        tensor: *mut ggml_tensor,
    ) -> usize;

    pub(crate) unsafe fn ggml_backend_buft_is_host(buft: ggml_backend_buffer_type_t) -> bool;

    pub(crate) unsafe fn ggml_backend_buft_get_device(
        buft: ggml_backend_buffer_type_t,
    ) -> ggml_backend_dev_t;

    pub(crate) unsafe fn ggml_backend_buffer_name(
        buffer: ggml_backend_buffer_t,
    ) -> *const core::ffi::c_char;

    pub(crate) unsafe fn ggml_backend_buffer_free(buffer: ggml_backend_buffer_t);

    pub(crate) unsafe fn ggml_backend_buffer_get_base(
        buffer: ggml_backend_buffer_t,
    ) -> *mut core::ffi::c_void;

    pub(crate) unsafe fn ggml_backend_buffer_get_size(buffer: ggml_backend_buffer_t) -> usize;

    pub(crate) unsafe fn ggml_backend_buffer_init_tensor(
        buffer: ggml_backend_buffer_t,
        tensor: *mut ggml_tensor,
    );

    pub(crate) unsafe fn ggml_backend_buffer_get_alignment(buffer: ggml_backend_buffer_t) -> usize;

    pub(crate) unsafe fn ggml_backend_buffer_get_max_size(buffer: ggml_backend_buffer_t) -> usize;

    pub(crate) unsafe fn ggml_backend_buffer_get_alloc_size(
        buffer: ggml_backend_buffer_t,
        tensor: *mut ggml_tensor,
    ) -> usize;

    pub(crate) unsafe fn ggml_backend_buffer_clear(buffer: ggml_backend_buffer_t, value: u8);

    pub(crate) unsafe fn ggml_backend_buffer_is_host(buffer: ggml_backend_buffer_t) -> bool;

    pub(crate) unsafe fn ggml_backend_buffer_set_usage(
        buffer: ggml_backend_buffer_t,
        usage: ggml_backend_buffer_usage,
    );

    pub(crate) unsafe fn ggml_backend_buffer_get_usage(
        buffer: ggml_backend_buffer_t,
    ) -> ggml_backend_buffer_usage;

    pub(crate) unsafe fn ggml_backend_buffer_get_type(
        buffer: ggml_backend_buffer_t,
    ) -> ggml_backend_buffer_type_t;

    pub(crate) unsafe fn ggml_backend_buffer_reset(buffer: ggml_backend_buffer_t);

    pub(crate) unsafe fn ggml_backend_tensor_copy(src: *mut ggml_tensor, dst: *mut ggml_tensor);

    pub(crate) unsafe fn ggml_backend_guid(backend: ggml_backend_t) -> ggml_guid_t;

    pub(crate) unsafe fn ggml_backend_name(backend: ggml_backend_t) -> *const core::ffi::c_char;

    pub(crate) unsafe fn ggml_backend_free(backend: ggml_backend_t);

    pub(crate) unsafe fn ggml_backend_get_default_buffer_type(
        backend: ggml_backend_t,
    ) -> ggml_backend_buffer_type_t;

    pub(crate) unsafe fn ggml_backend_alloc_buffer(
        backend: ggml_backend_t,
        size: usize,
    ) -> ggml_backend_buffer_t;

    pub(crate) unsafe fn ggml_backend_get_alignment(backend: ggml_backend_t) -> usize;

    pub(crate) unsafe fn ggml_backend_get_max_size(backend: ggml_backend_t) -> usize;

    pub(crate) unsafe fn ggml_backend_tensor_set_async(
        backend: ggml_backend_t,
        tensor: *mut ggml_tensor,
        data: *const core::ffi::c_void,
        offset: usize,
        size: usize,
    );

    pub(crate) unsafe fn ggml_backend_tensor_get_async(
        backend: ggml_backend_t,
        tensor: *const ggml_tensor,
        data: *mut core::ffi::c_void,
        offset: usize,
        size: usize,
    );

    pub(crate) unsafe fn ggml_backend_tensor_set(
        tensor: *mut ggml_tensor,
        data: *const core::ffi::c_void,
        offset: usize,
        size: usize,
    );

    pub(crate) unsafe fn ggml_backend_tensor_get(
        tensor: *const ggml_tensor,
        data: *mut core::ffi::c_void,
        offset: usize,
        size: usize,
    );

    pub(crate) unsafe fn ggml_backend_tensor_memset(
        tensor: *mut ggml_tensor,
        value: u8,
        offset: usize,
        size: usize,
    );

    pub(crate) unsafe fn ggml_backend_synchronize(backend: ggml_backend_t);

    pub(crate) unsafe fn ggml_backend_graph_plan_create(
        backend: ggml_backend_t,
        cgraph: *mut ggml_cgraph,
    ) -> ggml_backend_graph_plan_t;

    pub(crate) unsafe fn ggml_backend_graph_plan_free(
        backend: ggml_backend_t,
        plan: ggml_backend_graph_plan_t,
    );

    pub(crate) unsafe fn ggml_backend_graph_plan_compute(
        backend: ggml_backend_t,
        plan: ggml_backend_graph_plan_t,
    ) -> ggml_status;

    pub(crate) unsafe fn ggml_backend_graph_compute(
        backend: ggml_backend_t,
        cgraph: *mut ggml_cgraph,
    ) -> ggml_status;

    pub(crate) unsafe fn ggml_backend_graph_compute_async(
        backend: ggml_backend_t,
        cgraph: *mut ggml_cgraph,
    ) -> ggml_status;

    pub(crate) unsafe fn ggml_backend_supports_op(
        backend: ggml_backend_t,
        op: *const ggml_tensor,
    ) -> bool;

    pub(crate) unsafe fn ggml_backend_supports_buft(
        backend: ggml_backend_t,
        buft: ggml_backend_buffer_type_t,
    ) -> bool;

    pub(crate) unsafe fn ggml_backend_offload_op(
        backend: ggml_backend_t,
        op: *const ggml_tensor,
    ) -> bool;

    pub(crate) unsafe fn ggml_backend_tensor_copy_async(
        backend_src: ggml_backend_t,
        backend_dst: ggml_backend_t,
        src: *mut ggml_tensor,
        dst: *mut ggml_tensor,
    );

    pub(crate) unsafe fn ggml_backend_get_device(backend: ggml_backend_t) -> ggml_backend_dev_t;

    pub(crate) unsafe fn ggml_backend_event_new(device: ggml_backend_dev_t)
    -> ggml_backend_event_t;

    pub(crate) unsafe fn ggml_backend_event_free(event: ggml_backend_event_t);

    pub(crate) unsafe fn ggml_backend_event_record(
        event: ggml_backend_event_t,
        backend: ggml_backend_t,
    );

    pub(crate) unsafe fn ggml_backend_event_synchronize(event: ggml_backend_event_t);

    pub(crate) unsafe fn ggml_backend_event_wait(
        backend: ggml_backend_t,
        event: ggml_backend_event_t,
    );

    pub(crate) unsafe fn ggml_backend_dev_name(
        device: ggml_backend_dev_t,
    ) -> *const core::ffi::c_char;

    pub(crate) unsafe fn ggml_backend_dev_description(
        device: ggml_backend_dev_t,
    ) -> *const core::ffi::c_char;

    pub(crate) unsafe fn ggml_backend_dev_memory(
        device: ggml_backend_dev_t,
        free: *mut usize,
        total: *mut usize,
    );

    pub(crate) unsafe fn ggml_backend_dev_type(device: ggml_backend_dev_t)
    -> ggml_backend_dev_type;

    pub(crate) unsafe fn ggml_backend_dev_get_props(
        device: ggml_backend_dev_t,
        props: *mut ggml_backend_dev_props,
    );

    pub(crate) unsafe fn ggml_backend_dev_backend_reg(
        device: ggml_backend_dev_t,
    ) -> ggml_backend_reg_t;

    pub(crate) unsafe fn ggml_backend_dev_init(
        device: ggml_backend_dev_t,
        params: *const core::ffi::c_char,
    ) -> ggml_backend_t;

    pub(crate) unsafe fn ggml_backend_dev_buffer_type(
        device: ggml_backend_dev_t,
    ) -> ggml_backend_buffer_type_t;

    pub(crate) unsafe fn ggml_backend_dev_host_buffer_type(
        device: ggml_backend_dev_t,
    ) -> ggml_backend_buffer_type_t;

    pub(crate) unsafe fn ggml_backend_dev_buffer_from_host_ptr(
        device: ggml_backend_dev_t,
        ptr: *mut core::ffi::c_void,
        size: usize,
        max_tensor_size: usize,
    ) -> ggml_backend_buffer_t;

    pub(crate) unsafe fn ggml_backend_dev_supports_op(
        device: ggml_backend_dev_t,
        op: *const ggml_tensor,
    ) -> bool;

    pub(crate) unsafe fn ggml_backend_dev_supports_buft(
        device: ggml_backend_dev_t,
        buft: ggml_backend_buffer_type_t,
    ) -> bool;

    pub(crate) unsafe fn ggml_backend_dev_offload_op(
        device: ggml_backend_dev_t,
        op: *const ggml_tensor,
    ) -> bool;

    pub(crate) unsafe fn ggml_backend_reg_name(reg: ggml_backend_reg_t)
    -> *const core::ffi::c_char;

    pub(crate) unsafe fn ggml_backend_reg_dev_count(reg: ggml_backend_reg_t) -> usize;

    pub(crate) unsafe fn ggml_backend_reg_dev_get(
        reg: ggml_backend_reg_t,
        index: usize,
    ) -> ggml_backend_dev_t;

    pub(crate) unsafe fn ggml_backend_reg_get_proc_address(
        reg: ggml_backend_reg_t,
        name: *const core::ffi::c_char,
    ) -> *mut core::ffi::c_void;

    pub(crate) unsafe fn ggml_backend_device_register(device: ggml_backend_dev_t);

    pub(crate) unsafe fn ggml_backend_reg_count() -> usize;

    pub(crate) unsafe fn ggml_backend_reg_get(index: usize) -> ggml_backend_reg_t;

    pub(crate) unsafe fn ggml_backend_reg_by_name(
        name: *const core::ffi::c_char,
    ) -> ggml_backend_reg_t;

    pub(crate) unsafe fn ggml_backend_dev_count() -> usize;

    pub(crate) unsafe fn ggml_backend_dev_get(index: usize) -> ggml_backend_dev_t;

    pub(crate) unsafe fn ggml_backend_dev_by_name(
        name: *const core::ffi::c_char,
    ) -> ggml_backend_dev_t;

    pub(crate) unsafe fn ggml_backend_dev_by_type(
        type_: ggml_backend_dev_type,
    ) -> ggml_backend_dev_t;

    pub(crate) unsafe fn ggml_backend_init_by_name(
        name: *const core::ffi::c_char,
        params: *const core::ffi::c_char,
    ) -> ggml_backend_t;

    pub(crate) unsafe fn ggml_backend_init_by_type(
        type_: ggml_backend_dev_type,
        params: *const core::ffi::c_char,
    ) -> ggml_backend_t;

    pub(crate) unsafe fn ggml_backend_init_best() -> ggml_backend_t;

    pub(crate) unsafe fn ggml_backend_load(path: *const core::ffi::c_char) -> ggml_backend_reg_t;

    pub(crate) unsafe fn ggml_backend_unload(reg: ggml_backend_reg_t);

    pub(crate) unsafe fn ggml_backend_load_all();

    pub(crate) unsafe fn ggml_backend_load_all_from_path(dir_path: *const core::ffi::c_char);

    pub(crate) unsafe fn ggml_backend_sched_new(
        backends: *mut ggml_backend_t,
        bufts: *mut ggml_backend_buffer_type_t,
        n_backends: core::ffi::c_int,
        graph_size: usize,
        parallel: bool,
    ) -> ggml_backend_sched_t;

    pub(crate) unsafe fn ggml_backend_sched_free(sched: ggml_backend_sched_t);

    pub(crate) unsafe fn ggml_backend_sched_reserve(
        sched: ggml_backend_sched_t,
        measure_graph: *mut ggml_cgraph,
    ) -> bool;

    pub(crate) unsafe fn ggml_backend_sched_get_n_backends(
        sched: ggml_backend_sched_t,
    ) -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_backend_sched_get_backend(
        sched: ggml_backend_sched_t,
        i: core::ffi::c_int,
    ) -> ggml_backend_t;

    pub(crate) unsafe fn ggml_backend_sched_get_n_splits(
        sched: ggml_backend_sched_t,
    ) -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_backend_sched_get_n_copies(
        sched: ggml_backend_sched_t,
    ) -> core::ffi::c_int;

    pub(crate) unsafe fn ggml_backend_sched_get_buffer_size(
        sched: ggml_backend_sched_t,
        backend: ggml_backend_t,
    ) -> usize;

    pub(crate) unsafe fn ggml_backend_sched_set_tensor_backend(
        sched: ggml_backend_sched_t,
        node: *mut ggml_tensor,
        backend: ggml_backend_t,
    );

    pub(crate) unsafe fn ggml_backend_sched_get_tensor_backend(
        sched: ggml_backend_sched_t,
        node: *mut ggml_tensor,
    ) -> ggml_backend_t;

    pub(crate) unsafe fn ggml_backend_sched_alloc_graph(
        sched: ggml_backend_sched_t,
        graph: *mut ggml_cgraph,
    ) -> bool;

    pub(crate) unsafe fn ggml_backend_sched_graph_compute(
        sched: ggml_backend_sched_t,
        graph: *mut ggml_cgraph,
    ) -> ggml_status;

    pub(crate) unsafe fn ggml_backend_sched_graph_compute_async(
        sched: ggml_backend_sched_t,
        graph: *mut ggml_cgraph,
    ) -> ggml_status;

    pub(crate) unsafe fn ggml_backend_sched_synchronize(sched: ggml_backend_sched_t);

    pub(crate) unsafe fn ggml_backend_sched_reset(sched: ggml_backend_sched_t);

    pub(crate) unsafe fn ggml_backend_sched_set_eval_callback(
        sched: ggml_backend_sched_t,
        callback: ggml_backend_sched_eval_callback,
        user_data: *mut core::ffi::c_void,
    );

    pub(crate) unsafe fn ggml_backend_graph_copy(
        backend: ggml_backend_t,
        graph: *mut ggml_cgraph,
    ) -> ggml_backend_graph_copy;

    pub(crate) unsafe fn ggml_backend_graph_copy_free(copy: ggml_backend_graph_copy);

    pub(crate) unsafe fn ggml_backend_compare_graph_backend(
        backend1: ggml_backend_t,
        backend2: ggml_backend_t,
        graph: *mut ggml_cgraph,
        callback: ggml_backend_eval_callback,
        user_data: *mut core::ffi::c_void,
    ) -> bool;

    pub(crate) unsafe fn ggml_backend_tensor_alloc(
        buffer: ggml_backend_buffer_t,
        tensor: *mut ggml_tensor,
        addr: *mut core::ffi::c_void,
    );

    pub(crate) unsafe fn ggml_backend_view_init(tensor: *mut ggml_tensor);

    pub(crate) unsafe fn ggml_backend_cpu_buffer_from_ptr(
        ptr: *mut core::ffi::c_void,
        size: usize,
    ) -> ggml_backend_buffer_t;

    pub(crate) unsafe fn ggml_backend_cpu_buffer_type() -> ggml_backend_buffer_type_t;
}
