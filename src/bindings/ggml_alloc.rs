#![allow(non_camel_case_types, unused, clippy::upper_case_acronyms)]

use super::ggml::{ggml_cgraph, ggml_context, ggml_gallocr_t, ggml_tensor};

use super::ggml_backend::{ggml_backend_buffer_t, ggml_backend_buffer_type_t, ggml_backend_t};

#[repr(C)]
pub(crate) struct ggml_backend_buffer_type {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub(crate) struct ggml_backend_buffer {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub(crate) struct ggml_backend {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub(crate) struct ggml_tallocr {
    pub(crate) buffer: ggml_backend_buffer_t,
    pub(crate) base: *mut core::ffi::c_void,
    pub(crate) alignment: usize,
    pub(crate) offset: usize,
}

#[repr(C)]
pub(crate) struct ggml_gallocr {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

unsafe extern "C" {

    pub(crate) unsafe fn ggml_tallocr_new(buffer: ggml_backend_buffer_t) -> ggml_tallocr;

    pub(crate) unsafe fn ggml_tallocr_alloc(talloc: *mut ggml_tallocr, tensor: *mut ggml_tensor);

    pub(crate) unsafe fn ggml_gallocr_new(buft: ggml_backend_buffer_type_t) -> ggml_gallocr_t;

    pub(crate) unsafe fn ggml_gallocr_new_n(
        bufts: *mut ggml_backend_buffer_type_t,
        n_bufs: core::ffi::c_int,
    ) -> ggml_gallocr_t;

    pub(crate) unsafe fn ggml_gallocr_free(galloc: ggml_gallocr_t);

    pub(crate) unsafe fn ggml_gallocr_reserve(
        galloc: ggml_gallocr_t,
        graph: *mut ggml_cgraph,
    ) -> bool;

    pub(crate) unsafe fn ggml_gallocr_reserve_n(
        galloc: ggml_gallocr_t,
        graph: *mut ggml_cgraph,
        node_buffer_ids: *const core::ffi::c_int,
        leaf_buffer_ids: *const core::ffi::c_int,
    ) -> bool;

    pub(crate) unsafe fn ggml_gallocr_alloc_graph(
        galloc: ggml_gallocr_t,
        graph: *mut ggml_cgraph,
    ) -> bool;

    pub(crate) unsafe fn ggml_gallocr_get_buffer_size(
        galloc: ggml_gallocr_t,
        buffer_id: core::ffi::c_int,
    ) -> usize;

    pub(crate) unsafe fn ggml_backend_alloc_ctx_tensors_from_buft(
        ctx: *mut ggml_context,
        buft: ggml_backend_buffer_type_t,
    ) -> *mut ggml_backend_buffer;

    pub(crate) unsafe fn ggml_backend_alloc_ctx_tensors(
        ctx: *mut ggml_context,
        backend: ggml_backend_t,
    ) -> *mut ggml_backend_buffer;
}
