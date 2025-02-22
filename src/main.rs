mod bindings {
    pub(crate) mod ggml;
    pub(crate) mod llama;
    pub(crate) mod whisper;
}

use bindings::whisper::{
    whisper_context, whisper_context_default_params, whisper_context_params,
    whisper_init_from_file_with_params,
};

fn main() {

    // let mut cparams: whisper_context_params = unsafe { whisper_context_default_params() };

    // let ctx: *mut whisper_context = unsafe {whisper_init_from_file_with_params(path_model, cparams)};
}
