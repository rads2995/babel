#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct __BindgenBitfieldUnit<Storage> {
    storage: Storage,
}
impl<Storage> __BindgenBitfieldUnit<Storage> {
    #[inline]
    pub const fn new(storage: Storage) -> Self {
        Self { storage }
    }
}
impl<Storage> __BindgenBitfieldUnit<Storage>
where
    Storage: AsRef<[u8]> + AsMut<[u8]>,
{
    #[inline]
    fn extract_bit(byte: u8, index: usize) -> bool {
        let bit_index = if cfg!(target_endian = "big") {
            7 - (index % 8)
        } else {
            index % 8
        };
        let mask = 1 << bit_index;
        byte & mask == mask
    }
    #[inline]
    pub fn get_bit(&self, index: usize) -> bool {
        debug_assert!(index / 8 < self.storage.as_ref().len());
        let byte_index = index / 8;
        let byte = self.storage.as_ref()[byte_index];
        Self::extract_bit(byte, index)
    }
    #[inline]
    pub unsafe fn raw_get_bit(this: *const Self, index: usize) -> bool {
        debug_assert!(index / 8 < core::mem::size_of::<Storage>());
        let byte_index = index / 8;
        let byte = *(core::ptr::addr_of!((*this).storage) as *const u8).offset(byte_index as isize);
        Self::extract_bit(byte, index)
    }
    #[inline]
    fn change_bit(byte: u8, index: usize, val: bool) -> u8 {
        let bit_index = if cfg!(target_endian = "big") {
            7 - (index % 8)
        } else {
            index % 8
        };
        let mask = 1 << bit_index;
        if val {
            byte | mask
        } else {
            byte & !mask
        }
    }
    #[inline]
    pub fn set_bit(&mut self, index: usize, val: bool) {
        debug_assert!(index / 8 < self.storage.as_ref().len());
        let byte_index = index / 8;
        let byte = &mut self.storage.as_mut()[byte_index];
        *byte = Self::change_bit(*byte, index, val);
    }
    #[inline]
    pub unsafe fn raw_set_bit(this: *mut Self, index: usize, val: bool) {
        debug_assert!(index / 8 < core::mem::size_of::<Storage>());
        let byte_index = index / 8;
        let byte =
            (core::ptr::addr_of_mut!((*this).storage) as *mut u8).offset(byte_index as isize);
        *byte = Self::change_bit(*byte, index, val);
    }
    #[inline]
    pub fn get(&self, bit_offset: usize, bit_width: u8) -> u64 {
        debug_assert!(bit_width <= 64);
        debug_assert!(bit_offset / 8 < self.storage.as_ref().len());
        debug_assert!((bit_offset + (bit_width as usize)) / 8 <= self.storage.as_ref().len());
        let mut val = 0;
        for i in 0..(bit_width as usize) {
            if self.get_bit(i + bit_offset) {
                let index = if cfg!(target_endian = "big") {
                    bit_width as usize - 1 - i
                } else {
                    i
                };
                val |= 1 << index;
            }
        }
        val
    }
    #[inline]
    pub unsafe fn raw_get(this: *const Self, bit_offset: usize, bit_width: u8) -> u64 {
        debug_assert!(bit_width <= 64);
        debug_assert!(bit_offset / 8 < core::mem::size_of::<Storage>());
        debug_assert!((bit_offset + (bit_width as usize)) / 8 <= core::mem::size_of::<Storage>());
        let mut val = 0;
        for i in 0..(bit_width as usize) {
            if Self::raw_get_bit(this, i + bit_offset) {
                let index = if cfg!(target_endian = "big") {
                    bit_width as usize - 1 - i
                } else {
                    i
                };
                val |= 1 << index;
            }
        }
        val
    }
    #[inline]
    pub fn set(&mut self, bit_offset: usize, bit_width: u8, val: u64) {
        debug_assert!(bit_width <= 64);
        debug_assert!(bit_offset / 8 < self.storage.as_ref().len());
        debug_assert!((bit_offset + (bit_width as usize)) / 8 <= self.storage.as_ref().len());
        for i in 0..(bit_width as usize) {
            let mask = 1 << i;
            let val_bit_is_set = val & mask == mask;
            let index = if cfg!(target_endian = "big") {
                bit_width as usize - 1 - i
            } else {
                i
            };
            self.set_bit(index + bit_offset, val_bit_is_set);
        }
    }
    #[inline]
    pub unsafe fn raw_set(this: *mut Self, bit_offset: usize, bit_width: u8, val: u64) {
        debug_assert!(bit_width <= 64);
        debug_assert!(bit_offset / 8 < core::mem::size_of::<Storage>());
        debug_assert!((bit_offset + (bit_width as usize)) / 8 <= core::mem::size_of::<Storage>());
        for i in 0..(bit_width as usize) {
            let mask = 1 << i;
            let val_bit_is_set = val & mask == mask;
            let index = if cfg!(target_endian = "big") {
                bit_width as usize - 1 - i
            } else {
                i
            };
            Self::raw_set_bit(this, index + bit_offset, val_bit_is_set);
        }
    }
}
#[derive(PartialEq, Copy, Clone, Hash, Debug, Default)]
#[repr(C)]
pub struct __BindgenComplex<T> {
    pub re: T,
    pub im: T,
}
pub const __bool_true_false_are_defined: u32 = 1;
pub const true_: u32 = 1;
pub const false_: u32 = 0;
pub const _STDINT_H: u32 = 1;
pub const _FEATURES_H: u32 = 1;
pub const _DEFAULT_SOURCE: u32 = 1;
pub const __GLIBC_USE_ISOC2Y: u32 = 0;
pub const __GLIBC_USE_ISOC23: u32 = 0;
pub const __USE_ISOC11: u32 = 1;
pub const __USE_ISOC99: u32 = 1;
pub const __USE_ISOC95: u32 = 1;
pub const __USE_POSIX_IMPLICITLY: u32 = 1;
pub const _POSIX_SOURCE: u32 = 1;
pub const _POSIX_C_SOURCE: u32 = 200809;
pub const __USE_POSIX: u32 = 1;
pub const __USE_POSIX2: u32 = 1;
pub const __USE_POSIX199309: u32 = 1;
pub const __USE_POSIX199506: u32 = 1;
pub const __USE_XOPEN2K: u32 = 1;
pub const __USE_XOPEN2K8: u32 = 1;
pub const _ATFILE_SOURCE: u32 = 1;
pub const __WORDSIZE: u32 = 64;
pub const __WORDSIZE_TIME64_COMPAT32: u32 = 1;
pub const __SYSCALL_WORDSIZE: u32 = 64;
pub const __TIMESIZE: u32 = 64;
pub const __USE_TIME_BITS64: u32 = 1;
pub const __USE_MISC: u32 = 1;
pub const __USE_ATFILE: u32 = 1;
pub const __USE_FORTIFY_LEVEL: u32 = 0;
pub const __GLIBC_USE_DEPRECATED_GETS: u32 = 0;
pub const __GLIBC_USE_DEPRECATED_SCANF: u32 = 0;
pub const __GLIBC_USE_C23_STRTOL: u32 = 0;
pub const _STDC_PREDEF_H: u32 = 1;
pub const __STDC_IEC_559__: u32 = 1;
pub const __STDC_IEC_60559_BFP__: u32 = 201404;
pub const __STDC_IEC_559_COMPLEX__: u32 = 1;
pub const __STDC_IEC_60559_COMPLEX__: u32 = 201404;
pub const __STDC_ISO_10646__: u32 = 201706;
pub const __GNU_LIBRARY__: u32 = 6;
pub const __GLIBC__: u32 = 2;
pub const __GLIBC_MINOR__: u32 = 41;
pub const _SYS_CDEFS_H: u32 = 1;
pub const __glibc_c99_flexarr_available: u32 = 1;
pub const __LDOUBLE_REDIRECTS_TO_FLOAT128_ABI: u32 = 0;
pub const __HAVE_GENERIC_SELECTION: u32 = 1;
pub const __GLIBC_USE_LIB_EXT2: u32 = 0;
pub const __GLIBC_USE_IEC_60559_BFP_EXT: u32 = 0;
pub const __GLIBC_USE_IEC_60559_BFP_EXT_C23: u32 = 0;
pub const __GLIBC_USE_IEC_60559_EXT: u32 = 0;
pub const __GLIBC_USE_IEC_60559_FUNCS_EXT: u32 = 0;
pub const __GLIBC_USE_IEC_60559_FUNCS_EXT_C23: u32 = 0;
pub const __GLIBC_USE_IEC_60559_TYPES_EXT: u32 = 0;
pub const _BITS_TYPES_H: u32 = 1;
pub const _BITS_TYPESIZES_H: u32 = 1;
pub const __OFF_T_MATCHES_OFF64_T: u32 = 1;
pub const __INO_T_MATCHES_INO64_T: u32 = 1;
pub const __RLIM_T_MATCHES_RLIM64_T: u32 = 1;
pub const __STATFS_MATCHES_STATFS64: u32 = 1;
pub const __KERNEL_OLD_TIMEVAL_MATCHES_TIMEVAL64: u32 = 1;
pub const __FD_SETSIZE: u32 = 1024;
pub const _BITS_TIME64_H: u32 = 1;
pub const _BITS_WCHAR_H: u32 = 1;
pub const _BITS_STDINT_INTN_H: u32 = 1;
pub const _BITS_STDINT_UINTN_H: u32 = 1;
pub const _BITS_STDINT_LEAST_H: u32 = 1;
pub const INT8_MIN: i32 = -128;
pub const INT16_MIN: i32 = -32768;
pub const INT32_MIN: i32 = -2147483648;
pub const INT8_MAX: u32 = 127;
pub const INT16_MAX: u32 = 32767;
pub const INT32_MAX: u32 = 2147483647;
pub const UINT8_MAX: u32 = 255;
pub const UINT16_MAX: u32 = 65535;
pub const UINT32_MAX: u32 = 4294967295;
pub const INT_LEAST8_MIN: i32 = -128;
pub const INT_LEAST16_MIN: i32 = -32768;
pub const INT_LEAST32_MIN: i32 = -2147483648;
pub const INT_LEAST8_MAX: u32 = 127;
pub const INT_LEAST16_MAX: u32 = 32767;
pub const INT_LEAST32_MAX: u32 = 2147483647;
pub const UINT_LEAST8_MAX: u32 = 255;
pub const UINT_LEAST16_MAX: u32 = 65535;
pub const UINT_LEAST32_MAX: u32 = 4294967295;
pub const INT_FAST8_MIN: i32 = -128;
pub const INT_FAST16_MIN: i64 = -9223372036854775808;
pub const INT_FAST32_MIN: i64 = -9223372036854775808;
pub const INT_FAST8_MAX: u32 = 127;
pub const INT_FAST16_MAX: u64 = 9223372036854775807;
pub const INT_FAST32_MAX: u64 = 9223372036854775807;
pub const UINT_FAST8_MAX: u32 = 255;
pub const UINT_FAST16_MAX: i32 = -1;
pub const UINT_FAST32_MAX: i32 = -1;
pub const INTPTR_MIN: i64 = -9223372036854775808;
pub const INTPTR_MAX: u64 = 9223372036854775807;
pub const UINTPTR_MAX: i32 = -1;
pub const PTRDIFF_MIN: i64 = -9223372036854775808;
pub const PTRDIFF_MAX: u64 = 9223372036854775807;
pub const SIG_ATOMIC_MIN: i32 = -2147483648;
pub const SIG_ATOMIC_MAX: u32 = 2147483647;
pub const SIZE_MAX: i32 = -1;
pub const WINT_MIN: u32 = 0;
pub const WINT_MAX: u32 = 4294967295;
pub const _STDIO_H: u32 = 1;
pub const _____fpos_t_defined: u32 = 1;
pub const ____mbstate_t_defined: u32 = 1;
pub const _____fpos64_t_defined: u32 = 1;
pub const ____FILE_defined: u32 = 1;
pub const __FILE_defined: u32 = 1;
pub const __struct_FILE_defined: u32 = 1;
pub const _IO_EOF_SEEN: u32 = 16;
pub const _IO_ERR_SEEN: u32 = 32;
pub const _IO_USER_LOCK: u32 = 32768;
pub const __cookie_io_functions_t_defined: u32 = 1;
pub const _IOFBF: u32 = 0;
pub const _IOLBF: u32 = 1;
pub const _IONBF: u32 = 2;
pub const BUFSIZ: u32 = 8192;
pub const EOF: i32 = -1;
pub const SEEK_SET: u32 = 0;
pub const SEEK_CUR: u32 = 1;
pub const SEEK_END: u32 = 2;
pub const P_tmpdir: &[u8; 5] = b"/tmp\0";
pub const L_tmpnam: u32 = 20;
pub const TMP_MAX: u32 = 238328;
pub const _BITS_STDIO_LIM_H: u32 = 1;
pub const FILENAME_MAX: u32 = 4096;
pub const L_ctermid: u32 = 9;
pub const FOPEN_MAX: u32 = 16;
pub const __HAVE_FLOAT128: u32 = 1;
pub const __HAVE_DISTINCT_FLOAT128: u32 = 1;
pub const __HAVE_FLOAT64X: u32 = 1;
pub const __HAVE_FLOAT64X_LONG_DOUBLE: u32 = 1;
pub const __HAVE_FLOAT16: u32 = 0;
pub const __HAVE_FLOAT32: u32 = 1;
pub const __HAVE_FLOAT64: u32 = 1;
pub const __HAVE_FLOAT32X: u32 = 1;
pub const __HAVE_FLOAT128X: u32 = 0;
pub const __HAVE_DISTINCT_FLOAT16: u32 = 0;
pub const __HAVE_DISTINCT_FLOAT32: u32 = 0;
pub const __HAVE_DISTINCT_FLOAT64: u32 = 0;
pub const __HAVE_DISTINCT_FLOAT32X: u32 = 0;
pub const __HAVE_DISTINCT_FLOAT64X: u32 = 0;
pub const __HAVE_DISTINCT_FLOAT128X: u32 = 0;
pub const __HAVE_FLOATN_NOT_TYPEDEF: u32 = 0;
pub const GGML_FILE_MAGIC: u32 = 1734831468;
pub const GGML_FILE_VERSION: u32 = 2;
pub const GGML_QNT_VERSION: u32 = 2;
pub const GGML_QNT_VERSION_FACTOR: u32 = 1000;
pub const GGML_MAX_DIMS: u32 = 4;
pub const GGML_MAX_PARAMS: u32 = 2048;
pub const GGML_MAX_SRC: u32 = 10;
pub const GGML_MAX_N_THREADS: u32 = 512;
pub const GGML_MAX_OP_PARAMS: u32 = 64;
pub const GGML_MAX_NAME: u32 = 64;
pub const GGML_DEFAULT_N_THREADS: u32 = 4;
pub const GGML_DEFAULT_GRAPH_SIZE: u32 = 2048;
pub const GGML_MEM_ALIGN: u32 = 16;
pub const GGML_EXIT_SUCCESS: u32 = 0;
pub const GGML_EXIT_ABORTED: u32 = 1;
pub const GGML_ROPE_TYPE_NEOX: u32 = 2;
pub const GGML_ROPE_TYPE_MROPE: u32 = 8;
pub const GGML_ROPE_TYPE_VISION: u32 = 24;
pub const GGML_KQ_MASK_PAD: u32 = 64;
pub const GGML_N_TASKS_MAX: i32 = -1;
pub const WHISPER_SAMPLE_RATE: u32 = 16000;
pub const WHISPER_N_FFT: u32 = 400;
pub const WHISPER_HOP_LENGTH: u32 = 160;
pub const WHISPER_CHUNK_SIZE: u32 = 30;
pub type wchar_t = ::std::os::raw::c_int;
#[repr(C)]
#[repr(align(16))]
#[derive(Debug, Copy, Clone)]
pub struct max_align_t {
    pub __clang_max_align_nonce1: ::std::os::raw::c_longlong,
    pub __bindgen_padding_0: u64,
    pub __clang_max_align_nonce2: u128,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of max_align_t"][::std::mem::size_of::<max_align_t>() - 32usize];
    ["Alignment of max_align_t"][::std::mem::align_of::<max_align_t>() - 16usize];
    ["Offset of field: max_align_t::__clang_max_align_nonce1"]
        [::std::mem::offset_of!(max_align_t, __clang_max_align_nonce1) - 0usize];
    ["Offset of field: max_align_t::__clang_max_align_nonce2"]
        [::std::mem::offset_of!(max_align_t, __clang_max_align_nonce2) - 16usize];
};
pub type __u_char = ::std::os::raw::c_uchar;
pub type __u_short = ::std::os::raw::c_ushort;
pub type __u_int = ::std::os::raw::c_uint;
pub type __u_long = ::std::os::raw::c_ulong;
pub type __int8_t = ::std::os::raw::c_schar;
pub type __uint8_t = ::std::os::raw::c_uchar;
pub type __int16_t = ::std::os::raw::c_short;
pub type __uint16_t = ::std::os::raw::c_ushort;
pub type __int32_t = ::std::os::raw::c_int;
pub type __uint32_t = ::std::os::raw::c_uint;
pub type __int64_t = ::std::os::raw::c_long;
pub type __uint64_t = ::std::os::raw::c_ulong;
pub type __int_least8_t = __int8_t;
pub type __uint_least8_t = __uint8_t;
pub type __int_least16_t = __int16_t;
pub type __uint_least16_t = __uint16_t;
pub type __int_least32_t = __int32_t;
pub type __uint_least32_t = __uint32_t;
pub type __int_least64_t = __int64_t;
pub type __uint_least64_t = __uint64_t;
pub type __quad_t = ::std::os::raw::c_long;
pub type __u_quad_t = ::std::os::raw::c_ulong;
pub type __intmax_t = ::std::os::raw::c_long;
pub type __uintmax_t = ::std::os::raw::c_ulong;
pub type __dev_t = ::std::os::raw::c_ulong;
pub type __uid_t = ::std::os::raw::c_uint;
pub type __gid_t = ::std::os::raw::c_uint;
pub type __ino_t = ::std::os::raw::c_ulong;
pub type __ino64_t = ::std::os::raw::c_ulong;
pub type __mode_t = ::std::os::raw::c_uint;
pub type __nlink_t = ::std::os::raw::c_ulong;
pub type __off_t = ::std::os::raw::c_long;
pub type __off64_t = ::std::os::raw::c_long;
pub type __pid_t = ::std::os::raw::c_int;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct __fsid_t {
    pub __val: [::std::os::raw::c_int; 2usize],
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of __fsid_t"][::std::mem::size_of::<__fsid_t>() - 8usize];
    ["Alignment of __fsid_t"][::std::mem::align_of::<__fsid_t>() - 4usize];
    ["Offset of field: __fsid_t::__val"][::std::mem::offset_of!(__fsid_t, __val) - 0usize];
};
pub type __clock_t = ::std::os::raw::c_long;
pub type __rlim_t = ::std::os::raw::c_ulong;
pub type __rlim64_t = ::std::os::raw::c_ulong;
pub type __id_t = ::std::os::raw::c_uint;
pub type __time_t = ::std::os::raw::c_long;
pub type __useconds_t = ::std::os::raw::c_uint;
pub type __suseconds_t = ::std::os::raw::c_long;
pub type __suseconds64_t = ::std::os::raw::c_long;
pub type __daddr_t = ::std::os::raw::c_int;
pub type __key_t = ::std::os::raw::c_int;
pub type __clockid_t = ::std::os::raw::c_int;
pub type __timer_t = *mut ::std::os::raw::c_void;
pub type __blksize_t = ::std::os::raw::c_long;
pub type __blkcnt_t = ::std::os::raw::c_long;
pub type __blkcnt64_t = ::std::os::raw::c_long;
pub type __fsblkcnt_t = ::std::os::raw::c_ulong;
pub type __fsblkcnt64_t = ::std::os::raw::c_ulong;
pub type __fsfilcnt_t = ::std::os::raw::c_ulong;
pub type __fsfilcnt64_t = ::std::os::raw::c_ulong;
pub type __fsword_t = ::std::os::raw::c_long;
pub type __ssize_t = ::std::os::raw::c_long;
pub type __syscall_slong_t = ::std::os::raw::c_long;
pub type __syscall_ulong_t = ::std::os::raw::c_ulong;
pub type __loff_t = __off64_t;
pub type __caddr_t = *mut ::std::os::raw::c_char;
pub type __intptr_t = ::std::os::raw::c_long;
pub type __socklen_t = ::std::os::raw::c_uint;
pub type __sig_atomic_t = ::std::os::raw::c_int;
pub type int_least8_t = __int_least8_t;
pub type int_least16_t = __int_least16_t;
pub type int_least32_t = __int_least32_t;
pub type int_least64_t = __int_least64_t;
pub type uint_least8_t = __uint_least8_t;
pub type uint_least16_t = __uint_least16_t;
pub type uint_least32_t = __uint_least32_t;
pub type uint_least64_t = __uint_least64_t;
pub type int_fast8_t = ::std::os::raw::c_schar;
pub type int_fast16_t = ::std::os::raw::c_long;
pub type int_fast32_t = ::std::os::raw::c_long;
pub type int_fast64_t = ::std::os::raw::c_long;
pub type uint_fast8_t = ::std::os::raw::c_uchar;
pub type uint_fast16_t = ::std::os::raw::c_ulong;
pub type uint_fast32_t = ::std::os::raw::c_ulong;
pub type uint_fast64_t = ::std::os::raw::c_ulong;
pub type intmax_t = __intmax_t;
pub type uintmax_t = __uintmax_t;
pub type __gnuc_va_list = __builtin_va_list;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct __mbstate_t {
    pub __count: ::std::os::raw::c_int,
    pub __value: __mbstate_t__bindgen_ty_1,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union __mbstate_t__bindgen_ty_1 {
    pub __wch: ::std::os::raw::c_uint,
    pub __wchb: [::std::os::raw::c_char; 4usize],
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of __mbstate_t__bindgen_ty_1"]
        [::std::mem::size_of::<__mbstate_t__bindgen_ty_1>() - 4usize];
    ["Alignment of __mbstate_t__bindgen_ty_1"]
        [::std::mem::align_of::<__mbstate_t__bindgen_ty_1>() - 4usize];
    ["Offset of field: __mbstate_t__bindgen_ty_1::__wch"]
        [::std::mem::offset_of!(__mbstate_t__bindgen_ty_1, __wch) - 0usize];
    ["Offset of field: __mbstate_t__bindgen_ty_1::__wchb"]
        [::std::mem::offset_of!(__mbstate_t__bindgen_ty_1, __wchb) - 0usize];
};
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of __mbstate_t"][::std::mem::size_of::<__mbstate_t>() - 8usize];
    ["Alignment of __mbstate_t"][::std::mem::align_of::<__mbstate_t>() - 4usize];
    ["Offset of field: __mbstate_t::__count"]
        [::std::mem::offset_of!(__mbstate_t, __count) - 0usize];
    ["Offset of field: __mbstate_t::__value"]
        [::std::mem::offset_of!(__mbstate_t, __value) - 4usize];
};
#[repr(C)]
#[derive(Copy, Clone)]
pub struct _G_fpos_t {
    pub __pos: __off_t,
    pub __state: __mbstate_t,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of _G_fpos_t"][::std::mem::size_of::<_G_fpos_t>() - 16usize];
    ["Alignment of _G_fpos_t"][::std::mem::align_of::<_G_fpos_t>() - 8usize];
    ["Offset of field: _G_fpos_t::__pos"][::std::mem::offset_of!(_G_fpos_t, __pos) - 0usize];
    ["Offset of field: _G_fpos_t::__state"][::std::mem::offset_of!(_G_fpos_t, __state) - 8usize];
};
pub type __fpos_t = _G_fpos_t;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct _G_fpos64_t {
    pub __pos: __off64_t,
    pub __state: __mbstate_t,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of _G_fpos64_t"][::std::mem::size_of::<_G_fpos64_t>() - 16usize];
    ["Alignment of _G_fpos64_t"][::std::mem::align_of::<_G_fpos64_t>() - 8usize];
    ["Offset of field: _G_fpos64_t::__pos"][::std::mem::offset_of!(_G_fpos64_t, __pos) - 0usize];
    ["Offset of field: _G_fpos64_t::__state"]
        [::std::mem::offset_of!(_G_fpos64_t, __state) - 8usize];
};
pub type __fpos64_t = _G_fpos64_t;
pub type __FILE = _IO_FILE;
pub type FILE = _IO_FILE;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _IO_marker {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _IO_codecvt {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _IO_wide_data {
    _unused: [u8; 0],
}
pub type _IO_lock_t = ::std::os::raw::c_void;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _IO_FILE {
    pub _flags: ::std::os::raw::c_int,
    pub _IO_read_ptr: *mut ::std::os::raw::c_char,
    pub _IO_read_end: *mut ::std::os::raw::c_char,
    pub _IO_read_base: *mut ::std::os::raw::c_char,
    pub _IO_write_base: *mut ::std::os::raw::c_char,
    pub _IO_write_ptr: *mut ::std::os::raw::c_char,
    pub _IO_write_end: *mut ::std::os::raw::c_char,
    pub _IO_buf_base: *mut ::std::os::raw::c_char,
    pub _IO_buf_end: *mut ::std::os::raw::c_char,
    pub _IO_save_base: *mut ::std::os::raw::c_char,
    pub _IO_backup_base: *mut ::std::os::raw::c_char,
    pub _IO_save_end: *mut ::std::os::raw::c_char,
    pub _markers: *mut _IO_marker,
    pub _chain: *mut _IO_FILE,
    pub _fileno: ::std::os::raw::c_int,
    pub _bitfield_align_1: [u32; 0],
    pub _bitfield_1: __BindgenBitfieldUnit<[u8; 3usize]>,
    pub _short_backupbuf: [::std::os::raw::c_char; 1usize],
    pub _old_offset: __off_t,
    pub _cur_column: ::std::os::raw::c_ushort,
    pub _vtable_offset: ::std::os::raw::c_schar,
    pub _shortbuf: [::std::os::raw::c_char; 1usize],
    pub _lock: *mut _IO_lock_t,
    pub _offset: __off64_t,
    pub _codecvt: *mut _IO_codecvt,
    pub _wide_data: *mut _IO_wide_data,
    pub _freeres_list: *mut _IO_FILE,
    pub _freeres_buf: *mut ::std::os::raw::c_void,
    pub _prevchain: *mut *mut _IO_FILE,
    pub _mode: ::std::os::raw::c_int,
    pub _unused2: [::std::os::raw::c_char; 20usize],
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of _IO_FILE"][::std::mem::size_of::<_IO_FILE>() - 216usize];
    ["Alignment of _IO_FILE"][::std::mem::align_of::<_IO_FILE>() - 8usize];
    ["Offset of field: _IO_FILE::_flags"][::std::mem::offset_of!(_IO_FILE, _flags) - 0usize];
    ["Offset of field: _IO_FILE::_IO_read_ptr"]
        [::std::mem::offset_of!(_IO_FILE, _IO_read_ptr) - 8usize];
    ["Offset of field: _IO_FILE::_IO_read_end"]
        [::std::mem::offset_of!(_IO_FILE, _IO_read_end) - 16usize];
    ["Offset of field: _IO_FILE::_IO_read_base"]
        [::std::mem::offset_of!(_IO_FILE, _IO_read_base) - 24usize];
    ["Offset of field: _IO_FILE::_IO_write_base"]
        [::std::mem::offset_of!(_IO_FILE, _IO_write_base) - 32usize];
    ["Offset of field: _IO_FILE::_IO_write_ptr"]
        [::std::mem::offset_of!(_IO_FILE, _IO_write_ptr) - 40usize];
    ["Offset of field: _IO_FILE::_IO_write_end"]
        [::std::mem::offset_of!(_IO_FILE, _IO_write_end) - 48usize];
    ["Offset of field: _IO_FILE::_IO_buf_base"]
        [::std::mem::offset_of!(_IO_FILE, _IO_buf_base) - 56usize];
    ["Offset of field: _IO_FILE::_IO_buf_end"]
        [::std::mem::offset_of!(_IO_FILE, _IO_buf_end) - 64usize];
    ["Offset of field: _IO_FILE::_IO_save_base"]
        [::std::mem::offset_of!(_IO_FILE, _IO_save_base) - 72usize];
    ["Offset of field: _IO_FILE::_IO_backup_base"]
        [::std::mem::offset_of!(_IO_FILE, _IO_backup_base) - 80usize];
    ["Offset of field: _IO_FILE::_IO_save_end"]
        [::std::mem::offset_of!(_IO_FILE, _IO_save_end) - 88usize];
    ["Offset of field: _IO_FILE::_markers"][::std::mem::offset_of!(_IO_FILE, _markers) - 96usize];
    ["Offset of field: _IO_FILE::_chain"][::std::mem::offset_of!(_IO_FILE, _chain) - 104usize];
    ["Offset of field: _IO_FILE::_fileno"][::std::mem::offset_of!(_IO_FILE, _fileno) - 112usize];
    ["Offset of field: _IO_FILE::_short_backupbuf"]
        [::std::mem::offset_of!(_IO_FILE, _short_backupbuf) - 119usize];
    ["Offset of field: _IO_FILE::_old_offset"]
        [::std::mem::offset_of!(_IO_FILE, _old_offset) - 120usize];
    ["Offset of field: _IO_FILE::_cur_column"]
        [::std::mem::offset_of!(_IO_FILE, _cur_column) - 128usize];
    ["Offset of field: _IO_FILE::_vtable_offset"]
        [::std::mem::offset_of!(_IO_FILE, _vtable_offset) - 130usize];
    ["Offset of field: _IO_FILE::_shortbuf"]
        [::std::mem::offset_of!(_IO_FILE, _shortbuf) - 131usize];
    ["Offset of field: _IO_FILE::_lock"][::std::mem::offset_of!(_IO_FILE, _lock) - 136usize];
    ["Offset of field: _IO_FILE::_offset"][::std::mem::offset_of!(_IO_FILE, _offset) - 144usize];
    ["Offset of field: _IO_FILE::_codecvt"][::std::mem::offset_of!(_IO_FILE, _codecvt) - 152usize];
    ["Offset of field: _IO_FILE::_wide_data"]
        [::std::mem::offset_of!(_IO_FILE, _wide_data) - 160usize];
    ["Offset of field: _IO_FILE::_freeres_list"]
        [::std::mem::offset_of!(_IO_FILE, _freeres_list) - 168usize];
    ["Offset of field: _IO_FILE::_freeres_buf"]
        [::std::mem::offset_of!(_IO_FILE, _freeres_buf) - 176usize];
    ["Offset of field: _IO_FILE::_prevchain"]
        [::std::mem::offset_of!(_IO_FILE, _prevchain) - 184usize];
    ["Offset of field: _IO_FILE::_mode"][::std::mem::offset_of!(_IO_FILE, _mode) - 192usize];
    ["Offset of field: _IO_FILE::_unused2"][::std::mem::offset_of!(_IO_FILE, _unused2) - 196usize];
};
impl _IO_FILE {
    #[inline]
    pub fn _flags2(&self) -> ::std::os::raw::c_int {
        unsafe { ::std::mem::transmute(self._bitfield_1.get(0usize, 24u8) as u32) }
    }
    #[inline]
    pub fn set__flags2(&mut self, val: ::std::os::raw::c_int) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            self._bitfield_1.set(0usize, 24u8, val as u64)
        }
    }
    #[inline]
    pub unsafe fn _flags2_raw(this: *const Self) -> ::std::os::raw::c_int {
        unsafe {
            ::std::mem::transmute(<__BindgenBitfieldUnit<[u8; 3usize]>>::raw_get(
                ::std::ptr::addr_of!((*this)._bitfield_1),
                0usize,
                24u8,
            ) as u32)
        }
    }
    #[inline]
    pub unsafe fn set__flags2_raw(this: *mut Self, val: ::std::os::raw::c_int) {
        unsafe {
            let val: u32 = ::std::mem::transmute(val);
            <__BindgenBitfieldUnit<[u8; 3usize]>>::raw_set(
                ::std::ptr::addr_of_mut!((*this)._bitfield_1),
                0usize,
                24u8,
                val as u64,
            )
        }
    }
    #[inline]
    pub fn new_bitfield_1(_flags2: ::std::os::raw::c_int) -> __BindgenBitfieldUnit<[u8; 3usize]> {
        let mut __bindgen_bitfield_unit: __BindgenBitfieldUnit<[u8; 3usize]> = Default::default();
        __bindgen_bitfield_unit.set(0usize, 24u8, {
            let _flags2: u32 = unsafe { ::std::mem::transmute(_flags2) };
            _flags2 as u64
        });
        __bindgen_bitfield_unit
    }
}
pub type cookie_read_function_t = ::std::option::Option<
    unsafe extern "C" fn(
        __cookie: *mut ::std::os::raw::c_void,
        __buf: *mut ::std::os::raw::c_char,
        __nbytes: usize,
    ) -> __ssize_t,
>;
pub type cookie_write_function_t = ::std::option::Option<
    unsafe extern "C" fn(
        __cookie: *mut ::std::os::raw::c_void,
        __buf: *const ::std::os::raw::c_char,
        __nbytes: usize,
    ) -> __ssize_t,
>;
pub type cookie_seek_function_t = ::std::option::Option<
    unsafe extern "C" fn(
        __cookie: *mut ::std::os::raw::c_void,
        __pos: *mut __off64_t,
        __w: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int,
>;
pub type cookie_close_function_t = ::std::option::Option<
    unsafe extern "C" fn(__cookie: *mut ::std::os::raw::c_void) -> ::std::os::raw::c_int,
>;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _IO_cookie_io_functions_t {
    pub read: cookie_read_function_t,
    pub write: cookie_write_function_t,
    pub seek: cookie_seek_function_t,
    pub close: cookie_close_function_t,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of _IO_cookie_io_functions_t"]
        [::std::mem::size_of::<_IO_cookie_io_functions_t>() - 32usize];
    ["Alignment of _IO_cookie_io_functions_t"]
        [::std::mem::align_of::<_IO_cookie_io_functions_t>() - 8usize];
    ["Offset of field: _IO_cookie_io_functions_t::read"]
        [::std::mem::offset_of!(_IO_cookie_io_functions_t, read) - 0usize];
    ["Offset of field: _IO_cookie_io_functions_t::write"]
        [::std::mem::offset_of!(_IO_cookie_io_functions_t, write) - 8usize];
    ["Offset of field: _IO_cookie_io_functions_t::seek"]
        [::std::mem::offset_of!(_IO_cookie_io_functions_t, seek) - 16usize];
    ["Offset of field: _IO_cookie_io_functions_t::close"]
        [::std::mem::offset_of!(_IO_cookie_io_functions_t, close) - 24usize];
};
pub type cookie_io_functions_t = _IO_cookie_io_functions_t;
pub type va_list = __gnuc_va_list;
pub type off_t = __off_t;
pub type fpos_t = __fpos_t;
unsafe extern "C" {
    pub static mut stdin: *mut FILE;
}
unsafe extern "C" {
    pub static mut stdout: *mut FILE;
}
unsafe extern "C" {
    pub static mut stderr: *mut FILE;
}
unsafe extern "C" {
    pub fn remove(__filename: *const ::std::os::raw::c_char) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn rename(
        __old: *const ::std::os::raw::c_char,
        __new: *const ::std::os::raw::c_char,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn renameat(
        __oldfd: ::std::os::raw::c_int,
        __old: *const ::std::os::raw::c_char,
        __newfd: ::std::os::raw::c_int,
        __new: *const ::std::os::raw::c_char,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn fclose(__stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn tmpfile() -> *mut FILE;
}
unsafe extern "C" {
    pub fn tmpnam(arg1: *mut ::std::os::raw::c_char) -> *mut ::std::os::raw::c_char;
}
unsafe extern "C" {
    pub fn tmpnam_r(__s: *mut ::std::os::raw::c_char) -> *mut ::std::os::raw::c_char;
}
unsafe extern "C" {
    pub fn tempnam(
        __dir: *const ::std::os::raw::c_char,
        __pfx: *const ::std::os::raw::c_char,
    ) -> *mut ::std::os::raw::c_char;
}
unsafe extern "C" {
    pub fn fflush(__stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn fflush_unlocked(__stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn fopen(
        __filename: *const ::std::os::raw::c_char,
        __modes: *const ::std::os::raw::c_char,
    ) -> *mut FILE;
}
unsafe extern "C" {
    pub fn freopen(
        __filename: *const ::std::os::raw::c_char,
        __modes: *const ::std::os::raw::c_char,
        __stream: *mut FILE,
    ) -> *mut FILE;
}
unsafe extern "C" {
    pub fn fdopen(__fd: ::std::os::raw::c_int, __modes: *const ::std::os::raw::c_char)
        -> *mut FILE;
}
unsafe extern "C" {
    pub fn fopencookie(
        __magic_cookie: *mut ::std::os::raw::c_void,
        __modes: *const ::std::os::raw::c_char,
        __io_funcs: cookie_io_functions_t,
    ) -> *mut FILE;
}
unsafe extern "C" {
    pub fn fmemopen(
        __s: *mut ::std::os::raw::c_void,
        __len: usize,
        __modes: *const ::std::os::raw::c_char,
    ) -> *mut FILE;
}
unsafe extern "C" {
    pub fn open_memstream(
        __bufloc: *mut *mut ::std::os::raw::c_char,
        __sizeloc: *mut usize,
    ) -> *mut FILE;
}
unsafe extern "C" {
    pub fn setbuf(__stream: *mut FILE, __buf: *mut ::std::os::raw::c_char);
}
unsafe extern "C" {
    pub fn setvbuf(
        __stream: *mut FILE,
        __buf: *mut ::std::os::raw::c_char,
        __modes: ::std::os::raw::c_int,
        __n: usize,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn setbuffer(__stream: *mut FILE, __buf: *mut ::std::os::raw::c_char, __size: usize);
}
unsafe extern "C" {
    pub fn setlinebuf(__stream: *mut FILE);
}
unsafe extern "C" {
    pub fn fprintf(
        __stream: *mut FILE,
        __format: *const ::std::os::raw::c_char,
        ...
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn printf(__format: *const ::std::os::raw::c_char, ...) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn sprintf(
        __s: *mut ::std::os::raw::c_char,
        __format: *const ::std::os::raw::c_char,
        ...
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn vfprintf(
        __s: *mut FILE,
        __format: *const ::std::os::raw::c_char,
        __arg: *mut __va_list_tag,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn vprintf(
        __format: *const ::std::os::raw::c_char,
        __arg: *mut __va_list_tag,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn vsprintf(
        __s: *mut ::std::os::raw::c_char,
        __format: *const ::std::os::raw::c_char,
        __arg: *mut __va_list_tag,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn snprintf(
        __s: *mut ::std::os::raw::c_char,
        __maxlen: ::std::os::raw::c_ulong,
        __format: *const ::std::os::raw::c_char,
        ...
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn vsnprintf(
        __s: *mut ::std::os::raw::c_char,
        __maxlen: ::std::os::raw::c_ulong,
        __format: *const ::std::os::raw::c_char,
        __arg: *mut __va_list_tag,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn vasprintf(
        __ptr: *mut *mut ::std::os::raw::c_char,
        __f: *const ::std::os::raw::c_char,
        __arg: *mut __va_list_tag,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn __asprintf(
        __ptr: *mut *mut ::std::os::raw::c_char,
        __fmt: *const ::std::os::raw::c_char,
        ...
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn asprintf(
        __ptr: *mut *mut ::std::os::raw::c_char,
        __fmt: *const ::std::os::raw::c_char,
        ...
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn vdprintf(
        __fd: ::std::os::raw::c_int,
        __fmt: *const ::std::os::raw::c_char,
        __arg: *mut __va_list_tag,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn dprintf(
        __fd: ::std::os::raw::c_int,
        __fmt: *const ::std::os::raw::c_char,
        ...
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn fscanf(
        __stream: *mut FILE,
        __format: *const ::std::os::raw::c_char,
        ...
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn scanf(__format: *const ::std::os::raw::c_char, ...) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn sscanf(
        __s: *const ::std::os::raw::c_char,
        __format: *const ::std::os::raw::c_char,
        ...
    ) -> ::std::os::raw::c_int;
}
pub type __cfloat128 = __BindgenComplex<u128>;
pub type _Float128 = u128;
pub type _Float32 = f32;
pub type _Float64 = f64;
pub type _Float32x = f64;
pub type _Float64x = u128;
unsafe extern "C" {
    #[link_name = "\u{1}__isoc99_fscanf"]
    pub fn fscanf1(
        __stream: *mut FILE,
        __format: *const ::std::os::raw::c_char,
        ...
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    #[link_name = "\u{1}__isoc99_scanf"]
    pub fn scanf1(__format: *const ::std::os::raw::c_char, ...) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    #[link_name = "\u{1}__isoc99_sscanf"]
    pub fn sscanf1(
        __s: *const ::std::os::raw::c_char,
        __format: *const ::std::os::raw::c_char,
        ...
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn vfscanf(
        __s: *mut FILE,
        __format: *const ::std::os::raw::c_char,
        __arg: *mut __va_list_tag,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn vscanf(
        __format: *const ::std::os::raw::c_char,
        __arg: *mut __va_list_tag,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn vsscanf(
        __s: *const ::std::os::raw::c_char,
        __format: *const ::std::os::raw::c_char,
        __arg: *mut __va_list_tag,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    #[link_name = "\u{1}__isoc99_vfscanf"]
    pub fn vfscanf1(
        __s: *mut FILE,
        __format: *const ::std::os::raw::c_char,
        __arg: *mut __va_list_tag,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    #[link_name = "\u{1}__isoc99_vscanf"]
    pub fn vscanf1(
        __format: *const ::std::os::raw::c_char,
        __arg: *mut __va_list_tag,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    #[link_name = "\u{1}__isoc99_vsscanf"]
    pub fn vsscanf1(
        __s: *const ::std::os::raw::c_char,
        __format: *const ::std::os::raw::c_char,
        __arg: *mut __va_list_tag,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn fgetc(__stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn getc(__stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn getchar() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn getc_unlocked(__stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn getchar_unlocked() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn fgetc_unlocked(__stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn fputc(__c: ::std::os::raw::c_int, __stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn putc(__c: ::std::os::raw::c_int, __stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn putchar(__c: ::std::os::raw::c_int) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn fputc_unlocked(__c: ::std::os::raw::c_int, __stream: *mut FILE)
        -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn putc_unlocked(__c: ::std::os::raw::c_int, __stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn putchar_unlocked(__c: ::std::os::raw::c_int) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn getw(__stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn putw(__w: ::std::os::raw::c_int, __stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn fgets(
        __s: *mut ::std::os::raw::c_char,
        __n: ::std::os::raw::c_int,
        __stream: *mut FILE,
    ) -> *mut ::std::os::raw::c_char;
}
unsafe extern "C" {
    pub fn __getdelim(
        __lineptr: *mut *mut ::std::os::raw::c_char,
        __n: *mut usize,
        __delimiter: ::std::os::raw::c_int,
        __stream: *mut FILE,
    ) -> __ssize_t;
}
unsafe extern "C" {
    pub fn getdelim(
        __lineptr: *mut *mut ::std::os::raw::c_char,
        __n: *mut usize,
        __delimiter: ::std::os::raw::c_int,
        __stream: *mut FILE,
    ) -> __ssize_t;
}
unsafe extern "C" {
    pub fn getline(
        __lineptr: *mut *mut ::std::os::raw::c_char,
        __n: *mut usize,
        __stream: *mut FILE,
    ) -> __ssize_t;
}
unsafe extern "C" {
    pub fn fputs(__s: *const ::std::os::raw::c_char, __stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn puts(__s: *const ::std::os::raw::c_char) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ungetc(__c: ::std::os::raw::c_int, __stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn fread(
        __ptr: *mut ::std::os::raw::c_void,
        __size: ::std::os::raw::c_ulong,
        __n: ::std::os::raw::c_ulong,
        __stream: *mut FILE,
    ) -> ::std::os::raw::c_ulong;
}
unsafe extern "C" {
    pub fn fwrite(
        __ptr: *const ::std::os::raw::c_void,
        __size: ::std::os::raw::c_ulong,
        __n: ::std::os::raw::c_ulong,
        __s: *mut FILE,
    ) -> ::std::os::raw::c_ulong;
}
unsafe extern "C" {
    pub fn fread_unlocked(
        __ptr: *mut ::std::os::raw::c_void,
        __size: usize,
        __n: usize,
        __stream: *mut FILE,
    ) -> usize;
}
unsafe extern "C" {
    pub fn fwrite_unlocked(
        __ptr: *const ::std::os::raw::c_void,
        __size: usize,
        __n: usize,
        __stream: *mut FILE,
    ) -> usize;
}
unsafe extern "C" {
    pub fn fseek(
        __stream: *mut FILE,
        __off: ::std::os::raw::c_long,
        __whence: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ftell(__stream: *mut FILE) -> ::std::os::raw::c_long;
}
unsafe extern "C" {
    pub fn rewind(__stream: *mut FILE);
}
unsafe extern "C" {
    pub fn fseeko(
        __stream: *mut FILE,
        __off: __off_t,
        __whence: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ftello(__stream: *mut FILE) -> __off_t;
}
unsafe extern "C" {
    pub fn fgetpos(__stream: *mut FILE, __pos: *mut fpos_t) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn fsetpos(__stream: *mut FILE, __pos: *const fpos_t) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn clearerr(__stream: *mut FILE);
}
unsafe extern "C" {
    pub fn feof(__stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ferror(__stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn clearerr_unlocked(__stream: *mut FILE);
}
unsafe extern "C" {
    pub fn feof_unlocked(__stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ferror_unlocked(__stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn perror(__s: *const ::std::os::raw::c_char);
}
unsafe extern "C" {
    pub fn fileno(__stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn fileno_unlocked(__stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn pclose(__stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn popen(
        __command: *const ::std::os::raw::c_char,
        __modes: *const ::std::os::raw::c_char,
    ) -> *mut FILE;
}
unsafe extern "C" {
    pub fn ctermid(__s: *mut ::std::os::raw::c_char) -> *mut ::std::os::raw::c_char;
}
unsafe extern "C" {
    pub fn flockfile(__stream: *mut FILE);
}
unsafe extern "C" {
    pub fn ftrylockfile(__stream: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn funlockfile(__stream: *mut FILE);
}
unsafe extern "C" {
    pub fn __uflow(arg1: *mut FILE) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn __overflow(arg1: *mut FILE, arg2: ::std::os::raw::c_int) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_abort(
        file: *const ::std::os::raw::c_char,
        line: ::std::os::raw::c_int,
        fmt: *const ::std::os::raw::c_char,
        ...
    );
}
pub const ggml_status_GGML_STATUS_ALLOC_FAILED: ggml_status = -2;
pub const ggml_status_GGML_STATUS_FAILED: ggml_status = -1;
pub const ggml_status_GGML_STATUS_SUCCESS: ggml_status = 0;
pub const ggml_status_GGML_STATUS_ABORTED: ggml_status = 1;
pub type ggml_status = ::std::os::raw::c_int;
unsafe extern "C" {
    pub fn ggml_status_to_string(status: ggml_status) -> *const ::std::os::raw::c_char;
}
pub type ggml_fp16_t = u16;
unsafe extern "C" {
    pub fn ggml_fp16_to_fp32(arg1: ggml_fp16_t) -> f32;
}
unsafe extern "C" {
    pub fn ggml_fp32_to_fp16(arg1: f32) -> ggml_fp16_t;
}
unsafe extern "C" {
    pub fn ggml_fp16_to_fp32_row(arg1: *const ggml_fp16_t, arg2: *mut f32, arg3: i64);
}
unsafe extern "C" {
    pub fn ggml_fp32_to_fp16_row(arg1: *const f32, arg2: *mut ggml_fp16_t, arg3: i64);
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_bf16_t {
    pub bits: u16,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of ggml_bf16_t"][::std::mem::size_of::<ggml_bf16_t>() - 2usize];
    ["Alignment of ggml_bf16_t"][::std::mem::align_of::<ggml_bf16_t>() - 2usize];
    ["Offset of field: ggml_bf16_t::bits"][::std::mem::offset_of!(ggml_bf16_t, bits) - 0usize];
};
unsafe extern "C" {
    pub fn ggml_fp32_to_bf16(arg1: f32) -> ggml_bf16_t;
}
unsafe extern "C" {
    pub fn ggml_bf16_to_fp32(arg1: ggml_bf16_t) -> f32;
}
unsafe extern "C" {
    pub fn ggml_bf16_to_fp32_row(arg1: *const ggml_bf16_t, arg2: *mut f32, arg3: i64);
}
unsafe extern "C" {
    pub fn ggml_fp32_to_bf16_row_ref(arg1: *const f32, arg2: *mut ggml_bf16_t, arg3: i64);
}
unsafe extern "C" {
    pub fn ggml_fp32_to_bf16_row(arg1: *const f32, arg2: *mut ggml_bf16_t, arg3: i64);
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_object {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_context {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_cgraph {
    _unused: [u8; 0],
}
pub const ggml_type_GGML_TYPE_F32: ggml_type = 0;
pub const ggml_type_GGML_TYPE_F16: ggml_type = 1;
pub const ggml_type_GGML_TYPE_Q4_0: ggml_type = 2;
pub const ggml_type_GGML_TYPE_Q4_1: ggml_type = 3;
pub const ggml_type_GGML_TYPE_Q5_0: ggml_type = 6;
pub const ggml_type_GGML_TYPE_Q5_1: ggml_type = 7;
pub const ggml_type_GGML_TYPE_Q8_0: ggml_type = 8;
pub const ggml_type_GGML_TYPE_Q8_1: ggml_type = 9;
pub const ggml_type_GGML_TYPE_Q2_K: ggml_type = 10;
pub const ggml_type_GGML_TYPE_Q3_K: ggml_type = 11;
pub const ggml_type_GGML_TYPE_Q4_K: ggml_type = 12;
pub const ggml_type_GGML_TYPE_Q5_K: ggml_type = 13;
pub const ggml_type_GGML_TYPE_Q6_K: ggml_type = 14;
pub const ggml_type_GGML_TYPE_Q8_K: ggml_type = 15;
pub const ggml_type_GGML_TYPE_IQ2_XXS: ggml_type = 16;
pub const ggml_type_GGML_TYPE_IQ2_XS: ggml_type = 17;
pub const ggml_type_GGML_TYPE_IQ3_XXS: ggml_type = 18;
pub const ggml_type_GGML_TYPE_IQ1_S: ggml_type = 19;
pub const ggml_type_GGML_TYPE_IQ4_NL: ggml_type = 20;
pub const ggml_type_GGML_TYPE_IQ3_S: ggml_type = 21;
pub const ggml_type_GGML_TYPE_IQ2_S: ggml_type = 22;
pub const ggml_type_GGML_TYPE_IQ4_XS: ggml_type = 23;
pub const ggml_type_GGML_TYPE_I8: ggml_type = 24;
pub const ggml_type_GGML_TYPE_I16: ggml_type = 25;
pub const ggml_type_GGML_TYPE_I32: ggml_type = 26;
pub const ggml_type_GGML_TYPE_I64: ggml_type = 27;
pub const ggml_type_GGML_TYPE_F64: ggml_type = 28;
pub const ggml_type_GGML_TYPE_IQ1_M: ggml_type = 29;
pub const ggml_type_GGML_TYPE_BF16: ggml_type = 30;
pub const ggml_type_GGML_TYPE_TQ1_0: ggml_type = 34;
pub const ggml_type_GGML_TYPE_TQ2_0: ggml_type = 35;
pub const ggml_type_GGML_TYPE_COUNT: ggml_type = 39;
pub type ggml_type = ::std::os::raw::c_uint;
pub const ggml_prec_GGML_PREC_DEFAULT: ggml_prec = 0;
pub const ggml_prec_GGML_PREC_F32: ggml_prec = 1;
pub type ggml_prec = ::std::os::raw::c_uint;
pub const ggml_ftype_GGML_FTYPE_UNKNOWN: ggml_ftype = -1;
pub const ggml_ftype_GGML_FTYPE_ALL_F32: ggml_ftype = 0;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_F16: ggml_ftype = 1;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_Q4_0: ggml_ftype = 2;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_Q4_1: ggml_ftype = 3;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: ggml_ftype = 4;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_Q8_0: ggml_ftype = 7;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_Q5_0: ggml_ftype = 8;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_Q5_1: ggml_ftype = 9;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_Q2_K: ggml_ftype = 10;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_Q3_K: ggml_ftype = 11;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_Q4_K: ggml_ftype = 12;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_Q5_K: ggml_ftype = 13;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_Q6_K: ggml_ftype = 14;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_IQ2_XXS: ggml_ftype = 15;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_IQ2_XS: ggml_ftype = 16;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_IQ3_XXS: ggml_ftype = 17;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_IQ1_S: ggml_ftype = 18;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_IQ4_NL: ggml_ftype = 19;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_IQ3_S: ggml_ftype = 20;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_IQ2_S: ggml_ftype = 21;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_IQ4_XS: ggml_ftype = 22;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_IQ1_M: ggml_ftype = 23;
pub const ggml_ftype_GGML_FTYPE_MOSTLY_BF16: ggml_ftype = 24;
pub type ggml_ftype = ::std::os::raw::c_int;
pub const ggml_op_GGML_OP_NONE: ggml_op = 0;
pub const ggml_op_GGML_OP_DUP: ggml_op = 1;
pub const ggml_op_GGML_OP_ADD: ggml_op = 2;
pub const ggml_op_GGML_OP_ADD1: ggml_op = 3;
pub const ggml_op_GGML_OP_ACC: ggml_op = 4;
pub const ggml_op_GGML_OP_SUB: ggml_op = 5;
pub const ggml_op_GGML_OP_MUL: ggml_op = 6;
pub const ggml_op_GGML_OP_DIV: ggml_op = 7;
pub const ggml_op_GGML_OP_SQR: ggml_op = 8;
pub const ggml_op_GGML_OP_SQRT: ggml_op = 9;
pub const ggml_op_GGML_OP_LOG: ggml_op = 10;
pub const ggml_op_GGML_OP_SIN: ggml_op = 11;
pub const ggml_op_GGML_OP_COS: ggml_op = 12;
pub const ggml_op_GGML_OP_SUM: ggml_op = 13;
pub const ggml_op_GGML_OP_SUM_ROWS: ggml_op = 14;
pub const ggml_op_GGML_OP_MEAN: ggml_op = 15;
pub const ggml_op_GGML_OP_ARGMAX: ggml_op = 16;
pub const ggml_op_GGML_OP_COUNT_EQUAL: ggml_op = 17;
pub const ggml_op_GGML_OP_REPEAT: ggml_op = 18;
pub const ggml_op_GGML_OP_REPEAT_BACK: ggml_op = 19;
pub const ggml_op_GGML_OP_CONCAT: ggml_op = 20;
pub const ggml_op_GGML_OP_SILU_BACK: ggml_op = 21;
pub const ggml_op_GGML_OP_NORM: ggml_op = 22;
pub const ggml_op_GGML_OP_RMS_NORM: ggml_op = 23;
pub const ggml_op_GGML_OP_RMS_NORM_BACK: ggml_op = 24;
pub const ggml_op_GGML_OP_GROUP_NORM: ggml_op = 25;
pub const ggml_op_GGML_OP_MUL_MAT: ggml_op = 26;
pub const ggml_op_GGML_OP_MUL_MAT_ID: ggml_op = 27;
pub const ggml_op_GGML_OP_OUT_PROD: ggml_op = 28;
pub const ggml_op_GGML_OP_SCALE: ggml_op = 29;
pub const ggml_op_GGML_OP_SET: ggml_op = 30;
pub const ggml_op_GGML_OP_CPY: ggml_op = 31;
pub const ggml_op_GGML_OP_CONT: ggml_op = 32;
pub const ggml_op_GGML_OP_RESHAPE: ggml_op = 33;
pub const ggml_op_GGML_OP_VIEW: ggml_op = 34;
pub const ggml_op_GGML_OP_PERMUTE: ggml_op = 35;
pub const ggml_op_GGML_OP_TRANSPOSE: ggml_op = 36;
pub const ggml_op_GGML_OP_GET_ROWS: ggml_op = 37;
pub const ggml_op_GGML_OP_GET_ROWS_BACK: ggml_op = 38;
pub const ggml_op_GGML_OP_DIAG: ggml_op = 39;
pub const ggml_op_GGML_OP_DIAG_MASK_INF: ggml_op = 40;
pub const ggml_op_GGML_OP_DIAG_MASK_ZERO: ggml_op = 41;
pub const ggml_op_GGML_OP_SOFT_MAX: ggml_op = 42;
pub const ggml_op_GGML_OP_SOFT_MAX_BACK: ggml_op = 43;
pub const ggml_op_GGML_OP_ROPE: ggml_op = 44;
pub const ggml_op_GGML_OP_ROPE_BACK: ggml_op = 45;
pub const ggml_op_GGML_OP_CLAMP: ggml_op = 46;
pub const ggml_op_GGML_OP_CONV_TRANSPOSE_1D: ggml_op = 47;
pub const ggml_op_GGML_OP_IM2COL: ggml_op = 48;
pub const ggml_op_GGML_OP_IM2COL_BACK: ggml_op = 49;
pub const ggml_op_GGML_OP_CONV_TRANSPOSE_2D: ggml_op = 50;
pub const ggml_op_GGML_OP_POOL_1D: ggml_op = 51;
pub const ggml_op_GGML_OP_POOL_2D: ggml_op = 52;
pub const ggml_op_GGML_OP_POOL_2D_BACK: ggml_op = 53;
pub const ggml_op_GGML_OP_UPSCALE: ggml_op = 54;
pub const ggml_op_GGML_OP_PAD: ggml_op = 55;
pub const ggml_op_GGML_OP_PAD_REFLECT_1D: ggml_op = 56;
pub const ggml_op_GGML_OP_ARANGE: ggml_op = 57;
pub const ggml_op_GGML_OP_TIMESTEP_EMBEDDING: ggml_op = 58;
pub const ggml_op_GGML_OP_ARGSORT: ggml_op = 59;
pub const ggml_op_GGML_OP_LEAKY_RELU: ggml_op = 60;
pub const ggml_op_GGML_OP_FLASH_ATTN_EXT: ggml_op = 61;
pub const ggml_op_GGML_OP_FLASH_ATTN_BACK: ggml_op = 62;
pub const ggml_op_GGML_OP_SSM_CONV: ggml_op = 63;
pub const ggml_op_GGML_OP_SSM_SCAN: ggml_op = 64;
pub const ggml_op_GGML_OP_WIN_PART: ggml_op = 65;
pub const ggml_op_GGML_OP_WIN_UNPART: ggml_op = 66;
pub const ggml_op_GGML_OP_GET_REL_POS: ggml_op = 67;
pub const ggml_op_GGML_OP_ADD_REL_POS: ggml_op = 68;
pub const ggml_op_GGML_OP_RWKV_WKV6: ggml_op = 69;
pub const ggml_op_GGML_OP_GATED_LINEAR_ATTN: ggml_op = 70;
pub const ggml_op_GGML_OP_UNARY: ggml_op = 71;
pub const ggml_op_GGML_OP_MAP_UNARY: ggml_op = 72;
pub const ggml_op_GGML_OP_MAP_BINARY: ggml_op = 73;
pub const ggml_op_GGML_OP_MAP_CUSTOM1_F32: ggml_op = 74;
pub const ggml_op_GGML_OP_MAP_CUSTOM2_F32: ggml_op = 75;
pub const ggml_op_GGML_OP_MAP_CUSTOM3_F32: ggml_op = 76;
pub const ggml_op_GGML_OP_MAP_CUSTOM1: ggml_op = 77;
pub const ggml_op_GGML_OP_MAP_CUSTOM2: ggml_op = 78;
pub const ggml_op_GGML_OP_MAP_CUSTOM3: ggml_op = 79;
pub const ggml_op_GGML_OP_CROSS_ENTROPY_LOSS: ggml_op = 80;
pub const ggml_op_GGML_OP_CROSS_ENTROPY_LOSS_BACK: ggml_op = 81;
pub const ggml_op_GGML_OP_OPT_STEP_ADAMW: ggml_op = 82;
pub const ggml_op_GGML_OP_COUNT: ggml_op = 83;
pub type ggml_op = ::std::os::raw::c_uint;
pub const ggml_unary_op_GGML_UNARY_OP_ABS: ggml_unary_op = 0;
pub const ggml_unary_op_GGML_UNARY_OP_SGN: ggml_unary_op = 1;
pub const ggml_unary_op_GGML_UNARY_OP_NEG: ggml_unary_op = 2;
pub const ggml_unary_op_GGML_UNARY_OP_STEP: ggml_unary_op = 3;
pub const ggml_unary_op_GGML_UNARY_OP_TANH: ggml_unary_op = 4;
pub const ggml_unary_op_GGML_UNARY_OP_ELU: ggml_unary_op = 5;
pub const ggml_unary_op_GGML_UNARY_OP_RELU: ggml_unary_op = 6;
pub const ggml_unary_op_GGML_UNARY_OP_SIGMOID: ggml_unary_op = 7;
pub const ggml_unary_op_GGML_UNARY_OP_GELU: ggml_unary_op = 8;
pub const ggml_unary_op_GGML_UNARY_OP_GELU_QUICK: ggml_unary_op = 9;
pub const ggml_unary_op_GGML_UNARY_OP_SILU: ggml_unary_op = 10;
pub const ggml_unary_op_GGML_UNARY_OP_HARDSWISH: ggml_unary_op = 11;
pub const ggml_unary_op_GGML_UNARY_OP_HARDSIGMOID: ggml_unary_op = 12;
pub const ggml_unary_op_GGML_UNARY_OP_EXP: ggml_unary_op = 13;
pub const ggml_unary_op_GGML_UNARY_OP_COUNT: ggml_unary_op = 14;
pub type ggml_unary_op = ::std::os::raw::c_uint;
pub const ggml_object_type_GGML_OBJECT_TYPE_TENSOR: ggml_object_type = 0;
pub const ggml_object_type_GGML_OBJECT_TYPE_GRAPH: ggml_object_type = 1;
pub const ggml_object_type_GGML_OBJECT_TYPE_WORK_BUFFER: ggml_object_type = 2;
pub type ggml_object_type = ::std::os::raw::c_uint;
pub const ggml_log_level_GGML_LOG_LEVEL_NONE: ggml_log_level = 0;
pub const ggml_log_level_GGML_LOG_LEVEL_DEBUG: ggml_log_level = 1;
pub const ggml_log_level_GGML_LOG_LEVEL_INFO: ggml_log_level = 2;
pub const ggml_log_level_GGML_LOG_LEVEL_WARN: ggml_log_level = 3;
pub const ggml_log_level_GGML_LOG_LEVEL_ERROR: ggml_log_level = 4;
pub const ggml_log_level_GGML_LOG_LEVEL_CONT: ggml_log_level = 5;
pub type ggml_log_level = ::std::os::raw::c_uint;
pub const ggml_tensor_flag_GGML_TENSOR_FLAG_INPUT: ggml_tensor_flag = 1;
pub const ggml_tensor_flag_GGML_TENSOR_FLAG_OUTPUT: ggml_tensor_flag = 2;
pub const ggml_tensor_flag_GGML_TENSOR_FLAG_PARAM: ggml_tensor_flag = 4;
pub const ggml_tensor_flag_GGML_TENSOR_FLAG_LOSS: ggml_tensor_flag = 8;
pub type ggml_tensor_flag = ::std::os::raw::c_uint;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_init_params {
    pub mem_size: usize,
    pub mem_buffer: *mut ::std::os::raw::c_void,
    pub no_alloc: bool,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of ggml_init_params"][::std::mem::size_of::<ggml_init_params>() - 24usize];
    ["Alignment of ggml_init_params"][::std::mem::align_of::<ggml_init_params>() - 8usize];
    ["Offset of field: ggml_init_params::mem_size"]
        [::std::mem::offset_of!(ggml_init_params, mem_size) - 0usize];
    ["Offset of field: ggml_init_params::mem_buffer"]
        [::std::mem::offset_of!(ggml_init_params, mem_buffer) - 8usize];
    ["Offset of field: ggml_init_params::no_alloc"]
        [::std::mem::offset_of!(ggml_init_params, no_alloc) - 16usize];
};
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_tensor {
    pub type_: ggml_type,
    pub buffer: *mut ggml_backend_buffer,
    pub ne: [i64; 4usize],
    pub nb: [usize; 4usize],
    pub op: ggml_op,
    pub op_params: [i32; 16usize],
    pub flags: i32,
    pub src: [*mut ggml_tensor; 10usize],
    pub view_src: *mut ggml_tensor,
    pub view_offs: usize,
    pub data: *mut ::std::os::raw::c_void,
    pub name: [::std::os::raw::c_char; 64usize],
    pub extra: *mut ::std::os::raw::c_void,
    pub padding: [::std::os::raw::c_char; 8usize],
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of ggml_tensor"][::std::mem::size_of::<ggml_tensor>() - 336usize];
    ["Alignment of ggml_tensor"][::std::mem::align_of::<ggml_tensor>() - 8usize];
    ["Offset of field: ggml_tensor::type_"][::std::mem::offset_of!(ggml_tensor, type_) - 0usize];
    ["Offset of field: ggml_tensor::buffer"][::std::mem::offset_of!(ggml_tensor, buffer) - 8usize];
    ["Offset of field: ggml_tensor::ne"][::std::mem::offset_of!(ggml_tensor, ne) - 16usize];
    ["Offset of field: ggml_tensor::nb"][::std::mem::offset_of!(ggml_tensor, nb) - 48usize];
    ["Offset of field: ggml_tensor::op"][::std::mem::offset_of!(ggml_tensor, op) - 80usize];
    ["Offset of field: ggml_tensor::op_params"]
        [::std::mem::offset_of!(ggml_tensor, op_params) - 84usize];
    ["Offset of field: ggml_tensor::flags"][::std::mem::offset_of!(ggml_tensor, flags) - 148usize];
    ["Offset of field: ggml_tensor::src"][::std::mem::offset_of!(ggml_tensor, src) - 152usize];
    ["Offset of field: ggml_tensor::view_src"]
        [::std::mem::offset_of!(ggml_tensor, view_src) - 232usize];
    ["Offset of field: ggml_tensor::view_offs"]
        [::std::mem::offset_of!(ggml_tensor, view_offs) - 240usize];
    ["Offset of field: ggml_tensor::data"][::std::mem::offset_of!(ggml_tensor, data) - 248usize];
    ["Offset of field: ggml_tensor::name"][::std::mem::offset_of!(ggml_tensor, name) - 256usize];
    ["Offset of field: ggml_tensor::extra"][::std::mem::offset_of!(ggml_tensor, extra) - 320usize];
    ["Offset of field: ggml_tensor::padding"]
        [::std::mem::offset_of!(ggml_tensor, padding) - 328usize];
};
pub const GGML_TENSOR_SIZE: usize = 336;
pub type ggml_abort_callback =
    ::std::option::Option<unsafe extern "C" fn(data: *mut ::std::os::raw::c_void) -> bool>;
pub type ggml_guid = [u8; 16usize];
pub type ggml_guid_t = *mut ggml_guid;
unsafe extern "C" {
    pub fn ggml_guid_matches(guid_a: ggml_guid_t, guid_b: ggml_guid_t) -> bool;
}
unsafe extern "C" {
    pub fn ggml_time_init();
}
unsafe extern "C" {
    pub fn ggml_time_ms() -> i64;
}
unsafe extern "C" {
    pub fn ggml_time_us() -> i64;
}
unsafe extern "C" {
    pub fn ggml_cycles() -> i64;
}
unsafe extern "C" {
    pub fn ggml_cycles_per_ms() -> i64;
}
unsafe extern "C" {
    pub fn ggml_fopen(
        fname: *const ::std::os::raw::c_char,
        mode: *const ::std::os::raw::c_char,
    ) -> *mut FILE;
}
unsafe extern "C" {
    pub fn ggml_print_object(obj: *const ggml_object);
}
unsafe extern "C" {
    pub fn ggml_print_objects(ctx: *const ggml_context);
}
unsafe extern "C" {
    pub fn ggml_nelements(tensor: *const ggml_tensor) -> i64;
}
unsafe extern "C" {
    pub fn ggml_nrows(tensor: *const ggml_tensor) -> i64;
}
unsafe extern "C" {
    pub fn ggml_nbytes(tensor: *const ggml_tensor) -> usize;
}
unsafe extern "C" {
    pub fn ggml_nbytes_pad(tensor: *const ggml_tensor) -> usize;
}
unsafe extern "C" {
    pub fn ggml_blck_size(type_: ggml_type) -> i64;
}
unsafe extern "C" {
    pub fn ggml_type_size(type_: ggml_type) -> usize;
}
unsafe extern "C" {
    pub fn ggml_row_size(type_: ggml_type, ne: i64) -> usize;
}
unsafe extern "C" {
    pub fn ggml_type_sizef(type_: ggml_type) -> f64;
}
unsafe extern "C" {
    pub fn ggml_type_name(type_: ggml_type) -> *const ::std::os::raw::c_char;
}
unsafe extern "C" {
    pub fn ggml_op_name(op: ggml_op) -> *const ::std::os::raw::c_char;
}
unsafe extern "C" {
    pub fn ggml_op_symbol(op: ggml_op) -> *const ::std::os::raw::c_char;
}
unsafe extern "C" {
    pub fn ggml_unary_op_name(op: ggml_unary_op) -> *const ::std::os::raw::c_char;
}
unsafe extern "C" {
    pub fn ggml_op_desc(t: *const ggml_tensor) -> *const ::std::os::raw::c_char;
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
    pub fn ggml_n_dims(tensor: *const ggml_tensor) -> ::std::os::raw::c_int;
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
        data: *const ::std::os::raw::c_void,
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
    pub fn ggml_get_mem_buffer(ctx: *const ggml_context) -> *mut ::std::os::raw::c_void;
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
        n_dims: ::std::os::raw::c_int,
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
    pub fn ggml_new_buffer(ctx: *mut ggml_context, nbytes: usize) -> *mut ::std::os::raw::c_void;
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
        name: *const ::std::os::raw::c_char,
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
    pub fn ggml_get_data(tensor: *const ggml_tensor) -> *mut ::std::os::raw::c_void;
}
unsafe extern "C" {
    pub fn ggml_get_data_f32(tensor: *const ggml_tensor) -> *mut f32;
}
unsafe extern "C" {
    pub fn ggml_get_name(tensor: *const ggml_tensor) -> *const ::std::os::raw::c_char;
}
unsafe extern "C" {
    pub fn ggml_set_name(
        tensor: *mut ggml_tensor,
        name: *const ::std::os::raw::c_char,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_format_name(
        tensor: *mut ggml_tensor,
        fmt: *const ::std::os::raw::c_char,
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
        dim: ::std::os::raw::c_int,
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
        n_groups: ::std::os::raw::c_int,
        eps: f32,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_group_norm_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        n_groups: ::std::os::raw::c_int,
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
        axis0: ::std::os::raw::c_int,
        axis1: ::std::os::raw::c_int,
        axis2: ::std::os::raw::c_int,
        axis3: ::std::os::raw::c_int,
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
        n_past: ::std::os::raw::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_diag_mask_inf_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        n_past: ::std::os::raw::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_diag_mask_zero(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        n_past: ::std::os::raw::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_diag_mask_zero_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        n_past: ::std::os::raw::c_int,
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
        n_dims: ::std::os::raw::c_int,
        mode: ::std::os::raw::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_rope_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        n_dims: ::std::os::raw::c_int,
        mode: ::std::os::raw::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_rope_ext(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
        n_dims: ::std::os::raw::c_int,
        mode: ::std::os::raw::c_int,
        n_ctx_orig: ::std::os::raw::c_int,
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
        n_dims: ::std::os::raw::c_int,
        sections: *mut ::std::os::raw::c_int,
        mode: ::std::os::raw::c_int,
        n_ctx_orig: ::std::os::raw::c_int,
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
        n_dims: ::std::os::raw::c_int,
        mode: ::std::os::raw::c_int,
        n_ctx_orig: ::std::os::raw::c_int,
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
        n_dims: ::std::os::raw::c_int,
        mode: ::std::os::raw::c_int,
        n_ctx_orig: ::std::os::raw::c_int,
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
        n_dims: ::std::os::raw::c_int,
        mode: ::std::os::raw::c_int,
        n_ctx_orig: ::std::os::raw::c_int,
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
        n_dims: ::std::os::raw::c_int,
        n_ctx_orig: ::std::os::raw::c_int,
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
        n_dims: ::std::os::raw::c_int,
        mode: ::std::os::raw::c_int,
        n_ctx_orig: ::std::os::raw::c_int,
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
        n_dims: ::std::os::raw::c_int,
        sections: *mut ::std::os::raw::c_int,
        mode: ::std::os::raw::c_int,
        n_ctx_orig: ::std::os::raw::c_int,
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
        s0: ::std::os::raw::c_int,
        s1: ::std::os::raw::c_int,
        p0: ::std::os::raw::c_int,
        p1: ::std::os::raw::c_int,
        d0: ::std::os::raw::c_int,
        d1: ::std::os::raw::c_int,
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
        s0: ::std::os::raw::c_int,
        s1: ::std::os::raw::c_int,
        p0: ::std::os::raw::c_int,
        p1: ::std::os::raw::c_int,
        d0: ::std::os::raw::c_int,
        d1: ::std::os::raw::c_int,
        is_2D: bool,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_conv_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        s0: ::std::os::raw::c_int,
        p0: ::std::os::raw::c_int,
        d0: ::std::os::raw::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_conv_1d_ph(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        s: ::std::os::raw::c_int,
        d: ::std::os::raw::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_conv_1d_dw(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        s0: ::std::os::raw::c_int,
        p0: ::std::os::raw::c_int,
        d0: ::std::os::raw::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_conv_1d_dw_ph(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        s0: ::std::os::raw::c_int,
        d0: ::std::os::raw::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_conv_transpose_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        s0: ::std::os::raw::c_int,
        p0: ::std::os::raw::c_int,
        d0: ::std::os::raw::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_conv_2d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        s0: ::std::os::raw::c_int,
        s1: ::std::os::raw::c_int,
        p0: ::std::os::raw::c_int,
        p1: ::std::os::raw::c_int,
        d0: ::std::os::raw::c_int,
        d1: ::std::os::raw::c_int,
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
        s0: ::std::os::raw::c_int,
        s1: ::std::os::raw::c_int,
        p0: ::std::os::raw::c_int,
        p1: ::std::os::raw::c_int,
        d0: ::std::os::raw::c_int,
        d1: ::std::os::raw::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_conv_transpose_2d_p0(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        stride: ::std::os::raw::c_int,
    ) -> *mut ggml_tensor;
}
pub const ggml_op_pool_GGML_OP_POOL_MAX: ggml_op_pool = 0;
pub const ggml_op_pool_GGML_OP_POOL_AVG: ggml_op_pool = 1;
pub const ggml_op_pool_GGML_OP_POOL_COUNT: ggml_op_pool = 2;
pub type ggml_op_pool = ::std::os::raw::c_uint;
unsafe extern "C" {
    pub fn ggml_pool_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        op: ggml_op_pool,
        k0: ::std::os::raw::c_int,
        s0: ::std::os::raw::c_int,
        p0: ::std::os::raw::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_pool_2d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        op: ggml_op_pool,
        k0: ::std::os::raw::c_int,
        k1: ::std::os::raw::c_int,
        s0: ::std::os::raw::c_int,
        s1: ::std::os::raw::c_int,
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
        k0: ::std::os::raw::c_int,
        k1: ::std::os::raw::c_int,
        s0: ::std::os::raw::c_int,
        s1: ::std::os::raw::c_int,
        p0: f32,
        p1: f32,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_upscale(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        scale_factor: ::std::os::raw::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_upscale_ext(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: ::std::os::raw::c_int,
        ne1: ::std::os::raw::c_int,
        ne2: ::std::os::raw::c_int,
        ne3: ::std::os::raw::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_pad(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        p0: ::std::os::raw::c_int,
        p1: ::std::os::raw::c_int,
        p2: ::std::os::raw::c_int,
        p3: ::std::os::raw::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_pad_reflect_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        p0: ::std::os::raw::c_int,
        p1: ::std::os::raw::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_timestep_embedding(
        ctx: *mut ggml_context,
        timesteps: *mut ggml_tensor,
        dim: ::std::os::raw::c_int,
        max_period: ::std::os::raw::c_int,
    ) -> *mut ggml_tensor;
}
pub const ggml_sort_order_GGML_SORT_ORDER_ASC: ggml_sort_order = 0;
pub const ggml_sort_order_GGML_SORT_ORDER_DESC: ggml_sort_order = 1;
pub type ggml_sort_order = ::std::os::raw::c_uint;
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
        k: ::std::os::raw::c_int,
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
        w: ::std::os::raw::c_int,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_win_unpart(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        w0: ::std::os::raw::c_int,
        h0: ::std::os::raw::c_int,
        w: ::std::os::raw::c_int,
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
        qh: ::std::os::raw::c_int,
        kh: ::std::os::raw::c_int,
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
pub type ggml_unary_op_f32_t = ::std::option::Option<
    unsafe extern "C" fn(arg1: ::std::os::raw::c_int, arg2: *mut f32, arg3: *const f32),
>;
pub type ggml_binary_op_f32_t = ::std::option::Option<
    unsafe extern "C" fn(
        arg1: ::std::os::raw::c_int,
        arg2: *mut f32,
        arg3: *const f32,
        arg4: *const f32,
    ),
>;
pub type ggml_custom1_op_f32_t =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ggml_tensor, arg2: *const ggml_tensor)>;
pub type ggml_custom2_op_f32_t = ::std::option::Option<
    unsafe extern "C" fn(
        arg1: *mut ggml_tensor,
        arg2: *const ggml_tensor,
        arg3: *const ggml_tensor,
    ),
>;
pub type ggml_custom3_op_f32_t = ::std::option::Option<
    unsafe extern "C" fn(
        arg1: *mut ggml_tensor,
        arg2: *const ggml_tensor,
        arg3: *const ggml_tensor,
        arg4: *const ggml_tensor,
    ),
>;
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
pub type ggml_custom1_op_t = ::std::option::Option<
    unsafe extern "C" fn(
        dst: *mut ggml_tensor,
        a: *const ggml_tensor,
        ith: ::std::os::raw::c_int,
        nth: ::std::os::raw::c_int,
        userdata: *mut ::std::os::raw::c_void,
    ),
>;
pub type ggml_custom2_op_t = ::std::option::Option<
    unsafe extern "C" fn(
        dst: *mut ggml_tensor,
        a: *const ggml_tensor,
        b: *const ggml_tensor,
        ith: ::std::os::raw::c_int,
        nth: ::std::os::raw::c_int,
        userdata: *mut ::std::os::raw::c_void,
    ),
>;
pub type ggml_custom3_op_t = ::std::option::Option<
    unsafe extern "C" fn(
        dst: *mut ggml_tensor,
        a: *const ggml_tensor,
        b: *const ggml_tensor,
        c: *const ggml_tensor,
        ith: ::std::os::raw::c_int,
        nth: ::std::os::raw::c_int,
        userdata: *mut ::std::os::raw::c_void,
    ),
>;
unsafe extern "C" {
    pub fn ggml_map_custom1(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        fun: ggml_custom1_op_t,
        n_tasks: ::std::os::raw::c_int,
        userdata: *mut ::std::os::raw::c_void,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_map_custom1_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        fun: ggml_custom1_op_t,
        n_tasks: ::std::os::raw::c_int,
        userdata: *mut ::std::os::raw::c_void,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_map_custom2(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        fun: ggml_custom2_op_t,
        n_tasks: ::std::os::raw::c_int,
        userdata: *mut ::std::os::raw::c_void,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_map_custom2_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        fun: ggml_custom2_op_t,
        n_tasks: ::std::os::raw::c_int,
        userdata: *mut ::std::os::raw::c_void,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_map_custom3(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
        fun: ggml_custom3_op_t,
        n_tasks: ::std::os::raw::c_int,
        userdata: *mut ::std::os::raw::c_void,
    ) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_map_custom3_inplace(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
        fun: ggml_custom3_op_t,
        n_tasks: ::std::os::raw::c_int,
        userdata: *mut ::std::os::raw::c_void,
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
    pub fn ggml_graph_size(cgraph: *mut ggml_cgraph) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_graph_node(cgraph: *mut ggml_cgraph, i: ::std::os::raw::c_int) -> *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_graph_nodes(cgraph: *mut ggml_cgraph) -> *mut *mut ggml_tensor;
}
unsafe extern "C" {
    pub fn ggml_graph_n_nodes(cgraph: *mut ggml_cgraph) -> ::std::os::raw::c_int;
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
        name: *const ::std::os::raw::c_char,
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
    pub fn ggml_graph_export(cgraph: *const ggml_cgraph, fname: *const ::std::os::raw::c_char);
}
unsafe extern "C" {
    pub fn ggml_graph_import(
        fname: *const ::std::os::raw::c_char,
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
        filename: *const ::std::os::raw::c_char,
    );
}
pub type ggml_log_callback = ::std::option::Option<
    unsafe extern "C" fn(
        level: ggml_log_level,
        text: *const ::std::os::raw::c_char,
        user_data: *mut ::std::os::raw::c_void,
    ),
>;
unsafe extern "C" {
    pub fn ggml_log_set(log_callback: ggml_log_callback, user_data: *mut ::std::os::raw::c_void);
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
        dst: *mut ::std::os::raw::c_void,
        start: i64,
        nrows: i64,
        n_per_row: i64,
        imatrix: *const f32,
    ) -> usize;
}
pub type ggml_to_float_t = ::std::option::Option<
    unsafe extern "C" fn(x: *const ::std::os::raw::c_void, y: *mut f32, k: i64),
>;
pub type ggml_from_float_t = ::std::option::Option<
    unsafe extern "C" fn(x: *const f32, y: *mut ::std::os::raw::c_void, k: i64),
>;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_type_traits {
    pub type_name: *const ::std::os::raw::c_char,
    pub blck_size: i64,
    pub blck_size_interleave: i64,
    pub type_size: usize,
    pub is_quantized: bool,
    pub to_float: ggml_to_float_t,
    pub from_float_ref: ggml_from_float_t,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of ggml_type_traits"][::std::mem::size_of::<ggml_type_traits>() - 56usize];
    ["Alignment of ggml_type_traits"][::std::mem::align_of::<ggml_type_traits>() - 8usize];
    ["Offset of field: ggml_type_traits::type_name"]
        [::std::mem::offset_of!(ggml_type_traits, type_name) - 0usize];
    ["Offset of field: ggml_type_traits::blck_size"]
        [::std::mem::offset_of!(ggml_type_traits, blck_size) - 8usize];
    ["Offset of field: ggml_type_traits::blck_size_interleave"]
        [::std::mem::offset_of!(ggml_type_traits, blck_size_interleave) - 16usize];
    ["Offset of field: ggml_type_traits::type_size"]
        [::std::mem::offset_of!(ggml_type_traits, type_size) - 24usize];
    ["Offset of field: ggml_type_traits::is_quantized"]
        [::std::mem::offset_of!(ggml_type_traits, is_quantized) - 32usize];
    ["Offset of field: ggml_type_traits::to_float"]
        [::std::mem::offset_of!(ggml_type_traits, to_float) - 40usize];
    ["Offset of field: ggml_type_traits::from_float_ref"]
        [::std::mem::offset_of!(ggml_type_traits, from_float_ref) - 48usize];
};
unsafe extern "C" {
    pub fn ggml_get_type_traits(type_: ggml_type) -> *const ggml_type_traits;
}
pub const ggml_sched_priority_GGML_SCHED_PRIO_NORMAL: ggml_sched_priority = 0;
pub const ggml_sched_priority_GGML_SCHED_PRIO_MEDIUM: ggml_sched_priority = 1;
pub const ggml_sched_priority_GGML_SCHED_PRIO_HIGH: ggml_sched_priority = 2;
pub const ggml_sched_priority_GGML_SCHED_PRIO_REALTIME: ggml_sched_priority = 3;
pub type ggml_sched_priority = ::std::os::raw::c_uint;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_threadpool_params {
    pub cpumask: [bool; 512usize],
    pub n_threads: ::std::os::raw::c_int,
    pub prio: ggml_sched_priority,
    pub poll: u32,
    pub strict_cpu: bool,
    pub paused: bool,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of ggml_threadpool_params"][::std::mem::size_of::<ggml_threadpool_params>() - 528usize];
    ["Alignment of ggml_threadpool_params"]
        [::std::mem::align_of::<ggml_threadpool_params>() - 4usize];
    ["Offset of field: ggml_threadpool_params::cpumask"]
        [::std::mem::offset_of!(ggml_threadpool_params, cpumask) - 0usize];
    ["Offset of field: ggml_threadpool_params::n_threads"]
        [::std::mem::offset_of!(ggml_threadpool_params, n_threads) - 512usize];
    ["Offset of field: ggml_threadpool_params::prio"]
        [::std::mem::offset_of!(ggml_threadpool_params, prio) - 516usize];
    ["Offset of field: ggml_threadpool_params::poll"]
        [::std::mem::offset_of!(ggml_threadpool_params, poll) - 520usize];
    ["Offset of field: ggml_threadpool_params::strict_cpu"]
        [::std::mem::offset_of!(ggml_threadpool_params, strict_cpu) - 524usize];
    ["Offset of field: ggml_threadpool_params::paused"]
        [::std::mem::offset_of!(ggml_threadpool_params, paused) - 525usize];
};
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_threadpool {
    _unused: [u8; 0],
}
pub type ggml_threadpool_t = *mut ggml_threadpool;
unsafe extern "C" {
    pub fn ggml_threadpool_params_default(
        n_threads: ::std::os::raw::c_int,
    ) -> ggml_threadpool_params;
}
unsafe extern "C" {
    pub fn ggml_threadpool_params_init(
        p: *mut ggml_threadpool_params,
        n_threads: ::std::os::raw::c_int,
    );
}
unsafe extern "C" {
    pub fn ggml_threadpool_params_match(
        p0: *const ggml_threadpool_params,
        p1: *const ggml_threadpool_params,
    ) -> bool;
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_backend_buffer_type {
    _unused: [u8; 0],
}
pub type ggml_backend_buffer_type_t = *mut ggml_backend_buffer_type;
pub type ggml_backend_buffer_t = *mut ggml_backend_buffer;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_backend {
    _unused: [u8; 0],
}
pub type ggml_backend_t = *mut ggml_backend;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_tallocr {
    pub buffer: ggml_backend_buffer_t,
    pub base: *mut ::std::os::raw::c_void,
    pub alignment: usize,
    pub offset: usize,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of ggml_tallocr"][::std::mem::size_of::<ggml_tallocr>() - 32usize];
    ["Alignment of ggml_tallocr"][::std::mem::align_of::<ggml_tallocr>() - 8usize];
    ["Offset of field: ggml_tallocr::buffer"]
        [::std::mem::offset_of!(ggml_tallocr, buffer) - 0usize];
    ["Offset of field: ggml_tallocr::base"][::std::mem::offset_of!(ggml_tallocr, base) - 8usize];
    ["Offset of field: ggml_tallocr::alignment"]
        [::std::mem::offset_of!(ggml_tallocr, alignment) - 16usize];
    ["Offset of field: ggml_tallocr::offset"]
        [::std::mem::offset_of!(ggml_tallocr, offset) - 24usize];
};
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
pub type ggml_gallocr_t = *mut ggml_gallocr;
unsafe extern "C" {
    pub fn ggml_gallocr_new(buft: ggml_backend_buffer_type_t) -> ggml_gallocr_t;
}
unsafe extern "C" {
    pub fn ggml_gallocr_new_n(
        bufts: *mut ggml_backend_buffer_type_t,
        n_bufs: ::std::os::raw::c_int,
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
        node_buffer_ids: *const ::std::os::raw::c_int,
        leaf_buffer_ids: *const ::std::os::raw::c_int,
    ) -> bool;
}
unsafe extern "C" {
    pub fn ggml_gallocr_alloc_graph(galloc: ggml_gallocr_t, graph: *mut ggml_cgraph) -> bool;
}
unsafe extern "C" {
    pub fn ggml_gallocr_get_buffer_size(
        galloc: ggml_gallocr_t,
        buffer_id: ::std::os::raw::c_int,
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
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_backend_event {
    _unused: [u8; 0],
}
pub type ggml_backend_event_t = *mut ggml_backend_event;
pub type ggml_backend_graph_plan_t = *mut ::std::os::raw::c_void;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_backend_reg {
    _unused: [u8; 0],
}
pub type ggml_backend_reg_t = *mut ggml_backend_reg;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_backend_device {
    _unused: [u8; 0],
}
pub type ggml_backend_dev_t = *mut ggml_backend_device;
unsafe extern "C" {
    pub fn ggml_backend_buft_name(
        buft: ggml_backend_buffer_type_t,
    ) -> *const ::std::os::raw::c_char;
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
pub const ggml_backend_buffer_usage_GGML_BACKEND_BUFFER_USAGE_ANY: ggml_backend_buffer_usage = 0;
pub const ggml_backend_buffer_usage_GGML_BACKEND_BUFFER_USAGE_WEIGHTS: ggml_backend_buffer_usage =
    1;
pub const ggml_backend_buffer_usage_GGML_BACKEND_BUFFER_USAGE_COMPUTE: ggml_backend_buffer_usage =
    2;
pub type ggml_backend_buffer_usage = ::std::os::raw::c_uint;
unsafe extern "C" {
    pub fn ggml_backend_buffer_name(buffer: ggml_backend_buffer_t)
        -> *const ::std::os::raw::c_char;
}
unsafe extern "C" {
    pub fn ggml_backend_buffer_free(buffer: ggml_backend_buffer_t);
}
unsafe extern "C" {
    pub fn ggml_backend_buffer_get_base(
        buffer: ggml_backend_buffer_t,
    ) -> *mut ::std::os::raw::c_void;
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
    pub fn ggml_backend_name(backend: ggml_backend_t) -> *const ::std::os::raw::c_char;
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
        data: *const ::std::os::raw::c_void,
        offset: usize,
        size: usize,
    );
}
unsafe extern "C" {
    pub fn ggml_backend_tensor_get_async(
        backend: ggml_backend_t,
        tensor: *const ggml_tensor,
        data: *mut ::std::os::raw::c_void,
        offset: usize,
        size: usize,
    );
}
unsafe extern "C" {
    pub fn ggml_backend_tensor_set(
        tensor: *mut ggml_tensor,
        data: *const ::std::os::raw::c_void,
        offset: usize,
        size: usize,
    );
}
unsafe extern "C" {
    pub fn ggml_backend_tensor_get(
        tensor: *const ggml_tensor,
        data: *mut ::std::os::raw::c_void,
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
pub type ggml_backend_dev_type = ::std::os::raw::c_uint;
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
    pub name: *const ::std::os::raw::c_char,
    pub description: *const ::std::os::raw::c_char,
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
    pub fn ggml_backend_dev_name(device: ggml_backend_dev_t) -> *const ::std::os::raw::c_char;
}
unsafe extern "C" {
    pub fn ggml_backend_dev_description(
        device: ggml_backend_dev_t,
    ) -> *const ::std::os::raw::c_char;
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
        params: *const ::std::os::raw::c_char,
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
        ptr: *mut ::std::os::raw::c_void,
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
    pub fn ggml_backend_reg_name(reg: ggml_backend_reg_t) -> *const ::std::os::raw::c_char;
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
        name: *const ::std::os::raw::c_char,
    ) -> *mut ::std::os::raw::c_void;
}
pub type ggml_backend_split_buffer_type_t = ::std::option::Option<
    unsafe extern "C" fn(
        main_device: ::std::os::raw::c_int,
        tensor_split: *const f32,
    ) -> ggml_backend_buffer_type_t,
>;
pub type ggml_backend_set_n_threads_t = ::std::option::Option<
    unsafe extern "C" fn(backend: ggml_backend_t, n_threads: ::std::os::raw::c_int),
>;
pub type ggml_backend_dev_get_extra_bufts_t = ::std::option::Option<
    unsafe extern "C" fn(device: ggml_backend_dev_t) -> *mut ggml_backend_buffer_type_t,
>;
pub type ggml_backend_set_abort_callback_t = ::std::option::Option<
    unsafe extern "C" fn(
        backend: ggml_backend_t,
        abort_callback: ggml_abort_callback,
        abort_callback_data: *mut ::std::os::raw::c_void,
    ),
>;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_backend_feature {
    pub name: *const ::std::os::raw::c_char,
    pub value: *const ::std::os::raw::c_char,
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
pub type ggml_backend_get_features_t = ::std::option::Option<
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
    pub fn ggml_backend_reg_by_name(name: *const ::std::os::raw::c_char) -> ggml_backend_reg_t;
}
unsafe extern "C" {
    pub fn ggml_backend_dev_count() -> usize;
}
unsafe extern "C" {
    pub fn ggml_backend_dev_get(index: usize) -> ggml_backend_dev_t;
}
unsafe extern "C" {
    pub fn ggml_backend_dev_by_name(name: *const ::std::os::raw::c_char) -> ggml_backend_dev_t;
}
unsafe extern "C" {
    pub fn ggml_backend_dev_by_type(type_: ggml_backend_dev_type) -> ggml_backend_dev_t;
}
unsafe extern "C" {
    pub fn ggml_backend_init_by_name(
        name: *const ::std::os::raw::c_char,
        params: *const ::std::os::raw::c_char,
    ) -> ggml_backend_t;
}
unsafe extern "C" {
    pub fn ggml_backend_init_by_type(
        type_: ggml_backend_dev_type,
        params: *const ::std::os::raw::c_char,
    ) -> ggml_backend_t;
}
unsafe extern "C" {
    pub fn ggml_backend_init_best() -> ggml_backend_t;
}
unsafe extern "C" {
    pub fn ggml_backend_load(path: *const ::std::os::raw::c_char) -> ggml_backend_reg_t;
}
unsafe extern "C" {
    pub fn ggml_backend_unload(reg: ggml_backend_reg_t);
}
unsafe extern "C" {
    pub fn ggml_backend_load_all();
}
unsafe extern "C" {
    pub fn ggml_backend_load_all_from_path(dir_path: *const ::std::os::raw::c_char);
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_backend_sched {
    _unused: [u8; 0],
}
pub type ggml_backend_sched_t = *mut ggml_backend_sched;
pub type ggml_backend_sched_eval_callback = ::std::option::Option<
    unsafe extern "C" fn(
        t: *mut ggml_tensor,
        ask: bool,
        user_data: *mut ::std::os::raw::c_void,
    ) -> bool,
>;
unsafe extern "C" {
    pub fn ggml_backend_sched_new(
        backends: *mut ggml_backend_t,
        bufts: *mut ggml_backend_buffer_type_t,
        n_backends: ::std::os::raw::c_int,
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
    pub fn ggml_backend_sched_get_n_backends(sched: ggml_backend_sched_t) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_backend_sched_get_backend(
        sched: ggml_backend_sched_t,
        i: ::std::os::raw::c_int,
    ) -> ggml_backend_t;
}
unsafe extern "C" {
    pub fn ggml_backend_sched_get_n_splits(sched: ggml_backend_sched_t) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_backend_sched_get_n_copies(sched: ggml_backend_sched_t) -> ::std::os::raw::c_int;
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
        user_data: *mut ::std::os::raw::c_void,
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
pub type ggml_backend_eval_callback = ::std::option::Option<
    unsafe extern "C" fn(
        node_index: ::std::os::raw::c_int,
        t1: *mut ggml_tensor,
        t2: *mut ggml_tensor,
        user_data: *mut ::std::os::raw::c_void,
    ) -> bool,
>;
unsafe extern "C" {
    pub fn ggml_backend_compare_graph_backend(
        backend1: ggml_backend_t,
        backend2: ggml_backend_t,
        graph: *mut ggml_cgraph,
        callback: ggml_backend_eval_callback,
        user_data: *mut ::std::os::raw::c_void,
    ) -> bool;
}
unsafe extern "C" {
    pub fn ggml_backend_tensor_alloc(
        buffer: ggml_backend_buffer_t,
        tensor: *mut ggml_tensor,
        addr: *mut ::std::os::raw::c_void,
    );
}
unsafe extern "C" {
    pub fn ggml_backend_view_init(tensor: *mut ggml_tensor);
}
unsafe extern "C" {
    pub fn ggml_backend_cpu_buffer_from_ptr(
        ptr: *mut ::std::os::raw::c_void,
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
    pub n_threads: ::std::os::raw::c_int,
    pub threadpool: *mut ggml_threadpool,
    pub abort_callback: ggml_abort_callback,
    pub abort_callback_data: *mut ::std::os::raw::c_void,
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
pub type ggml_numa_strategy = ::std::os::raw::c_uint;
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
    pub fn ggml_get_i32_1d(tensor: *const ggml_tensor, i: ::std::os::raw::c_int) -> i32;
}
unsafe extern "C" {
    pub fn ggml_set_i32_1d(tensor: *const ggml_tensor, i: ::std::os::raw::c_int, value: i32);
}
unsafe extern "C" {
    pub fn ggml_get_i32_nd(
        tensor: *const ggml_tensor,
        i0: ::std::os::raw::c_int,
        i1: ::std::os::raw::c_int,
        i2: ::std::os::raw::c_int,
        i3: ::std::os::raw::c_int,
    ) -> i32;
}
unsafe extern "C" {
    pub fn ggml_set_i32_nd(
        tensor: *const ggml_tensor,
        i0: ::std::os::raw::c_int,
        i1: ::std::os::raw::c_int,
        i2: ::std::os::raw::c_int,
        i3: ::std::os::raw::c_int,
        value: i32,
    );
}
unsafe extern "C" {
    pub fn ggml_get_f32_1d(tensor: *const ggml_tensor, i: ::std::os::raw::c_int) -> f32;
}
unsafe extern "C" {
    pub fn ggml_set_f32_1d(tensor: *const ggml_tensor, i: ::std::os::raw::c_int, value: f32);
}
unsafe extern "C" {
    pub fn ggml_get_f32_nd(
        tensor: *const ggml_tensor,
        i0: ::std::os::raw::c_int,
        i1: ::std::os::raw::c_int,
        i2: ::std::os::raw::c_int,
        i3: ::std::os::raw::c_int,
    ) -> f32;
}
unsafe extern "C" {
    pub fn ggml_set_f32_nd(
        tensor: *const ggml_tensor,
        i0: ::std::os::raw::c_int,
        i1: ::std::os::raw::c_int,
        i2: ::std::os::raw::c_int,
        i3: ::std::os::raw::c_int,
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
    pub fn ggml_threadpool_get_n_threads(threadpool: *mut ggml_threadpool)
        -> ::std::os::raw::c_int;
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
        n_threads: ::std::os::raw::c_int,
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
        n_threads: ::std::os::raw::c_int,
    ) -> ggml_status;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_sse3() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_ssse3() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_avx() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_avx_vnni() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_avx2() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_f16c() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_fma() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_avx512() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_avx512_vbmi() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_avx512_vnni() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_avx512_bf16() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_amx_int8() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_neon() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_arm_fma() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_fp16_va() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_dotprod() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_matmul_int8() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_sve() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_get_sve_cnt() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_riscv_v() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_vsx() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_wasm_simd() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn ggml_cpu_has_llamafile() -> ::std::os::raw::c_int;
}
pub type ggml_vec_dot_t = ::std::option::Option<
    unsafe extern "C" fn(
        n: ::std::os::raw::c_int,
        s: *mut f32,
        bs: usize,
        x: *const ::std::os::raw::c_void,
        bx: usize,
        y: *const ::std::os::raw::c_void,
        by: usize,
        nrc: ::std::os::raw::c_int,
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
    pub fn ggml_backend_cpu_set_n_threads(
        backend_cpu: ggml_backend_t,
        n_threads: ::std::os::raw::c_int,
    );
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
        abort_callback_data: *mut ::std::os::raw::c_void,
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
pub type whisper_alignment_heads_preset = ::std::os::raw::c_uint;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct whisper_ahead {
    pub n_text_layer: ::std::os::raw::c_int,
    pub n_head: ::std::os::raw::c_int,
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
    pub gpu_device: ::std::os::raw::c_int,
    pub dtw_token_timestamps: bool,
    pub dtw_aheads_preset: whisper_alignment_heads_preset,
    pub dtw_n_top: ::std::os::raw::c_int,
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
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of whisper_token_data"][::std::mem::size_of::<whisper_token_data>() - 56usize];
    ["Alignment of whisper_token_data"][::std::mem::align_of::<whisper_token_data>() - 8usize];
    ["Offset of field: whisper_token_data::id"]
        [::std::mem::offset_of!(whisper_token_data, id) - 0usize];
    ["Offset of field: whisper_token_data::tid"]
        [::std::mem::offset_of!(whisper_token_data, tid) - 4usize];
    ["Offset of field: whisper_token_data::p"]
        [::std::mem::offset_of!(whisper_token_data, p) - 8usize];
    ["Offset of field: whisper_token_data::plog"]
        [::std::mem::offset_of!(whisper_token_data, plog) - 12usize];
    ["Offset of field: whisper_token_data::pt"]
        [::std::mem::offset_of!(whisper_token_data, pt) - 16usize];
    ["Offset of field: whisper_token_data::ptsum"]
        [::std::mem::offset_of!(whisper_token_data, ptsum) - 20usize];
    ["Offset of field: whisper_token_data::t0"]
        [::std::mem::offset_of!(whisper_token_data, t0) - 24usize];
    ["Offset of field: whisper_token_data::t1"]
        [::std::mem::offset_of!(whisper_token_data, t1) - 32usize];
    ["Offset of field: whisper_token_data::t_dtw"]
        [::std::mem::offset_of!(whisper_token_data, t_dtw) - 40usize];
    ["Offset of field: whisper_token_data::vlen"]
        [::std::mem::offset_of!(whisper_token_data, vlen) - 48usize];
};
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct whisper_model_loader {
    pub context: *mut ::std::os::raw::c_void,
    pub read: ::std::option::Option<
        unsafe extern "C" fn(
            ctx: *mut ::std::os::raw::c_void,
            output: *mut ::std::os::raw::c_void,
            read_size: usize,
        ) -> usize,
    >,
    pub eof: ::std::option::Option<unsafe extern "C" fn(ctx: *mut ::std::os::raw::c_void) -> bool>,
    pub close: ::std::option::Option<unsafe extern "C" fn(ctx: *mut ::std::os::raw::c_void)>,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of whisper_model_loader"][::std::mem::size_of::<whisper_model_loader>() - 32usize];
    ["Alignment of whisper_model_loader"][::std::mem::align_of::<whisper_model_loader>() - 8usize];
    ["Offset of field: whisper_model_loader::context"]
        [::std::mem::offset_of!(whisper_model_loader, context) - 0usize];
    ["Offset of field: whisper_model_loader::read"]
        [::std::mem::offset_of!(whisper_model_loader, read) - 8usize];
    ["Offset of field: whisper_model_loader::eof"]
        [::std::mem::offset_of!(whisper_model_loader, eof) - 16usize];
    ["Offset of field: whisper_model_loader::close"]
        [::std::mem::offset_of!(whisper_model_loader, close) - 24usize];
};
pub const whisper_gretype_WHISPER_GRETYPE_END: whisper_gretype = 0;
pub const whisper_gretype_WHISPER_GRETYPE_ALT: whisper_gretype = 1;
pub const whisper_gretype_WHISPER_GRETYPE_RULE_REF: whisper_gretype = 2;
pub const whisper_gretype_WHISPER_GRETYPE_CHAR: whisper_gretype = 3;
pub const whisper_gretype_WHISPER_GRETYPE_CHAR_NOT: whisper_gretype = 4;
pub const whisper_gretype_WHISPER_GRETYPE_CHAR_RNG_UPPER: whisper_gretype = 5;
pub const whisper_gretype_WHISPER_GRETYPE_CHAR_ALT: whisper_gretype = 6;
pub type whisper_gretype = ::std::os::raw::c_uint;
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
        path_model: *const ::std::os::raw::c_char,
        params: whisper_context_params,
    ) -> *mut whisper_context;
}
unsafe extern "C" {
    pub fn whisper_init_from_buffer_with_params(
        buffer: *mut ::std::os::raw::c_void,
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
        path_model: *const ::std::os::raw::c_char,
        params: whisper_context_params,
    ) -> *mut whisper_context;
}
unsafe extern "C" {
    pub fn whisper_init_from_buffer_with_params_no_state(
        buffer: *mut ::std::os::raw::c_void,
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
    pub fn whisper_init_from_file(
        path_model: *const ::std::os::raw::c_char,
    ) -> *mut whisper_context;
}
unsafe extern "C" {
    pub fn whisper_init_from_buffer(
        buffer: *mut ::std::os::raw::c_void,
        buffer_size: usize,
    ) -> *mut whisper_context;
}
unsafe extern "C" {
    pub fn whisper_init(loader: *mut whisper_model_loader) -> *mut whisper_context;
}
unsafe extern "C" {
    pub fn whisper_init_from_file_no_state(
        path_model: *const ::std::os::raw::c_char,
    ) -> *mut whisper_context;
}
unsafe extern "C" {
    pub fn whisper_init_from_buffer_no_state(
        buffer: *mut ::std::os::raw::c_void,
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
        model_path: *const ::std::os::raw::c_char,
        device: *const ::std::os::raw::c_char,
        cache_dir: *const ::std::os::raw::c_char,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_ctx_init_openvino_encoder(
        ctx: *mut whisper_context,
        model_path: *const ::std::os::raw::c_char,
        device: *const ::std::os::raw::c_char,
        cache_dir: *const ::std::os::raw::c_char,
    ) -> ::std::os::raw::c_int;
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
        n_samples: ::std::os::raw::c_int,
        n_threads: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_pcm_to_mel_with_state(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        samples: *const f32,
        n_samples: ::std::os::raw::c_int,
        n_threads: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_set_mel(
        ctx: *mut whisper_context,
        data: *const f32,
        n_len: ::std::os::raw::c_int,
        n_mel: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_set_mel_with_state(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        data: *const f32,
        n_len: ::std::os::raw::c_int,
        n_mel: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_encode(
        ctx: *mut whisper_context,
        offset: ::std::os::raw::c_int,
        n_threads: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_encode_with_state(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        offset: ::std::os::raw::c_int,
        n_threads: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_decode(
        ctx: *mut whisper_context,
        tokens: *const whisper_token,
        n_tokens: ::std::os::raw::c_int,
        n_past: ::std::os::raw::c_int,
        n_threads: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_decode_with_state(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        tokens: *const whisper_token,
        n_tokens: ::std::os::raw::c_int,
        n_past: ::std::os::raw::c_int,
        n_threads: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_tokenize(
        ctx: *mut whisper_context,
        text: *const ::std::os::raw::c_char,
        tokens: *mut whisper_token,
        n_max_tokens: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_token_count(
        ctx: *mut whisper_context,
        text: *const ::std::os::raw::c_char,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_lang_max_id() -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_lang_id(lang: *const ::std::os::raw::c_char) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_lang_str(id: ::std::os::raw::c_int) -> *const ::std::os::raw::c_char;
}
unsafe extern "C" {
    pub fn whisper_lang_str_full(id: ::std::os::raw::c_int) -> *const ::std::os::raw::c_char;
}
unsafe extern "C" {
    pub fn whisper_lang_auto_detect(
        ctx: *mut whisper_context,
        offset_ms: ::std::os::raw::c_int,
        n_threads: ::std::os::raw::c_int,
        lang_probs: *mut f32,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_lang_auto_detect_with_state(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        offset_ms: ::std::os::raw::c_int,
        n_threads: ::std::os::raw::c_int,
        lang_probs: *mut f32,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_n_len(ctx: *mut whisper_context) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_n_len_from_state(state: *mut whisper_state) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_n_vocab(ctx: *mut whisper_context) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_n_text_ctx(ctx: *mut whisper_context) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_n_audio_ctx(ctx: *mut whisper_context) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_is_multilingual(ctx: *mut whisper_context) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_n_vocab(ctx: *mut whisper_context) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_n_audio_ctx(ctx: *mut whisper_context) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_n_audio_state(ctx: *mut whisper_context) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_n_audio_head(ctx: *mut whisper_context) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_n_audio_layer(ctx: *mut whisper_context) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_n_text_ctx(ctx: *mut whisper_context) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_n_text_state(ctx: *mut whisper_context) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_n_text_head(ctx: *mut whisper_context) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_n_text_layer(ctx: *mut whisper_context) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_n_mels(ctx: *mut whisper_context) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_ftype(ctx: *mut whisper_context) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_model_type(ctx: *mut whisper_context) -> ::std::os::raw::c_int;
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
    ) -> *const ::std::os::raw::c_char;
}
unsafe extern "C" {
    pub fn whisper_model_type_readable(ctx: *mut whisper_context) -> *const ::std::os::raw::c_char;
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
        lang_id: ::std::os::raw::c_int,
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
    pub fn whisper_print_system_info() -> *const ::std::os::raw::c_char;
}
pub const whisper_sampling_strategy_WHISPER_SAMPLING_GREEDY: whisper_sampling_strategy = 0;
pub const whisper_sampling_strategy_WHISPER_SAMPLING_BEAM_SEARCH: whisper_sampling_strategy = 1;
pub type whisper_sampling_strategy = ::std::os::raw::c_uint;
pub type whisper_new_segment_callback = ::std::option::Option<
    unsafe extern "C" fn(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        n_new: ::std::os::raw::c_int,
        user_data: *mut ::std::os::raw::c_void,
    ),
>;
pub type whisper_progress_callback = ::std::option::Option<
    unsafe extern "C" fn(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        progress: ::std::os::raw::c_int,
        user_data: *mut ::std::os::raw::c_void,
    ),
>;
pub type whisper_encoder_begin_callback = ::std::option::Option<
    unsafe extern "C" fn(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        user_data: *mut ::std::os::raw::c_void,
    ) -> bool,
>;
pub type whisper_logits_filter_callback = ::std::option::Option<
    unsafe extern "C" fn(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        tokens: *const whisper_token_data,
        n_tokens: ::std::os::raw::c_int,
        logits: *mut f32,
        user_data: *mut ::std::os::raw::c_void,
    ),
>;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct whisper_full_params {
    pub strategy: whisper_sampling_strategy,
    pub n_threads: ::std::os::raw::c_int,
    pub n_max_text_ctx: ::std::os::raw::c_int,
    pub offset_ms: ::std::os::raw::c_int,
    pub duration_ms: ::std::os::raw::c_int,
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
    pub max_len: ::std::os::raw::c_int,
    pub split_on_word: bool,
    pub max_tokens: ::std::os::raw::c_int,
    pub debug_mode: bool,
    pub audio_ctx: ::std::os::raw::c_int,
    pub tdrz_enable: bool,
    pub suppress_regex: *const ::std::os::raw::c_char,
    pub initial_prompt: *const ::std::os::raw::c_char,
    pub prompt_tokens: *const whisper_token,
    pub prompt_n_tokens: ::std::os::raw::c_int,
    pub language: *const ::std::os::raw::c_char,
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
    pub new_segment_callback_user_data: *mut ::std::os::raw::c_void,
    pub progress_callback: whisper_progress_callback,
    pub progress_callback_user_data: *mut ::std::os::raw::c_void,
    pub encoder_begin_callback: whisper_encoder_begin_callback,
    pub encoder_begin_callback_user_data: *mut ::std::os::raw::c_void,
    pub abort_callback: ggml_abort_callback,
    pub abort_callback_user_data: *mut ::std::os::raw::c_void,
    pub logits_filter_callback: whisper_logits_filter_callback,
    pub logits_filter_callback_user_data: *mut ::std::os::raw::c_void,
    pub grammar_rules: *mut *const whisper_grammar_element,
    pub n_grammar_rules: usize,
    pub i_start_rule: usize,
    pub grammar_penalty: f32,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct whisper_full_params__bindgen_ty_1 {
    pub best_of: ::std::os::raw::c_int,
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
    pub beam_size: ::std::os::raw::c_int,
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
        n_samples: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_full_with_state(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        params: whisper_full_params,
        samples: *const f32,
        n_samples: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_full_parallel(
        ctx: *mut whisper_context,
        params: whisper_full_params,
        samples: *const f32,
        n_samples: ::std::os::raw::c_int,
        n_processors: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_full_n_segments(ctx: *mut whisper_context) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_full_n_segments_from_state(state: *mut whisper_state) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_full_lang_id(ctx: *mut whisper_context) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_full_lang_id_from_state(state: *mut whisper_state) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_full_get_segment_t0(
        ctx: *mut whisper_context,
        i_segment: ::std::os::raw::c_int,
    ) -> i64;
}
unsafe extern "C" {
    pub fn whisper_full_get_segment_t0_from_state(
        state: *mut whisper_state,
        i_segment: ::std::os::raw::c_int,
    ) -> i64;
}
unsafe extern "C" {
    pub fn whisper_full_get_segment_t1(
        ctx: *mut whisper_context,
        i_segment: ::std::os::raw::c_int,
    ) -> i64;
}
unsafe extern "C" {
    pub fn whisper_full_get_segment_t1_from_state(
        state: *mut whisper_state,
        i_segment: ::std::os::raw::c_int,
    ) -> i64;
}
unsafe extern "C" {
    pub fn whisper_full_get_segment_speaker_turn_next(
        ctx: *mut whisper_context,
        i_segment: ::std::os::raw::c_int,
    ) -> bool;
}
unsafe extern "C" {
    pub fn whisper_full_get_segment_speaker_turn_next_from_state(
        state: *mut whisper_state,
        i_segment: ::std::os::raw::c_int,
    ) -> bool;
}
unsafe extern "C" {
    pub fn whisper_full_get_segment_text(
        ctx: *mut whisper_context,
        i_segment: ::std::os::raw::c_int,
    ) -> *const ::std::os::raw::c_char;
}
unsafe extern "C" {
    pub fn whisper_full_get_segment_text_from_state(
        state: *mut whisper_state,
        i_segment: ::std::os::raw::c_int,
    ) -> *const ::std::os::raw::c_char;
}
unsafe extern "C" {
    pub fn whisper_full_n_tokens(
        ctx: *mut whisper_context,
        i_segment: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_full_n_tokens_from_state(
        state: *mut whisper_state,
        i_segment: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_full_get_token_text(
        ctx: *mut whisper_context,
        i_segment: ::std::os::raw::c_int,
        i_token: ::std::os::raw::c_int,
    ) -> *const ::std::os::raw::c_char;
}
unsafe extern "C" {
    pub fn whisper_full_get_token_text_from_state(
        ctx: *mut whisper_context,
        state: *mut whisper_state,
        i_segment: ::std::os::raw::c_int,
        i_token: ::std::os::raw::c_int,
    ) -> *const ::std::os::raw::c_char;
}
unsafe extern "C" {
    pub fn whisper_full_get_token_id(
        ctx: *mut whisper_context,
        i_segment: ::std::os::raw::c_int,
        i_token: ::std::os::raw::c_int,
    ) -> whisper_token;
}
unsafe extern "C" {
    pub fn whisper_full_get_token_id_from_state(
        state: *mut whisper_state,
        i_segment: ::std::os::raw::c_int,
        i_token: ::std::os::raw::c_int,
    ) -> whisper_token;
}
unsafe extern "C" {
    pub fn whisper_full_get_token_data(
        ctx: *mut whisper_context,
        i_segment: ::std::os::raw::c_int,
        i_token: ::std::os::raw::c_int,
    ) -> whisper_token_data;
}
unsafe extern "C" {
    pub fn whisper_full_get_token_data_from_state(
        state: *mut whisper_state,
        i_segment: ::std::os::raw::c_int,
        i_token: ::std::os::raw::c_int,
    ) -> whisper_token_data;
}
unsafe extern "C" {
    pub fn whisper_full_get_token_p(
        ctx: *mut whisper_context,
        i_segment: ::std::os::raw::c_int,
        i_token: ::std::os::raw::c_int,
    ) -> f32;
}
unsafe extern "C" {
    pub fn whisper_full_get_token_p_from_state(
        state: *mut whisper_state,
        i_segment: ::std::os::raw::c_int,
        i_token: ::std::os::raw::c_int,
    ) -> f32;
}
unsafe extern "C" {
    pub fn whisper_bench_memcpy(n_threads: ::std::os::raw::c_int) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_bench_memcpy_str(
        n_threads: ::std::os::raw::c_int,
    ) -> *const ::std::os::raw::c_char;
}
unsafe extern "C" {
    pub fn whisper_bench_ggml_mul_mat(n_threads: ::std::os::raw::c_int) -> ::std::os::raw::c_int;
}
unsafe extern "C" {
    pub fn whisper_bench_ggml_mul_mat_str(
        n_threads: ::std::os::raw::c_int,
    ) -> *const ::std::os::raw::c_char;
}
unsafe extern "C" {
    pub fn whisper_log_set(log_callback: ggml_log_callback, user_data: *mut ::std::os::raw::c_void);
}
unsafe extern "C" {
    pub fn whisper_full_get_segment_no_speech_prob(
        ctx: *mut whisper_context,
        i_segment: ::std::os::raw::c_int,
    ) -> f32;
}
unsafe extern "C" {
    pub fn whisper_full_get_segment_no_speech_prob_from_state(
        state: *mut whisper_state,
        i_segment: ::std::os::raw::c_int,
    ) -> f32;
}
pub type __builtin_va_list = [__va_list_tag; 1usize];
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct __va_list_tag {
    pub gp_offset: ::std::os::raw::c_uint,
    pub fp_offset: ::std::os::raw::c_uint,
    pub overflow_arg_area: *mut ::std::os::raw::c_void,
    pub reg_save_area: *mut ::std::os::raw::c_void,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of __va_list_tag"][::std::mem::size_of::<__va_list_tag>() - 24usize];
    ["Alignment of __va_list_tag"][::std::mem::align_of::<__va_list_tag>() - 8usize];
    ["Offset of field: __va_list_tag::gp_offset"]
        [::std::mem::offset_of!(__va_list_tag, gp_offset) - 0usize];
    ["Offset of field: __va_list_tag::fp_offset"]
        [::std::mem::offset_of!(__va_list_tag, fp_offset) - 4usize];
    ["Offset of field: __va_list_tag::overflow_arg_area"]
        [::std::mem::offset_of!(__va_list_tag, overflow_arg_area) - 8usize];
    ["Offset of field: __va_list_tag::reg_save_area"]
        [::std::mem::offset_of!(__va_list_tag, reg_save_area) - 16usize];
};
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_backend_buffer {
    pub _address: u8,
}
