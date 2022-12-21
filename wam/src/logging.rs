#[cfg(feature = "logging")]
#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => (
        $crate::log::error!($($arg)*)
    )
}

#[cfg(not(feature = "logging"))]
#[macro_export]
macro_rules! log_error {
    ($($arg:expr),*) => {$(let _ = &$arg;)*};
}

#[cfg(feature = "logging")]
#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => (
        $crate::log::warn!($($arg)*)
    )
}

#[cfg(not(feature = "logging"))]
#[macro_export]
macro_rules! log_warn {
    ($($arg:expr),*) => {$(let _ = &$arg;)*};
}

#[cfg(feature = "logging")]
#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => (
        $crate::log::info!($($arg)*)
    )
}

#[cfg(not(feature = "logging"))]
#[macro_export]
macro_rules! log_info {
    ($($arg:expr),*) => {$(let _ = &$arg;)*};
}

#[cfg(feature = "logging")]
#[macro_export]
macro_rules! log_debug {
    ($($arg:tt)*) => (
        $crate::log::debug!($($arg)*)
    )
}

#[cfg(not(feature = "logging"))]
#[macro_export]
macro_rules! log_debug {
    ($($arg:expr),*) => {$(let _ = &$arg;)*};
}

#[cfg(feature = "logging")]
#[macro_export]
macro_rules! log_trace {
    ($($arg:tt)*) => (
        $crate::log::trace!($($arg)*)
    )
}

#[cfg(not(feature = "logging"))]
#[macro_export]
macro_rules! log_trace {
    ($($arg:expr),*) => {$(let _ = &$arg;)*};
}
