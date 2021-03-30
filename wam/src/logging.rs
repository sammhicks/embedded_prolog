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
    ($($arg:tt)*) => {};
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
    ($($arg:tt)*) => {};
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
    ($($arg:tt)*) => {};
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
    ($($arg:tt)*) => {};
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
    ($($arg:tt)*) => {};
}
