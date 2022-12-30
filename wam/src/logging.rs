#[macro_export]
macro_rules! log_error {
    ($($arg:expr),*) => {{
        $(let _ = &$arg;)*

        #[cfg(feature = "logging")]
        $crate::log::error!($($arg),*);

        #[cfg(feature = "defmt-logging")]
        $crate::defmt::error!($($arg),*);
    }};
}

#[macro_export]
macro_rules! log_warn {
    ($($arg:expr),*) => {{
        $(let _ = &$arg;)*

        #[cfg(feature = "logging")]
        $crate::log::warn!($($arg),*);

        #[cfg(feature = "defmt-logging")]
        $crate::defmt::warn!($($arg),*);
    }};
}

#[macro_export]
macro_rules! log_info {
    ($($arg:expr),*) => {{
        $(let _ = &$arg;)*

        #[cfg(feature = "logging")]
        $crate::log::info!($($arg),*);

        #[cfg(feature = "defmt-logging")]
        $crate::defmt::info!($($arg),*);
    }};
}

#[macro_export]
macro_rules! log_debug {
    ($($arg:expr),*) => {{
        $(let _ = &$arg;)*

        #[cfg(feature = "logging")]
        $crate::log::debug!($($arg),*);

        #[cfg(feature = "defmt-logging")]
        $crate::defmt::debug!($($arg),*);
    }};
}

#[macro_export]
macro_rules! log_trace {
    ($($arg:expr),*) => {{
        $(let _ = &$arg;)*

        #[cfg(feature = "logging")]
        $crate::log::trace!($($arg),*);

        #[cfg(feature = "defmt-logging")]
        $crate::defmt::trace!($($arg),*);
    }};
}
