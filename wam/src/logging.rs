#[macro_export]
macro_rules! log_error {
    ($($arg:expr),*) => {{
        $(let _ = &$arg;)*

        #[cfg(feature = "logging")]
        ::log::error!($($arg),*);

        #[cfg(feature = "defmt-logging")]
        ::defmt::error!($($arg),*);
    }};
}

#[macro_export]
macro_rules! log_warn {
    ($($arg:expr),*) => {{
        $(let _ = &$arg;)*

        #[cfg(feature = "logging")]
        ::log::warn!($($arg),*);

        #[cfg(feature = "defmt-logging")]
        ::defmt::warn!($($arg),*);
    }};
}

#[macro_export]
macro_rules! log_info {
    ($($arg:expr),*) => {{
        $(let _ = &$arg;)*

        #[cfg(feature = "logging")]
        ::log::info!($($arg),*);

        #[cfg(feature = "defmt-logging")]
        ::defmt::info!($($arg),*);
    }};
}

#[macro_export]
macro_rules! log_debug {
    ($($arg:expr),*) => {{
        $(let _ = &$arg;)*

        #[cfg(feature = "logging")]
        ::log::debug!($($arg),*);

        #[cfg(feature = "defmt-logging")]
        ::defmt::debug!($($arg),*);
    }};
}

#[macro_export]
macro_rules! log_trace {
    ($($arg:expr),*) => {{
        $(let _ = &$arg;)*

        #[cfg(feature = "logging")]
        ::log::trace!($($arg),*);

        #[cfg(feature = "defmt-logging")]
        ::defmt::trace!($($arg),*);
    }};
}
