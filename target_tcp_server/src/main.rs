use std::io::{Read, Write};

fn handle_channel_error(err: std::io::Error) -> wam::nb::Error<std::io::Error> {
    if let std::io::ErrorKind::WouldBlock = err.kind() {
        wam::nb::Error::WouldBlock
    } else {
        wam::nb::Error::Other(err)
    }
}

struct Channels(std::net::TcpStream);

impl wam::SerialRead<u8> for Channels {
    type Error = std::io::Error;

    fn read(&mut self) -> wam::nb::Result<u8, Self::Error> {
        let mut buffer = [0; 1];
        self.0
            .read_exact(&mut buffer)
            .map_err(handle_channel_error)?;
        Ok(buffer[0])
    }
}

impl wam::SerialWrite<u8> for Channels {
    type Error = std::io::Error;

    fn write(&mut self, word: u8) -> wam::nb::Result<(), Self::Error> {
        let buffer = [word];
        self.0.write_all(&buffer).map_err(handle_channel_error)
    }

    fn flush(&mut self) -> wam::nb::Result<(), Self::Error> {
        self.0.flush().map_err(handle_channel_error)
    }
}

#[derive(Default)]
struct SystemState {
    value: i32,
}

impl SystemState {
    fn hello() {
        println!("Hello");
    }

    fn get(&self) -> i32 {
        self.value
    }

    fn put(&mut self, value: i32) {
        self.value = value;
    }

    fn increment(&mut self) {
        self.value += 1;
    }
}

macro_rules! system_calls {
    ($($call:ident),*) => {
        ::wam::system_calls(
            &[$(&::wam::system_call_handler(stringify!($call), SystemState::$call)),*],
            SystemState::default(),
        )
    };
}

#[derive(Debug)]
struct DisplayError(Box<dyn std::error::Error>);

impl<E: std::error::Error + 'static> From<E> for DisplayError {
    fn from(error: E) -> Self {
        Self(error.into())
    }
}

fn main() -> Result<(), DisplayError> {
    pretty_env_logger::init();

    let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 8080))?;

    loop {
        let (stream, _) = listener.accept()?;

        let mut memory = [0; 32768];

        match (wam::Device {
            memory: &mut memory,
            serial_connection: wam::SerialConnection(Channels(stream)),
            system_calls: system_calls!(hello, get, put, increment),
        })
        .run()
        {
            Ok(never) => match never {},
            Err(wam::IoError) => {}
        }
    }
}
