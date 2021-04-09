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

fn do_main() -> std::io::Result<wam::Never> {
    let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 8080))?;

    loop {
        let (stream, remote) = listener.accept()?;

        wam::log::info!("Connection from {}", remote);

        let mut memory = [0; 32768];

        let _ = wam::Device {
            memory: &mut memory,
            serial_connection: wam::SerialConnection(Channels(stream)),
        }
        .run();
    }
}

fn main() {
    pretty_env_logger::init();

    if let Err(err) = do_main() {
        eprintln!("{}", err);
    }
}
