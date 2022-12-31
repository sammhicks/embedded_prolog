use std::io::{BufRead, Read, Write};

struct Channels {
    stream: std::io::BufReader<std::net::TcpStream>,
    read_buffer: Vec<u8>,
}

impl wam::SerialReadWrite for Channels {
    type Error = std::io::Error;

    fn read_one(&mut self) -> Result<u8, Self::Error> {
        let mut byte = 0;

        self.stream.read_exact(std::slice::from_mut(&mut byte))?;

        Ok(byte)
    }

    fn read_until_zero<T, F: FnOnce(&mut [u8]) -> T>(&mut self, f: F) -> Result<T, Self::Error> {
        self.read_buffer.clear();
        self.stream.read_until(0, &mut self.read_buffer)?;

        Ok(f(&mut self.read_buffer))
    }

    fn read_exact(&mut self, buffer: &mut [u8]) -> Result<(), Self::Error> {
        self.stream.read_exact(buffer)
    }

    fn write_all(&mut self, buffer: &[u8]) -> Result<(), Self::Error> {
        self.stream.get_mut().write_all(buffer)
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        self.stream.get_mut().flush()
    }
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
            serial_connection: wam::SerialConnection::new(Channels {
                stream: std::io::BufReader::new(stream),
                read_buffer: Vec::new(),
            }),
            system_calls: wam::system_calls(
                &[&wam::system_call_handler("hello", || println!("Hello"))],
                (),
            ),
        })
        .run()
        {
            Ok(never) => match never {},
            Err(io_error) => {
                wam::log_error!("{}", io_error)
            }
        }
    }
}
