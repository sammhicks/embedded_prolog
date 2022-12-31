use std::sync::mpsc;

pub struct Channels {
    tx: mpsc::Sender<u8>,
    rx: mpsc::Receiver<u8>,
}

impl std::io::Read for Channels {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        Ok(match self.rx.recv() {
            Ok(byte) => {
                buf[0] = byte;
                1
            }
            Err(mpsc::RecvError) => 0,
        })
    }
}

impl std::io::Write for Channels {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        for &byte in buf {
            match self.tx.send(byte) {
                Ok(()) => (),
                Err(mpsc::SendError(_)) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::ConnectionReset,
                        "Simulator has stopped",
                    ))
                }
            }
        }

        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[derive(Debug)]
pub struct Disconnection;

impl From<mpsc::RecvError> for Disconnection {
    fn from(mpsc::RecvError: mpsc::RecvError) -> Self {
        Self
    }
}

impl From<mpsc::SendError<u8>> for Disconnection {
    fn from(mpsc::SendError(_): mpsc::SendError<u8>) -> Self {
        Self
    }
}

impl wam::SerialReadWrite for Channels {
    type Error = Disconnection;

    fn read_until_zero<T, F: FnOnce(&mut [u8]) -> T>(&mut self, f: F) -> Result<T, Self::Error> {
        let mut buffer = Vec::new();

        loop {
            let byte = self.rx.recv()?;

            buffer.push(byte);

            if byte == 0 {
                return Ok(f(&mut buffer));
            }
        }
    }

    fn read_exact(&mut self, buffer: &mut [u8]) -> Result<(), Self::Error> {
        for slot in buffer {
            *slot = self.rx.recv()?;
        }

        Ok(())
    }

    fn write_all(&mut self, buffer: &[u8]) -> Result<(), Self::Error> {
        for &byte in buffer {
            self.tx.send(byte)?;
        }

        Ok(())
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

pub fn run() -> Channels {
    let (to_simulator, from_terminal) = mpsc::channel();
    let (to_terminal, from_simulator) = mpsc::channel();

    std::thread::spawn(move || {
        let mut memory = [0; 32768];

        match (wam::Device {
            memory: &mut memory,
            serial_connection: wam::SerialConnection::new(Channels {
                tx: to_terminal,
                rx: from_terminal,
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
    });

    Channels {
        tx: to_simulator,
        rx: from_simulator,
    }
}
