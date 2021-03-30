use std::io::{Read, Write};

/*use std::io::{self, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
// use std::thread;

use embedded_hal::serial;

fn handle_client(
    mut stream: TcpStream,
    sender: crossbeam_channel::Sender<u8>,
    receiver: crossbeam_channel::Receiver<u8>,
) -> io::Result<()> {
    loop {
        {
            let mut buffer = [0; 1];
            match stream.read(&mut buffer) {
                Ok(0) => {
                    println!("Disconnection");
                    return Ok(());
                }
                Ok(_) => {
                    println!("Data: {:?}", buffer[0] as char);
                    sender
                        .send(buffer[0])
                        .map_err(|e| io::Error::new(ErrorKind::Other, e))?;
                }
                Err(e) => {
                    if let std::io::ErrorKind::WouldBlock = e.kind() {
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        match receiver.try_recv() {
            Ok(b) => {
                let buffer = [b; 1];
                stream.write_all(&buffer)?;
                stream.flush()?;
            }
            Err(crossbeam_channel::TryRecvError::Empty) => {}
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                return Err(io::Error::new(
                    ErrorKind::Other,
                    crossbeam_channel::TryRecvError::Disconnected,
                ))
            }
        }
    }
}

#[derive(Clone)]
struct ChannelRead(crossbeam_channel::Receiver<u8>);

impl serial::Read<u8> for ChannelRead {
    type Error = !;

    fn read(&mut self) -> nb::Result<u8, !> {
        self.0.try_recv().or(Err(nb::Error::WouldBlock))
    }
}

#[derive(Clone)]
struct ChannelWrite(crossbeam_channel::Sender<u8>);

impl serial::Write<u8> for ChannelWrite {
    type Error = !;

    fn write(&mut self, word: u8) -> nb::Result<(), !> {
        self.0.try_send(word).or(Err(nb::Error::WouldBlock))
    }

    fn flush(&mut self) -> nb::Result<(), !> {
        Ok(())
    }
}

fn main() -> io::Result<()> {
    let (read_sender, read_receiver) = crossbeam_channel::unbounded();
    let (write_sender, write_receiver) = crossbeam_channel::unbounded();

    std::thread::spawn(move || {
        let mut memory = [0; 2048];

        wam::run_wam(
            ChannelRead(read_receiver.clone()),
            ChannelWrite(write_sender.clone()),
            &mut memory,
        )
    });

    let listener = TcpListener::bind("127.0.0.1:8080")?;

    // accept connections and process them serially
    for stream in listener.incoming() {
        println!("Connection");
        let stream = stream?;
        stream.set_nonblocking(true)?;
        if let Err(e) = handle_client(stream, read_sender.clone(), write_receiver.clone()) {
            println!("Error: {}", e);
        }
    }

    Ok(())
}
*/

/*use std::sync::mpsc;

struct Channels {
    tx: mpsc::SyncSender<u8>,
    rx: mpsc::Receiver<u8>,
}

impl wam::SerialRead<u8> for Channels {
    type Error = mpsc::RecvError;

    fn read(&mut self) -> wam::nb::Result<u8, Self::Error> {
        self.rx.try_recv().map_err(|err| match err {
            mpsc::TryRecvError::Empty => wam::nb::Error::WouldBlock,
            mpsc::TryRecvError::Disconnected => wam::nb::Error::Other(mpsc::RecvError),
        })
    }
}

impl wam::SerialWrite<u8> for Channels {
    type Error = mpsc::SendError<u8>;

    fn write(&mut self, word: u8) -> wam::nb::Result<(), Self::Error> {
        self.tx.try_send(word).map_err(|err| match err {
            mpsc::TrySendError::Full(_) => wam::nb::Error::WouldBlock,
            mpsc::TrySendError::Disconnected(word) => wam::nb::Error::Other(mpsc::SendError(word)),
        })
    }

    fn flush(&mut self) -> wam::nb::Result<(), Self::Error> {
        Ok(())
    }
}*/

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

        let mut memory = [0; 2048];

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
