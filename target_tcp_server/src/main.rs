#![feature(never_type)]

use std::io::{self, ErrorKind, Read, Write};
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
