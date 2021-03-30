#![cfg_attr(not(feature = "logging"), no_std)]

use core::convert::TryFrom;

pub use embedded_hal::serial::{Read as SerialRead, Write as SerialWrite};
pub use nb;
use num_enum::{IntoPrimitive, TryFromPrimitive, TryFromPrimitiveError};
use sha2::Digest;

#[cfg(feature = "logging")]
pub use log;

mod errors;
mod hex;
mod logging;
mod machine;

#[derive(Debug)]
pub enum Never {}

const SUCCESS: char = 'S';

#[derive(Debug)]
pub struct IoError;

impl core::fmt::Display for IoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("WAM IO Error")
    }
}

#[derive(Debug)]
pub enum ProcessInputError {
    Unexpected(u8),
    ReadError,
}

impl From<IoError> for ProcessInputError {
    fn from(_: IoError) -> Self {
        Self::ReadError
    }
}

pub struct SerialConnection<T>(pub T);

impl<R: SerialRead<u8>> SerialConnection<R> {
    fn read_ascii_char(&mut self) -> Result<u8, IoError> {
        loop {
            match self.0.read() {
                Ok(b) => {
                    if !b.is_ascii_whitespace() {
                        return Ok(b);
                    }
                }
                Err(nb::Error::Other(_)) => return Err(IoError),
                Err(nb::Error::WouldBlock) => (),
            }
        }
    }

    fn read_hex_u4(&mut self) -> Result<u8, ProcessInputError> {
        let b = self.read_ascii_char()?;
        let c = b as char;
        match c.to_digit(16) {
            Some(digit) => Ok(digit as u8),
            None => Err(ProcessInputError::Unexpected(b)),
        }
    }

    fn read_be_u8_hex(&mut self) -> Result<u8, ProcessInputError> {
        let a = self.read_hex_u4()?;
        let b = self.read_hex_u4()?;
        Ok((a << 4) + b)
    }

    fn read_be_u16_hex(&mut self) -> Result<u16, ProcessInputError> {
        let mut buffer = [0; 2];
        for b in (&mut buffer).iter_mut() {
            *b = self.read_be_u8_hex()?;
        }
        Ok(u16::from_be_bytes(buffer))
    }

    fn read_be_u32_hex(&mut self) -> Result<u32, ProcessInputError> {
        let mut buffer = [0; 4];
        for b in (&mut buffer).iter_mut() {
            *b = self.read_be_u8_hex()?;
        }
        Ok(u32::from_be_bytes(buffer))
    }
}

impl<W: SerialWrite<u8>> SerialConnection<W> {
    fn flush(&mut self) -> Result<(), IoError> {
        nb::block!(self.0.flush()).map_err(|_| IoError)
    }

    fn write_byte(&mut self, b: u8) -> Result<(), IoError> {
        nb::block!(self.0.write(b)).map_err(|_| IoError)
    }

    fn write_char(&mut self, c: char) -> Result<(), IoError> {
        let mut buffer = [0; 4];
        for b in c.encode_utf8(&mut buffer).bytes() {
            self.write_byte(b)?;
        }

        Ok(())
    }

    fn write_single_char(&mut self, c: char) -> Result<(), IoError> {
        self.write_char(c)?;
        self.flush()?;

        Ok(())
    }

    fn write_be_u4_hex(&mut self, b: u8) -> Result<(), IoError> {
        self.write_char(core::char::from_digit(b as u32, 16).unwrap())
    }

    fn write_be_u8_hex(&mut self, b: u8) -> Result<(), IoError> {
        self.write_be_u4_hex((b >> 4) & 0xF)?;
        self.write_be_u4_hex(b & 0xF)?;
        Ok(())
    }

    fn write_be_u8_slice_hex(&mut self, bytes: &[u8]) -> Result<(), IoError> {
        for &b in bytes {
            self.write_be_u8_hex(b)?;
        }
        Ok(())
    }

    fn write_be_u16_hex(&mut self, w: u16) -> Result<(), IoError> {
        self.write_be_u8_slice_hex(&w.to_be_bytes())
    }

    // fn write_be_u32_hex(&mut self, w: u32) -> Result<(), IoError> {
    //     self.write_be_u8_slice_hex(&w.to_be_bytes())
    // }

    fn write_value(&mut self, value: machine::Value) -> Result<(), IoError> {
        match value {
            machine::Value::Reference(address) => {
                self.write_char('R')?;
                self.write_be_u16_hex(address.0)?;
            }
            machine::Value::Constant(c) => {
                self.write_char('C')?;
                self.write_be_u16_hex(c.0)?;
            }
        }

        Ok(())
    }
}

struct LoadedCode<'a, 'b> {
    code_section: &'a [u32],
    rest_of_memory: &'b mut [u32],
}

fn load_code<
    'code,
    'rest,
    'memory: 'code + 'rest,
    'memory_ref: 'code + 'rest,
    S: SerialRead<u8> + SerialWrite<u8>,
    SC: core::borrow::BorrowMut<SerialConnection<S>>,
>(
    memory: &'memory_ref mut &'memory mut [u32],
    serial_connection: &mut SC,
) -> Result<Option<LoadedCode<'code, 'rest>>, ProcessInputError> {
    let mut serial_connection = serial_connection.borrow_mut();

    let code_length = serial_connection.read_be_u32_hex()? as usize;
    let memory_capacity = memory.len();

    log_debug!("Code length:     {}", code_length);
    log_debug!("Memory capacity: {}", memory_capacity);

    if code_length >= memory_capacity {
        error!(
            &mut serial_connection,
            "Code length ({}) is more than memory capacity ({})", code_length, memory_capacity
        )?;
        return Ok(None);
    }

    let (code_section, rest_of_memory) = memory.split_at_mut(code_length);

    let mut hasher = sha2::Sha256::new();

    for memory_word in code_section.iter_mut() {
        let mut bytes = [0; 4];
        for b in bytes.as_mut() {
            *b = serial_connection.read_be_u8_hex()?;
        }
        hasher.update(&bytes);
        let word = u32::from_be_bytes(bytes);
        log_trace!("Word: {:X}", word);
        *memory_word = word;
    }

    let mut received_hash = [0; 32];

    for b in received_hash.iter_mut() {
        *b = serial_connection.read_be_u8_hex()?;
    }

    let calculated_hash = hasher.finalize();

    log_debug!("Received Hash:   {:X}", hex::Hex(&received_hash[..]));
    log_debug!("Calculated Hash: {:X}", calculated_hash);

    if received_hash
        .iter()
        .zip(calculated_hash.iter())
        .any(|(&a, &b)| a != b)
    {
        error!(&mut serial_connection, "Hashes don't match")?;
        return Ok(None);
    }

    Ok(Some(LoadedCode {
        code_section,
        rest_of_memory,
    }))
}

#[derive(Debug, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum CommandHeader {
    ReportStatus = b'S',
    SubmitProgram = b'P',
    SubmitQuery = b'Q',
    LookupMemory = b'M',
    NextSolution = b'C',
}

impl CommandHeader {
    fn parse(value: u8) -> Result<Self, ProcessInputError> {
        Self::try_from(value).map_err(|_| ProcessInputError::Unexpected(value))
    }
}

mod device_with_query {
    use super::{CommandHeader, ProcessInputError, SerialConnection, SerialRead, SerialWrite};

    #[derive(Debug)]
    pub enum UnhandledCommand {
        ProcessNextCommand,
        SubmitProgram,
        SubmitQuery,
    }

    pub struct Device<'a, 's1, 's2, S> {
        pub program: &'a [u32],
        pub query: &'a [u32],
        pub memory: &'a mut [u32],
        pub serial_connection: &'s1 mut &'s2 mut SerialConnection<S>,
    }

    impl<'a, 's1, 's2, S: SerialRead<u8> + SerialWrite<u8>> Device<'a, 's1, 's2, S> {
        pub fn run(self) -> Result<UnhandledCommand, ProcessInputError> {
            let (machine, success) =
                match super::machine::Machine::run(super::machine::MachineMemory {
                    program: self.program,
                    query: self.query,
                    memory: self.memory,
                }) {
                    Ok(success) => success,
                    Err(failure) => {
                        self.serial_connection
                            .write_single_char(failure as u8 as char)?;
                        return Ok(UnhandledCommand::ProcessNextCommand);
                    }
                };

            self.serial_connection.write_char(success as u8 as char)?;
            for &value in machine.solution_registers() {
                self.serial_connection.write_value(value)?;
            }
            self.serial_connection.flush()?;

            loop {
                match CommandHeader::parse(self.serial_connection.read_ascii_char()?)? {
                    CommandHeader::ReportStatus => {
                        log::debug!("Status: Waiting for Query");
                        self.serial_connection
                            .write_single_char(success as u8 as char)?;
                    }
                    CommandHeader::SubmitProgram => return Ok(UnhandledCommand::SubmitProgram),
                    CommandHeader::SubmitQuery => return Ok(UnhandledCommand::SubmitQuery),
                    CommandHeader::LookupMemory => {
                        let value = machine.lookup_memory(super::machine::Address(
                            self.serial_connection.read_be_u16_hex()?,
                        ));
                        self.serial_connection.write_value(value)?;
                        self.serial_connection.flush()?;
                        if let super::machine::ExecutionSuccess::SingleAnswer = success {
                            return Ok(UnhandledCommand::ProcessNextCommand);
                        }
                    }
                    CommandHeader::NextSolution => {
                        todo!()
                    }
                }
            }
        }
    }
}

mod device_with_program {
    use super::{
        log_info, CommandHeader, ProcessInputError, SerialConnection, SerialRead, SerialWrite,
    };

    type EscalatedCommand = super::device_with_query::UnhandledCommand;

    #[derive(Debug)]
    enum HandledCommand {
        ReportStatus,
        SubmitQuery,
    }

    #[derive(Debug)]
    pub enum UnhandledCommand {
        SubmitProgram,
    }

    #[derive(Debug)]
    enum Action {
        ProcessNextCommand,
        ProcessCommand(HandledCommand),
        Escalate(UnhandledCommand),
    }

    pub struct Device<'a, 's, S> {
        pub program: &'a [u32],
        pub memory: &'a mut [u32],
        pub serial_connection: &'s mut SerialConnection<S>,
    }

    impl<'a, 's, S: SerialRead<u8> + SerialWrite<u8>> Device<'a, 's, S> {
        fn handle_submit_query(&mut self) -> Result<Action, ProcessInputError> {
            log_info!("Loading query");

            let super::LoadedCode {
                code_section: query,
                rest_of_memory: memory,
            } = match super::load_code(&mut self.memory, &mut self.serial_connection)? {
                Some(c) => c,
                None => return Ok(Action::ProcessNextCommand),
            };

            let unhandled_command = super::device_with_query::Device {
                program: self.program,
                query,
                memory,
                serial_connection: &mut self.serial_connection,
            }
            .run()?;

            Ok(match unhandled_command {
                EscalatedCommand::ProcessNextCommand => Action::ProcessNextCommand,
                EscalatedCommand::SubmitProgram => {
                    Action::Escalate(UnhandledCommand::SubmitProgram)
                }
                EscalatedCommand::SubmitQuery => {
                    Action::ProcessCommand(HandledCommand::SubmitQuery)
                }
            })
        }

        fn process_command(
            &mut self,
            command: HandledCommand,
        ) -> Result<Action, ProcessInputError> {
            match command {
                HandledCommand::ReportStatus => {
                    log::debug!("Status: Waiting for Query");
                    self.serial_connection.write_single_char('Q')?;
                    Ok(Action::ProcessNextCommand)
                }
                HandledCommand::SubmitQuery => self.handle_submit_query(),
            }
        }

        pub fn run(mut self) -> Result<UnhandledCommand, ProcessInputError> {
            log_info!("Waiting for Query");
            loop {
                let command = CommandHeader::parse(self.serial_connection.read_ascii_char()?)?;
                let mut command = match command {
                    CommandHeader::ReportStatus => HandledCommand::ReportStatus,
                    CommandHeader::SubmitProgram => return Ok(UnhandledCommand::SubmitProgram),
                    CommandHeader::SubmitQuery => HandledCommand::SubmitQuery,
                    CommandHeader::LookupMemory | CommandHeader::NextSolution => {
                        return Err(ProcessInputError::Unexpected(command as u8))
                    }
                };

                loop {
                    match self.process_command(command)? {
                        Action::ProcessNextCommand => break,
                        Action::ProcessCommand(next_command) => {
                            command = next_command;
                        }
                        Action::Escalate(UnhandledCommand::SubmitProgram) => {
                            return Ok(UnhandledCommand::SubmitProgram)
                        }
                    }
                }
            }
        }
    }
}

enum Action {
    ProcessNextCommand,
    ProcessCommand(CommandHeader),
}

pub struct Device<'a, S> {
    pub memory: &'a mut [u32],
    pub serial_connection: SerialConnection<S>,
}

impl<'a, S: SerialRead<u8> + SerialWrite<u8>> Device<'a, S> {
    fn handle_submit_program(&mut self) -> Result<Action, ProcessInputError> {
        log_info!("Loading program");

        let LoadedCode {
            code_section: program,
            rest_of_memory: memory,
        } = match load_code(&mut self.memory, &mut self.serial_connection)? {
            Some(c) => c,
            None => return Ok(Action::ProcessNextCommand),
        };

        self.serial_connection.write_single_char(SUCCESS)?;

        let unhandled_command = device_with_program::Device {
            program,
            memory,
            serial_connection: &mut self.serial_connection,
        }
        .run()?;

        match unhandled_command {
            device_with_program::UnhandledCommand::SubmitProgram => {
                Ok(Action::ProcessCommand(CommandHeader::SubmitProgram))
            }
        }
    }

    fn process_command_byte(&mut self, command_byte: u8) -> Result<Action, ProcessInputError> {
        let command_header = match CommandHeader::try_from(command_byte) {
            Ok(header) => header,
            Err(TryFromPrimitiveError { number }) => {
                error!(
                    &mut self.serial_connection,
                    "Invalid command: {:?}", number as char
                )?;
                return Ok(Action::ProcessNextCommand);
            }
        };

        log::debug!("Processing command {:?}", command_header);

        match command_header {
            CommandHeader::ReportStatus => {
                log::debug!("Status: Waiting for program");
                self.serial_connection.write_single_char('P')?;
                Ok(Action::ProcessNextCommand)
            }
            CommandHeader::SubmitProgram => self.handle_submit_program(),
            CommandHeader::SubmitQuery => {
                error!(&mut self.serial_connection, "No Program",)?;
                Ok(Action::ProcessNextCommand)
            }
            CommandHeader::LookupMemory | CommandHeader::NextSolution => {
                Err(ProcessInputError::Unexpected(command_header as u8))
            }
        }
    }

    pub fn run(mut self) -> Result<Never, IoError> {
        log_info!("Running");
        let mut command_header = self.serial_connection.read_ascii_char()?;
        loop {
            let next_command_header = match self.process_command_byte(command_header) {
                Ok(Action::ProcessNextCommand) => self.serial_connection.read_ascii_char()?,
                Ok(Action::ProcessCommand(command)) => command as u8,
                Err(ProcessInputError::Unexpected(b)) => b,
                Err(ProcessInputError::ReadError) => return Err(IoError),
            };

            command_header = next_command_header;
        }
    }
}
