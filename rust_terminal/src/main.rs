use std::{
    borrow::Borrow,
    collections::BTreeMap,
    fmt,
    io::{Read, Write},
    net::TcpStream,
    path::PathBuf,
    rc::Rc,
};

use anyhow::Context;
use clap::Parser;
use compiler::{Functor, ProgramInfo, Query};
use crossterm::{
    event::{Event, KeyCode, KeyEvent, KeyEventKind},
    style::{style, Print, Stylize},
    ExecutableCommand,
};
use num_bigint::BigInt;
use serialport::SerialPortType;
use sha2::{digest::FixedOutput, Digest};

mod compiler;

#[derive(Debug, thiserror::Error)]
#[error("Bad Status: {0}")]
struct BadStatus(StatusCode);

#[derive(Debug)]
enum StatusCode {
    WaitingForProgram,
    WaitingForQuery,
    SingleAnswer,
    ChoicePoint,
    Error(String),
}

impl StatusCode {
    fn assert_can_submit_program(self) -> Result<(), BadStatus> {
        match self {
            Self::WaitingForProgram
            | Self::WaitingForQuery
            | Self::SingleAnswer
            | Self::ChoicePoint => Ok(()),
            Self::Error(_) => Err(BadStatus(self)),
        }
    }

    fn assert_can_submit_query(self) -> Result<(), BadStatus> {
        match self {
            Self::WaitingForQuery | Self::SingleAnswer | Self::ChoicePoint => Ok(()),
            Self::WaitingForProgram | Self::Error(_) => Err(BadStatus(self)),
        }
    }
}

impl fmt::Display for StatusCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WaitingForProgram => write!(f, "Waiting For Program"),
            Self::WaitingForQuery => write!(f, "Waiting For Query"),
            Self::SingleAnswer => write!(f, "Single Answer"),
            Self::ChoicePoint => write!(f, "Choice Point"),
            Self::Error(error) => write!(f, "{error}"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("Expected hex, got {0:?}")]
pub struct NotHex(u8);

#[derive(Debug, thiserror::Error)]
pub enum ReadHexError {
    #[error(transparent)]
    IO(#[from] std::io::Error),
    #[error(transparent)]
    NotHex(#[from] NotHex),
}

#[derive(Debug, thiserror::Error)]
enum ReadHexStringError {
    #[error(transparent)]
    IO(#[from] std::io::Error),
    #[error(transparent)]
    NotHex(#[from] NotHex),
    #[error(transparent)]
    NotUtf8(#[from] std::string::FromUtf8Error),
}

impl From<ReadHexError> for ReadHexStringError {
    fn from(inner: ReadHexError) -> Self {
        match inner {
            ReadHexError::IO(inner) => inner.into(),
            ReadHexError::NotHex(inner) => inner.into(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum ReadStatusError {
    #[error(transparent)]
    IO(#[from] std::io::Error),
    #[error(transparent)]
    ReadHexError(#[from] ReadHexError),
    #[error(transparent)]
    ReadHexStringError(#[from] ReadHexStringError),
    #[error("Bad status code {0:?}")]
    BadStatus(char),
}

type Address = u16;

#[derive(Debug, Clone, PartialEq, Eq)]
enum Value {
    Reference(Address),
    Structure(Functor, Rc<[Address]>),
    List(Address, Address),
    Constant(Functor),
    Integer(BigInt),
    Error(String),
}

#[derive(Debug, thiserror::Error)]
enum GetValueError {
    #[error(transparent)]
    IO(#[from] std::io::Error),
    #[error(transparent)]
    ReadHexError(#[from] ReadHexError),
    #[error(transparent)]
    ReadHexStringError(#[from] ReadHexStringError),
    #[error("Bad value type {0}")]
    BadValueType(u8),
}

struct DisplayCommaSeparated<I>(I);

impl<I> fmt::Display for DisplayCommaSeparated<I>
where
    I: Clone,
    I: Iterator,
    I::Item: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut i = self.0.clone();

        if let Some(item) = i.next() {
            write!(f, "{item}")?;
        }

        i.try_for_each(|item| write!(f, ", {item}"))
    }
}

struct DisplayFunctorName<'a> {
    answer: &'a Answer<'a>,
    name: &'a Functor,
}

impl<'a> fmt::Display for DisplayFunctorName<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.answer.program_info.lookup_functor(self.name) {
            Some(name) => write!(f, "{name}"),
            None => write!(f, "<Unknown functor {}>", self.name),
        }
    }
}

struct DisplayValue<'a> {
    answer: &'a Answer<'a>,
    address: &'a Address,
}

impl<'a> fmt::Display for DisplayValue<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let answer = self.answer;

        match self.answer.values.get(self.address) {
            Some(Value::Reference(reference)) => {
                if reference == self.address {
                    write!(f, "_{reference}")
                } else {
                    DisplayValue {
                        answer,
                        address: reference,
                    }
                    .fmt(f)
                }
            }
            Some(Value::Structure(name, terms)) => {
                write!(
                    f,
                    "{}",
                    DisplayStructure {
                        answer,
                        name,
                        terms
                    }
                )
            }
            Some(Value::List(head, tail)) => write!(
                f,
                "[{}{}]",
                DisplayValue {
                    answer,
                    address: head
                },
                DisplayListTail {
                    answer,
                    address: tail
                }
            ),
            Some(Value::Constant(name)) => {
                if self.answer.program_info.lookup_functor(name)
                    == Some(compiler::EMPTY_LIST.as_str())
                {
                    write!(f, "[]")
                } else {
                    write!(f, "{}", DisplayFunctorName { answer, name })
                }
            }
            Some(Value::Integer(i)) => write!(f, "{i}"),
            Some(Value::Error(error)) => write!(f, "<{error}>"),
            None => write!(f, "<Unknown value at {}>", self.address),
        }
    }
}

struct DisplayStructure<'a> {
    answer: &'a Answer<'a>,
    name: &'a Functor,
    terms: &'a [Address],
}

impl<'a> fmt::Display for DisplayStructure<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let answer = self.answer;

        match self.answer.program_info.lookup_functor(self.name) {
            None => write!(
                f,
                "<Unknown functor {}>({})",
                self.name,
                DisplayCommaSeparated(
                    self.terms
                        .iter()
                        .map(|address| DisplayValue { answer, address })
                )
            ),
            Some(name) => match (name, self.terms) {
                ("*" | "//" | "div" | "mod" | "+" | "-" | ":", [lhs, rhs]) => {
                    write!(
                        f,
                        "({} {name} {})",
                        DisplayValue {
                            answer,
                            address: lhs
                        },
                        DisplayValue {
                            answer,
                            address: rhs
                        }
                    )
                }
                _ => write!(f, "{name}"),
            },
        }
    }
}

struct DisplayListTail<'a> {
    answer: &'a Answer<'a>,
    address: &'a Address,
}

impl<'a> fmt::Display for DisplayListTail<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let answer = self.answer;

        match self.answer.values.get(self.address) {
            Some(Value::Reference(reference)) => {
                if reference == self.address {
                    write!(f, "|_{reference}")
                } else {
                    DisplayValue {
                        answer,
                        address: reference,
                    }
                    .fmt(f)
                }
            }
            Some(Value::Structure(name, terms)) => write!(
                f,
                "|{}",
                DisplayStructure {
                    answer,
                    name,
                    terms
                }
            ),
            Some(Value::List(head, tail)) => write!(
                f,
                ",{}{}",
                DisplayValue {
                    answer,
                    address: head
                },
                DisplayListTail {
                    answer,
                    address: tail
                }
            ),
            Some(Value::Constant(name)) => {
                if self.answer.program_info.lookup_functor(name)
                    == Some(compiler::EMPTY_LIST.as_str())
                {
                    Ok(())
                } else {
                    write!(f, "|{}", DisplayFunctorName { answer, name })
                }
            }
            Some(Value::Integer(i)) => write!(f, "|{i}"),
            Some(Value::Error(error)) => write!(f, "|<{error}>"),
            None => write!(f, "|<Unknown value at {}>", self.address),
        }
    }
}

struct Answer<'a> {
    program_info: &'a ProgramInfo,
    query: &'a Query,
    registers: Vec<Address>,
    values: BTreeMap<Address, Value>,
}

impl<'a> fmt::Display for Answer<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}({})",
            self.query.name,
            DisplayCommaSeparated(self.registers.iter().map(|address| DisplayValue {
                answer: self,
                address
            }))
        )
    }
}

#[derive(Debug, thiserror::Error)]
enum GetAnswerError {
    #[error(transparent)]
    IO(#[from] std::io::Error),
    #[error(transparent)]
    ReadHexError(#[from] ReadHexError),
    #[error(transparent)]
    GetValueError(#[from] GetValueError),
}

trait TryIteratorExt<T, E>: Sized + Iterator<Item = Result<T, E>> {
    fn try_collect_vec(self) -> Result<Vec<T>, E> {
        self.collect()
    }
}

impl<T, E, I: Iterator<Item = Result<T, E>>> TryIteratorExt<T, E> for I {}

trait ReadExt: Read {
    fn read_one(&mut self) -> std::io::Result<u8> {
        let mut buffer = 0;
        self.read_exact(std::slice::from_mut(&mut buffer))?;
        Ok(buffer)
    }

    fn read_hex_u4(&mut self) -> Result<u8, ReadHexError> {
        let b = self.read_one()?;

        std::str::from_utf8(std::slice::from_ref(&b))
            .ok()
            .and_then(|src| u8::from_str_radix(src, 16).ok())
            .ok_or(ReadHexError::NotHex(NotHex(b)))
    }

    fn read_hex_u8(&mut self) -> Result<u8, ReadHexError> {
        let a = self.read_hex_u4()?;
        let b = self.read_hex_u4()?;
        Ok((a << 4) + b)
    }

    fn read_hex_u16(&mut self) -> Result<u16, ReadHexError> {
        let a = self.read_hex_u8()?;
        let b = self.read_hex_u8()?;
        Ok(u16::from_be_bytes([a, b]))
    }

    fn read_hex_u32(&mut self) -> Result<u32, ReadHexError> {
        let a = self.read_hex_u8()?;
        let b = self.read_hex_u8()?;
        let c = self.read_hex_u8()?;
        let d = self.read_hex_u8()?;

        Ok(u32::from_be_bytes([a, b, c, d]))
    }

    fn read_hex_string(&mut self) -> Result<String, ReadHexStringError> {
        let length = self.read_hex_u8()?;
        let buffer = (0..length).map(|_| self.read_hex_u8()).try_collect_vec()?;

        Ok(String::from_utf8(buffer)?)
    }

    fn read_error_message(&mut self) -> Result<String, ReadHexStringError> {
        std::iter::from_fn(|| match self.read_hex_u8() {
            Ok(b) => Some(Ok(b)),
            Err(ReadHexError::IO(io)) => Some(Err(ReadHexError::IO(io))),
            Err(ReadHexError::NotHex(NotHex(b'S'))) => None,
            Err(ReadHexError::NotHex(not_hex)) => Some(Err(ReadHexError::NotHex(not_hex))),
        })
        .try_collect_vec()
        .map_err(ReadHexStringError::from)
        .and_then(|bytes| Ok(String::from_utf8(bytes)?))
    }
}

impl<T: Read> ReadExt for T {}

trait WriteExt: Write {
    fn write_one(&mut self, b: u8) -> std::io::Result<()> {
        self.write_all(std::slice::from_ref(&b))?;
        self.flush()
    }

    fn write_char(&mut self, c: char) -> std::io::Result<()> {
        let mut buffer = [0; 4];
        self.write_all(c.encode_utf8(&mut buffer).as_bytes())
    }

    fn write_single_char(&mut self, c: char) -> std::io::Result<()> {
        self.write_char(c)?;
        self.flush()?;

        Ok(())
    }

    fn write_be_u4_hex(&mut self, b: u8) -> std::io::Result<()> {
        self.write_char(char::from_digit(b.into(), 16).unwrap())
    }

    fn write_be_u8_hex(&mut self, b: u8) -> std::io::Result<()> {
        self.write_be_u4_hex((b >> 4) & 0xF)?;
        self.write_be_u4_hex(b & 0xF)?;
        Ok(())
    }

    fn write_be_u16_hex(&mut self, v: u16) -> std::io::Result<()> {
        self.write_be_u8_hex_iter(v.to_be_bytes())
    }

    fn write_be_u32_hex(&mut self, v: u32) -> std::io::Result<()> {
        self.write_be_u8_hex_iter(v.to_be_bytes())
    }

    fn write_be_u8_hex_iter<I>(&mut self, i: I) -> std::io::Result<()>
    where
        I: IntoIterator,
        I::Item: Borrow<u8>,
    {
        i.into_iter()
            .try_for_each(|b| self.write_be_u8_hex(*b.borrow()))
    }

    fn write_words_with_hash(&mut self, words: &[[u8; 4]]) -> std::io::Result<()> {
        self.write_be_u32_hex(words.len() as u32)?;

        self.write_be_u8_hex_iter(words.iter().flatten())?;

        let mut hasher = sha2::Sha256::new();

        for word in words {
            hasher.update(word);
        }

        let hash = hasher.finalize_fixed();

        self.write_be_u8_hex_iter(hash)?;

        self.flush()
    }
}

impl<T: Write> WriteExt for T {}

trait ReadWriteExt: Sized + Read + Write {
    fn get_status(&mut self) -> Result<StatusCode, ReadStatusError> {
        self.write_one(b'S')?;

        Ok(match self.read_one()? {
            b'P' => StatusCode::WaitingForProgram,
            b'Q' => StatusCode::WaitingForQuery,
            b'A' => StatusCode::SingleAnswer,
            b'C' => StatusCode::ChoicePoint,
            b'E' => StatusCode::Error(self.read_error_message()?),
            status_code => return Err(ReadStatusError::BadStatus(status_code as char)),
        })
    }

    fn get_value(&mut self, address: Address) -> Result<(Address, Value), GetValueError> {
        self.write_all(b"M")?;
        self.write_be_u16_hex(address)?;
        self.flush()?;

        let value = match self.read_one()? {
            b'R' => Value::Reference(self.read_hex_u16()?),
            b'S' => {
                let f = self.read_hex_u16()?;
                let n = self.read_hex_u8()?;
                Value::Structure(
                    f,
                    (0..n)
                        .map(|_| self.read_hex_u16())
                        .try_collect_vec()?
                        .into(),
                )
            }
            b'L' => {
                let head = self.read_hex_u16()?;
                let tail = self.read_hex_u16()?;
                Value::List(head, tail)
            }
            b'C' => Value::Constant(self.read_hex_u16()?),
            b'I' => {
                let sign = match self.read_one()? {
                    b'-' => num_bigint::Sign::Minus,
                    b'+' => num_bigint::Sign::Plus,
                    sign => panic!("Bad sign {sign}"),
                };
                let n = self.read_hex_u32()?;

                let be_bytes = (0..n).map(|_| self.read_hex_u8()).try_collect_vec()?;

                Value::Integer(if be_bytes.iter().copied().all(|n| n == 0) {
                    BigInt::from(0)
                } else {
                    BigInt::from_bytes_be(sign, &be_bytes)
                })
            }
            b'E' => return Ok((address, Value::Error(self.read_error_message()?))),
            code => return Err(GetValueError::BadValueType(code)),
        };

        let address = self.read_hex_u16()?;

        Ok((address, value))
    }

    fn get_answer<'a>(
        &mut self,
        program_info: &'a ProgramInfo,
        query: &'a Query,
    ) -> Result<Answer<'a>, GetAnswerError> {
        let length = self.read_hex_u8()?;

        let answer_registers = (0..length).map(|_| self.read_hex_u16()).try_collect_vec()?;

        let mut known_values = BTreeMap::new();

        let mut values_to_get = answer_registers.clone();

        while !values_to_get.is_empty() {
            for address in std::mem::take(&mut values_to_get) {
                if known_values.contains_key(&address) {
                    continue;
                }

                let (deref_address, value) = self.get_value(address)?;

                for address in [address, deref_address] {
                    if let Some(current_value) = known_values.insert(address, value.clone()) {
                        assert_eq!(current_value, value);
                    }
                }

                match value {
                    Value::Reference(reference) => {
                        if reference != deref_address {
                            values_to_get.push(reference);
                        }
                    }
                    Value::Structure(_, terms) => {
                        for term in terms.as_ref().iter().copied() {
                            values_to_get.push(term);
                        }
                    }
                    Value::List(head, tail) => {
                        values_to_get.push(head);
                        values_to_get.push(tail);
                    }
                    Value::Constant(_) | Value::Integer(_) | Value::Error(_) => {}
                }
            }
        }

        Ok(Answer {
            program_info,
            query,
            registers: answer_registers,
            values: known_values,
        })
    }
}

impl<T: Sized + Read + Write> ReadWriteExt for T {}

struct DisplayKeyCode(KeyCode);

impl fmt::Debug for DisplayKeyCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl fmt::Display for DisplayKeyCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let KeyCode::Char(c) = &self.0 {
            fmt::Display::fmt(c, f)
        } else {
            fmt::Debug::fmt(&self.0, f)
        }
    }
}

struct EndOfLine;

impl fmt::Display for EndOfLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)
    }
}

#[derive(Parser)]
struct Cli {
    /// The program to parse
    program: PathBuf,

    /// Connect to the simulator
    #[arg(short = 's', long = "simulator")]
    connect_to_simulator: bool,
}

enum Port {
    TcpStream(TcpStream),
    Serial(Box<dyn serialport::SerialPort>),
}

impl Read for Port {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::TcpStream(stream) => stream.read(buf),
            Self::Serial(serial) => serial.read(buf),
        }
    }
}

impl Write for Port {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Self::TcpStream(stream) => stream.write(buf),
            Self::Serial(serial) => serial.write(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Self::TcpStream(stream) => stream.flush(),
            Self::Serial(serial) => serial.flush(),
        }
    }
}

fn main() -> anyhow::Result<()> {
    let Cli {
        program,
        connect_to_simulator,
    } = Cli::parse();

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    let mut port = if connect_to_simulator {
        stdout
            .execute(Print("Connecting to device"))?
            .execute(Print(EndOfLine))?;
        Port::TcpStream(TcpStream::connect("localhost:8080").context("Failed to connect")?)
    } else {
        Port::Serial(
            serialport::new(
                serialport::available_ports()?
                    .into_iter()
                    .find_map(|port| match port.port_type {
                        SerialPortType::UsbPort(_) => Some(port.port_name),
                        _ => None,
                    })
                    .context("Device not connected")?,
                9600,
            )
            .open()?,
        )
    };

    port.get_status()?.assert_can_submit_program()?;

    port.write_one(b'P')?;

    let system_call_count = port.read_hex_u16()?;

    let system_calls = (0..system_call_count)
        .map(|_| {
            let name = port.read_hex_string()?;
            let arity = port.read_hex_u8()?;

            Ok::<_, ReadHexStringError>((name, arity))
        })
        .try_collect_vec()?;

    let (program_info, program_words) = compiler::compile_program(system_calls, program)?;

    port.write_words_with_hash(&program_words)?;

    match port.read_one()? {
        b'S' => (),
        b'E' => anyhow::bail!("Failed to send program: {}", port.read_error_message()?),
        b => anyhow::bail!("Unknown response {b}"),
    }

    loop {
        port.get_status()?.assert_can_submit_query()?;

        let (program_info, query, query_words) = loop {
            stdout.execute(Print("?- "))?;

            let mut line = String::new();
            stdin.read_line(&mut line)?;
            let query = line.trim();

            if query.is_empty() {
                continue;
            }

            match compiler::compile_query(query, &program_info) {
                Ok(result) => break result,
                Err(err) => {
                    stdout.execute(Print(err))?.execute(Print(EndOfLine))?;
                    continue;
                }
            }
        };

        port.write_one(b'Q')?;

        port.write_words_with_hash(&query_words)?;

        'outer: loop {
            break match port.read_one()? {
                b'A' => stdout.execute(Print(port.get_answer(&program_info, &query)?))?,
                b'C' => {
                    stdout.execute(Print(port.get_answer(&program_info, &query)?))?;

                    loop {
                        if let Event::Key(KeyEvent {
                            code,
                            kind: KeyEventKind::Press,
                            ..
                        }) = crossterm::event::read().unwrap()
                        {
                            match code {
                                KeyCode::Esc | KeyCode::Enter => break 'outer &mut stdout,
                                KeyCode::Char(' ') => {
                                    stdout.execute(Print(EndOfLine))?;
                                    break;
                                }
                                code => {
                                    stdout.execute(Print(EndOfLine))?.execute(Print(
                                        format_args!("Unhandled key {}", DisplayKeyCode(code)),
                                    ))?;

                                    continue;
                                }
                            }
                        }
                    }

                    port.write_one(b'C')?;
                    continue;
                }
                b'F' => stdout.execute(Print("fail".bold()))?,
                b'E' => stdout.execute(Print(
                    style(format_args!("error: {}", port.read_error_message()?)).red(),
                ))?,
                code => stdout.execute(Print(format_args!("Bad code {code}")))?,
            };
        }
        .execute(Print(EndOfLine))?;
    }
}
