use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt,
    io::{Read, Write},
    net::TcpStream,
    num::NonZeroU16,
    path::PathBuf,
    rc::Rc,
};

use anyhow::Context;
use arcstr::ArcStr;
use clap::Parser;
use comms::minicbor;
use crossterm::{
    event::{Event, KeyCode, KeyEvent, KeyEventKind},
    style::{Print, Stylize},
    ExecutableCommand, QueueableCommand,
};
use num_bigint::BigInt;
use serialport::{SerialPortType, UsbPortInfo};

mod compiler;
use compiler::{Arity, Functor, ProgramInfo, Query, Term};

type Address = NonZeroU16;

#[derive(Debug, Clone, PartialEq, Eq)]
enum Value {
    Reference(Address),
    Structure(Functor, Rc<[Address]>),
    List(Address, Address),
    Constant(Functor),
    Integer(BigInt),
    Error(String),
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

#[derive(Clone, Copy)]
struct Precedence(u8);

impl Precedence {
    fn decrement(self) -> Self {
        Self(self.0 - 1)
    }

    fn display_brackets(&self, parent_precedence: Option<Precedence>) -> bool {
        parent_precedence.map_or(false, |parent_precedence| self.0 > parent_precedence.0)
    }
}

struct DisplayValue<'a> {
    answer: &'a Answer<'a>,
    parent: Option<&'a Address>,
    address: &'a Address,
    precedence: Option<Precedence>,
}

impl<'a> fmt::Display for DisplayValue<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let answer = self.answer;

        if let Some(parent) = self.parent.copied() {
            if answer.loops.contains(&LoopLink {
                parent,
                child: *self.address,
            }) {
                return DisplayReference {
                    answer,
                    reference: self.address,
                }
                .fmt(f);
            }
        }

        match self.answer.values.get(self.address) {
            Some(Value::Reference(reference)) => {
                if reference == self.address {
                    DisplayReference { answer, reference }.fmt(f)
                } else {
                    DisplayValue {
                        answer,
                        parent: Some(self.address),
                        address: reference,
                        precedence: self.precedence,
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
                        address: self.address,
                        name,
                        terms,
                        precedence: self.precedence,
                    }
                )
            }
            Some(Value::List(head, tail)) => write!(
                f,
                "[{}{}]",
                DisplayValue {
                    answer,
                    address: head,
                    parent: Some(self.address),
                    precedence: None,
                },
                DisplayListTail {
                    answer,
                    parent: Some(self.address),
                    address: tail,
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

struct DisplayReference<'a> {
    answer: &'a Answer<'a>,
    reference: &'a Address,
}

impl<'a> fmt::Display for DisplayReference<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
            .answer
            .variables
            .iter()
            .find_map(|(name, address)| (address == self.reference).then_some(name))
        {
            Some(name) => name.fmt(f),
            None => write!(f, "_{}", self.reference),
        }
    }
}

struct DisplayStructure<'a> {
    answer: &'a Answer<'a>,
    address: &'a Address,
    name: &'a Functor,
    terms: &'a [Address],
    precedence: Option<Precedence>,
}

impl<'a> fmt::Display for DisplayStructure<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let answer = self.answer;

        match self.answer.program_info.lookup_functor(self.name) {
            None => write!(
                f,
                "<Unknown functor {}>({})",
                self.name,
                DisplayCommaSeparated(self.terms.iter().map(|address| DisplayValue {
                    answer,
                    parent: Some(self.address),
                    address,
                    precedence: None
                }))
            ),
            Some(operator) => match (operator, self.terms) {
                (":", [lhs, rhs]) => DisplayInfix {
                    answer,
                    address: self.address,
                    operator,
                    precedence: Precedence(6),
                    parent_precedence: self.precedence,
                    lhs,
                    rhs,
                }
                .fmt(f),
                ("+" | "-", [lhs, rhs]) => DisplayInfix {
                    answer,
                    address: self.address,
                    operator,
                    precedence: Precedence(5),
                    parent_precedence: self.precedence,
                    lhs,
                    rhs,
                }
                .fmt(f),
                ("*" | "//" | "div" | "mod", [lhs, rhs]) => DisplayInfix {
                    answer,
                    address: self.address,
                    operator,
                    precedence: Precedence(4),
                    parent_precedence: self.precedence,
                    lhs,
                    rhs,
                }
                .fmt(f),
                ("+" | "-", [term]) => DisplayPrefix {
                    answer,
                    address: self.address,
                    operator,
                    precedence: Precedence(2),
                    parent_precedence: self.precedence,
                    term,
                }
                .fmt(f),
                _ => write!(
                    f,
                    "{operator}({})",
                    DisplayCommaSeparated(self.terms.iter().map(|address| DisplayValue {
                        answer,
                        parent: Some(self.address),
                        address,
                        precedence: None
                    }))
                ),
            },
        }
    }
}

struct DisplayPrefix<'a> {
    answer: &'a Answer<'a>,
    address: &'a Address,
    operator: &'a str,
    precedence: Precedence,
    parent_precedence: Option<Precedence>,
    term: &'a Address,
}

impl<'a> fmt::Display for DisplayPrefix<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            answer,
            address,
            operator,
            precedence,
            parent_precedence,
            term,
        } = *self;

        let term = DisplayValue {
            answer,
            parent: Some(address),
            address: term,
            precedence: Some(precedence),
        };

        if precedence.display_brackets(parent_precedence) {
            write!(f, "({operator}{term})")
        } else {
            write!(f, "{operator}{term}")
        }
    }
}

struct DisplayInfix<'a> {
    answer: &'a Answer<'a>,
    address: &'a Address,
    operator: &'a str,
    precedence: Precedence,
    parent_precedence: Option<Precedence>,
    lhs: &'a Address,
    rhs: &'a Address,
}

impl<'a> fmt::Display for DisplayInfix<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            answer,
            address,
            operator,
            precedence,
            parent_precedence,
            lhs,
            rhs,
        } = *self;

        let lhs = DisplayValue {
            answer,
            parent: Some(address),
            address: lhs,
            precedence: Some(precedence),
        };
        let rhs = DisplayValue {
            answer,
            parent: Some(address),
            address: rhs,
            precedence: Some(precedence.decrement()),
        };

        if precedence.display_brackets(parent_precedence) {
            write!(f, "({lhs} {operator} {rhs})")
        } else {
            write!(f, "{lhs} {operator} {rhs}")
        }
    }
}

struct DisplayListTail<'a> {
    answer: &'a Answer<'a>,
    parent: Option<&'a Address>,
    address: &'a Address,
}

impl<'a> fmt::Display for DisplayListTail<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let answer = self.answer;

        if let Some(parent) = self.parent.copied() {
            if answer.loops.contains(&LoopLink {
                parent,
                child: *self.address,
            }) {
                return write!(
                    f,
                    "|{}",
                    DisplayReference {
                        answer,
                        reference: self.address,
                    }
                );
            }
        }

        match self.answer.values.get(self.address) {
            Some(Value::Reference(reference)) => {
                if reference == self.address {
                    write!(f, "|{}", DisplayReference { answer, reference })
                } else {
                    DisplayValue {
                        answer,
                        parent: Some(self.address),
                        address: reference,
                        precedence: None,
                    }
                    .fmt(f)
                }
            }
            Some(Value::Structure(name, terms)) => write!(
                f,
                "|{}",
                DisplayStructure {
                    answer,
                    address: self.address,
                    name,
                    terms,
                    precedence: None,
                }
            ),
            Some(Value::List(head, tail)) => write!(
                f,
                ",{}{}",
                DisplayValue {
                    answer,
                    parent: Some(self.address),
                    address: head,
                    precedence: None,
                },
                DisplayListTail {
                    answer,
                    parent: Some(self.address),
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

#[derive(Debug, PartialEq, Eq, Hash)]
struct LoopLink {
    parent: Address,
    child: Address,
}

struct LoopScannerState<'a> {
    values: &'a BTreeMap<Address, Value>,
    loops: HashSet<LoopLink>,
    descendants: HashMap<Address, HashSet<Address>>,
}

impl<'a> LoopScannerState<'a> {
    fn check_node_for_loops(self, parent: Address) -> Self {
        match self.values.get(&parent).unwrap() {
            Value::Reference(child) => self.check_link_for_loops((parent, *child)),
            Value::Structure(_, children) => children
                .as_ref()
                .iter()
                .map(move |child| (parent, *child))
                .fold(self, Self::check_link_for_loops),
            Value::List(lhs, rhs) => self
                .check_link_for_loops((parent, *lhs))
                .check_link_for_loops((parent, *rhs)),
            Value::Constant(_) | Value::Integer(_) | Value::Error(_) => self,
        }
    }

    fn check_link_for_loops(mut self, (parent, child): (Address, Address)) -> Self {
        self.descendants.entry(parent).or_default().insert(child);

        for entry in self.descendants.values_mut() {
            if entry.contains(&parent) {
                entry.insert(child);
            }
        }

        if let Some(entry) = self.descendants.get(&child) {
            if entry.contains(&parent) {
                self.loops.insert(LoopLink { parent, child });
                return self;
            }
        }

        self.check_node_for_loops(child)
    }
}

struct VariableBindingsState<'a> {
    values: &'a BTreeMap<Address, Value>,
    variable_bindings: BTreeMap<ArcStr, Address>,
}

impl<'a> VariableBindingsState<'a> {
    fn get_value(&self, address: &Address) -> &'a Value {
        match self
            .values
            .get(address)
            .unwrap_or_else(|| panic!("No value for address {address}"))
        {
            value @ Value::Reference(reference) => {
                if address == reference {
                    value
                } else {
                    self.get_value(reference)
                }
            }
            value => value,
        }
    }

    fn calculate_bindings(mut self, (term, address): (&'a Term, &Address)) -> Self {
        let value = self.get_value(address);
        match (term, value) {
            (Term::Variable { name }, _) => {
                self.variable_bindings.insert(name.as_string(), *address);
                self
            }
            (Term::Structure { terms, .. }, Value::Structure(_, term_addresses)) => terms
                .iter()
                .zip(term_addresses.as_ref())
                .fold(self, Self::calculate_bindings),
            (Term::List { head, tail }, Value::List(head_address, tail_address)) => self
                .calculate_bindings((head, head_address))
                .calculate_bindings((tail, tail_address)),
            (Term::Constant { .. }, Value::Constant(..))
            | (Term::Integer { .. }, Value::Integer(..))
            | (Term::Void, _) => self,
            _ => panic!("Failed to unify {term:?} with {value:?}"),
        }
    }

    fn generate_loop_variables(mut self, loops: &HashSet<LoopLink>) -> Self {
        let loop_variables = loops
            .iter()
            .filter_map(|LoopLink { child, .. }| {
                (!self
                    .variable_bindings
                    .iter()
                    .any(|(_, variable)| variable == child))
                .then_some(*child)
            })
            .collect::<BTreeSet<_>>();

        self.variable_bindings
            .extend((1..).map(|n| arcstr::format!("_S{n}")).zip(loop_variables));

        self
    }
}

struct Answer<'a> {
    program_info: &'a ProgramInfo,
    values: BTreeMap<Address, Value>,
    variables: BTreeMap<ArcStr, Address>,
    loops: HashSet<LoopLink>,
}

impl<'a> fmt::Display for Answer<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let printed_answers_count =
            self.variables
                .iter()
                .try_fold(0, |printed_answers_count, (name, address)| {
                    if let Some(Value::Reference(reference)) = self.values.get(address) {
                        if Some(name)
                            == self
                                .variables
                                .iter()
                                .find_map(|(name, address)| (reference == address).then_some(name))
                        {
                            return Ok(printed_answers_count);
                        }
                    }

                    if printed_answers_count > 0 {
                        writeln!(f, ",")?;
                    }

                    write!(
                        f,
                        "{name} = {}",
                        DisplayValue {
                            answer: self,
                            parent: None,
                            address,
                            precedence: None
                        }
                    )
                    .map(|()| printed_answers_count + 1)
                })?;

        match printed_answers_count {
            0 => "true".bold().fmt(f),
            1 => Ok(()),
            _ => " ".fmt(f),
        }
    }
}

trait TryCollectVec<T, E>: Sized + Iterator<Item = Result<T, E>> {
    fn try_collect_vec(self) -> Result<Vec<T>, E> {
        self.collect()
    }
}

impl<T, E, I: Iterator<Item = Result<T, E>>> TryCollectVec<T, E> for I {}

trait TryCollectString<'a, E>: Sized + Iterator<Item = Result<&'a str, E>> {
    fn try_collect_string(self) -> Result<String, E> {
        self.collect()
    }
}

impl<'a, E, I: Iterator<Item = Result<&'a str, E>>> TryCollectString<'a, E> for I {}

struct ErrorMessage(String);

impl fmt::Debug for ErrorMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl fmt::Display for ErrorMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for ErrorMessage {}

impl<'b, C> minicbor::Decode<'b, C> for ErrorMessage {
    fn decode(
        d: &mut minicbor::Decoder<'b>,
        _ctx: &mut C,
    ) -> Result<Self, minicbor::decode::Error> {
        d.str_iter()?.try_collect_string().map(ErrorMessage)
    }
}

type SolutionRegisters = Vec<Option<Address>>;

#[derive(Debug)]
enum ReportStatusResponse {
    Error(ErrorMessage),
    WaitingForProgram,
    WaitingForQuery,
}

impl fmt::Display for ReportStatusResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReportStatusResponse::Error(error) => error.fmt(f),
            ReportStatusResponse::WaitingForQuery => "Waiting for Query".fmt(f),
            ReportStatusResponse::WaitingForProgram => "Waiting for Program".fmt(f),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("Bad Status: {0}")]
struct BadStatus(ReportStatusResponse);

impl ReportStatusResponse {
    fn assert_can_submit_program(self) -> Result<(), BadStatus> {
        match self {
            Self::WaitingForProgram | Self::WaitingForQuery => Ok(()),
            Self::Error(_) => Err(BadStatus(self)),
        }
    }

    fn assert_can_submit_query(self) -> Result<(), BadStatus> {
        match self {
            Self::WaitingForQuery => Ok(()),
            Self::WaitingForProgram | Self::Error(_) => Err(BadStatus(self)),
        }
    }
}

type GetSystemCallsResponse = comms::GetSystemCallsResponse<Vec<(String, Arity)>>;

#[derive(Debug)]
struct StructureTerms(Vec<Option<Address>>);

impl<'b, C> minicbor::Decode<'b, C> for StructureTerms {
    fn decode(d: &mut minicbor::Decoder<'b>, ctx: &mut C) -> Result<Self, minicbor::decode::Error> {
        d.decode_with(ctx).map(Self)
    }
}

#[derive(Debug)]
struct IntegerLeBytes(Vec<u8>);

impl<'b, C> minicbor::Decode<'b, C> for IntegerLeBytes {
    fn decode(
        d: &mut minicbor::Decoder<'b>,
        _ctx: &mut C,
    ) -> Result<Self, minicbor::decode::Error> {
        d.array_iter()?.try_collect_vec().map(Self)
    }
}

impl AsRef<[u8]> for IntegerLeBytes {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

type LookupMemoryResponse = comms::LookupMemoryResponse<StructureTerms, IntegerLeBytes>;
type Solution = comms::Solution<SolutionRegisters>;
type SubmitQueryResponse = comms::SubmitQueryResponse<SolutionRegisters>;

trait ReadExt: Read {
    fn read_one(&mut self) -> std::io::Result<u8> {
        let mut buffer = 0;
        self.read_exact(std::slice::from_mut(&mut buffer))?;
        Ok(buffer)
    }

    fn decode_response<T: for<'b> minicbor::Decode<'b, ()>>(
        &mut self,
    ) -> anyhow::Result<Result<T, ErrorMessage>> {
        let mut buffer = Vec::new();
        loop {
            let block_size = self.read_one()?;

            let Some(block_size) = block_size.checked_sub(1) else {
                return Ok(comms::CommandResponse::into_response(minicbor::decode(&buffer)?));
            };

            let start = buffer.len();

            buffer.resize(start + usize::from(block_size), 0);
            self.read_exact(&mut buffer[start..])?;

            if block_size < 254 {
                buffer.push(0);
            }
        }
    }

    fn decode_response_ok<T: for<'b> minicbor::Decode<'b, ()>>(&mut self) -> anyhow::Result<T> {
        self.decode_response()?.map_err(anyhow::Error::new)
    }
}

impl<T: Read> ReadExt for T {}

trait WriteExt: Write {
    fn encode_command(&mut self, command: comms::Command) -> anyhow::Result<()> {
        let buffer = minicbor::to_vec(command)?;

        let mut buffer = cobs::encode_vec(&buffer);
        buffer.push(0);

        self.write_all(&buffer)?;
        Ok(self.flush()?)
    }

    fn write_words(&mut self, words: &[[u8; 4]]) -> std::io::Result<()> {
        for word in words {
            self.write_all(word)?;
        }
        self.flush()
    }
}

impl<T: Write> WriteExt for T {}

trait ReadWriteExt: Sized + Read + Write {
    fn get_status(&mut self) -> anyhow::Result<ReportStatusResponse> {
        self.encode_command(comms::Command::ReportStatus)?;
        self.decode_response().map(|response| match response {
            Err(error) => ReportStatusResponse::Error(error),
            Ok(comms::ReportStatusResponse::WaitingForProgram) => {
                ReportStatusResponse::WaitingForProgram
            }
            Ok(comms::ReportStatusResponse::WaitingForQuery) => {
                ReportStatusResponse::WaitingForQuery
            }
        })
    }

    fn get_value(&mut self, address: Address) -> anyhow::Result<(Address, Value)> {
        self.encode_command(comms::Command::LookupMemory { address })?;
        match self.decode_response()? {
            Err(ErrorMessage(error)) => Ok((address, Value::Error(error))),
            Ok(LookupMemoryResponse::MemoryValue { address, value }) => {
                let value = match value {
                    comms::Value::Reference(reference) => Value::Reference(reference),
                    comms::Value::Structure(f, terms) => {
                        Value::Structure(f, terms.0.into_iter().map(Option::unwrap).collect())
                    }
                    comms::Value::List(lhs, rhs) => Value::List(lhs.unwrap(), rhs.unwrap()),
                    comms::Value::Constant(c) => Value::Constant(c),
                    comms::Value::Integer { sign, le_bytes } => {
                        Value::Integer(BigInt::from_bytes_le(
                            match sign {
                                comms::IntegerSign::Negative => num_bigint::Sign::Minus,
                                comms::IntegerSign::Zero => num_bigint::Sign::NoSign,
                                comms::IntegerSign::Positive => num_bigint::Sign::Plus,
                            },
                            le_bytes.as_ref(),
                        ))
                    }
                };

                Ok((address, value))
            }
        }
    }

    fn get_answer<'a>(
        &mut self,
        program_info: &'a ProgramInfo,
        query: &'a Query,
        solution_registers: SolutionRegisters,
    ) -> anyhow::Result<Answer<'a>> {
        let solution_registers = solution_registers
            .into_iter()
            .map(Option::unwrap)
            .collect::<Vec<_>>();

        let mut known_values = BTreeMap::new();

        let mut values_to_get = solution_registers.clone();

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

        let loops = solution_registers
            .iter()
            .copied()
            .fold(
                LoopScannerState {
                    values: &known_values,
                    descendants: HashMap::new(),
                    loops: HashSet::new(),
                },
                LoopScannerState::check_node_for_loops,
            )
            .loops;

        let variables = query
            .terms
            .iter()
            .zip(solution_registers.as_slice())
            .fold(
                VariableBindingsState {
                    values: &known_values,
                    variable_bindings: BTreeMap::new(),
                },
                VariableBindingsState::calculate_bindings,
            )
            .generate_loop_variables(&loops)
            .variable_bindings;

        Ok(Answer {
            program_info,
            values: known_values,
            variables,
            loops,
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

struct PrintLn<T>(T);

impl<T: fmt::Display> crossterm::Command for PrintLn<T> {
    fn write_ansi(&self, f: &mut impl fmt::Write) -> fmt::Result {
        Print(&self.0).write_ansi(f)?;
        Print(EndOfLine).write_ansi(f)
    }

    fn execute_winapi(&self) -> crossterm::Result<()> {
        Print(&self.0).execute_winapi()?;
        Print(EndOfLine).execute_winapi()
    }

    fn is_ansi_code_supported(&self) -> bool {
        Print(&self.0).is_ansi_code_supported()
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
        stdout.execute(PrintLn("Connecting to device"))?;
        Port::TcpStream(TcpStream::connect("localhost:8080").context("Failed to connect")?)
    } else {
        Port::Serial(
            serialport::new(
                serialport::available_ports()?
                    .into_iter()
                    .find_map(|port| match port.port_type {
                        SerialPortType::UsbPort(UsbPortInfo {
                            serial_number: Some(serial_number),
                            ..
                        }) if serial_number.starts_with("BARE_METAL_PROLOG_") => {
                            Some(port.port_name)
                        }
                        _ => None,
                    })
                    .context("Device not connected")?,
                9600,
            )
            .open()?,
        )
    };

    port.get_status()?.assert_can_submit_program()?;

    port.encode_command(comms::Command::GetSystemCalls)?;
    let GetSystemCallsResponse::SystemCalls(system_calls) = port.decode_response_ok()?;

    let (program_info, program_words) = compiler::compile_program(system_calls, program)?;

    port.encode_command(comms::Command::SubmitProgram {
        code_submission: comms::CodeSubmission::new(&program_words),
    })?;

    port.write_words(&program_words)?;

    let comms::SubmitProgramResponse::Success =
        port.decode_response_ok::<comms::SubmitProgramResponse>()?;

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
                    stdout.execute(PrintLn(err))?;
                    continue;
                }
            }
        };

        port.encode_command(comms::Command::SubmitQuery {
            code_submission: comms::CodeSubmission::new(&query_words),
        })?;

        port.write_words(&query_words)?;

        'outer: loop {
            break match port.decode_response_ok::<SubmitQueryResponse>()? {
                SubmitQueryResponse::Solution(Solution::SingleSolution(solution_registers)) => {
                    let answer = port.get_answer(&program_info, &query, solution_registers)?;

                    stdout.queue(Print(answer))?
                }
                SubmitQueryResponse::Solution(Solution::MultipleSolutions(solution_registers)) => {
                    stdout.execute(Print(port.get_answer(
                        &program_info,
                        &query,
                        solution_registers,
                    )?))?;

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
                                    stdout.execute(PrintLn(";"))?;
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

                    port.encode_command(comms::Command::NextSolution)?;
                    continue;
                }
                SubmitQueryResponse::NoSolution => stdout.queue(Print("fail".bold()))?,
            };
        }
        .execute(PrintLn('.'))?;
    }
}
