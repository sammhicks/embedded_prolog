use core::fmt;

use crate::{
    log_debug, log_trace,
    serializable::{Serializable, SerializableWrapper},
};

#[derive(Clone, Copy, Debug)]
pub struct Xn {
    xn: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct Yn {
    yn: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct Ai {
    ai: u8,
}

impl Ai {
    fn iter(self, arity: Arity) -> impl Iterator<Item = Self> {
        (self.ai..).take(arity.value()).map(|ai| Self { ai })
    }
}

#[derive(Clone, Copy)]
pub struct RegisterIndex(u8);

impl fmt::Debug for RegisterIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RegisterIndex({})", self)
    }
}

impl fmt::Display for RegisterIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:02X}", self.0)
    }
}

impl RegisterIndex {
    fn to_index(self) -> usize {
        self.0 as usize
    }
}

impl From<Xn> for RegisterIndex {
    fn from(Xn { xn }: Xn) -> Self {
        RegisterIndex(xn)
    }
}

impl From<Ai> for RegisterIndex {
    fn from(Ai { ai }: Ai) -> Self {
        RegisterIndex(ai)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Functor(u16);

impl fmt::Debug for Functor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Functor({})", self)
    }
}

impl fmt::Display for Functor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04X}", self.0)
    }
}

impl SerializableWrapper for Functor {
    type Inner = u16;

    fn from_inner(inner: Self::Inner) -> Self {
        Self(inner)
    }

    fn into_inner(self) -> Self::Inner {
        self.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Arity(u8);

impl fmt::Debug for Arity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Arity({})", self)
    }
}

impl fmt::Display for Arity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:02X}", self.0)
    }
}

impl SerializableWrapper for Arity {
    type Inner = u8;

    fn from_inner(inner: Self::Inner) -> Self {
        Self(inner)
    }

    fn into_inner(self) -> Self::Inner {
        self.0
    }
}

impl Arity {
    fn value(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Constant(u16);

impl fmt::Debug for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Constant({})", self)
    }
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04X}", self.0)
    }
}

impl SerializableWrapper for Constant {
    type Inner = u16;

    fn from_inner(inner: Self::Inner) -> Self {
        Self(inner)
    }

    fn into_inner(self) -> Self::Inner {
        self.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Address(u16);

impl fmt::Debug for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Address({})", self)
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04X}", self.0)
    }
}

impl From<EnvironmentAddress> for Address {
    fn from(e: EnvironmentAddress) -> Self {
        Self(e.0)
    }
}

impl Address {
    const NULL: Self = Address(u16::MAX);

    pub fn offset(self, offset: u16) -> Self {
        Self(self.0 + offset)
    }
}

impl SerializableWrapper for Address {
    type Inner = u16;

    fn from_inner(inner: Self::Inner) -> Self {
        Self(inner)
    }

    fn into_inner(self) -> Self::Inner {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EnvironmentAddress(u16);

impl EnvironmentAddress {
    const NULL: Self = Self(u16::MAX);

    fn new(address: Address) -> Self {
        Self(address.0)
    }

    fn nth_variable_address(self, index: Yn) -> Address {
        Address(self.0 + Environment::HEADER_SIZE + index.yn as u16)
    }

    fn variable_addresses(self, count: Arity) -> impl Iterator<Item = Address> {
        (0..count.0).map(move |yn| self.nth_variable_address(Yn { yn }))
    }

    fn after(self, arity: Arity) -> Address {
        self.nth_variable_address(Yn { yn: arity.0 })
    }
}

impl SerializableWrapper for EnvironmentAddress {
    type Inner = u16;

    fn from_inner(inner: Self::Inner) -> Self {
        Self(inner)
    }

    fn into_inner(self) -> Self::Inner {
        self.0
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct ProgramCounter(u16);

impl fmt::Debug for ProgramCounter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ProgramCounter({})", self)
    }
}

impl fmt::Display for ProgramCounter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04X}", self.0)
    }
}

impl ProgramCounter {
    const NULL: Self = Self(u16::MAX);

    fn offset(self, offset: u16) -> Self {
        Self(self.0 + offset)
    }
}

impl SerializableWrapper for ProgramCounter {
    type Inner = u16;

    fn from_inner(inner: Self::Inner) -> Self {
        Self(inner)
    }

    fn into_inner(self) -> Self::Inner {
        self.0
    }
}

const REFERENCE_HEADER: u8 = b'R';
const STRUCTURE_HEADER: u8 = b'S';
const CONSTANT_HEADER: u8 = b'C';
const ENVIRONMENT_HEADER: u8 = b'E';

struct BadMemoryWord(u32);

impl fmt::Debug for BadMemoryWord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:08X}", self.0)
    }
}

pub struct BadMemoryRange<'m>(&'m [u32]);

impl fmt::Debug for BadMemoryRange<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for &word in self.0 {
            list.entry(&BadMemoryWord(word));
        }
        list.finish()
    }
}

#[derive(Debug)]
pub enum Error<'m> {
    BadInstruction(ProgramCounter, BadMemoryRange<'m>),
    BadValue(Address, BadMemoryRange<'m>),
    BadEnvironment(EnvironmentAddress, BadMemoryRange<'m>),
}

#[derive(Clone, Copy, Debug)]
pub enum Value {
    Reference(Address),
    Structure(Functor, Arity),
    Constant(Constant),
}

impl Value {
    fn decode<'a, 'm>(memory: &'a Heap<'m>, address: Address) -> Result<Self, Error<'a>> {
        Ok(match memory[address].to_be_bytes() {
            [REFERENCE_HEADER, 0, r1, r0] => Value::Reference(Address::from_be_bytes([r1, r0])),
            [STRUCTURE_HEADER, n, f1, f0] => {
                Value::Structure(Functor::from_be_bytes([f1, f0]), Arity(n))
            }
            [CONSTANT_HEADER, 0, c1, c0] => Value::Constant(Constant::from_be_bytes([c1, c0])),
            _ => {
                return Err(Error::BadValue(
                    address,
                    BadMemoryRange(memory.range(address, 1)),
                ))
            }
        })
    }

    fn encode(self) -> u32 {
        match self {
            Value::Reference(r) => {
                let [r1, r0] = r.into_be_bytes();
                u32::from_be_bytes([REFERENCE_HEADER, 0, r1, r0])
            }
            Value::Structure(f, n) => {
                let [f1, f0] = f.into_be_bytes();
                u32::from_be_bytes([STRUCTURE_HEADER, n.0, f1, f0])
            }
            Value::Constant(c) => {
                let [c1, c0] = c.into_be_bytes();
                u32::from_be_bytes([CONSTANT_HEADER, 0, c1, c0])
            }
        }
    }
}

fn structure_terms(header_address: Address, n: Arity) -> impl Iterator<Item = Address> {
    ((header_address.0 + 1)..).take(n.value()).map(Address)
}

struct Heap<'m>(&'m mut [u32]);

impl<'m, A: Into<Address>> std::ops::Index<A> for Heap<'m> {
    type Output = u32;

    fn index(&self, index: A) -> &Self::Output {
        &self.0[index.into().0 as usize]
    }
}

impl<'m, A: Into<Address>> std::ops::IndexMut<A> for Heap<'m> {
    fn index_mut(&mut self, index: A) -> &mut Self::Output {
        &mut self.0[index.into().0 as usize]
    }
}

impl<'m> Heap<'m> {
    fn range(&self, start: impl Into<Address>, count: usize) -> &[u32] {
        let start = start.into().0 as usize;
        &self.0[start..(start + count)]
    }
}

struct Environment {
    continuation_environment: EnvironmentAddress,
    continuation_point: ProgramCounter,
    number_of_active_permanent_variables: Arity,
    number_of_permanent_variables: Arity,
}

impl Environment {
    const HEADER_SIZE: u16 = 2;

    fn decode<'a, 'm>(
        memory: &'a Heap<'m>,
        e: EnvironmentAddress,
    ) -> Result<(Address, Self), Error<'a>> {
        Ok(match memory[e].to_be_bytes() {
            [ENVIRONMENT_HEADER, 0, a, n] => {
                let [ce1, ce0, cp1, cp0] = memory[Address(e.0 + 1)].to_be_bytes();
                (
                    e.after(Arity(n)),
                    Environment {
                        continuation_environment: EnvironmentAddress::from_be_bytes([ce1, ce0]),
                        continuation_point: ProgramCounter::from_be_bytes([cp1, cp0]),
                        number_of_active_permanent_variables: Arity(a),
                        number_of_permanent_variables: Arity(n),
                    },
                )
            }
            _ => return Err(Error::BadEnvironment(e, BadMemoryRange(memory.range(e, 1)))),
        })
    }

    fn encode(self, memory: &mut Heap<'_>, e: EnvironmentAddress) -> Address {
        memory[e] = u32::from_be_bytes([
            ENVIRONMENT_HEADER,
            0,
            self.number_of_active_permanent_variables.0,
            self.number_of_permanent_variables.0,
        ]);
        let [ce1, ce0] = self.continuation_environment.0.into_be_bytes();
        let [cp1, cp0] = self.continuation_point.into_be_bytes();
        memory[Address(e.0 + 1)] = u32::from_be_bytes([ce1, ce0, cp1, cp0]);

        e.after(self.number_of_permanent_variables)
    }

    fn read_variable(memory: &Heap<'_>, e: EnvironmentAddress, yn: Yn) -> Address {
        let environment = Self::decode(memory, e).unwrap().1;
        assert!(yn.yn < environment.number_of_permanent_variables.0);
        Address(memory[e.nth_variable_address(yn)] as u16)
    }

    fn write_variable(memory: &mut Heap<'_>, e: EnvironmentAddress, yn: Yn, address: Address) {
        let environment = Self::decode(memory, e).unwrap().1;
        assert!(yn.yn < environment.number_of_permanent_variables.0);
        memory[e.nth_variable_address(yn)] = address.0 as u32;
    }

    fn trim(memory: &mut Heap<'_>, e: EnvironmentAddress, n: Arity) {
        let mut environment = Self::decode(memory, e).unwrap().1;
        environment.number_of_active_permanent_variables = Arity(
            environment
                .number_of_active_permanent_variables
                .0
                .checked_sub(n.0)
                .unwrap_or_else(|| {
                    panic!(
                        "Too many variables trimmed: {} > {}",
                        n, environment.number_of_active_permanent_variables
                    )
                }),
        );
        environment.encode(memory, e);
    }
}

#[derive(Debug)]
enum Instruction {
    PutVariableXn { ai: Ai, xn: Xn },
    PutVariableYn { ai: Ai, yn: Yn },
    PutValueXn { ai: Ai, xn: Xn },
    PutValueYn { ai: Ai, yn: Yn },
    PutStructure { ai: Ai, f: Functor, n: Arity },
    PutConstant { ai: Ai, c: Constant },
    PutVoid { ai: Ai, n: Arity },
    GetVariableXn { ai: Ai, xn: Xn },
    GetVariableYn { ai: Ai, yn: Yn },
    GetValueXn { ai: Ai, xn: Xn },
    GetValueYn { ai: Ai, yn: Yn },
    GetStructure { ai: Ai, f: Functor, n: Arity },
    GetConstant { ai: Ai, c: Constant },
    SetVariableXn { xn: Xn },
    SetVariableYn { yn: Yn },
    SetValueXn { xn: Xn },
    SetValueYn { yn: Yn },
    SetConstant { c: Constant },
    SetVoid { n: Arity },
    UnifyVariableXn { xn: Xn },
    UnifyVariableYn { yn: Yn },
    UnifyValueXn { xn: Xn },
    UnifyValueYn { yn: Yn },
    UnifyConstant { c: Constant },
    UnifyVoid { n: Arity },
    Allocate { n: Arity },
    Trim { n: Arity },
    Deallocate,
    Call { p: ProgramCounter, n: Arity },
    Execute { p: ProgramCounter, n: Arity },
    Proceed,
}

impl Instruction {
    fn decode_structure_long_functor(memory: &[u32], pc: ProgramCounter) -> Result<Functor, Error> {
        if let [0, 0, f1, f0] = memory[pc.offset(1).0 as usize].to_be_bytes() {
            Ok(Functor(u16::from_be_bytes([f1, f0])))
        } else {
            Err(Error::BadInstruction(
                pc,
                BadMemoryRange(&memory[pc.0 as usize..pc.offset(2).0 as usize]),
            ))
        }
    }

    fn decode(memory: &[u32], pc: ProgramCounter) -> Result<(ProgramCounter, Self), Error> {
        Ok(match memory[pc.0 as usize].to_be_bytes() {
            [0x00, ai, 0, xn] => (
                pc.offset(1),
                Instruction::PutVariableXn {
                    ai: Ai { ai },
                    xn: Xn { xn },
                },
            ),
            [0x01, ai, 0, yn] => (
                pc.offset(1),
                Instruction::PutVariableYn {
                    ai: Ai { ai },
                    yn: Yn { yn },
                },
            ),
            [0x02, ai, 0, xn] => (
                pc.offset(1),
                Instruction::PutValueXn {
                    ai: Ai { ai },
                    xn: Xn { xn },
                },
            ),
            [0x03, ai, 0, yn] => (
                pc.offset(1),
                Instruction::PutValueYn {
                    ai: Ai { ai },
                    yn: Yn { yn },
                },
            ),
            [0x04, ai, f, n] => (
                pc.offset(1),
                Instruction::PutStructure {
                    ai: Ai { ai },
                    f: Functor(f as u16),
                    n: Arity(n),
                },
            ),
            [0x05, ai, 0, n] => (
                pc.offset(2),
                Instruction::PutStructure {
                    ai: Ai { ai },
                    f: Self::decode_structure_long_functor(memory, pc)?,
                    n: Arity(n),
                },
            ),
            [0x07, ai, c1, c0] => (
                pc.offset(1),
                Instruction::PutConstant {
                    ai: Ai { ai },
                    c: Constant::from_be_bytes([c1, c0]),
                },
            ),
            [0x0a, ai, 0, n] => (
                pc.offset(1),
                Instruction::PutVoid {
                    ai: Ai { ai },
                    n: Arity(n),
                },
            ),
            [0x10, ai, 0, xn] => (
                pc.offset(1),
                Instruction::GetVariableXn {
                    ai: Ai { ai },
                    xn: Xn { xn },
                },
            ),
            [0x11, ai, 0, yn] => (
                pc.offset(1),
                Instruction::GetVariableYn {
                    ai: Ai { ai },
                    yn: Yn { yn },
                },
            ),
            [0x12, ai, 0, xn] => (
                pc.offset(1),
                Instruction::GetValueXn {
                    ai: Ai { ai },
                    xn: Xn { xn },
                },
            ),
            [0x13, ai, 0, yn] => (
                pc.offset(1),
                Instruction::GetValueYn {
                    ai: Ai { ai },
                    yn: Yn { yn },
                },
            ),
            [0x14, ai, f, n] => (
                pc.offset(1),
                Instruction::GetStructure {
                    ai: Ai { ai },
                    f: Functor(f as u16),
                    n: Arity(n),
                },
            ),
            [0x15, ai, 0, n] => (
                pc.offset(2),
                Instruction::GetStructure {
                    ai: Ai { ai },
                    f: Self::decode_structure_long_functor(memory, pc)?,
                    n: Arity(n),
                },
            ),
            [0x17, ai, c1, c0] => (
                pc.offset(1),
                Instruction::GetConstant {
                    ai: Ai { ai },
                    c: Constant::from_be_bytes([c1, c0]),
                },
            ),
            [0x20, 0, 0, xn] => (pc.offset(1), Instruction::SetVariableXn { xn: Xn { xn } }),
            [0x21, 0, 0, yn] => (pc.offset(1), Instruction::SetVariableYn { yn: Yn { yn } }),
            [0x22, 0, 0, xn] => (pc.offset(1), Instruction::SetValueXn { xn: Xn { xn } }),
            [0x23, 0, 0, yn] => (pc.offset(1), Instruction::SetValueYn { yn: Yn { yn } }),
            [0x27, 0, c1, c0] => (
                pc.offset(1),
                Instruction::SetConstant {
                    c: Constant::from_be_bytes([c1, c0]),
                },
            ),
            [0x2a, 0, 0, n] => (pc.offset(1), Instruction::SetVoid { n: Arity(n) }),
            [0x30, 0, 0, xn] => (pc.offset(1), Instruction::UnifyVariableXn { xn: Xn { xn } }),
            [0x31, 0, 0, yn] => (pc.offset(1), Instruction::UnifyVariableYn { yn: Yn { yn } }),
            [0x32, 0, 0, xn] => (pc.offset(1), Instruction::UnifyValueXn { xn: Xn { xn } }),
            [0x33, 0, 0, yn] => (pc.offset(1), Instruction::UnifyValueYn { yn: Yn { yn } }),
            [0x37, 0, c1, c0] => (
                pc.offset(1),
                Instruction::UnifyConstant {
                    c: Constant::from_be_bytes([c1, c0]),
                },
            ),
            [0x3a, 0, 0, n] => (pc.offset(1), Instruction::UnifyVoid { n: Arity(n) }),
            [0x40, 0, 0, n] => (pc.offset(1), Instruction::Allocate { n: Arity(n) }),
            [0x41, 0, 0, n] => (pc.offset(1), Instruction::Trim { n: Arity(n) }),
            [0x42, 0, 0, 0] => (pc.offset(1), Instruction::Deallocate),
            [0x43, n, p1, p0] => (
                pc.offset(1),
                Instruction::Call {
                    p: ProgramCounter::from_be_bytes([p1, p0]),
                    n: Arity(n),
                },
            ),
            [0x44, n, p1, p0] => (
                pc.offset(1),
                Instruction::Execute {
                    p: ProgramCounter::from_be_bytes([p1, p0]),
                    n: Arity(n),
                },
            ),
            [0x45, 0, 0, 0] => (pc.offset(1), Instruction::Proceed),
            _ => {
                return Err(Error::BadInstruction(
                    pc,
                    BadMemoryRange(core::slice::from_ref(&memory[pc.0 as usize])),
                ))
            }
        })
    }
}

#[derive(Debug)]
struct RegisterBlock([Address; 32]);

impl RegisterBlock {
    fn load(&self, index: impl Into<RegisterIndex>) -> Address {
        let index = index.into();
        log_trace!("Loading Register {}", index);
        *self.0.get(index.to_index()).unwrap_or_else(|| {
            panic!(
                "Register Index {} out of range ({})",
                index,
                (&self.0[..]).len()
            );
        })
    }

    fn store(&mut self, index: impl Into<RegisterIndex>, address: Address) {
        let index = index.into();
        log_trace!("Storing {} in Register {}", address, index);
        let len = (&self.0[..]).len();
        *self.0.get_mut(index.to_index()).unwrap_or_else(|| {
            panic!("Register Index {} out of range ({})", index, len);
        }) = address;
    }
}

#[derive(Debug)]
enum CurrentlyExecuting {
    Query,
    Program,
}

#[derive(Copy, Clone, Debug)]
struct ReadMode {
    header_address: Address,
    index: Arity,
    arity: Arity,
}

impl ReadMode {
    fn next_term(mut self) -> (Address, ReadWriteMode) {
        if self.index == self.arity {
            panic!("No more terms to read");
        }

        let address = self.header_address.offset(1).offset(self.index.0 as u16);

        self.index = Arity(self.index.0 + 1);

        if self.index == self.arity {
            log_trace!("Finished reading structure");
            (address, ReadWriteMode::Unknown)
        } else {
            (address, ReadWriteMode::Read(self))
        }
    }

    fn skip_terms(mut self, n: Arity) -> ReadWriteMode {
        self.index = Arity(self.index.0 + n.0);
        if self.index > self.arity {
            panic!("No more terms to read");
        }

        if self.index == self.arity {
            log_trace!("Finished reading structure");
            ReadWriteMode::Unknown
        } else {
            ReadWriteMode::Read(self)
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum ReadWriteMode {
    Unknown,
    Read(ReadMode),
    Write,
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum ExecutionFailure {
    Failed = b'F',
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum ExecutionSuccess {
    SingleAnswer = b'A',
    MultipleAnswers = b'C',
}

#[derive(Debug)]
struct UnificationFailure;

pub struct MachineMemory<'m> {
    pub program: &'m [u32],
    pub query: &'m [u32],
    pub memory: &'m mut [u32],
}

pub struct Machine<'m> {
    currently_executing: CurrentlyExecuting,
    read_write_mode: ReadWriteMode,
    pc: ProgramCounter,
    cp: ProgramCounter,
    h: Address,
    e: EnvironmentAddress,
    argument_count: Arity,
    registers: RegisterBlock,
    program: &'m [u32],
    query: &'m [u32],
    memory: Heap<'m>,
}

impl<'m> Machine<'m> {
    pub fn run(
        MachineMemory {
            program,
            query,
            memory,
        }: MachineMemory<'m>,
    ) -> Result<(Self, ExecutionSuccess), ExecutionFailure> {
        let mut machine = Self {
            currently_executing: CurrentlyExecuting::Query,
            read_write_mode: ReadWriteMode::Unknown,
            pc: ProgramCounter(0),
            cp: ProgramCounter::NULL,
            h: Address(0),
            e: EnvironmentAddress::NULL,
            argument_count: Arity(0),
            registers: RegisterBlock([Address::NULL; 32]),
            program,
            query,
            memory: Heap(memory),
        };

        let execution_result = machine.continue_execution();
        execution_result.map(|success| (machine, success))
    }

    fn continue_execution(&mut self) -> Result<ExecutionSuccess, ExecutionFailure> {
        loop {
            if self.pc == ProgramCounter::NULL {
                return Ok(ExecutionSuccess::SingleAnswer);
            }

            let (new_pc, instruction) = match &self.currently_executing {
                CurrentlyExecuting::Query => Instruction::decode(self.query, self.pc).unwrap(),
                CurrentlyExecuting::Program => Instruction::decode(self.program, self.pc).unwrap(),
            };

            log_debug!(
                "{:?} Instruction @{} : {:?}",
                self.currently_executing,
                self.pc,
                instruction
            );

            self.pc = new_pc;

            self.execute_instruction(instruction)?;

            log_trace!("");
        }
    }

    fn execute_instruction(&mut self, instruction: Instruction) -> Result<(), ExecutionFailure> {
        match instruction {
            Instruction::PutVariableXn { ai, xn } => {
                let address = self.new_variable();
                self.registers.store(ai, address);
                self.registers.store(xn, address);
                Ok(())
            }
            Instruction::PutVariableYn { ai, yn } => {
                let address = self.new_variable();
                self.registers.store(ai, address);
                Environment::write_variable(&mut self.memory, self.e, yn, address);
                Ok(())
            }
            Instruction::PutValueXn { ai, xn } => {
                let address = self.registers.load(xn);
                self.registers.store(ai, address);
                Ok(())
            }
            Instruction::PutValueYn { ai, yn } => {
                let address = Environment::read_variable(&self.memory, self.e, yn);
                self.registers.store(ai, address);
                Ok(())
            }
            Instruction::PutStructure { ai, f, n } => {
                let address = self.new_structure(f, n);
                self.registers.store(ai, address);
                Ok(())
            }
            Instruction::PutConstant { ai, c } => {
                let address = self.new_constant(c);
                self.registers.store(ai, address);
                Ok(())
            }
            Instruction::PutVoid { ai, n } => {
                for ai in ai.iter(n) {
                    let address = self.new_variable();
                    self.registers.store(ai, address);
                }
                Ok(())
            }
            Instruction::GetVariableXn { ai, xn } => {
                self.registers.store(xn, self.registers.load(ai));
                Ok(())
            }
            Instruction::GetVariableYn { ai, yn } => {
                let address = self.registers.load(ai);
                Environment::write_variable(&mut self.memory, self.e, yn, address);
                Ok(())
            }
            Instruction::GetValueXn { ai, xn } => self
                .unify(self.registers.load(xn), self.registers.load(ai))
                .or_else(|UnificationFailure| self.backtrack()),
            Instruction::GetValueYn { ai, yn } => self
                .unify(
                    Environment::read_variable(&self.memory, self.e, yn),
                    self.registers.load(ai),
                )
                .or_else(|UnificationFailure| self.backtrack()),
            Instruction::GetStructure { ai, f, n } => {
                let (address, value) = self.lookup_register(ai);
                match value {
                    Value::Reference(r) => {
                        log_trace!("Writing structure {}/{}", f, n);

                        let structure_address = self.new_structure(f, n);

                        self.bind_variable(r, structure_address);
                        self.read_write_mode = ReadWriteMode::Write;
                    }
                    Value::Structure(found_f, found_n) => {
                        if f == found_f && n == found_n {
                            log_trace!("Reading structure {}/{}", f, n);

                            self.read_write_mode = ReadWriteMode::Read(ReadMode {
                                header_address: address,
                                index: Arity(0),
                                arity: n,
                            });
                        } else {
                            self.backtrack()?;
                        }
                    }
                    Value::Constant(_) => self.backtrack()?,
                }
                Ok(())
            }
            Instruction::GetConstant { ai, c } => match self.lookup_register(ai).1 {
                Value::Reference(address) => {
                    self.bind_to_constant(address, c);
                    Ok(())
                }
                Value::Constant(rc) if rc == c => Ok(()),
                _ => self.backtrack(),
            },
            Instruction::SetVariableXn { xn } => {
                let address = self.new_variable();
                self.registers.store(xn, address);
                Ok(())
            }
            Instruction::SetVariableYn { yn } => {
                let address = self.new_variable();
                Environment::write_variable(&mut self.memory, self.e, yn, address);
                Ok(())
            }
            Instruction::SetValueXn { xn } => {
                self.new_variable_with_value(self.registers.load(xn));
                Ok(())
            }
            Instruction::SetValueYn { yn } => {
                let address = Environment::read_variable(&self.memory, self.e, yn);
                self.new_variable_with_value(address);
                Ok(())
            }
            Instruction::SetConstant { c } => {
                self.new_constant(c);
                Ok(())
            }
            Instruction::SetVoid { n } => {
                for _ in 0..n.into_inner() {
                    self.new_variable();
                }
                Ok(())
            }
            Instruction::UnifyVariableXn { xn } => {
                self.unify_variable(xn, |this, xn, address| this.registers.store(xn, address))
            }
            Instruction::UnifyVariableYn { yn } => self.unify_variable(yn, |this, yn, address| {
                Environment::write_variable(&mut this.memory, this.e, yn, address)
            }),
            Instruction::UnifyValueXn { xn } => {
                self.unify_value(xn, |this, xn| this.registers.load(xn))
            }
            Instruction::UnifyValueYn { yn } => self.unify_value(yn, |this, yn| {
                Environment::read_variable(&this.memory, this.e, yn)
            }),
            Instruction::UnifyConstant { c } => {
                match self.read_write_mode {
                    ReadWriteMode::Unknown => panic!("Unknown read/write mode"),
                    ReadWriteMode::Read(read_mode) => {
                        let (term_address, next_mode) = read_mode.next_term();

                        self.read_write_mode = next_mode;

                        match self.lookup_memory(term_address).1 {
                            Value::Reference(address) => {
                                self.memory[address] = Value::Constant(c).encode();
                                // trail(address)
                                Ok(())
                            }
                            Value::Constant(c1) => {
                                if c == c1 {
                                    Ok(())
                                } else {
                                    self.backtrack()
                                }
                            }
                            Value::Structure(_, _) => self.backtrack(),
                        }
                    }
                    ReadWriteMode::Write => {
                        self.new_constant(c);

                        Ok(())
                    }
                }
            }
            Instruction::UnifyVoid { n } => match self.read_write_mode {
                ReadWriteMode::Unknown => panic!("Unknown read/write mode"),
                ReadWriteMode::Read(read_mode) => {
                    self.read_write_mode = read_mode.skip_terms(n);
                    Ok(())
                }
                ReadWriteMode::Write => {
                    for _ in 0..n.0 {
                        self.new_variable();
                    }
                    Ok(())
                }
            },
            Instruction::Allocate { n } => {
                self.new_environment(n);
                Ok(())
            }
            Instruction::Trim { n } => {
                Environment::trim(&mut self.memory, self.e, n);
                Ok(())
            }
            Instruction::Deallocate => {
                let environment = Environment::decode(&self.memory, self.e).unwrap().1;
                self.cp = environment.continuation_point;
                self.e = environment.continuation_environment;

                log_trace!("CP => {}", self.cp);
                log_trace!("E => {}", self.e.0);

                Ok(())
            }
            Instruction::Call { p, n } => {
                match self.currently_executing {
                    CurrentlyExecuting::Query => {
                        self.currently_executing = CurrentlyExecuting::Program;
                        self.new_environment(n);
                        for index in 0..n.0 {
                            let value = self.lookup_register(Ai { ai: index }).0;
                            log_trace!("Saved Register Value: {}", value);
                            Environment::write_variable(
                                &mut self.memory,
                                self.e,
                                Yn { yn: index },
                                value,
                            );
                        }

                        self.cp = ProgramCounter::NULL;
                    }
                    CurrentlyExecuting::Program => {
                        self.cp = self.pc;
                    }
                }
                self.pc = p;
                self.argument_count = n;

                // TODO - Clear registers above "n"
                // TODO - Set cut point

                Ok(())
            }
            Instruction::Execute { p, n } => {
                self.pc = p;
                self.argument_count = n;

                // TODO - Clear registers above "n"
                // TODO - Set cut point

                Ok(())
            }
            Instruction::Proceed => {
                log_trace!("Proceeding to {}", self.cp);
                self.pc = self.cp;
                Ok(())
            }
        }
    }

    pub fn lookup_register(&self, index: Ai) -> (Address, Value) {
        self.lookup_memory(self.registers.load(index))
    }

    pub fn lookup_memory(&self, mut address: Address) -> (Address, Value) {
        log_trace!("Looking up memory at {}", address);
        let value = loop {
            let value = Value::decode(&self.memory, address).unwrap();
            address = if let Value::Reference(new_address) = value {
                if new_address == address {
                    break value;
                } else {
                    new_address
                }
            } else {
                break value;
            };
            log_trace!("Looking up memory at {}", address);
        };

        log_trace!("Value: {:?}", value);

        (address, value)
    }

    pub fn solution_registers(&self) -> impl Iterator<Item = Address> + '_ {
        let environment = Environment::decode(&self.memory, self.e).unwrap().1;
        self.e
            .variable_addresses(environment.number_of_permanent_variables)
            .map(move |address| Address(self.memory[address] as u16))
    }

    fn new_value(&mut self, factory: impl FnOnce(Address) -> Value) -> Address {
        let address = self.h;
        let new_value = factory(address);
        log_trace!("New value at {}: {:?}", self.h, new_value);
        self.memory[address] = new_value.encode();
        self.h = self.h.offset(1);
        address
    }

    fn new_variable(&mut self) -> Address {
        self.new_value(Value::Reference)
    }

    fn new_variable_with_value(&mut self, address: Address) -> Address {
        self.new_value(|_| Value::Reference(address))
    }

    fn new_structure(&mut self, f: Functor, n: Arity) -> Address {
        self.new_value(|_| Value::Structure(f, n))
    }

    fn new_constant(&mut self, c: Constant) -> Address {
        self.new_value(|_| Value::Constant(c))
    }

    fn new_environment(&mut self, number_of_permanent_variables: Arity) {
        let environment = Environment {
            continuation_environment: self.e,
            continuation_point: self.cp,
            number_of_active_permanent_variables: number_of_permanent_variables,
            number_of_permanent_variables,
        };

        self.e = EnvironmentAddress::new(self.h);
        self.h = environment.encode(&mut self.memory, EnvironmentAddress::new(self.h));
    }

    fn bind_variable(&mut self, variable_address: Address, value_address: Address) {
        assert!(matches!(
            Value::decode(&self.memory, variable_address),
            Ok(Value::Reference(_))
        ));

        log_trace!(
            "Binding memory {} to value at {}",
            variable_address,
            value_address
        );

        self.memory[variable_address] = Value::Reference(value_address).encode();
    }

    fn bind_to_constant(&mut self, address: Address, c: Constant) {
        match Value::decode(&self.memory, address).unwrap() {
            Value::Reference(r) => assert_eq!(r, address),
            value => panic!("Value at {} not a reference but a {:?}", address, value),
        };
        self.memory[address] = Value::Constant(c).encode();
    }

    fn unify_variable<I>(
        &mut self,
        index: I,
        store: impl FnOnce(&mut Self, I, Address),
    ) -> Result<(), ExecutionFailure> {
        match self.read_write_mode {
            ReadWriteMode::Unknown => panic!("Unknown read/write mode"),
            ReadWriteMode::Read(read_mode) => {
                let (term_address, next_mode) = read_mode.next_term();

                self.read_write_mode = next_mode;

                store(self, index, term_address);

                Ok(())
            }
            ReadWriteMode::Write => {
                let address = self.new_variable();
                store(self, index, address);

                Ok(())
            }
        }
    }

    fn unify_value<I>(
        &mut self,
        index: I,
        load: impl FnOnce(&mut Self, I) -> Address,
    ) -> Result<(), ExecutionFailure> {
        match self.read_write_mode {
            ReadWriteMode::Unknown => panic!("Unknown read/write mode"),
            ReadWriteMode::Read(read_mode) => {
                let (term_address, next_mode) = read_mode.next_term();

                self.read_write_mode = next_mode;

                let register_address = load(self, index);

                self.unify(register_address, term_address)
                    .or_else(|UnificationFailure| self.backtrack())
            }
            ReadWriteMode::Write => {
                let address = load(self, index);
                self.new_variable_with_value(address);
                Ok(())
            }
        }
    }

    fn unify(&mut self, a1: Address, a2: Address) -> Result<(), UnificationFailure> {
        log_trace!("Unifying {} and {}", a1, a2);
        let e1 = self.lookup_memory(a1);
        let e2 = self.lookup_memory(a2);
        log_trace!("Resolved to {:?} and {:?}", e1, e2);
        match (self.lookup_memory(a1), self.lookup_memory(a2)) {
            ((a1, Value::Reference(r1)), (a2, Value::Reference(r2))) => {
                assert_eq!(a1, r1);
                assert_eq!(a2, r2);
                if a2 < a1 {
                    self.bind_variable(a1, a2);
                } else {
                    self.bind_variable(a2, a1);
                }
                Ok(())
            }
            ((a1, Value::Reference(r1)), (a2, value)) => {
                assert_eq!(a1, r1);
                assert!(!matches!(value, Value::Reference(_)));
                self.bind_variable(a1, a2);
                Ok(())
            }
            ((a1, value), (a2, Value::Reference(r2))) => {
                assert_eq!(a2, r2);
                assert!(!matches!(value, Value::Reference(_)));
                self.bind_variable(a2, a1);
                Ok(())
            }
            ((a1, Value::Structure(f1, n1)), (a2, Value::Structure(f2, n2))) => {
                if f1 == f2 && n1 == n2 {
                    let terms_1 = structure_terms(a1, n1);
                    let terms_2 = structure_terms(a2, n2);
                    for (a1, a2) in terms_1.zip(terms_2) {
                        self.unify(a1, a2)?;
                    }

                    Ok(())
                } else {
                    Err(UnificationFailure)
                }
            }
            ((_, Value::Constant(c1)), (_, Value::Constant(c2))) => {
                if c1 == c2 {
                    Ok(())
                } else {
                    Err(UnificationFailure)
                }
            }
            _ => Err(UnificationFailure),
        }
    }

    fn backtrack(&mut self) -> Result<(), ExecutionFailure> {
        log_debug!("BACKTRACK!");
        Err(ExecutionFailure::Failed)
    }
}
