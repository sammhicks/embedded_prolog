use core::fmt;

use crate::log_trace;

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

    fn iter(self, arity: Arity) -> impl Iterator<Item = Self> {
        (self.0..).take(arity.value()).map(Self)
    }
}

#[derive(Clone, Copy)]
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

impl Arity {
    fn value(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Constant(pub u16);

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

impl Constant {
    fn from_be_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_be_bytes(bytes))
    }

    fn to_be_bytes(self) -> [u8; 2] {
        self.0.to_be_bytes()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Address(pub u16);

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

impl Address {
    fn to_index(self) -> usize {
        self.0 as usize
    }

    fn offset(self, offset: u16) -> Self {
        Self(self.0 + offset)
    }

    fn from_be_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_be_bytes(bytes))
    }

    fn to_be_bytes(self) -> [u8; 2] {
        self.0.to_be_bytes()
    }
}

#[derive(Debug)]
pub enum Error<'m> {
    BadInstruction(Address, &'m [u32]),
    BadValue(Address, &'m [u32]),
}

#[derive(Clone, Copy, Debug)]
pub enum Value {
    Reference(Address),
    Constant(Constant),
}

impl Value {
    fn decode(memory: &[u32], address: Address) -> Result<Self, Error> {
        Ok(match memory[address.to_index()].to_be_bytes() {
            [b'R', 0x00, r1, r0] => Value::Reference(Address::from_be_bytes([r1, r0])),
            [b'C', 0x00, c1, c0] => Value::Constant(Constant::from_be_bytes([c1, c0])),
            _ => {
                return Err(Error::BadValue(
                    address,
                    core::slice::from_ref(&memory[address.to_index()]),
                ))
            }
        })
    }

    fn encode(self) -> u32 {
        match self {
            Value::Reference(r) => {
                let [r1, r0] = r.to_be_bytes();
                u32::from_be_bytes([b'R', 0x00, r1, r0])
            }
            Value::Constant(c) => {
                let [c1, c0] = c.to_be_bytes();
                u32::from_be_bytes([b'C', 0x00, c1, c0])
            }
        }
    }
}

#[derive(Debug)]
enum Instruction {
    PutConstant {
        ai: RegisterIndex,
        c: Constant,
    },
    PutVoid {
        ai: RegisterIndex,
        n: Arity,
    },
    GetVariableXn {
        ai: RegisterIndex,
        xn: RegisterIndex,
    },
    GetValueXn {
        ai: RegisterIndex,
        xn: RegisterIndex,
    },
    GetConstant {
        ai: RegisterIndex,
        c: Constant,
    },
    Call {
        p: Address,
        n: Arity,
    },
    Proceed,
}

impl Instruction {
    fn decode(memory: &[u32], pc: Address) -> Result<(Address, Self), Error> {
        Ok(match memory[pc.to_index()].to_be_bytes() {
            [0x07, ai, c1, c0] => (
                pc.offset(1),
                Instruction::PutConstant {
                    ai: RegisterIndex(ai),
                    c: Constant::from_be_bytes([c1, c0]),
                },
            ),
            [0x0a, ai, 0, n] => (
                pc.offset(1),
                Instruction::PutVoid {
                    ai: RegisterIndex(ai),
                    n: Arity(n),
                },
            ),
            [0x10, ai, 0, xn] => (
                pc.offset(1),
                Instruction::GetVariableXn {
                    ai: RegisterIndex(ai),
                    xn: RegisterIndex(xn),
                },
            ),
            [0x12, ai, 0, xn] => (
                pc.offset(1),
                Instruction::GetValueXn {
                    ai: RegisterIndex(ai),
                    xn: RegisterIndex(xn),
                },
            ),
            [0x17, ai, c1, c0] => (
                pc.offset(1),
                Instruction::GetConstant {
                    ai: RegisterIndex(ai),
                    c: Constant::from_be_bytes([c1, c0]),
                },
            ),
            [0x43, n, p1, p0] => (
                pc.offset(1),
                Instruction::Call {
                    p: Address::from_be_bytes([p1, p0]),
                    n: Arity(n),
                },
            ),
            [0x45, 0, 0, 0] => (pc.offset(1), Instruction::Proceed),
            _ => {
                return Err(Error::BadInstruction(
                    pc,
                    core::slice::from_ref(&memory[pc.to_index()]),
                ))
            }
        })
    }
}

#[derive(Debug)]
enum CurrentlyExecuting {
    Query,
    Program,
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

#[derive(Debug)]
pub struct Machine<'m> {
    currently_executing: CurrentlyExecuting,
    pc: Address,
    h: Address,
    argument_count: Arity,
    registers: [Address; 32],
    program: &'m [u32],
    query: &'m [u32],
    memory: &'m mut [u32],
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
            pc: Address(0),
            h: Address(0),
            argument_count: Arity(0),
            registers: [Address(u16::MAX); 32],
            program,
            query,
            memory,
        };

        let execution_result = machine.continue_execution();
        execution_result.map(|success| (machine, success))
    }

    fn continue_execution(&mut self) -> Result<ExecutionSuccess, ExecutionFailure> {
        loop {
            let (new_pc, instruction) = match &self.currently_executing {
                CurrentlyExecuting::Query => Instruction::decode(self.query, self.pc).unwrap(),
                CurrentlyExecuting::Program => Instruction::decode(self.program, self.pc).unwrap(),
            };

            self.pc = new_pc;

            log_trace!("Instruction: {:?}", instruction);

            match instruction {
                Instruction::PutConstant { ai, c } => {
                    self.registers[ai.to_index()] = self.new_constant(c);
                }
                Instruction::PutVoid { ai, n } => {
                    for ai in ai.iter(n) {
                        self.registers[ai.to_index()] = self.new_reference()
                    }
                }
                Instruction::GetVariableXn { ai, xn } => {
                    self.registers[xn.to_index()] = self.registers[ai.to_index()];
                }
                Instruction::GetValueXn { ai, xn } => {
                    if let Err(UnificationFailure) =
                        self.unify(self.registers[xn.to_index()], self.registers[ai.to_index()])
                    {
                        self.backtrack()?;
                    }
                }
                Instruction::GetConstant { ai, c } => match self.lookup_register(ai).1 {
                    Value::Reference(address) => self.bind_to_constant(address, c),
                    Value::Constant(rc) if rc == c => (),
                    _ => self.backtrack()?,
                },
                Instruction::Call { p, n } => {
                    self.currently_executing = CurrentlyExecuting::Program;
                    self.pc = p;
                    self.argument_count = n;
                }
                Instruction::Proceed => return Ok(ExecutionSuccess::SingleAnswer),
            }
        }
    }

    pub fn lookup_register(&self, index: RegisterIndex) -> (Address, Value) {
        log_trace!("Looking up register {}", index);
        self.lookup_memory(self.registers[index.to_index()])
    }

    pub fn lookup_memory(&self, mut address: Address) -> (Address, Value) {
        log_trace!("Looking up memory at {}", address);
        let value = loop {
            let value = Value::decode(self.memory, address).unwrap();
            address = match value {
                Value::Reference(new_address) => {
                    if new_address == address {
                        break value;
                    } else {
                        new_address
                    }
                }
                Value::Constant(_) => break value,
            };
            log_trace!("Looking up memory at {}", address);
        };

        (address, value)
    }

    pub fn solution_registers(&self) -> &[Address] {
        &self.registers[..self.argument_count.value()]
    }

    fn new_value(&mut self, factory: impl FnOnce(Address) -> Value) -> Address {
        let address = self.h;
        let new_value = factory(address);
        log_trace!("New value at {}: {:?}", self.h, new_value);
        self.memory[address.to_index()] = new_value.encode();
        self.h = self.h.offset(1);
        address
    }

    fn new_reference(&mut self) -> Address {
        self.new_value(Value::Reference)
    }

    fn new_constant(&mut self, c: Constant) -> Address {
        self.new_value(|_| Value::Constant(c))
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

        self.memory[variable_address.to_index()] = Value::Reference(value_address).encode();
    }

    fn bind_to_constant(&mut self, address: Address, c: Constant) {
        match Value::decode(&self.memory, address).unwrap() {
            Value::Reference(r) => assert_eq!(r, address),
            value => panic!("Value at {} not a reference but a {:?}", address, value),
        };
        self.memory[address.to_index()] = Value::Constant(c).encode();
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
            ((_, Value::Constant(c1)), (_, Value::Constant(c2))) => {
                if c1 == c2 {
                    Ok(())
                } else {
                    Err(UnificationFailure)
                }
            }
        }
    }

    fn backtrack(&mut self) -> Result<(), ExecutionFailure> {
        Err(ExecutionFailure::Failed)
    }
}
