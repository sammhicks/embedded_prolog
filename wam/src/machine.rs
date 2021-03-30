use crate::log_trace;

#[derive(Clone, Copy, Debug)]
pub struct RegisterIndex(u8);

impl RegisterIndex {
    fn to_index(self) -> usize {
        self.0 as usize
    }

    fn offset(self, offset: u8) -> Self {
        Self(self.0 + offset)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Arity(u8);

impl Arity {
    fn to_index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Constant(pub u16);

impl Constant {
    fn from_be_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_be_bytes(bytes))
    }

    fn to_be_bytes(self) -> [u8; 2] {
        self.0.to_be_bytes()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Address(pub u16);

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
    fn parse(memory: &[u32], address: Address) -> Result<(Address, Self), Error> {
        Ok(match memory[address.to_index()].to_be_bytes() {
            [b'R', 0x00, r1, r0] => (
                address.offset(1),
                Value::Reference(Address::from_be_bytes([r1, r0])),
            ),
            [b'C', 0x00, c1, c0] => (
                address.offset(1),
                Value::Constant(Constant::from_be_bytes([c1, c0])),
            ),
            _ => {
                return Err(Error::BadValue(
                    address,
                    core::slice::from_ref(&memory[address.to_index()]),
                ))
            }
        })
    }

    fn write(self, memory: &mut [u32], address: Address) -> Address {
        match self {
            Value::Reference(r) => {
                let [r1, r0] = r.to_be_bytes();
                memory[address.to_index()] = u32::from_be_bytes([b'R', 0x00, r1, r0]);
                address.offset(1)
            }
            Value::Constant(c) => {
                let [c1, c0] = c.to_be_bytes();
                memory[address.to_index()] = u32::from_be_bytes([b'R', 0x00, c1, c0]);
                address.offset(1)
            }
        }
    }
}

#[derive(Debug)]
enum Instruction {
    PutConstant { ai: RegisterIndex, c: Constant },
    PutVoid { ai: RegisterIndex, n: Arity },
    GetConstant { ai: RegisterIndex, c: Constant },
    Call { p: Address, n: Arity },
    Proceed,
}

impl Instruction {
    fn parse(memory: &[u32], pc: Address) -> Result<(Address, Self), Error> {
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
    registers: [Value; 32],
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
            registers: [Value::Reference(Address(u16::MAX)); 32],
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
                CurrentlyExecuting::Query => Instruction::parse(self.query, self.pc).unwrap(),
                CurrentlyExecuting::Program => Instruction::parse(self.program, self.pc).unwrap(),
            };

            self.pc = new_pc;

            log_trace!("Instruction: {:?}", instruction);

            match instruction {
                Instruction::PutConstant { ai, c } => {
                    self.registers[ai.to_index()] = Value::Constant(c)
                }
                Instruction::PutVoid { mut ai, n } => {
                    for _ in 0..n.to_index() {
                        let new_value = Value::Reference(self.h);
                        self.h = new_value.write(&mut self.memory, self.h);
                        self.registers[ai.to_index()] = new_value;
                        ai = ai.offset(1);
                    }
                }
                Instruction::GetConstant { ai, c } => match self.registers[ai.to_index()] {
                    // TODO - Handle Reference
                    Value::Constant(rc) if rc == c => (),
                    _ => return Err(ExecutionFailure::Failed),
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

    pub fn lookup_memory(&self, address: Address) -> Value {
        Value::parse(self.memory, address).unwrap().1
    }

    pub fn solution_registers(&self) -> &[Value] {
        &self.registers[..self.argument_count.to_index()]
    }
}
