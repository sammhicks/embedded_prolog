use core::fmt;

use crate::{
    log_debug, log_trace,
    serializable::{Serializable, SerializableWrapper},
};

mod basic_types;
mod heap;

pub use basic_types::OptionDisplay;
use basic_types::{Ai, Arity, Constant, Functor, ProgramCounter, RegisterIndex, Xn, Yn};
use heap::{
    structure_iteration::{ReadWriteMode, StructureIterationState},
    Heap,
};
pub use heap::{Address, Value};

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
    RegisterBlock(RegisterBlockError),
    Memory(heap::MemoryError),
    PermanentVariable(heap::PermanentVariableError),
    StructureIterationState(heap::structure_iteration::Error),
}

#[derive(Debug)]
enum Instruction {
    PutVariableXn { ai: Ai, xn: Xn },
    PutVariableYn { ai: Ai, yn: Yn },
    PutValueXn { ai: Ai, xn: Xn },
    PutValueYn { ai: Ai, yn: Yn },
    PutStructure { ai: Ai, f: Functor, n: Arity },
    PutList { ai: Ai },
    PutConstant { ai: Ai, c: Constant },
    PutVoid { ai: Ai, n: Arity },
    GetVariableXn { ai: Ai, xn: Xn },
    GetVariableYn { ai: Ai, yn: Yn },
    GetValueXn { ai: Ai, xn: Xn },
    GetValueYn { ai: Ai, yn: Yn },
    GetStructure { ai: Ai, f: Functor, n: Arity },
    GetList { ai: Ai },
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
    TryMeElse { p: ProgramCounter },
    RetryMeElse { p: ProgramCounter },
    TrustMe,
    NeckCut,
    GetLevel { yn: Yn },
    Cut { yn: Yn },
    True,
    Fail,
}

impl Instruction {
    fn decode_structure_long_functor(memory: &[u32], pc: ProgramCounter) -> Result<Functor, Error> {
        if let [0, 0, f1, f0] = memory[pc.offset(1).into_usize()].to_be_bytes() {
            Ok(Functor(u16::from_be_bytes([f1, f0])))
        } else {
            Err(Error::BadInstruction(
                pc,
                BadMemoryRange(&memory[pc.into_usize()..pc.offset(2).into_usize()]),
            ))
        }
    }

    fn decode(memory: &[u32], pc: ProgramCounter) -> Result<(ProgramCounter, Self), Error> {
        Ok(match memory[pc.into_usize()].to_be_bytes() {
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
            [0x06, ai, 0, 0] => (pc.offset(1), Instruction::PutList { ai: Ai { ai } }),
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
            [0x16, ai, 0, 0] => (pc.offset(1), Instruction::GetList { ai: Ai { ai } }),
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
            [0x50, 0, p1, p0] => (
                pc.offset(1),
                Instruction::TryMeElse {
                    p: ProgramCounter::from_be_bytes([p1, p0]),
                },
            ),
            [0x51, 0, p1, p0] => (
                pc.offset(1),
                Instruction::RetryMeElse {
                    p: ProgramCounter::from_be_bytes([p1, p0]),
                },
            ),
            [0x52, 0, 0, 0] => (pc.offset(1), Instruction::TrustMe),
            [0x53, 0, 0, 0] => (pc.offset(1), Instruction::NeckCut),
            [0x54, 0, 0, yn] => (pc.offset(1), Instruction::GetLevel { yn: Yn { yn } }),
            [0x55, 0, 0, yn] => (pc.offset(1), Instruction::Cut { yn: Yn { yn } }),
            [0x70, 0, 0, 0] => (pc.offset(1), Instruction::True),
            [0x71, 0, 0, 0] => (pc.offset(1), Instruction::Fail),
            _ => {
                return Err(Error::BadInstruction(
                    pc,
                    BadMemoryRange(core::slice::from_ref(&memory[pc.into_usize()])),
                ))
            }
        })
    }
}

#[derive(Debug)]
pub enum RegisterBlockError {
    IndexOutOfRange {
        index: RegisterIndex,
        register_count: usize,
    },
    NoValue {
        index: RegisterIndex,
    },
}

#[derive(Debug)]
struct RegisterBlock([Option<Address>; 32]);

impl RegisterBlock {
    fn new() -> Self {
        Self(Default::default())
    }

    fn load(&self, index: impl Into<RegisterIndex>) -> Result<Address, RegisterBlockError> {
        let index = index.into();
        log_trace!("Loading Register {}", index);
        self.0
            .get(index.0 as usize)
            .ok_or_else(|| RegisterBlockError::IndexOutOfRange {
                index,
                register_count: (self.0[..]).len(),
            })?
            .ok_or(RegisterBlockError::NoValue { index })
    }

    fn store(
        &mut self,
        index: impl Into<RegisterIndex>,
        address: Address,
    ) -> Result<(), RegisterBlockError> {
        let index = index.into();
        log_trace!("Storing {} in Register {}", address, index);
        let register_count = (self.0[..]).len();
        let register =
            self.0
                .get_mut(index.0 as usize)
                .ok_or(RegisterBlockError::IndexOutOfRange {
                    index,
                    register_count,
                })?;
        *register = Some(address);
        Ok(())
    }

    fn clear_above(&mut self, n: Arity) {
        for register in &mut self.0[n.0.into()..] {
            *register = None;
        }
    }

    fn all(&self) -> &[Option<Address>] {
        &self.0
    }

    fn all_mut(&mut self) -> &mut [Option<Address>] {
        &mut self.0
    }
}

impl core::ops::Index<Arity> for RegisterBlock {
    type Output = [Option<Address>];

    fn index(&self, n: Arity) -> &Self::Output {
        &self.0[0..n.0 as usize]
    }
}

#[derive(Debug)]
enum CurrentlyExecuting {
    Query,
    Program,
}

#[derive(Debug)]
pub enum ExecutionFailure<'m> {
    Failed,
    Error(Error<'m>),
}

impl<'m> From<Error<'m>> for ExecutionFailure<'m> {
    fn from(err: Error<'m>) -> Self {
        Self::Error(err)
    }
}

impl<'m> From<RegisterBlockError> for ExecutionFailure<'m> {
    fn from(err: RegisterBlockError) -> Self {
        Self::Error(Error::RegisterBlock(err))
    }
}

impl<'m> From<heap::TupleMemoryError> for ExecutionFailure<'m> {
    fn from(err: heap::TupleMemoryError) -> Self {
        Self::Error(Error::Memory(heap::MemoryError::from(err)))
    }
}

impl<'m> From<heap::MemoryError> for ExecutionFailure<'m> {
    fn from(err: heap::MemoryError) -> Self {
        Self::Error(Error::Memory(err))
    }
}

impl<'m> From<heap::PermanentVariableError> for ExecutionFailure<'m> {
    fn from(err: heap::PermanentVariableError) -> Self {
        Self::Error(Error::PermanentVariable(err))
    }
}

impl<'m> From<heap::CutError> for ExecutionFailure<'m> {
    fn from(cut_error: heap::CutError) -> Self {
        match cut_error {
            heap::CutError::PermanentVariable(inner) => inner.into(),
            heap::CutError::Memory(inner) => inner.into(),
        }
    }
}

impl<'m> From<heap::structure_iteration::Error> for ExecutionFailure<'m> {
    fn from(err: heap::structure_iteration::Error) -> Self {
        Self::Error(Error::StructureIterationState(err))
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ExecutionSuccess {
    SingleAnswer,
    MultipleAnswers,
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
    structure_iteration_state: StructureIterationState,
    pc: Option<ProgramCounter>,
    cp: Option<ProgramCounter>,
    argument_count: Arity,
    registers: RegisterBlock,
    program: &'m [u32],
    query: &'m [u32],
    memory: Heap<'m>,
}

impl<'m> Machine<'m> {
    pub fn new(
        MachineMemory {
            program,
            query,
            memory,
        }: MachineMemory<'m>,
    ) -> Self {
        Self {
            currently_executing: CurrentlyExecuting::Query,
            structure_iteration_state: StructureIterationState::new(),
            pc: Some(ProgramCounter::START),
            cp: None,
            argument_count: Arity::ZERO,
            registers: RegisterBlock::new(),
            program,
            query,
            memory: Heap::init(memory),
        }
    }

    fn continue_execution(&mut self) -> Result<ExecutionSuccess, ExecutionFailure> {
        loop {
            let pc = match self.pc {
                Some(pc) => pc,
                None => {
                    return Ok(if self.memory.query_has_multiple_solutions() {
                        ExecutionSuccess::MultipleAnswers
                    } else {
                        ExecutionSuccess::SingleAnswer
                    });
                }
            };

            let (new_pc, instruction) = match &self.currently_executing {
                CurrentlyExecuting::Query => Instruction::decode(self.query, pc)?,
                CurrentlyExecuting::Program => Instruction::decode(self.program, pc)?,
            };

            log_debug!(
                "{:?} Instruction @{} : {:?}",
                self.currently_executing,
                pc,
                instruction
            );

            self.pc = Some(new_pc);

            self.execute_instruction(instruction)?;

            self.run_garbage_collection()?;

            log_trace!("");
        }
    }

    fn execute_instruction(
        &mut self,
        instruction: Instruction,
    ) -> Result<(), ExecutionFailure<'m>> {
        match instruction {
            Instruction::PutVariableXn { ai, xn } => {
                let address = self.new_variable()?;
                self.registers.store(ai, address)?;
                self.registers.store(xn, address)?;
                Ok(())
            }
            Instruction::PutVariableYn { ai, yn } => {
                let address = self.new_variable()?;
                self.registers.store(ai, address)?;
                self.memory.store_permanent_variable(yn, address)?;
                Ok(())
            }
            Instruction::PutValueXn { ai, xn } => {
                let address = self.registers.load(xn)?;
                self.registers.store(ai, address)?;
                Ok(())
            }
            Instruction::PutValueYn { ai, yn } => {
                let address = self.memory.load_permanent_variable(yn)?;
                self.registers.store(ai, address)?;
                Ok(())
            }
            Instruction::PutStructure { ai, f, n } => {
                let address = self.new_structure(f, n)?;
                self.registers.store(ai, address)?;
                Ok(())
            }
            Instruction::PutList { ai } => {
                let address = self.new_list()?;
                self.registers.store(ai, address)?;
                Ok(())
            }
            Instruction::PutConstant { ai, c } => {
                let address = self.new_constant(c)?;
                self.registers.store(ai, address)?;
                Ok(())
            }
            Instruction::PutVoid { ai, n } => {
                for ai in (ai.ai..).take(n.0 as usize).map(|ai| Ai { ai }) {
                    let address = self.new_variable()?;
                    self.registers.store(ai, address)?;
                }
                Ok(())
            }
            Instruction::GetVariableXn { ai, xn } => {
                self.registers.store(xn, self.registers.load(ai)?)?;
                Ok(())
            }
            Instruction::GetVariableYn { ai, yn } => {
                let address = self.registers.load(ai)?;
                self.memory.store_permanent_variable(yn, address)?;
                Ok(())
            }
            Instruction::GetValueXn { ai, xn } => self
                .unify(self.registers.load(xn)?, self.registers.load(ai)?)?
                .or_else(|UnificationFailure| self.backtrack()),
            Instruction::GetValueYn { ai, yn } => self
                .unify(
                    self.memory.load_permanent_variable(yn)?,
                    self.registers.load(ai)?,
                )?
                .or_else(|UnificationFailure| self.backtrack()),
            Instruction::GetStructure { ai, f, n } => {
                let (address, value) = self.get_register_value(ai)?;
                match value {
                    Value::Reference(variable_address) => {
                        log_trace!("Writing structure {}/{}", f, n);

                        let value_address = self.new_structure(f, n)?;

                        self.memory
                            .bind_variable_to_value(variable_address, value_address)?;
                    }
                    Value::Structure(found_f, found_n) => {
                        if f == found_f && n == found_n {
                            log_trace!("Reading structure {}/{}", f, n);

                            self.structure_iteration_state.start_reading(address)?;
                        } else {
                            self.backtrack()?;
                        }
                    }
                    Value::List | Value::Constant(_) => self.backtrack()?,
                }
                Ok(())
            }
            Instruction::GetList { ai } => {
                let (address, value) = self.get_register_value(ai)?;
                match value {
                    Value::Reference(variable_address) => {
                        log_trace!("Writing list");

                        let value_address = self.new_list()?;

                        self.memory
                            .bind_variable_to_value(variable_address, value_address)?;
                    }
                    Value::List => {
                        log_trace!("Reading list");
                        self.structure_iteration_state.start_reading(address)?;
                    }
                    Value::Structure(..) | Value::Constant(_) => self.backtrack()?,
                }
                Ok(())
            }
            Instruction::GetConstant { ai, c } => match self.get_register_value(ai)?.1 {
                Value::Reference(variable_address) => {
                    let value_address = self.new_constant(c)?;
                    self.memory
                        .bind_variable_to_value(variable_address, value_address)?;
                    Ok(())
                }
                Value::Constant(rc) if rc == c => Ok(()),
                _ => self.backtrack(),
            },
            Instruction::SetVariableXn { xn } => {
                let address = self.new_variable()?;
                self.registers.store(xn, address)?;
                self.structure_iteration_state
                    .write_next(&mut self.memory, address)?;
                Ok(())
            }
            Instruction::SetVariableYn { yn } => {
                let address = self.new_variable()?;
                self.memory.store_permanent_variable(yn, address)?;
                self.structure_iteration_state
                    .write_next(&mut self.memory, address)?;
                Ok(())
            }
            Instruction::SetValueXn { xn } => {
                let address = self.registers.load(xn)?;
                self.structure_iteration_state
                    .write_next(&mut self.memory, address)?;
                Ok(())
            }
            Instruction::SetValueYn { yn } => {
                let address = self.memory.load_permanent_variable(yn)?;
                self.structure_iteration_state
                    .write_next(&mut self.memory, address)?;
                Ok(())
            }
            Instruction::SetConstant { c } => {
                let address = self.new_constant(c)?;
                self.structure_iteration_state
                    .write_next(&mut self.memory, address)?;
                Ok(())
            }
            Instruction::SetVoid { n } => {
                for _ in 0..n.into_inner() {
                    let address = self.new_variable()?;
                    self.structure_iteration_state
                        .write_next(&mut self.memory, address)?;
                }
                Ok(())
            }
            Instruction::UnifyVariableXn { xn } => {
                self.unify_variable(xn, |this, xn, address| this.registers.store(xn, address))
            }
            Instruction::UnifyVariableYn { yn } => self.unify_variable(yn, |this, yn, address| {
                this.memory.store_permanent_variable(yn, address)
            }),
            Instruction::UnifyValueXn { xn } => {
                self.unify_value(xn, |this, xn| this.registers.load(xn))
            }
            Instruction::UnifyValueYn { yn } => {
                self.unify_value(yn, |this, yn| this.memory.load_permanent_variable(yn))
            }
            Instruction::UnifyConstant { c } => {
                match self.structure_iteration_state.read_write_mode()? {
                    ReadWriteMode::Read => {
                        let term_address =
                            self.structure_iteration_state.read_next(&self.memory)?;

                        let value = self.memory.get_value(term_address)?.1;

                        match value {
                            Value::Reference(variable_address) => {
                                let value_address = self.new_constant(c)?;
                                self.memory
                                    .bind_variable_to_value(variable_address, value_address)?;
                                Ok(())
                            }
                            Value::Constant(c1) => {
                                if c == c1 {
                                    Ok(())
                                } else {
                                    self.backtrack()
                                }
                            }
                            Value::Structure(..) | Value::List => self.backtrack(),
                        }
                    }
                    ReadWriteMode::Write => {
                        let address = self.new_constant(c)?;
                        self.structure_iteration_state
                            .write_next(&mut self.memory, address)?;

                        Ok(())
                    }
                }
            }
            Instruction::UnifyVoid { n } => {
                match self.structure_iteration_state.read_write_mode()? {
                    ReadWriteMode::Read => {
                        self.structure_iteration_state.skip(&self.memory, n)?;
                        Ok(())
                    }
                    ReadWriteMode::Write => {
                        for _ in 0..n.0 {
                            let address = self.new_variable()?;
                            self.structure_iteration_state
                                .write_next(&mut self.memory, address)?;
                        }
                        Ok(())
                    }
                }
            }
            Instruction::Allocate { n } => Ok(self.memory.allocate(n, self.cp, &[])?),
            Instruction::Trim { n } => Ok(self.memory.trim(n)?),
            Instruction::Deallocate => Ok(self.memory.deallocate(&mut self.cp)?),
            Instruction::Call { p, n } => {
                match self.currently_executing {
                    CurrentlyExecuting::Query => {
                        self.currently_executing = CurrentlyExecuting::Program;

                        let registers = &self.registers[n];

                        for &value in registers {
                            log_trace!("Saved Register Value: {}", OptionDisplay(value));
                        }

                        self.memory.allocate(n, None, registers)?;

                        self.cp = None;
                    }
                    CurrentlyExecuting::Program => {
                        self.cp = self.pc;
                    }
                }

                self.memory.update_cut_register();
                self.pc = Some(p);
                self.argument_count = n;

                self.registers.clear_above(n);

                Ok(())
            }
            Instruction::Execute { p, n } => {
                self.memory.update_cut_register();
                self.pc = Some(p);
                self.argument_count = n;

                self.registers.clear_above(n);

                Ok(())
            }
            Instruction::Proceed => {
                log_trace!("Proceeding to {}", OptionDisplay(self.cp));
                self.pc = self.cp;
                Ok(())
            }
            Instruction::TryMeElse { p } => {
                self.structure_iteration_state.verify_not_active()?;
                self.memory
                    .new_choice_point(p, self.cp, &self.registers[self.argument_count])?;
                Ok(())
            }
            Instruction::RetryMeElse { p } => {
                self.structure_iteration_state.verify_not_active()?;
                self.memory
                    .retry_choice_point(self.registers.all_mut(), &mut self.cp, p)?;
                Ok(())
            }
            Instruction::TrustMe => {
                self.structure_iteration_state.verify_not_active()?;
                self.memory
                    .remove_choice_point(self.registers.all_mut(), &mut self.cp)?;
                Ok(())
            }
            Instruction::NeckCut => {
                self.memory.neck_cut()?;
                Ok(())
            }
            Instruction::GetLevel { yn } => {
                self.memory.get_level(yn)?;
                Ok(())
            }
            Instruction::Cut { yn } => {
                self.memory.cut(yn)?;
                Ok(())
            }
            Instruction::True => Ok(()),
            Instruction::Fail => self.backtrack(),
        }
    }

    fn run_garbage_collection(&mut self) -> Result<(), heap::MemoryError> {
        while let heap::GarbageCollectionIsRunning::Running =
            self.memory.run_garbage_collection(self.registers.all())?
        {}

        Ok(())
    }

    fn get_register_value(&self, index: Ai) -> Result<(Address, Value), ExecutionFailure<'m>> {
        let (address, value, _) = self.memory.get_value(self.registers.load(index)?)?;
        Ok((address, value))
    }

    pub fn solution_registers(
        &self,
    ) -> Result<impl Iterator<Item = Option<Address>> + '_, heap::MemoryError> {
        self.memory.solution_registers()
    }

    pub fn lookup_memory(
        &self,
        address: Option<Address>,
    ) -> Result<(Address, Value, impl Iterator<Item = Option<Address>> + '_), heap::MemoryError>
    {
        self.memory
            .get_value(address.ok_or(heap::MemoryError::NoRegistryEntryAt {
                address: address.into_inner(),
            })?)
    }

    fn new_variable(&mut self) -> Result<Address, ExecutionFailure<'m>> {
        Ok(self.memory.new_variable()?)
    }

    fn new_structure(&mut self, f: Functor, n: Arity) -> Result<Address, ExecutionFailure<'m>> {
        let address = self.memory.new_structure(f, n)?;
        self.structure_iteration_state.start_writing(address)?;

        Ok(address)
    }

    fn new_list(&mut self) -> Result<Address, ExecutionFailure<'m>> {
        let address = self.memory.new_list()?;
        self.structure_iteration_state.start_writing(address)?;

        Ok(address)
    }

    fn new_constant(&mut self, c: Constant) -> Result<Address, ExecutionFailure<'m>> {
        Ok(self.memory.new_constant(c)?)
    }

    fn unify_variable<I, E>(
        &mut self,
        index: I,
        store: impl FnOnce(&mut Self, I, Address) -> Result<(), E>,
    ) -> Result<(), ExecutionFailure<'m>>
    where
        ExecutionFailure<'m>: From<E>,
    {
        match self.structure_iteration_state.read_write_mode()? {
            ReadWriteMode::Read => {
                let term_address = self.structure_iteration_state.read_next(&self.memory)?;

                store(self, index, term_address)?;
                Ok(())
            }
            ReadWriteMode::Write => {
                let address = self.new_variable()?;
                store(self, index, address)?;
                self.structure_iteration_state
                    .write_next(&mut self.memory, address)?;
                Ok(())
            }
        }
    }

    fn unify_value<I, E>(
        &mut self,
        index: I,
        load: impl FnOnce(&mut Self, I) -> Result<Address, E>,
    ) -> Result<(), ExecutionFailure<'m>>
    where
        ExecutionFailure<'m>: From<E>,
    {
        match self.structure_iteration_state.read_write_mode()? {
            ReadWriteMode::Read => {
                let term_address = self.structure_iteration_state.read_next(&self.memory)?;

                let register_address = load(self, index)?;

                self.unify(register_address, term_address)?
                    .or_else(|UnificationFailure| self.backtrack())
            }
            ReadWriteMode::Write => {
                let address = load(self, index)?;
                self.structure_iteration_state
                    .write_next(&mut self.memory, address)?;
                Ok(())
            }
        }
    }

    fn unify(
        &mut self,
        a1: Address,
        a2: Address,
    ) -> Result<Result<(), UnificationFailure>, ExecutionFailure<'m>> {
        log_trace!("Unifying {} and {}", a1, a2);
        let (a1, v1, _) = self.memory.get_value(a1)?;
        let (a2, v2, _) = self.memory.get_value(a2)?;
        log_trace!("Resolved to {:?} @ {} and {:?} @ {}", v1, a1, v2, a2);
        Ok(match (a1, v1, a2, v2) {
            (a1, Value::Reference(_), a2, Value::Reference(_)) => {
                self.memory.bind_variables(a1, a2)?;
                Ok(())
            }
            (a1, Value::Reference(_), a2, value) => {
                assert!(!matches!(value, Value::Reference(_)));
                self.memory.bind_variable_to_value(a1, a2)?;
                Ok(())
            }
            (a1, value, a2, Value::Reference(_)) => {
                assert!(!matches!(value, Value::Reference(_)));
                self.memory.bind_variable_to_value(a2, a1)?;
                Ok(())
            }
            (a1, Value::Structure(f1, n1), a2, Value::Structure(f2, n2)) => {
                if f1 == f2 && n1 == n2 {
                    let mut terms_1 = StructureIterationState::structure_reader(a1);
                    let mut terms_2 = StructureIterationState::structure_reader(a2);
                    for _ in 0..n1.0 {
                        let a1 = terms_1.read_next(&self.memory)?;
                        let a2 = terms_2.read_next(&self.memory)?;
                        match self.unify(a1, a2) {
                            Ok(Ok(())) => (),
                            Ok(Err(err)) => return Ok(Err(err)),
                            Err(err) => return Err(err),
                        }
                    }

                    Ok(())
                } else {
                    Err(UnificationFailure)
                }
            }
            (a1, Value::List, a2, Value::List) => {
                let mut terms_1 = StructureIterationState::structure_reader(a1);
                let mut terms_2 = StructureIterationState::structure_reader(a2);
                for _ in 0..2 {
                    let a1 = terms_1.read_next(&self.memory)?;
                    let a2 = terms_2.read_next(&self.memory)?;
                    match self.unify(a1, a2) {
                        Ok(Ok(())) => (),
                        Ok(Err(err)) => return Ok(Err(err)),
                        Err(err) => return Err(err),
                    }
                }

                Ok(())
            }
            (_, Value::Constant(c1), _, Value::Constant(c2)) => {
                if c1 == c2 {
                    Ok(())
                } else {
                    Err(UnificationFailure)
                }
            }
            _ => Err(UnificationFailure),
        })
    }

    fn backtrack(&mut self) -> Result<(), ExecutionFailure<'static>> {
        self.structure_iteration_state.reset();
        self.memory.backtrack(&mut self.pc)
    }

    pub fn next_solution(&mut self) -> Result<ExecutionSuccess, ExecutionFailure> {
        if let CurrentlyExecuting::Program = self.currently_executing {
            self.backtrack()?;
        }
        self.continue_execution()
    }
}
