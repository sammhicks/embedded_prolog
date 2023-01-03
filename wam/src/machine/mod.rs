use core::fmt;

use crate::{log_debug, log_trace, log_warn, CommaSeparated};

mod basic_types;
mod heap;
mod system_call;

use basic_types::{
    Ai, Arity, Constant, Functor, LongInteger, ProgramCounter, RegisterIndex, ShortInteger, Xn, Yn,
};
pub use basic_types::{IntegerSign, OptionDisplay};
use heap::{
    structure_iteration::{ReadWriteMode, State as StructureIterationState},
    Heap, IntegerLeBytes,
};
pub use heap::{Address, MaybeFree, Solution, TermsList, Value, ValueHead};
pub use system_call::{system_call_handler, system_calls, SystemCallEncoder, SystemCalls};

fn assert_fused<I: core::iter::FusedIterator>(i: I) -> I {
    i
}

struct ProgramCounterOutOfRange {
    pc: ProgramCounter,
    offset: u16,
    memory_size: usize,
}

#[derive(Clone, Copy)]
pub struct Instructions<'m> {
    memory: &'m [[u8; 4]],
}

impl<'m> Instructions<'m> {
    pub fn new(memory: &'m [[u8; 4]]) -> Self {
        Self { memory }
    }

    fn get(
        &self,
        pc: ProgramCounter,
        offset: u16,
    ) -> Result<([u8; 4], &'m [[u8; 4]]), ProgramCounterOutOfRange> {
        let first_word = pc.into_usize();
        let last_word = pc.offset(offset).into_usize();
        self.memory
            .get(last_word)
            .copied()
            .zip(self.memory.get(first_word..=last_word))
            .ok_or(ProgramCounterOutOfRange {
                pc,
                offset,
                memory_size: self.memory.len(),
            })
    }

    fn get_tail(
        &self,
        pc: ProgramCounter,
        length: u8,
    ) -> Result<&'m [[u8; 4]], ProgramCounterOutOfRange> {
        let start = pc.offset(1).into_usize();
        let end = pc.offset(1).offset(u16::from(length)).into_usize();
        self.memory.get(start..end).ok_or(ProgramCounterOutOfRange {
            pc,
            offset: u16::from(length + 1),
            memory_size: self.memory.len(),
        })
    }
}

pub struct BadMemoryRange<'m>(pub &'m [[u8; 4]]);

impl fmt::Debug for BadMemoryRange<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:02X}",
            CommaSeparated(self.0.iter().map(CommaSeparated))
        )
    }
}

#[cfg(feature = "defmt-logging")]
impl<'m> defmt::Format for BadMemoryRange<'m> {
    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(fmt, "{:02X}", &self.0)
    }
}

struct UnificationFailure;

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
pub enum Error<'m> {
    ProgramCounterOutOfRange {
        pc: ProgramCounter,
        offset: u16,
        memory_size: usize,
    },
    BadInstruction(ProgramCounter, BadMemoryRange<'m>),
    RegisterBlock(RegisterBlockError),
    Memory(heap::MemoryError),
    PermanentVariable(heap::PermanentVariableError),
    StructureIterationState(heap::structure_iteration::Error),
    ExpressionEvaluation(heap::ExpressionEvaluationError),
    SystemCallIndexOutOfRange(system_call::SystemCallIndexOutOfRange),
    OutOfMemory(heap::OutOfMemory),
}

impl<'m> fmt::Display for Error<'m> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl<'m> From<ProgramCounterOutOfRange> for Error<'m> {
    fn from(
        ProgramCounterOutOfRange {
            pc,
            offset,
            memory_size,
        }: ProgramCounterOutOfRange,
    ) -> Self {
        Self::ProgramCounterOutOfRange {
            pc,
            offset,
            memory_size,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
pub enum Comparison {
    GreaterThan,
    LessThan,
    LessThanOrEqualTo,
    GreaterThanOrEqualTo,
    NotEqualTo,
    EqualTo,
}

impl Comparison {
    fn from_u8(n: u8) -> Option<Self> {
        Some(match n {
            0 => Self::GreaterThan,
            1 => Self::LessThan,
            2 => Self::LessThanOrEqualTo,
            3 => Self::GreaterThanOrEqualTo,
            4 => Self::NotEqualTo,
            5 => Self::EqualTo,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
struct SwitchOnTerm {
    variable: ProgramCounter,
    structure: Option<ProgramCounter>,
    list: Option<ProgramCounter>,
    constant: Option<ProgramCounter>,
    integer: Option<ProgramCounter>,
}

impl SwitchOnTerm {
    fn branch(&self, value: MaybeFree<Value>) -> Result<ProgramCounter, UnificationFailure> {
        match value {
            MaybeFree::FreeVariable => Some(self.variable),
            MaybeFree::BoundTo(Value::Structure(..)) => self.structure,
            MaybeFree::BoundTo(Value::List(..)) => self.list,
            MaybeFree::BoundTo(Value::Constant(..)) => self.constant,
            MaybeFree::BoundTo(Value::Integer { .. }) => self.integer,
        }
        .ok_or(UnificationFailure)
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
struct SwitchOnStructureCase {
    f: Functor,
    n: Arity,
    pc: Option<ProgramCounter>,
}

impl SwitchOnStructureCase {
    fn new([[_, _, p1, p0], [f1, f0, _, n0]]: [[u8; 4]; 2]) -> Self {
        Self {
            f: Functor(u16::from_le_bytes([f0, f1])),
            n: Arity(n0),
            pc: ProgramCounter::from_le_bytes([p0, p1]),
        }
    }

    fn matches(self, f: Functor, n: Arity) -> Option<Option<ProgramCounter>> {
        ((f, n) == (self.f, self.n)).then_some(self.pc)
    }
}

#[derive(Clone, Copy)]
struct SwitchOnStructure<'memory> {
    words: &'memory [[u8; 4]],
}

impl<'memory> SwitchOnStructure<'memory> {
    fn cases(&self) -> impl Iterator<Item = SwitchOnStructureCase> + Clone + 'memory {
        let mut words = assert_fused(self.words.iter());

        core::iter::from_fn(move || {
            let &a = words.next()?;
            let &b = words.next()?;
            Some(SwitchOnStructureCase::new([a, b]))
        })
    }

    fn branch(&self, value: MaybeFree<Value>) -> Result<ProgramCounter, UnificationFailure> {
        let MaybeFree::BoundTo(Value::Structure(f, n, _)) = value else {
            return Err(UnificationFailure)
        };

        self.cases()
            .find_map(|case| case.matches(f, n))
            .flatten()
            .ok_or(UnificationFailure)
    }
}

impl<'memory> fmt::Debug for SwitchOnStructure<'memory> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        CommaSeparated(self.cases()).fmt(f)
    }
}

#[cfg(feature = "defmt-logging")]
impl<'memory> defmt::Format for SwitchOnStructure<'memory> {
    fn format(&self, fmt: defmt::Formatter) {
        CommaSeparated(self.cases()).format(fmt)
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
struct SwitchOnConstantCase {
    c: Constant,
    pc: Option<ProgramCounter>,
}

impl SwitchOnConstantCase {
    fn new([c1, c0, p1, p0]: [u8; 4]) -> Self {
        Self {
            c: Constant::from_le_bytes([c0, c1]),
            pc: ProgramCounter::from_le_bytes([p0, p1]),
        }
    }

    fn matches(self, c: Constant) -> Option<Option<ProgramCounter>> {
        (c == self.c).then_some(self.pc)
    }
}

#[derive(Clone, Copy)]
struct SwitchOnConstant<'memory> {
    words: &'memory [[u8; 4]],
}

impl<'memory> SwitchOnConstant<'memory> {
    fn cases(&self) -> impl Iterator<Item = SwitchOnConstantCase> + Clone + 'memory {
        self.words.iter().copied().map(SwitchOnConstantCase::new)
    }

    fn branch(&self, value: MaybeFree<Value>) -> Result<ProgramCounter, UnificationFailure> {
        let MaybeFree::BoundTo(Value::Constant(c)) = value else {
            return Err(UnificationFailure)
        };

        self.cases()
            .find_map(|case| case.matches(c))
            .flatten()
            .ok_or(UnificationFailure)
    }
}

impl<'memory> fmt::Debug for SwitchOnConstant<'memory> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        CommaSeparated(self.cases()).fmt(f)
    }
}

#[cfg(feature = "defmt-logging")]
impl<'memory> defmt::Format for SwitchOnConstant<'memory> {
    fn format(&self, fmt: defmt::Formatter) {
        CommaSeparated(self.cases()).format(fmt)
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
struct SwitchOnIntegerCase {
    i: ShortInteger,
    pc: Option<ProgramCounter>,
}

impl SwitchOnIntegerCase {
    fn new([i1, i0, p1, p0]: [u8; 4]) -> Self {
        Self {
            i: ShortInteger::new(i16::from_le_bytes([i0, i1])),
            pc: ProgramCounter::from_le_bytes([p0, p1]),
        }
    }

    fn matches(
        self,
        sign: IntegerSign,
        le_bytes: IntegerLeBytes,
    ) -> Option<Option<ProgramCounter>> {
        (self.i.as_long().equals(sign, le_bytes)).then_some(self.pc)
    }
}

#[derive(Clone, Copy)]
struct SwitchOnInteger<'memory> {
    words: &'memory [[u8; 4]],
}

impl<'memory> SwitchOnInteger<'memory> {
    fn cases(&self) -> impl Iterator<Item = SwitchOnIntegerCase> + Clone + 'memory {
        self.words.iter().copied().map(SwitchOnIntegerCase::new)
    }

    fn branch(&self, value: MaybeFree<Value>) -> Result<ProgramCounter, UnificationFailure> {
        let MaybeFree::BoundTo(Value::Integer { sign, le_bytes }) = value else {
            return Err(UnificationFailure)
        };

        self.cases()
            .find_map(|case| case.matches(sign, le_bytes))
            .flatten()
            .ok_or(UnificationFailure)
    }
}

impl<'memory> fmt::Debug for SwitchOnInteger<'memory> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        CommaSeparated(self.cases()).fmt(f)
    }
}

#[cfg(feature = "defmt-logging")]
impl<'memory> defmt::Format for SwitchOnInteger<'memory> {
    fn format(&self, fmt: defmt::Formatter) {
        CommaSeparated(self.cases()).format(fmt)
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
enum Instruction<'memory> {
    PutVariableXn { ai: Ai, xn: Xn },
    PutVariableYn { ai: Ai, yn: Yn },
    PutValueXn { ai: Ai, xn: Xn },
    PutValueYn { ai: Ai, yn: Yn },
    PutStructure { ai: Ai, f: Functor, n: Arity },
    PutList { ai: Ai },
    PutConstant { ai: Ai, c: Constant },
    PutShortInteger { ai: Ai, i: ShortInteger },
    PutInteger { ai: Ai, i: LongInteger<'memory> },
    PutVoid { ai: Ai, n: Arity },
    GetVariableXn { ai: Ai, xn: Xn },
    GetVariableYn { ai: Ai, yn: Yn },
    GetValueXn { ai: Ai, xn: Xn },
    GetValueYn { ai: Ai, yn: Yn },
    GetStructure { ai: Ai, f: Functor, n: Arity },
    GetList { ai: Ai },
    GetConstant { ai: Ai, c: Constant },
    GetShortInteger { ai: Ai, i: ShortInteger },
    GetInteger { ai: Ai, i: LongInteger<'memory> },
    SetVariableXn { xn: Xn },
    SetVariableYn { yn: Yn },
    SetValueXn { xn: Xn },
    SetValueYn { yn: Yn },
    SetConstant { c: Constant },
    SetShortInteger { i: ShortInteger },
    SetInteger { i: LongInteger<'memory> },
    SetVoid { n: Arity },
    UnifyVariableXn { xn: Xn },
    UnifyVariableYn { yn: Yn },
    UnifyValueXn { xn: Xn },
    UnifyValueYn { yn: Yn },
    UnifyConstant { c: Constant },
    UnifyShortInteger { i: ShortInteger },
    UnifyInteger { i: LongInteger<'memory> },
    UnifyVoid { n: Arity },
    Allocate { n: Arity },
    Trim { n: Arity },
    Deallocate,
    Call { p: ProgramCounter, n: Arity },
    Execute { p: ProgramCounter, n: Arity },
    Proceed,
    True,
    Fail,
    Unify,
    Is,
    Comparison(Comparison),
    SystemCall { i: system_call::SystemCallIndex },
    TryMeElse { p: ProgramCounter },
    RetryMeElse { p: ProgramCounter },
    TrustMe,
    Try { p: ProgramCounter },
    Retry { p: ProgramCounter },
    Trust { p: ProgramCounter },
    SwitchOnTerm(SwitchOnTerm),
    SwitchOnStructure(SwitchOnStructure<'memory>),
    SwitchOnConstant(SwitchOnConstant<'memory>),
    SwitchOnInteger(SwitchOnInteger<'memory>),
    NeckCut,
    GetLevel { yn: Yn },
    Cut { yn: Yn },
}

impl<'memory> Instruction<'memory> {
    fn decode_structure_long_functor(
        instructions: Instructions,
        pc: ProgramCounter,
    ) -> Result<Functor, Error> {
        let (word, range) = instructions.get(pc, 1)?;
        let [0,0,f1, f0] = word else {
            return Err(Error::BadInstruction(pc, BadMemoryRange(range)));
        };

        Ok(Functor(u16::from_le_bytes([f0, f1])))
    }

    fn decode_long_integer_words(
        instructions: Instructions,
        pc: ProgramCounter,
        n: u8,
    ) -> Result<&[[u8; 4]], Error> {
        Ok(instructions.get_tail(pc, n)?)
    }

    fn decode(
        instructions: Instructions<'memory>,
        pc: ProgramCounter,
    ) -> Result<(ProgramCounter, Self), Error> {
        let (word, range) = instructions.get(pc, 0)?;
        log_trace!("{:02X}", CommaSeparated(word));
        Ok(match word {
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
                    f: Functor(u16::from(f)),
                    n: Arity(n),
                },
            ),
            [0x05, ai, 0, n] => (
                pc.offset(2),
                Instruction::PutStructure {
                    ai: Ai { ai },
                    f: Self::decode_structure_long_functor(instructions, pc)?,
                    n: Arity(n),
                },
            ),
            [0x06, ai, 0, 0] => (pc.offset(1), Instruction::PutList { ai: Ai { ai } }),
            [0x07, ai, c1, c0] => (
                pc.offset(1),
                Instruction::PutConstant {
                    ai: Ai { ai },
                    c: Constant::from_le_bytes([c0, c1]),
                },
            ),
            [0x08, ai, i1, i0] => (
                pc.offset(1),
                Instruction::PutShortInteger {
                    ai: Ai { ai },
                    i: ShortInteger::from_le_bytes([i0, i1]),
                },
            ),
            [0x09, ai, s, n] => (
                pc.offset(1 + u16::from(n)),
                Instruction::PutInteger {
                    ai: Ai { ai },
                    i: LongInteger {
                        sign: IntegerSign::from_u8(s),
                        words: Self::decode_long_integer_words(instructions, pc, n)?,
                    },
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
                    f: Functor(u16::from(f)),
                    n: Arity(n),
                },
            ),
            [0x15, ai, 0, n] => (
                pc.offset(2),
                Instruction::GetStructure {
                    ai: Ai { ai },
                    f: Self::decode_structure_long_functor(instructions, pc)?,
                    n: Arity(n),
                },
            ),
            [0x16, ai, 0, 0] => (pc.offset(1), Instruction::GetList { ai: Ai { ai } }),
            [0x17, ai, c1, c0] => (
                pc.offset(1),
                Instruction::GetConstant {
                    ai: Ai { ai },
                    c: Constant::from_le_bytes([c0, c1]),
                },
            ),
            [0x18, ai, i1, i0] => (
                pc.offset(1),
                Instruction::GetShortInteger {
                    ai: Ai { ai },
                    i: ShortInteger::from_le_bytes([i0, i1]),
                },
            ),
            [0x19, ai, s, n] => (
                pc.offset(1 + u16::from(n)),
                Instruction::GetInteger {
                    ai: Ai { ai },
                    i: LongInteger {
                        sign: IntegerSign::from_u8(s),
                        words: Self::decode_long_integer_words(instructions, pc, n)?,
                    },
                },
            ),
            [0x20, 0, 0, xn] => (pc.offset(1), Instruction::SetVariableXn { xn: Xn { xn } }),
            [0x21, 0, 0, yn] => (pc.offset(1), Instruction::SetVariableYn { yn: Yn { yn } }),
            [0x22, 0, 0, xn] => (pc.offset(1), Instruction::SetValueXn { xn: Xn { xn } }),
            [0x23, 0, 0, yn] => (pc.offset(1), Instruction::SetValueYn { yn: Yn { yn } }),
            [0x27, 0, c1, c0] => (
                pc.offset(1),
                Instruction::SetConstant {
                    c: Constant::from_le_bytes([c0, c1]),
                },
            ),
            [0x28, 0, i1, i0] => (
                pc.offset(1),
                Instruction::SetShortInteger {
                    i: ShortInteger::from_le_bytes([i0, i1]),
                },
            ),
            [0x29, 0, s, n] => (
                pc.offset(1 + u16::from(n)),
                Instruction::SetInteger {
                    i: LongInteger {
                        sign: IntegerSign::from_u8(s),
                        words: Self::decode_long_integer_words(instructions, pc, n)?,
                    },
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
                    c: Constant::from_le_bytes([c0, c1]),
                },
            ),
            [0x38, 0, i1, i0] => (
                pc.offset(1),
                Instruction::UnifyShortInteger {
                    i: ShortInteger::from_le_bytes([i0, i1]),
                },
            ),
            [0x39, 0, s, n] => (
                pc.offset(1 + u16::from(n)),
                Instruction::UnifyInteger {
                    i: LongInteger {
                        sign: IntegerSign::from_u8(s),
                        words: Self::decode_long_integer_words(instructions, pc, n)?,
                    },
                },
            ),
            [0x3a, 0, 0, n] => (pc.offset(1), Instruction::UnifyVoid { n: Arity(n) }),
            [0x40, 0, 0, n] => (pc.offset(1), Instruction::Allocate { n: Arity(n) }),
            [0x41, 0, 0, n] => (pc.offset(1), Instruction::Trim { n: Arity(n) }),
            [0x42, 0, 0, 0] => (pc.offset(1), Instruction::Deallocate),
            [0x43, n, p1, p0] => (
                pc.offset(1),
                Instruction::Call {
                    p: ProgramCounter::from_le_bytes([p0, p1])
                        .ok_or(Error::BadInstruction(pc, BadMemoryRange(range)))?,
                    n: Arity(n),
                },
            ),
            [0x44, n, p1, p0] => (
                pc.offset(1),
                Instruction::Execute {
                    p: ProgramCounter::from_le_bytes([p0, p1])
                        .ok_or(Error::BadInstruction(pc, BadMemoryRange(range)))?,
                    n: Arity(n),
                },
            ),
            [0x45, 0, 0, 0] => (pc.offset(1), Instruction::Proceed),
            [0x46, 0, 0, 0] => (pc.offset(1), Instruction::True),
            [0x47, 0, 0, 0] => (pc.offset(1), Instruction::Fail),
            [0x48, 0, 0, 0] => (pc.offset(1), Instruction::Unify),
            [0x49, 0, 0, 0] => (pc.offset(1), Instruction::Is),
            [0x4a, 0, 0, n] => (
                pc.offset(1),
                Instruction::Comparison(
                    Comparison::from_u8(n)
                        .ok_or(Error::BadInstruction(pc, BadMemoryRange(range)))?,
                ),
            ),
            [0x4b, 0, 0, n] => (
                pc.offset(1),
                Instruction::SystemCall {
                    i: system_call::SystemCallIndex(n),
                },
            ),
            [0x50, 0, p1, p0] => (
                pc.offset(1),
                Instruction::TryMeElse {
                    p: ProgramCounter::from_le_bytes([p0, p1])
                        .ok_or(Error::BadInstruction(pc, BadMemoryRange(range)))?,
                },
            ),
            [0x51, 0, p1, p0] => (
                pc.offset(1),
                Instruction::RetryMeElse {
                    p: ProgramCounter::from_le_bytes([p0, p1])
                        .ok_or(Error::BadInstruction(pc, BadMemoryRange(range)))?,
                },
            ),
            [0x52, 0, 0, 0] => (pc.offset(1), Instruction::TrustMe),
            [0x53, 0, p1, p0] => (
                pc.offset(1),
                Instruction::Try {
                    p: ProgramCounter::from_le_bytes([p0, p1])
                        .ok_or(Error::BadInstruction(pc, BadMemoryRange(range)))?,
                },
            ),
            [0x54, 0, p1, p0] => (
                pc.offset(1),
                Instruction::Retry {
                    p: ProgramCounter::from_le_bytes([p0, p1])
                        .ok_or(Error::BadInstruction(pc, BadMemoryRange(range)))?,
                },
            ),
            [0x55, 0, p1, p0] => (
                pc.offset(1),
                Instruction::Trust {
                    p: ProgramCounter::from_le_bytes([p0, p1])
                        .ok_or(Error::BadInstruction(pc, BadMemoryRange(range)))?,
                },
            ),
            [0x60, 0, v1, v0] => {
                if let &[[s1, s0, l1, l0], [c1, c0, i1, i0]] = instructions.get_tail(pc, 2)? {
                    (
                        pc.offset(3),
                        Instruction::SwitchOnTerm(SwitchOnTerm {
                            variable: ProgramCounter::from_le_bytes([v0, v1])
                                .ok_or(Error::BadInstruction(pc, BadMemoryRange(range)))?,
                            structure: ProgramCounter::from_le_bytes([s0, s1]),
                            list: ProgramCounter::from_le_bytes([l0, l1]),
                            constant: ProgramCounter::from_le_bytes([c0, c1]),
                            integer: ProgramCounter::from_le_bytes([i0, i1]),
                        }),
                    )
                } else {
                    return Err(Error::BadInstruction(pc, BadMemoryRange(range)));
                }
            }
            [0x61, 0, 0, n] => (
                pc.offset(1 + u16::from(n)),
                Instruction::SwitchOnStructure(SwitchOnStructure {
                    words: instructions.get_tail(pc, n)?,
                }),
            ),
            [0x62, 0, 0, n] => (
                pc.offset(1 + u16::from(n)),
                Instruction::SwitchOnConstant(SwitchOnConstant {
                    words: instructions.get_tail(pc, n)?,
                }),
            ),
            [0x63, 0, 0, n] => (
                pc.offset(1 + u16::from(n)),
                Instruction::SwitchOnInteger(SwitchOnInteger {
                    words: instructions.get_tail(pc, n)?,
                }),
            ),
            [0x70, 0, 0, 0] => (pc.offset(1), Instruction::NeckCut),
            [0x71, 0, 0, yn] => (pc.offset(1), Instruction::GetLevel { yn: Yn { yn } }),
            [0x72, 0, 0, yn] => (pc.offset(1), Instruction::Cut { yn: Yn { yn } }),
            _ => return Err(Error::BadInstruction(pc, BadMemoryRange(range))),
        })
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
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
pub struct RegisterBlock([Option<Address>; 32]);

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
        self.0[usize::from(n.0)..].fill(None);
    }

    fn all(&self) -> &[Option<Address>] {
        &self.0
    }

    fn all_mut(&mut self) -> &mut [Option<Address>] {
        &mut self.0
    }

    fn get(
        &self,
        core::ops::RangeTo { end: Arity(n) }: core::ops::RangeTo<Arity>,
    ) -> &[Option<Address>] {
        self.0.get(..usize::from(n)).unwrap_or(&[])
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
enum CurrentlyExecuting {
    Query,
    Program,
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
pub enum ExecutionFailure<'m> {
    Failed,
    Error(Error<'m>),
}

impl<'m> From<heap::OutOfMemory> for ExecutionFailure<'m> {
    fn from(inner: heap::OutOfMemory) -> Self {
        Self::Error(Error::OutOfMemory(inner))
    }
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

impl<'m> From<heap::ExpressionEvaluationError> for ExecutionFailure<'m> {
    fn from(expression_evaluation_error: heap::ExpressionEvaluationError) -> Self {
        Self::Error(Error::ExpressionEvaluation(expression_evaluation_error))
    }
}

impl<'m> From<heap::structure_iteration::Error> for ExecutionFailure<'m> {
    fn from(structure_iteration_error: heap::structure_iteration::Error) -> Self {
        Self::Error(Error::StructureIterationState(structure_iteration_error))
    }
}

pub struct Machine<'m, Calls: system_call::SystemCalls> {
    currently_executing: CurrentlyExecuting,
    structure_iteration_state: StructureIterationState,
    pc: Option<ProgramCounter>,
    cp: Option<ProgramCounter>,
    argument_count: Arity,
    registers: RegisterBlock,
    program: Instructions<'m>,
    query: Instructions<'m>,
    memory: Heap<'m>,
    system_calls: &'m mut Calls,
}

impl<'m, Calls: system_call::SystemCalls> Machine<'m, Calls> {
    pub fn new(
        program: Instructions<'m>,
        query: Instructions<'m>,
        memory: &'m mut [u32],
        system_calls: &'m mut Calls,
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
            system_calls,
        }
    }

    pub fn system_calls(&self) -> &Calls {
        self.system_calls
    }

    fn continue_execution(&mut self) -> Result<(), ExecutionFailure> {
        loop {
            let Some(pc) = self.pc else {
                self.do_full_garbage_collection()?;

                return Ok(());
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

            let saved_trail_top = self.memory.save_trail_top();
            let saved_structure_iteration_state = self.structure_iteration_state.clone();

            match self.execute_instruction(new_pc, &instruction) {
                Ok(()) => (),
                Err(ExecutionFailure::Error(Error::OutOfMemory(oom))) => {
                    log_warn!("Out of Memory while executing {:?}: {:?}", instruction, oom);

                    self.memory.restore_saved_trail(saved_trail_top)?;
                    self.structure_iteration_state = saved_structure_iteration_state;

                    self.do_full_garbage_collection()?;

                    self.execute_instruction(new_pc, &instruction)?;
                }
                Err(err) => return Err(err),
            }

            self.run_garbage_collection()?;

            log_trace!("");
        }
    }

    fn do_full_garbage_collection(&mut self) -> Result<(), heap::MemoryError> {
        while let heap::GarbageCollectionIsRunning::Running = self.run_garbage_collection()? {}

        self.memory.resume_garbage_collection();

        while let heap::GarbageCollectionIsRunning::Running = self.run_garbage_collection()? {}

        Ok(())
    }

    fn execute_instruction(
        &mut self,
        pc: ProgramCounter,
        instruction: &Instruction,
    ) -> Result<(), ExecutionFailure<'m>> {
        match *instruction {
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
            Instruction::PutShortInteger { ai, i } => {
                self.execute_instruction(pc, &Instruction::PutInteger { ai, i: i.as_long() })
            }
            Instruction::PutInteger { ai, i } => {
                let address = self.new_integer(i)?;
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
            Instruction::GetValueXn { ai, xn } => {
                self.unify(self.registers.load(xn)?, self.registers.load(ai)?)
            }
            Instruction::GetValueYn { ai, yn } => self.unify(
                self.memory.load_permanent_variable(yn)?,
                self.registers.load(ai)?,
            ),
            Instruction::GetStructure { ai, f, n } => {
                let (address, value) = self.get_register_value(ai)?;
                match value.head() {
                    MaybeFree::FreeVariable => {
                        log_trace!("Writing structure {}/{}", f, n);

                        let value_address = self.new_structure(f, n)?;

                        Ok(self
                            .memory
                            .bind_variable_to_value(address, value_address)??)
                    }
                    MaybeFree::BoundTo(value) => {
                        if let ValueHead::Structure(found_f, found_n) = value {
                            if f == found_f && n == found_n {
                                log_trace!("Reading structure {}/{}", f, n);

                                Ok(self.structure_iteration_state.start_reading(address)?)
                            } else {
                                self.backtrack()
                            }
                        } else {
                            self.backtrack()
                        }
                    }
                }
            }
            Instruction::GetList { ai } => {
                let (address, value) = self.get_register_value(ai)?;
                match value.head() {
                    MaybeFree::FreeVariable => {
                        log_trace!("Writing list");

                        let value_address = self.new_list()?;

                        Ok(self
                            .memory
                            .bind_variable_to_value(address, value_address)??)
                    }

                    MaybeFree::BoundTo(value) => {
                        if let ValueHead::List = value {
                            log_trace!("Reading list");
                            Ok(self.structure_iteration_state.start_reading(address)?)
                        } else {
                            self.backtrack()
                        }
                    }
                }
            }
            Instruction::GetConstant { ai, c } => {
                let (address, value) = self.get_register_value(ai)?;

                match value.head() {
                    MaybeFree::FreeVariable => {
                        let value_address = self.new_constant(c)?;
                        self.memory
                            .bind_variable_to_value(address, value_address)??;
                        Ok(())
                    }
                    MaybeFree::BoundTo(value) => {
                        if let ValueHead::Constant(c1) = value {
                            if c == c1 {
                                Ok(())
                            } else {
                                self.backtrack()
                            }
                        } else {
                            self.backtrack()
                        }
                    }
                }
            }
            Instruction::GetShortInteger { ai, i } => {
                self.execute_instruction(pc, &Instruction::GetInteger { ai, i: i.as_long() })
            }
            Instruction::GetInteger { ai, i } => ({
                let (address, value) = self.get_register_value(ai)?;
                match value {
                    MaybeFree::FreeVariable => {
                        let value_address = self.new_integer(i)?;
                        self.memory
                            .bind_variable_to_value(address, value_address)??;
                        Ok(())
                    }
                    MaybeFree::BoundTo(value) => {
                        if let Value::Integer { sign, le_bytes, .. } = value {
                            if i.equals(sign, le_bytes) {
                                Ok(())
                            } else {
                                Err(UnificationFailure)
                            }
                        } else {
                            Err(UnificationFailure)
                        }
                    }
                }
            })
            .or_else(|UnificationFailure| self.backtrack()),
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
            Instruction::SetShortInteger { i } => {
                self.execute_instruction(pc, &Instruction::SetInteger { i: i.as_long() })
            }
            Instruction::SetInteger { i } => {
                let address = self.new_integer(i)?;
                self.structure_iteration_state
                    .write_next(&mut self.memory, address)?;
                Ok(())
            }
            Instruction::SetVoid { n } => {
                for _ in 0..n.0 {
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

                        let (address, value) = self.memory.get_value(term_address)?;

                        match value.head() {
                            MaybeFree::FreeVariable => {
                                let value_address = self.new_constant(c)?;
                                self.memory
                                    .bind_variable_to_value(address, value_address)??;
                                Ok(())
                            }
                            MaybeFree::BoundTo(value) => {
                                if let ValueHead::Constant(c1) = value {
                                    if c == c1 {
                                        Ok(())
                                    } else {
                                        self.backtrack()
                                    }
                                } else {
                                    self.backtrack()
                                }
                            }
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
            Instruction::UnifyShortInteger { i } => {
                self.execute_instruction(pc, &Instruction::UnifyInteger { i: i.as_long() })
            }
            Instruction::UnifyInteger { i } => {
                match self.structure_iteration_state.read_write_mode()? {
                    ReadWriteMode::Read => {
                        let term_address =
                            self.structure_iteration_state.read_next(&self.memory)?;

                        ({
                            let (address, value) = self.memory.get_value(term_address)?;

                            match value {
                                MaybeFree::FreeVariable => {
                                    let value_address = self.new_integer(i)?;
                                    self.memory
                                        .bind_variable_to_value(address, value_address)??;
                                    Ok(())
                                }
                                MaybeFree::BoundTo(Value::Integer {
                                    sign,
                                    le_bytes: be_bytes,
                                    ..
                                }) => {
                                    if i.equals(sign, be_bytes) {
                                        Ok(())
                                    } else {
                                        Err(UnificationFailure)
                                    }
                                }
                                MaybeFree::BoundTo(
                                    Value::Structure(..) | Value::List(..) | Value::Constant(..),
                                ) => Err(UnificationFailure),
                            }
                        })
                        .or_else(|UnificationFailure| self.backtrack())
                    }
                    ReadWriteMode::Write => {
                        let address = self.new_integer(i)?;
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
            Instruction::Allocate { n } => Ok(self.memory.allocate(n, self.cp, &[])??),
            Instruction::Trim { n } => Ok(self.memory.trim(n)?),
            Instruction::Deallocate => Ok(self.memory.deallocate(&mut self.cp)?),
            Instruction::Call { p, n } => {
                match self.currently_executing {
                    CurrentlyExecuting::Query => {
                        self.save_query_registers(n)?;

                        self.currently_executing = CurrentlyExecuting::Program;
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
            Instruction::True => {
                self.special_functor::<0>()?;
                Ok(())
            }
            Instruction::Fail => self.backtrack(),
            Instruction::Unify => {
                let [a1, a2] = self.special_functor()?;
                self.unify(a1, a2)
            }
            Instruction::Is => {
                let [a1, a2] = self.special_functor()?;
                let a2 = self.memory.evaluate(a2)??;
                self.unify(a1, a2)
            }
            Instruction::Comparison(operation) => {
                let [a1, a2] = self.special_functor()?;
                let comparison = self.memory.compare(a1, a2)??;

                let success = match operation {
                    Comparison::GreaterThan => comparison.is_gt(),
                    Comparison::LessThan => comparison.is_lt(),
                    Comparison::LessThanOrEqualTo => comparison.is_le(),
                    Comparison::GreaterThanOrEqualTo => comparison.is_ge(),
                    Comparison::NotEqualTo => comparison.is_ne(),
                    Comparison::EqualTo => comparison.is_eq(),
                };

                if success {
                    Ok(())
                } else {
                    self.backtrack()
                }
            }
            Instruction::SystemCall { i } => {
                match self.system_calls.execute(
                    system_call::Machine {
                        registers: &mut self.registers,
                        memory: &mut self.memory,
                    },
                    i,
                ) {
                    Ok(()) => Ok(()),
                    Err(system_call::SystemCallError::UnificationFailure) => self.backtrack(),
                    Err(system_call::SystemCallError::RegisterBlockError(inner)) => {
                        Err(inner.into())
                    }
                    Err(system_call::SystemCallError::MemoryError(inner)) => Err(inner.into()),
                    Err(system_call::SystemCallError::StructureIteration(inner)) => {
                        Err(inner.into())
                    }
                    Err(system_call::SystemCallError::OutOfMemory(inner)) => Err(inner.into()),
                    Err(system_call::SystemCallError::SystemCallIndexOutOfRange(inner)) => Err(
                        ExecutionFailure::Error(Error::SystemCallIndexOutOfRange(inner)),
                    ),
                }
            }
            Instruction::TryMeElse { p } => {
                self.structure_iteration_state.verify_not_active()?;
                self.memory.new_choice_point(
                    p,
                    self.cp,
                    self.registers.get(..self.argument_count),
                )??;
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
            Instruction::Try { p } => {
                self.structure_iteration_state.verify_not_active()?;
                self.memory.new_choice_point(
                    pc,
                    self.cp,
                    self.registers.get(..self.argument_count),
                )??;
                self.pc = Some(p);
                Ok(())
            }
            Instruction::Retry { p } => {
                self.structure_iteration_state.verify_not_active()?;
                self.memory
                    .retry_choice_point(self.registers.all_mut(), &mut self.cp, pc)?;
                self.pc = Some(p);
                Ok(())
            }
            Instruction::Trust { p } => {
                self.structure_iteration_state.verify_not_active()?;
                self.memory
                    .remove_choice_point(self.registers.all_mut(), &mut self.cp)?;
                self.pc = Some(p);
                Ok(())
            }
            Instruction::SwitchOnTerm(switch_on_term) => {
                match switch_on_term.branch(self.get_register_value(Ai { ai: 0 })?.1) {
                    Ok(pc) => {
                        self.pc = Some(pc);
                        Ok(())
                    }
                    Err(UnificationFailure) => self.backtrack(),
                }
            }
            Instruction::SwitchOnStructure(switch_on_structure) => {
                match switch_on_structure.branch(self.get_register_value(Ai { ai: 0 })?.1) {
                    Ok(pc) => {
                        self.pc = Some(pc);
                        Ok(())
                    }
                    Err(UnificationFailure) => self.backtrack(),
                }
            }
            Instruction::SwitchOnConstant(switch_on_constant) => {
                match switch_on_constant.branch(self.get_register_value(Ai { ai: 0 })?.1) {
                    Ok(pc) => {
                        self.pc = Some(pc);
                        Ok(())
                    }
                    Err(UnificationFailure) => self.backtrack(),
                }
            }
            Instruction::SwitchOnInteger(switch_on_integer) => {
                match switch_on_integer.branch(self.get_register_value(Ai { ai: 0 })?.1) {
                    Ok(pc) => {
                        self.pc = Some(pc);
                        Ok(())
                    }
                    Err(UnificationFailure) => self.backtrack(),
                }
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
        }
    }

    fn run_garbage_collection(
        &mut self,
    ) -> Result<heap::GarbageCollectionIsRunning, heap::MemoryError> {
        self.memory.run_garbage_collection(self.registers.all())
    }

    fn get_register_value(
        &self,
        index: Ai,
    ) -> Result<(Address, MaybeFree<Value>), ExecutionFailure<'m>> {
        Ok(self.memory.get_value(self.registers.load(index)?)?)
    }

    fn save_query_registers(&mut self, n: Arity) -> Result<(), ExecutionFailure<'m>> {
        if let CurrentlyExecuting::Query = self.currently_executing {
            let saved_registers = self.registers.get(..n);

            for &saved_register in saved_registers {
                log_trace!("Saved Register Value: {}", OptionDisplay(saved_register));
            }

            self.memory.allocate(n, None, saved_registers)??;
        }

        Ok(())
    }

    fn special_functor<const N: usize>(&mut self) -> Result<[Address; N], ExecutionFailure<'m>> {
        self.save_query_registers(Arity(N as u8))?;

        // Safety: We then initialise the entire array, and Addresses are plain old data
        let mut registers = unsafe { core::mem::zeroed::<[Address; N]>() };

        for (ai, register) in (0..).zip(registers.iter_mut()) {
            *register = self.registers.load(Ai { ai })?;
        }

        Ok(registers)
    }

    pub fn solution(&self) -> Result<Solution, heap::MemoryError> {
        let solution_terms = self.memory.solution()?;

        Ok(if self.memory.query_has_multiple_solutions() {
            Solution::MultipleSolutions(solution_terms)
        } else {
            Solution::SingleSolution(solution_terms)
        })
    }

    pub fn lookup_memory(
        &self,
        address: Address,
    ) -> Result<(Address, MaybeFree<Value>), heap::MemoryError> {
        self.memory.get_value(address)
    }

    fn new_variable(&mut self) -> Result<Address, ExecutionFailure<'m>> {
        Ok(self.memory.new_variable()??)
    }

    fn new_structure(&mut self, f: Functor, n: Arity) -> Result<Address, ExecutionFailure<'m>> {
        let address = self.memory.new_structure(f, n)??;
        self.structure_iteration_state.start_writing(address)?;

        Ok(address)
    }

    fn new_list(&mut self) -> Result<Address, ExecutionFailure<'m>> {
        let address = self.memory.new_list()??;
        self.structure_iteration_state.start_writing(address)?;

        Ok(address)
    }

    fn new_constant(&mut self, c: Constant) -> Result<Address, ExecutionFailure<'m>> {
        Ok(self.memory.new_constant(c)??)
    }

    fn new_integer(&mut self, i: LongInteger) -> Result<Address, ExecutionFailure<'m>> {
        Ok(self.memory.new_integer(i)??)
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

                self.unify(register_address, term_address)
            }
            ReadWriteMode::Write => {
                let address = load(self, index)?;
                self.structure_iteration_state
                    .write_next(&mut self.memory, address)?;
                Ok(())
            }
        }
    }

    fn unify(&mut self, a1: Address, a2: Address) -> Result<(), ExecutionFailure<'m>> {
        match self.memory.unify(a1, a2) {
            Ok(()) => Ok(()),
            Err(heap::UnificationError::UnificationFailure) => self.backtrack(),
            Err(heap::UnificationError::OutOfMemory(inner)) => Err(inner.into()),
            Err(heap::UnificationError::MemoryError(inner)) => Err(inner.into()),
            Err(heap::UnificationError::StructureIteration(inner)) => Err(inner.into()),
        }
    }

    fn backtrack(&mut self) -> Result<(), ExecutionFailure<'static>> {
        self.structure_iteration_state.reset();
        self.memory.backtrack(&mut self.pc)
    }

    pub fn next_solution(&mut self) -> Result<(), ExecutionFailure> {
        if let CurrentlyExecuting::Program = self.currently_executing {
            self.backtrack()?;
        }
        self.continue_execution()
    }
}
