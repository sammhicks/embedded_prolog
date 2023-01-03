use std::{fmt, rc::Rc};

use num_bigint::BigInt;

use comms::{CommsFromInto, HexNewType};

use super::ast::Name;

#[derive(Debug, Clone, Copy)]
pub struct Ai {
    pub ai: u8,
}

impl fmt::Display for Ai {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "a({})", self.ai)
    }
}

impl From<Xn> for Ai {
    fn from(Xn { xn }: Xn) -> Self {
        Ai { ai: xn }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Xn {
    pub xn: u8,
}

impl fmt::Display for Xn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "x({})", self.xn)
    }
}

impl From<Ai> for Xn {
    fn from(Ai { ai }: Ai) -> Self {
        Xn { xn: ai }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Yn {
    pub yn: u8,
}

impl fmt::Display for Yn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "y({})", self.yn)
    }
}

pub type Word = u32;
pub type ShortInteger = i16;
pub type IntegerWord = u32;
pub type Arity = u8;
pub type ProgramCounter = u16;
pub type SystemCallIndex = u8;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, HexNewType, CommsFromInto)]
#[repr(transparent)]
pub struct Functor(pub u16);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, HexNewType, CommsFromInto)]
#[repr(transparent)]
pub struct Constant(pub u16);

impl std::borrow::Borrow<Functor> for Constant {
    fn borrow(&self) -> &Functor {
        //Safety: both Functor and Constant are transparent
        unsafe { std::mem::transmute(self) }
    }
}

#[derive(Debug)]
#[repr(i8)]
pub enum IntegerSign {
    Negative = -1,
    Zero = 0,
    Positive = 1,
}

impl fmt::Display for IntegerSign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Negative => "-",
            Self::Zero | Self::Positive => "",
        }
        .fmt(f)
    }
}

impl From<num_bigint::Sign> for IntegerSign {
    fn from(sign: num_bigint::Sign) -> Self {
        match sign {
            num_bigint::Sign::Minus => Self::Negative,
            num_bigint::Sign::NoSign => Self::Zero,
            num_bigint::Sign::Plus => Self::Positive,
        }
    }
}

#[derive(Debug)]
pub struct LongInteger {
    pub sign: IntegerSign,
    pub le_words: Vec<IntegerWord>,
}

impl fmt::Display for LongInteger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}",
            self.sign,
            crate::DisplayCommaSeparated(self.le_words.iter())
        )
    }
}

impl<'a> From<&'a BigInt> for LongInteger {
    fn from(int: &'a BigInt) -> Self {
        let (sign, bytes) = int.to_bytes_le();

        let le_words = bytes
            .chunks(std::mem::size_of::<IntegerWord>())
            .map(|bytes| {
                let mut buffer = [0; std::mem::size_of::<IntegerWord>()];

                buffer[0..bytes.len()].copy_from_slice(bytes);

                IntegerWord::from_le_bytes(buffer)
            })
            .collect();

        Self {
            sign: sign.into(),
            le_words,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Comparison {
    GreaterThan = 0,
    LessThan = 1,
    LessThanOrEqualTo = 2,
    GreaterThanOrEqualTo = 3,
    NotEqualTo = 4,
    EqualTo = 5,
}

impl fmt::Display for Comparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Comparison::GreaterThan => r">",
            Comparison::LessThan => r"<",
            Comparison::LessThanOrEqualTo => r"=<",
            Comparison::GreaterThanOrEqualTo => r">=",
            Comparison::NotEqualTo => r"=\=",
            Comparison::EqualTo => r"=:=",
        }
        .fmt(f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum LabelType {
    Named,
    TrySubsequenceStartingAt(usize),
    TryNthClause(usize),
    NthClause(usize),
    SwitchOnStructure(usize, usize),
    SwitchOnStructureValue(usize, usize, (Functor, Arity)),
    SwitchOnList(usize, usize),
    SwitchOnListValue(usize, usize),
    SwitchOnConstant(usize, usize),
    SwitchOnConstantValue(usize, usize, Constant),
    SwitchOnInteger(usize, usize),
    SwitchOnIntegerValue(usize, usize, i16),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct Label<'a> {
    pub name: &'a Name,
    pub arity: Arity,
    pub label_type: LabelType,
}

impl<'a> Label<'a> {
    pub fn named(name: &'a Name, arity: u8) -> Self {
        Self {
            name,
            arity,
            label_type: LabelType::Named,
        }
    }

    pub fn is_named(&self) -> Option<(Name, Arity)> {
        matches!(self.label_type, LabelType::Named).then(|| (self.name.clone(), self.arity))
    }

    pub fn as_owned(&self) -> OwnedLabel {
        let Label {
            name,
            arity,
            label_type,
        } = *self;
        OwnedLabel {
            name: Rc::new(name.clone()),
            arity,
            label_type,
        }
    }
}

impl<'a> fmt::Display for Label<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

pub(super) struct OwnedLabel {
    pub name: Rc<Name>,
    pub arity: Arity,
    pub label_type: LabelType,
}

impl OwnedLabel {
    pub fn named((name, arity): (Name, Arity)) -> Self {
        Self {
            name: Rc::new(name),
            arity,
            label_type: LabelType::Named,
        }
    }
}

impl fmt::Display for OwnedLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let OwnedLabel {
            ref name,
            arity,
            label_type,
        } = *self;
        Label {
            name,
            arity,
            label_type,
        }
        .fmt(f)
    }
}

#[derive(Debug)]
pub(super) enum LabelOr<'a, I> {
    Label(Label<'a>),
    Instruction(I),
}

pub(super) enum InstructionHalf<'a> {
    Literal([u8; 2]),
    LabelValue(Option<Label<'a>>),
}

impl<'a> InstructionHalf<'a> {
    fn resolve_labels<E>(
        self,
        get_label: impl Fn(Label<'a>) -> Result<ProgramCounter, E>,
    ) -> Result<[u8; 2], E> {
        Ok(match self {
            Self::Literal(literal) => literal,
            Self::LabelValue(label) => match label {
                Some(label) => get_label(label)?,
                None => u16::MAX,
            }
            .to_be_bytes(),
        })
    }
}

#[derive(Debug)]
pub(super) struct SwitchOnTerm<'a> {
    pub variable: Label<'a>,
    pub structure: Option<Label<'a>>,
    pub list: Option<Label<'a>>,
    pub constant: Option<Label<'a>>,
    pub integer: Option<Label<'a>>,
}

#[derive(Debug)]
pub(super) enum Instruction<'a> {
    PutVariableXn { xn: Xn, ai: Ai },
    PutVariableYn { yn: Yn, ai: Ai },
    PutValueXn { xn: Xn, ai: Ai },
    PutValueYn { yn: Yn, ai: Ai },
    PutStructure { f: Functor, n: Arity, ai: Ai },
    PutList { ai: Ai },
    PutConstant { c: Constant, ai: Ai },
    PutShortInteger { i: ShortInteger, ai: Ai },
    PutInteger { i: LongInteger, ai: Ai },
    PutVoid { n: Arity, ai: Ai },
    GetVariableXn { xn: Xn, ai: Ai },
    GetVariableYn { yn: Yn, ai: Ai },
    GetValueXn { xn: Xn, ai: Ai },
    GetValueYn { yn: Yn, ai: Ai },
    GetStructure { f: Functor, n: Arity, ai: Ai },
    GetList { ai: Ai },
    GetConstant { c: Constant, ai: Ai },
    GetShortInteger { i: ShortInteger, ai: Ai },
    GetInteger { i: LongInteger, ai: Ai },
    SetVariableXn { xn: Xn },
    SetVariableYn { yn: Yn },
    SetValueXn { xn: Xn },
    SetValueYn { yn: Yn },
    SetConstant { c: Constant },
    SetShortInteger { i: ShortInteger },
    SetInteger { i: LongInteger },
    SetVoid { n: Arity },
    UnifyVariableXn { xn: Xn },
    UnifyVariableYn { yn: Yn },
    UnifyValueXn { xn: Xn },
    UnifyValueYn { yn: Yn },
    UnifyConstant { c: Constant },
    UnifyShortInteger { i: ShortInteger },
    UnifyInteger { i: LongInteger },
    UnifyVoid { n: Arity },
    Allocate { n: Arity },
    Trim { n: Arity },
    Deallocate,
    Call { l: Label<'a> },
    Execute { l: Label<'a> },
    Proceed,
    True,
    Fail,
    Unify,
    Is,
    Comparison(Comparison),
    SystemCall { i: SystemCallIndex },
    TryMeElse { l: Label<'a> },
    RetryMeElse { l: Label<'a> },
    TrustMe,
    Try { l: Label<'a> },
    Retry { l: Label<'a> },
    Trust { l: Label<'a> },
    SwitchOnTerm(SwitchOnTerm<'a>),
    SwitchOnStructure(Vec<((Functor, Arity), Label<'a>)>),
    SwitchOnConstant(Vec<(Constant, Label<'a>)>),
    SwitchOnInteger(Vec<(i16, Label<'a>)>),
    NeckCut,
    GetLevel { yn: Yn },
    Cut { yn: Yn },
}

impl<'a> fmt::Display for Instruction<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::PutVariableXn { xn, ai } => write!(f, "put_variable({xn},{ai})"),
            Instruction::PutVariableYn { yn, ai } => write!(f, "put_variable({yn},{ai})"),
            Instruction::PutValueXn { xn, ai } => write!(f, "put_value({xn},{ai})"),
            Instruction::PutValueYn { yn, ai } => write!(f, "put_value({yn},{ai})"),
            Instruction::PutStructure { f: name, n, ai } => {
                write!(f, "put_structure({name}/{n},{ai})")
            }
            Instruction::PutList { ai } => write!(f, "put_list({ai})"),
            Instruction::PutConstant { c, ai } => write!(f, "put_constant({c},{ai})"),
            Instruction::PutShortInteger { i, ai } => write!(f, "put_short_integer({i},{ai})"),
            Instruction::PutInteger { i, ai } => write!(f, "put_integer({i},{ai})"),
            Instruction::PutVoid { n, ai } => write!(f, "put_void({n},{ai})"),
            Instruction::GetVariableXn { xn, ai } => write!(f, "get_variable({xn},{ai})"),
            Instruction::GetVariableYn { yn, ai } => write!(f, "get_variable({yn},{ai})"),
            Instruction::GetValueXn { xn, ai } => write!(f, "get_value({xn},{ai})"),
            Instruction::GetValueYn { yn, ai } => write!(f, "get_value({yn},{ai})"),
            Instruction::GetStructure { f: name, n, ai } => {
                write!(f, "get_structure({name}/{n},{ai})")
            }
            Instruction::GetList { ai } => write!(f, "get_list({ai})"),
            Instruction::GetConstant { c, ai } => write!(f, "get_constant({c},{ai})"),
            Instruction::GetShortInteger { i, ai } => write!(f, "get_short_integer({i},{ai})"),
            Instruction::GetInteger { i, ai } => write!(f, "get_integer({i},{ai})"),
            Instruction::SetVariableXn { xn } => write!(f, "set_variable({xn})"),
            Instruction::SetVariableYn { yn } => write!(f, "set_variable({yn})"),
            Instruction::SetValueXn { xn } => write!(f, "set_value({xn})"),
            Instruction::SetValueYn { yn } => write!(f, "set_value({yn})"),
            Instruction::SetConstant { c } => write!(f, "set_constant({c})"),
            Instruction::SetShortInteger { i } => write!(f, "set_short_integer({i})"),
            Instruction::SetInteger { i } => write!(f, "set_integer({i})"),
            Instruction::SetVoid { n } => write!(f, "set_void({n})"),
            Instruction::UnifyVariableXn { xn } => write!(f, "unify_variable({xn})"),
            Instruction::UnifyVariableYn { yn } => write!(f, "unify_variable({yn})"),
            Instruction::UnifyValueXn { xn } => write!(f, "unify_value({xn})"),
            Instruction::UnifyValueYn { yn } => write!(f, "unify_value({yn})"),
            Instruction::UnifyConstant { c } => write!(f, "unify_constant({c})"),
            Instruction::UnifyShortInteger { i } => write!(f, "unify_short_integer({i})"),
            Instruction::UnifyInteger { i } => write!(f, "unify_integer({i})"),
            Instruction::UnifyVoid { n } => write!(f, "unify_void({n})"),
            Instruction::Allocate { n } => write!(f, "allocate({n})"),
            Instruction::Trim { n } => write!(f, "trim({n})"),
            Instruction::Deallocate => write!(f, "deallocate"),
            Instruction::Call { l } => write!(f, "call({l})"),
            Instruction::Execute { l } => write!(f, "execute({l})"),
            Instruction::Proceed => write!(f, "proceed"),
            Instruction::True => write!(f, "true"),
            Instruction::Fail => write!(f, "fail"),
            Instruction::Unify => write!(f, "unify"),
            Instruction::Is => write!(f, "is"),
            Instruction::Comparison(comparison) => write!(f, "{comparison}"),
            Instruction::SystemCall { i } => write!(f, "system_call({i})"),
            Instruction::TryMeElse { l } => write!(f, "try_me_else({l})"),
            Instruction::RetryMeElse { l } => write!(f, "retry_me_else({l})"),
            Instruction::TrustMe => write!(f, "trust_me"),
            Instruction::Try { l } => write!(f, "try({l})"),
            Instruction::Retry { l } => write!(f, "retry({l})"),
            Instruction::Trust { l } => write!(f, "trust({l})"),
            Instruction::SwitchOnTerm(switch) => write!(f, "switch_on_term({switch:?})"),
            Instruction::SwitchOnStructure(switch) => write!(f, "switch_on_structure({switch:?})"),
            Instruction::SwitchOnConstant(switch) => write!(f, "switch_on_constant({switch:?})"),
            Instruction::SwitchOnInteger(switch) => write!(f, "switch_on_integer({switch:?})"),
            Instruction::NeckCut => write!(f, "neck_cut"),
            Instruction::GetLevel { yn } => write!(f, "get_level({yn})"),
            Instruction::Cut { yn } => write!(f, "cut({yn})"),
        }
    }
}

trait HalfWord {
    fn encode(self) -> [u8; 2];
}

impl HalfWord for u8 {
    fn encode(self) -> [u8; 2] {
        [0, self]
    }
}

impl HalfWord for u16 {
    fn encode(self) -> [u8; 2] {
        self.to_be_bytes()
    }
}

impl HalfWord for Functor {
    fn encode(self) -> [u8; 2] {
        self.0.encode()
    }
}

impl HalfWord for Constant {
    fn encode(self) -> [u8; 2] {
        self.0.encode()
    }
}

impl HalfWord for i16 {
    fn encode(self) -> [u8; 2] {
        self.to_be_bytes()
    }
}

impl HalfWord for (u8, u8) {
    fn encode(self) -> [u8; 2] {
        [self.0, self.1]
    }
}

impl HalfWord for Xn {
    fn encode(self) -> [u8; 2] {
        [0, self.xn]
    }
}

impl HalfWord for Yn {
    fn encode(self) -> [u8; 2] {
        [0, self.yn]
    }
}

struct Zero;

impl HalfWord for Zero {
    fn encode(self) -> [u8; 2] {
        [0, 0]
    }
}

pub struct InstructionHalfList<'a>(Vec<LabelOr<'a, [InstructionHalf<'a>; 2]>>);

impl<'a> InstructionHalfList<'a> {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    fn with_label(mut self, label: Label<'a>) -> Self {
        self.0.push(LabelOr::Label(label));
        self
    }

    fn with_instruction_half_pair(mut self, pair: [InstructionHalf<'a>; 2]) -> Self {
        self.0.push(LabelOr::Instruction(pair));
        self
    }

    fn with_maybe_label_value_pair(self, [l1, l2]: [Option<Label<'a>>; 2]) -> Self {
        self.with_instruction_half_pair([
            InstructionHalf::LabelValue(l1),
            InstructionHalf::LabelValue(l2),
        ])
    }

    fn with_literal_and_label_value(self, (literal, label): ([u8; 2], Label<'a>)) -> Self {
        self.with_instruction_half_pair([
            InstructionHalf::Literal(literal),
            InstructionHalf::LabelValue(Some(label)),
        ])
    }

    fn with_literal_pair(mut self, [a, b]: [[u8; 2]; 2]) -> Self {
        self.0.push(LabelOr::Instruction([
            InstructionHalf::Literal(a),
            InstructionHalf::Literal(b),
        ]));
        self
    }

    fn with_literal(self, [a, b, c, d]: [u8; 4]) -> Self {
        self.with_literal_pair([[a, b], [c, d]])
    }

    fn with_ai_instruction(self, opcode: u8, Ai { ai }: Ai, w: impl HalfWord) -> Self {
        self.with_literal_pair([[opcode, ai], w.encode()])
    }

    fn with_ai_structure(
        self,
        short_opcode: u8,
        long_opcode: u8,
        Ai { ai }: Ai,
        Functor(f): Functor,
        n: Arity,
    ) -> Self {
        match u8::try_from(f) {
            Ok(f) => self.with_literal([short_opcode, ai, f, n]),
            Err(_) => self
                .with_literal([long_opcode, ai, 0, n])
                .with_literal(Word::from(f).to_be_bytes()),
        }
    }

    fn with_long_integer(
        self,
        opcode: u8,
        ai: Option<Ai>,
        LongInteger {
            sign,
            le_words: words,
        }: LongInteger,
    ) -> Self {
        let len = words.len() as u8;
        words.into_iter().map(Word::to_le_bytes).fold(
            self.with_literal([opcode, ai.map_or(0, |Ai { ai }| ai), sign as u8, len]),
            Self::with_literal,
        )
    }

    fn with_full_word_instruction(self, opcode: u8, b: u8, w: impl HalfWord) -> Self {
        self.with_literal_pair([[opcode, b], w.encode()])
    }

    fn with_half_word_instruction(self, opcode: u8, w: impl HalfWord) -> Self {
        self.with_full_word_instruction(opcode, 0, w)
    }

    fn with_just(self, opcode: u8) -> Self {
        self.with_half_word_instruction(opcode, Zero)
    }

    fn with_instruction(self, instruction: Instruction<'a>) -> Self {
        match instruction {
            Instruction::PutVariableXn { xn, ai } => self.with_ai_instruction(0x00, ai, xn),
            Instruction::PutVariableYn { yn, ai } => self.with_ai_instruction(0x01, ai, yn),
            Instruction::PutValueXn { xn, ai } => self.with_ai_instruction(0x02, ai, xn),
            Instruction::PutValueYn { yn, ai } => self.with_ai_instruction(0x03, ai, yn),
            Instruction::PutStructure { f, n, ai } => self.with_ai_structure(0x04, 0x05, ai, f, n),
            Instruction::PutList { ai } => self.with_ai_instruction(0x06, ai, Zero),
            Instruction::PutConstant { c, ai } => self.with_ai_instruction(0x07, ai, c),
            Instruction::PutShortInteger { i, ai } => self.with_ai_instruction(0x08, ai, i),
            Instruction::PutInteger { i, ai } => self.with_long_integer(0x09, Some(ai), i),
            Instruction::PutVoid { ai, n } => self.with_ai_instruction(0x0a, ai, n),
            Instruction::GetVariableXn { xn, ai } => self.with_ai_instruction(0x10, ai, xn),
            Instruction::GetVariableYn { yn, ai } => self.with_ai_instruction(0x11, ai, yn),
            Instruction::GetValueXn { xn, ai } => self.with_ai_instruction(0x12, ai, xn),
            Instruction::GetValueYn { yn, ai } => self.with_ai_instruction(0x13, ai, yn),
            Instruction::GetStructure { f, n, ai } => self.with_ai_structure(0x14, 0x15, ai, f, n),
            Instruction::GetList { ai } => self.with_ai_instruction(0x16, ai, Zero),
            Instruction::GetConstant { c, ai } => self.with_ai_instruction(0x17, ai, c),
            Instruction::GetShortInteger { i, ai } => self.with_ai_instruction(0x18, ai, i),
            Instruction::GetInteger { i, ai } => self.with_long_integer(0x19, Some(ai), i),
            Instruction::SetVariableXn { xn } => self.with_half_word_instruction(0x20, xn),
            Instruction::SetVariableYn { yn } => self.with_half_word_instruction(0x21, yn),
            Instruction::SetValueXn { xn } => self.with_half_word_instruction(0x22, xn),
            Instruction::SetValueYn { yn } => self.with_half_word_instruction(0x23, yn),
            Instruction::SetConstant { c } => self.with_half_word_instruction(0x27, c),
            Instruction::SetShortInteger { i } => self.with_half_word_instruction(0x28, i),
            Instruction::SetInteger { i } => self.with_long_integer(0x29, None, i),
            Instruction::SetVoid { n } => self.with_half_word_instruction(0x2a, n),
            Instruction::UnifyVariableXn { xn } => self.with_half_word_instruction(0x30, xn),
            Instruction::UnifyVariableYn { yn } => self.with_half_word_instruction(0x31, yn),
            Instruction::UnifyValueXn { xn } => self.with_half_word_instruction(0x32, xn),
            Instruction::UnifyValueYn { yn } => self.with_half_word_instruction(0x33, yn),
            Instruction::UnifyConstant { c } => self.with_half_word_instruction(0x37, c),
            Instruction::UnifyShortInteger { i } => self.with_half_word_instruction(0x38, i),
            Instruction::UnifyInteger { i } => self.with_long_integer(0x39, None, i),
            Instruction::UnifyVoid { n } => self.with_half_word_instruction(0x3a, n),
            Instruction::Allocate { n } => self.with_half_word_instruction(0x40, n),
            Instruction::Trim { n } => self.with_half_word_instruction(0x41, n),
            Instruction::Deallocate => self.with_just(0x42),
            Instruction::Call { l } => self.with_literal_and_label_value(([0x43, l.arity], l)),
            Instruction::Execute { l } => self.with_literal_and_label_value(([0x44, l.arity], l)),
            Instruction::Proceed => self.with_just(0x45),
            Instruction::True => self.with_just(0x46),
            Instruction::Fail => self.with_just(0x47),
            Instruction::Unify => self.with_just(0x48),
            Instruction::Is => self.with_just(0x49),
            Instruction::Comparison(comparison) => {
                self.with_half_word_instruction(0x4a, comparison as u8)
            }
            Instruction::SystemCall { i } => self.with_half_word_instruction(0x4b, i),
            Instruction::TryMeElse { l } => self.with_literal_and_label_value(([0x50, 0], l)),
            Instruction::RetryMeElse { l } => self.with_literal_and_label_value(([0x51, 0], l)),
            Instruction::TrustMe => self.with_just(0x52),
            Instruction::Try { l } => self.with_literal_and_label_value(([0x53, 0], l)),
            Instruction::Retry { l } => self.with_literal_and_label_value(([0x54, 0], l)),
            Instruction::Trust { l } => self.with_literal_and_label_value(([0x55, 0], l)),
            Instruction::SwitchOnTerm(SwitchOnTerm {
                variable,
                structure,
                list,
                constant,
                integer,
            }) => self
                .with_literal_and_label_value(([0x60, 0], variable))
                .with_maybe_label_value_pair([structure, list])
                .with_maybe_label_value_pair([constant, integer]),
            Instruction::SwitchOnStructure(switch) => {
                let len = (2 * switch.len()) as u8;
                switch.into_iter().fold(
                    self.with_literal([0x61, 0, 0, len]),
                    |this, ((f, n), label)| {
                        this.with_literal_and_label_value(([0, 0], label))
                            .with_literal_pair([f.0.to_be_bytes(), [0, n]])
                    },
                )
            }
            Instruction::SwitchOnConstant(switch) => {
                let len = switch.len() as u8;
                switch
                    .into_iter()
                    .map(|(c, l)| (c.0.to_be_bytes(), l))
                    .fold(
                        self.with_literal([0x62, 0, 0, len]),
                        Self::with_literal_and_label_value,
                    )
            }
            Instruction::SwitchOnInteger(switch) => {
                let len = switch.len() as u8;
                switch.into_iter().map(|(i, l)| (i.to_be_bytes(), l)).fold(
                    self.with_literal([0x63, 0, 0, len]),
                    Self::with_literal_and_label_value,
                )
            }
            Instruction::NeckCut => self.with_just(0x70),
            Instruction::GetLevel { yn } => self.with_half_word_instruction(0x71, yn),
            Instruction::Cut { yn } => self.with_half_word_instruction(0x72, yn),
        }
    }

    pub(super) fn with_label_or_instruction(
        self,
        instruction: LabelOr<'a, Instruction<'a>>,
    ) -> Self {
        match instruction {
            LabelOr::Label(label) => self.with_label(label),
            LabelOr::Instruction(instruction) => self.with_instruction(instruction),
        }
    }

    pub(super) fn iter(&self) -> std::slice::Iter<LabelOr<'a, [InstructionHalf; 2]>> {
        self.0.iter()
    }

    pub(super) fn resolve_labels<E>(
        self,
        get_label: impl Copy + Fn(Label) -> Result<ProgramCounter, E>,
    ) -> Result<Vec<[u8; 4]>, E> {
        self.0
            .into_iter()
            .flat_map(|entry| match entry {
                LabelOr::Label(_) => None,
                LabelOr::Instruction(instruction) => Some(instruction),
            })
            .map(|[l, r]| {
                let [a, b] = l.resolve_labels(get_label)?;
                let [c, d] = r.resolve_labels(get_label)?;
                Ok([a, b, c, d])
            })
            .collect()
    }
}
