use std::fmt;

use num_bigint::BigInt;

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
pub type Functor = u16;
pub type Constant = u16;
pub type ShortInteger = i16;
pub type IntegerWord = u32;
pub type Arity = u8;
pub type ProgramCounter = u16;
pub type SystemCallIndex = u8;

#[derive(Debug)]
#[repr(u8)]
pub enum IntegerSign {
    Positive = 0,
    Negative = 1,
}

#[derive(Debug)]
pub struct LongInteger {
    pub sign: IntegerSign,
    pub be_words: Vec<IntegerWord>,
}

impl fmt::Display for LongInteger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}",
            match self.sign {
                IntegerSign::Positive => "",
                IntegerSign::Negative => "-",
            },
            crate::DisplayCommaSeparated(self.be_words.iter())
        )
    }
}

impl<'a> From<&'a BigInt> for LongInteger {
    fn from(int: &'a BigInt) -> Self {
        let (sign, bytes) = int.to_bytes_le();
        let sign = match sign {
            num_bigint::Sign::NoSign | num_bigint::Sign::Plus => IntegerSign::Positive,
            num_bigint::Sign::Minus => IntegerSign::Negative,
        };

        let be_words = bytes
            .chunks(std::mem::size_of::<IntegerWord>())
            .rev()
            .map(|bytes| {
                let mut buffer = [0; std::mem::size_of::<IntegerWord>()];

                buffer[0..bytes.len()].copy_from_slice(bytes);

                IntegerWord::from_le_bytes(buffer)
            })
            .collect();

        Self { sign, be_words }
    }
}

#[derive(Debug)]
pub enum Instruction {
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
    Call { p: ProgramCounter, n: Arity },
    Execute { p: ProgramCounter, n: Arity },
    Proceed,
    TryMeElse { p: ProgramCounter },
    RetryMeElse { p: ProgramCounter },
    TrustMe,
    NeckCut,
    GetLevel { yn: Yn },
    Cut { yn: Yn },
    Is,
    True,
    Fail,
    SystemCall { i: SystemCallIndex },
}

impl fmt::Display for Instruction {
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
            Instruction::Call { p, n } => write!(f, "call({p}/{n})"),
            Instruction::Execute { p, n } => write!(f, "execute({p}/{n})"),
            Instruction::Proceed => write!(f, "proceed"),
            Instruction::TryMeElse { p } => write!(f, "try_me_else({p})"),
            Instruction::RetryMeElse { p } => write!(f, "retry_me_else({p})"),
            Instruction::TrustMe => write!(f, "trust_me"),
            Instruction::NeckCut => write!(f, "neck_cut"),
            Instruction::GetLevel { yn } => write!(f, "get_level({yn})"),
            Instruction::Cut { yn } => write!(f, "cut({yn})"),
            Instruction::Is => write!(f, "is"),
            Instruction::True => write!(f, "true"),
            Instruction::Fail => write!(f, "fail"),
            Instruction::SystemCall { i } => write!(f, "system_call({i})"),
        }
    }
}

impl Instruction {
    pub fn serialize(self) -> impl Iterator<Item = [u8; 4]> {
        enum ABC<A, B, C> {
            A(A),
            B(B),
            C(C),
        }

        type WordBytes = [u8; 4];

        impl<
                A: Iterator<Item = WordBytes>,
                B: Iterator<Item = WordBytes>,
                C: Iterator<Item = WordBytes>,
            > Iterator for ABC<A, B, C>
        {
            type Item = WordBytes;

            fn next(&mut self) -> Option<WordBytes> {
                match self {
                    Self::A(a) => a.next(),
                    Self::B(b) => b.next(),
                    Self::C(c) => c.next(),
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

        fn ai_instruction<B, C>(
            opcode: u8,
            Ai { ai }: Ai,
            w: impl HalfWord,
        ) -> ABC<std::iter::Once<WordBytes>, B, C> {
            let [w1, w0] = w.encode();
            ABC::A(std::iter::once([opcode, ai, w1, w0]))
        }

        fn ai_structure<C>(
            short_opcode: u8,
            long_opcode: u8,
            Ai { ai }: Ai,
            f: Functor,
            n: Arity,
        ) -> ABC<std::iter::Once<WordBytes>, std::array::IntoIter<WordBytes, 2>, C> {
            match u8::try_from(f) {
                Ok(f) => ABC::A(std::iter::once([short_opcode, ai, f, n])),
                Err(_) => ABC::B(IntoIterator::into_iter([
                    [long_opcode, ai, 0, n],
                    Word::from(f).to_be_bytes(),
                ])),
            }
        }

        fn long_integer<A, B>(
            opcode: u8,
            ai: Option<Ai>,
            LongInteger {
                sign,
                be_words: words,
            }: LongInteger,
        ) -> ABC<A, B, impl Iterator<Item = WordBytes>> {
            let ai = ai.map_or(0, |Ai { ai }| ai);

            ABC::C(
                std::iter::once([opcode, ai, sign as u8, words.len() as u8])
                    .chain(words.into_iter().map(Word::to_be_bytes)),
            )
        }

        fn half_word_instruction<B, C>(
            opcode: u8,
            w: impl HalfWord,
        ) -> ABC<std::iter::Once<WordBytes>, B, C> {
            let [w1, w0] = w.encode();
            ABC::A(std::iter::once([opcode, 0, w1, w0]))
        }

        fn full_word_instruction<B, C>(
            opcode: u8,
            b: u8,
            w: impl HalfWord,
        ) -> ABC<std::iter::Once<WordBytes>, B, C> {
            let [w1, w0] = w.encode();
            ABC::A(std::iter::once([opcode, b, w1, w0]))
        }

        fn just<B, C>(opcode: u8) -> ABC<std::iter::Once<WordBytes>, B, C> {
            ABC::A(std::iter::once([opcode, 0, 0, 0]))
        }

        match self {
            Instruction::PutVariableXn { xn, ai } => ai_instruction(0x00, ai, xn),
            Instruction::PutVariableYn { yn, ai } => ai_instruction(0x01, ai, yn),
            Instruction::PutValueXn { xn, ai } => ai_instruction(0x02, ai, xn),
            Instruction::PutValueYn { yn, ai } => ai_instruction(0x03, ai, yn),
            Instruction::PutStructure { f, n, ai } => ai_structure(0x04, 0x05, ai, f, n),
            Instruction::PutList { ai } => ai_instruction(0x06, ai, Zero),
            Instruction::PutConstant { c, ai } => ai_instruction(0x07, ai, c),
            Instruction::PutShortInteger { i, ai } => ai_instruction(0x08, ai, i),
            Instruction::PutInteger { i, ai } => long_integer(0x09, Some(ai), i),
            Instruction::PutVoid { ai, n } => ai_instruction(0x0a, ai, n),
            Instruction::GetVariableXn { xn, ai } => ai_instruction(0x10, ai, xn),
            Instruction::GetVariableYn { yn, ai } => ai_instruction(0x11, ai, yn),
            Instruction::GetValueXn { xn, ai } => ai_instruction(0x12, ai, xn),
            Instruction::GetValueYn { yn, ai } => ai_instruction(0x13, ai, yn),
            Instruction::GetStructure { f, n, ai } => ai_structure(0x14, 0x15, ai, f, n),
            Instruction::GetList { ai } => ai_instruction(0x16, ai, Zero),
            Instruction::GetConstant { c, ai } => ai_instruction(0x17, ai, c),
            Instruction::GetShortInteger { i, ai } => ai_instruction(0x18, ai, i),
            Instruction::GetInteger { i, ai } => long_integer(0x19, Some(ai), i),
            Instruction::SetVariableXn { xn } => half_word_instruction(0x20, xn),
            Instruction::SetVariableYn { yn } => half_word_instruction(0x21, yn),
            Instruction::SetValueXn { xn } => half_word_instruction(0x22, xn),
            Instruction::SetValueYn { yn } => half_word_instruction(0x23, yn),
            Instruction::SetConstant { c } => half_word_instruction(0x27, c),
            Instruction::SetShortInteger { i } => half_word_instruction(0x28, i),
            Instruction::SetInteger { i } => long_integer(0x29, None, i),
            Instruction::SetVoid { n } => half_word_instruction(0x2a, n),
            Instruction::UnifyVariableXn { xn } => half_word_instruction(0x30, xn),
            Instruction::UnifyVariableYn { yn } => half_word_instruction(0x31, yn),
            Instruction::UnifyValueXn { xn } => half_word_instruction(0x32, xn),
            Instruction::UnifyValueYn { yn } => half_word_instruction(0x33, yn),
            Instruction::UnifyConstant { c } => half_word_instruction(0x37, c),
            Instruction::UnifyShortInteger { i } => half_word_instruction(0x38, i),
            Instruction::UnifyInteger { i } => long_integer(0x39, None, i),
            Instruction::UnifyVoid { n } => half_word_instruction(0x3a, n),
            Instruction::Allocate { n } => half_word_instruction(0x40, n),
            Instruction::Trim { n } => half_word_instruction(0x41, n),
            Instruction::Deallocate => just(0x42),
            Instruction::Call { p, n } => full_word_instruction(0x43, n, p),
            Instruction::Execute { p, n } => full_word_instruction(0x44, n, p),
            Instruction::Proceed => just(0x45),
            Instruction::TryMeElse { p } => half_word_instruction(0x50, p),
            Instruction::RetryMeElse { p } => half_word_instruction(0x51, p),
            Instruction::TrustMe => just(0x52),
            Instruction::NeckCut => just(0x53),
            Instruction::GetLevel { yn } => half_word_instruction(0x54, yn),
            Instruction::Cut { yn } => half_word_instruction(0x55, yn),
            Instruction::Is => just(0x66),
            Instruction::True => just(0x70),
            Instruction::Fail => just(0x71),
            Instruction::SystemCall { i } => half_word_instruction(0x80, i),
        }
    }
}
