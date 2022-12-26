use core::{fmt, num::NonZeroU16};

pub trait NoneRepresents: fmt::Display {
    const NONE_REPRESENTS: &'static str;
}

pub struct OptionDisplay<T: NoneRepresents>(pub Option<T>);

impl<T: NoneRepresents> fmt::Display for OptionDisplay<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0.as_ref() {
            Some(t) => fmt::Display::fmt(t, f),
            None => f.write_str(T::NONE_REPRESENTS),
        }
    }
}

/// A Register Index
#[derive(Clone, Copy, Debug)]
pub struct Xn {
    pub xn: u8,
}

/// An Environment "Register Index", i.e. the nth variable on the stack
#[derive(Clone, Copy, Debug)]
pub struct Yn {
    pub yn: u8,
}

impl fmt::Display for Yn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.yn)
    }
}

/// An Argument Index. Functionally the same as a [Register Index](Xn)
#[derive(Clone, Copy, Debug)]
pub struct Ai {
    pub ai: u8,
}

#[derive(Clone, Copy)]
pub struct RegisterIndex(pub u8);

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

#[derive(Clone, Copy, PartialEq, Eq, minicbor::Encode, minicbor::Decode)]
#[cbor(transparent)]
pub struct Functor(#[n(0)] pub u16);

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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, minicbor::Encode, minicbor::Decode)]
#[cbor(transparent)]
pub struct Arity(#[n(0)] pub u8);

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

impl core::ops::SubAssign for Arity {
    fn sub_assign(&mut self, Arity(n): Self) {
        self.0 -= n;
    }
}

impl Arity {
    pub const ZERO: Self = Self(0);
}

#[derive(Clone, Copy, PartialEq, Eq, minicbor::Encode, minicbor::Decode)]
#[cbor(transparent)]
pub struct Constant(#[n(0)] pub u16);

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
    pub fn from_le_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_le_bytes(bytes))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, minicbor::Encode)]
#[repr(u8)]
#[cbor(index_only)]
pub enum IntegerSign {
    #[n(0)]
    Positive = 0,
    #[n(1)]
    Negative = 1,
}

impl IntegerSign {
    pub fn from_u8(s: u8) -> Option<Self> {
        IntoIterator::into_iter([Self::Positive, Self::Negative])
            .find_map(|sign| ((sign as u8) == s).then_some(sign))
    }
}

#[derive(Clone, Copy)]
pub struct ShortInteger {
    sign: IntegerSign,
    value: [u8; 4],
}

impl ShortInteger {
    pub fn new(i: i16) -> Self {
        Self {
            sign: if i < 0 {
                IntegerSign::Negative
            } else {
                IntegerSign::Positive
            },
            value: u32::from(i.unsigned_abs()).to_le_bytes(),
        }
    }

    pub fn from_le_bytes(bytes: [u8; 2]) -> Self {
        Self::new(i16::from_le_bytes(bytes))
    }

    pub fn into_value(self) -> i16 {
        let sign = match self.sign {
            IntegerSign::Positive => 1,
            IntegerSign::Negative => -1,
        };

        sign * (u32::from_le_bytes(self.value) as i16)
    }

    pub fn as_long(&self) -> LongInteger {
        LongInteger {
            sign: self.sign,
            words: core::slice::from_ref(&self.value),
        }
    }
}

impl fmt::Debug for ShortInteger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ShortInteger({})", self)
    }
}

impl fmt::Display for ShortInteger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04X}", self.into_value())
    }
}

#[derive(Clone, Copy)]
pub struct LongInteger<'memory> {
    pub sign: IntegerSign,
    pub words: &'memory [[u8; 4]],
}

impl<'memory> LongInteger<'memory> {
    pub fn equals(&self, sign: IntegerSign, le_bytes: super::heap::IntegerLeBytes) -> bool {
        (self.sign == sign) && le_bytes.equals(self.words.iter().copied().flatten())
    }
}

struct DisplayLongIntegerWords<'memory>(&'memory [[u8; 4]]);

impl<'memory> fmt::Display for DisplayLongIntegerWords<'memory> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for &byte in self.0.iter().flatten().rev() {
            write!(f, "{:02X}", byte)?;
        }

        Ok(())
    }
}

impl<'memory> fmt::Debug for LongInteger<'memory> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LongInteger({})", self)
    }
}

impl<'memory> fmt::Display for LongInteger<'memory> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let &Self { sign, words } = self;

        let sign = match sign {
            IntegerSign::Positive => "",
            IntegerSign::Negative => "-",
        };

        write!(f, "{}{}", sign, DisplayLongIntegerWords(words))
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct ProgramCounter(NonZeroU16);

impl fmt::Debug for ProgramCounter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ProgramCounter({})", self)
    }
}

impl fmt::Display for ProgramCounter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04X}", self.into_word())
    }
}

impl NoneRepresents for ProgramCounter {
    const NONE_REPRESENTS: &'static str = "End of Program";
}

impl ProgramCounter {
    pub const START: Self = Self(
        // Safety: 1 is NonZero
        unsafe { NonZeroU16::new_unchecked(1) },
    );

    pub fn offset(self, offset: u16) -> Self {
        Self(
            // Safety: Programs of length 2^16 are not supported
            unsafe { self.0.checked_add(offset).unwrap_unchecked() },
        )
    }

    pub fn from_word(word: u16) -> Self {
        Self(
            // Safety: adding 1 makes word NonZero
            unsafe { NonZeroU16::new_unchecked(word + 1) },
        )
    }

    pub fn from_le_bytes(bytes: [u8; 2]) -> Self {
        Self::from_word(u16::from_le_bytes(bytes))
    }

    pub fn into_word(self) -> u16 {
        self.0.get() - 1
    }

    pub fn into_usize(self) -> usize {
        self.into_word().into()
    }
}

impl From<ProgramCounter> for usize {
    fn from(pc: ProgramCounter) -> Self {
        pc.into_usize()
    }
}

impl From<u16> for ProgramCounter {
    fn from(word: u16) -> Self {
        Self::from_word(word)
    }
}
