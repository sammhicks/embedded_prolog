use core::{fmt, num::NonZeroU16};

use comms::{CommsFromInto, HexNewType};

use crate::CommaSeparated;

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

#[cfg(feature = "defmt-logging")]
impl<T: defmt::Format + NoneRepresents> defmt::Format for OptionDisplay<T> {
    fn format(&self, fmt: defmt::Formatter) {
        match self.0.as_ref() {
            Some(t) => t.format(fmt),
            None => defmt::write!(fmt, "{}", T::NONE_REPRESENTS),
        }
    }
}

/// A Register Index
#[derive(Clone, Copy, HexNewType)]
#[cfg_attr(feature = "defmt-logging", derive(comms::HexDefmt))]
pub struct Xn {
    pub xn: u8,
}

/// An Environment "Register Index", i.e. the nth variable on the stack
#[derive(Clone, Copy, HexNewType)]
#[cfg_attr(feature = "defmt-logging", derive(comms::HexDefmt))]
pub struct Yn {
    pub yn: u8,
}

/// An Argument Index. Functionally the same as a [Register Index](Xn)
#[derive(Clone, Copy, HexNewType)]
#[cfg_attr(feature = "defmt-logging", derive(comms::HexDefmt))]
pub struct Ai {
    pub ai: u8,
}

#[derive(Clone, Copy, HexNewType)]
#[cfg_attr(feature = "defmt-logging", derive(comms::HexDefmt))]
pub struct RegisterIndex(pub u8);

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

#[derive(Clone, Copy, PartialEq, Eq, HexNewType, CommsFromInto)]
#[cfg_attr(feature = "defmt-logging", derive(comms::HexDefmt))]
pub struct Functor(pub u16);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, HexNewType, CommsFromInto)]
#[cfg_attr(feature = "defmt-logging", derive(comms::HexDefmt))]
pub struct Arity(pub u8);

impl core::ops::SubAssign for Arity {
    fn sub_assign(&mut self, Arity(n): Self) {
        self.0 -= n;
    }
}

impl Arity {
    pub const ZERO: Self = Self(0);
}

#[derive(Clone, Copy, PartialEq, Eq, HexNewType, CommsFromInto)]
#[cfg_attr(feature = "defmt-logging", derive(comms::HexDefmt))]
pub struct Constant(pub u16);

impl Constant {
    pub fn from_le_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_le_bytes(bytes))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
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

impl From<core::cmp::Ordering> for IntegerSign {
    fn from(ordering: core::cmp::Ordering) -> Self {
        match ordering {
            core::cmp::Ordering::Less => Self::Negative,
            core::cmp::Ordering::Equal => Self::Zero,
            core::cmp::Ordering::Greater => Self::Positive,
        }
    }
}

impl core::ops::Mul for IntegerSign {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::Zero, _) | (_, Self::Zero) => Self::Zero,
            (Self::Positive, Self::Positive) | (Self::Negative, Self::Negative) => Self::Positive,
            (Self::Positive, Self::Negative) | (Self::Negative, Self::Positive) => Self::Negative,
        }
    }
}

impl IntegerSign {
    pub fn from_u8(s: u8) -> Option<Self> {
        Self::from_i8(s as i8)
    }

    pub fn from_i8(s: i8) -> Option<Self> {
        Some(match s {
            -1 => Self::Negative,
            0 => Self::Zero,
            1 => Self::Positive,
            _ => return None,
        })
    }

    pub fn reverse(self) -> Self {
        match self {
            Self::Negative => Self::Positive,
            Self::Zero => Self::Zero,
            Self::Positive => Self::Negative,
        }
    }

    pub fn into_comms(self) -> comms::IntegerSign {
        match self {
            IntegerSign::Negative => comms::IntegerSign::Negative,
            IntegerSign::Zero => comms::IntegerSign::Zero,
            IntegerSign::Positive => comms::IntegerSign::Positive,
        }
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
        i16::from(self.sign as i8) * (u32::from_le_bytes(self.value) as i16)
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

#[cfg(feature = "defmt-logging")]
impl defmt::Format for ShortInteger {
    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(fmt, "{:04X}", self.into_value())
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

impl<'memory> fmt::Debug for LongInteger<'memory> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LongInteger({})", self)
    }
}

impl<'memory> fmt::Display for LongInteger<'memory> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let &Self { sign, words } = self;

        write!(
            f,
            "{}{:02X}",
            sign,
            CommaSeparated(words.iter().flatten().rev())
        )
    }
}

#[cfg(feature = "defmt-logging")]
impl<'memory> defmt::Format for LongInteger<'memory> {
    fn format(&self, fmt: defmt::Formatter) {
        let &Self { sign, words } = self;

        defmt::write!(
            fmt,
            "{}{:02X}",
            sign,
            CommaSeparated(words.iter().flatten().rev())
        )
    }
}

#[derive(Clone, Copy, Eq, PartialEq, HexNewType)]
#[cfg_attr(feature = "defmt-logging", derive(comms::HexDefmt))]
pub struct ProgramCounter(NonZeroU16);

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
