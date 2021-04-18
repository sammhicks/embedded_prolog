use core::fmt;

use crate::serializable::{Serializable, SerializableWrapper};

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

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Functor(pub u16);

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
pub struct Arity(pub u8);

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
    pub const ZERO: Self = Self(0);
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

impl SerializableWrapper for Constant {
    type Inner = u16;

    fn from_inner(inner: Self::Inner) -> Self {
        Self(inner)
    }

    fn into_inner(self) -> Self::Inner {
        self.0
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct ProgramCounter(pub u16);

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

impl NoneRepresents for ProgramCounter {
    const NONE_REPRESENTS: &'static str = "End of Program";
}

impl ProgramCounter {
    pub const END_OF_PROGRAM: u16 = u16::MAX;

    pub fn offset(self, offset: u16) -> Self {
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

impl Serializable for Option<ProgramCounter> {
    type Bytes = <ProgramCounter as Serializable>::Bytes;

    fn from_be_bytes(bytes: Self::Bytes) -> Self {
        let pc = ProgramCounter::from_be_bytes(bytes);
        if pc.0 == ProgramCounter::END_OF_PROGRAM {
            None
        } else {
            Some(pc)
        }
    }

    fn into_be_bytes(self) -> Self::Bytes {
        self.unwrap_or(ProgramCounter(ProgramCounter::END_OF_PROGRAM))
            .into_be_bytes()
    }
}
