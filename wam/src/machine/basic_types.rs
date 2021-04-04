use core::fmt;

use crate::serializable::SerializableWrapper;

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

impl ProgramCounter {
    pub const NULL: Self = Self(u16::MAX);

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
