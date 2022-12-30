use core::{fmt, iter::FusedIterator, num::NonZeroU16};

use crate::{log_trace, CommaSeparated};

use super::basic_types::{
    self, Arity, Constant, Functor, IntegerSign, LongInteger, NoneRepresents, OptionDisplay,
    ProgramCounter, Yn,
};

mod integer_operations;
pub mod structure_iteration;

use structure_iteration::State as StructureIterationState;

type IntegerWordUsage = u16;

pub enum UnificationError {
    UnificationFailure,
    OutOfMemory(OutOfMemory),
    MemoryError(MemoryError),
    StructureIteration(structure_iteration::Error),
}

impl From<OutOfMemory> for UnificationError {
    fn from(inner: OutOfMemory) -> Self {
        Self::OutOfMemory(inner)
    }
}

impl<E> From<E> for UnificationError
where
    MemoryError: From<E>,
{
    fn from(inner: E) -> Self {
        Self::MemoryError(inner.into())
    }
}

impl From<structure_iteration::Error> for UnificationError {
    fn from(inner: structure_iteration::Error) -> Self {
        Self::StructureIteration(inner)
    }
}

#[cfg(feature = "defmt-logging")]
trait DisplayAndFormat: core::fmt::Display + defmt::Format {}

#[cfg(feature = "defmt-logging")]
impl<T: core::fmt::Display + defmt::Format> DisplayAndFormat for T {}

#[cfg(not(feature = "defmt-logging"))]
trait DisplayAndFormat: core::fmt::Display {}

#[cfg(not(feature = "defmt-logging"))]
impl<T: core::fmt::Display> DisplayAndFormat for T {}

trait NeqAssign {
    fn neq_assign(&mut self, new: Self, message: &str) -> bool;
}

impl<T: DisplayAndFormat + PartialEq> NeqAssign for T {
    fn neq_assign(&mut self, new: Self, message: &str) -> bool {
        if self != &new {
            log_trace!("{}: {} => {}", message, self, new);

            *self = new;
            true
        } else {
            false
        }
    }
}

enum DivOrMod {
    Div,
    Mod,
}

enum SpecialFunctor {
    PrefixOperation {
        operation: fn(
            (
                integer_operations::UnsignedOutput,
                integer_operations::SignedInput,
            ),
        ) -> (IntegerSign, IntegerWordUsage),
    },
    InfixOperation {
        calculate_words_count: fn(TupleAddress, TupleAddress) -> TupleAddress,
        operation: fn(
            (
                integer_operations::UnsignedOutput,
                integer_operations::SignedInput,
                integer_operations::SignedInput,
            ),
        ) -> (IntegerSign, IntegerWordUsage),
    },
    DivMod(DivOrMod),
    MinMax {
        select_first_if: fn(
            (
                integer_operations::SignedInput,
                integer_operations::SignedInput,
            ),
        ) -> bool,
    },
}

impl TryFrom<StructureValue> for SpecialFunctor {
    type Error = ExpressionEvaluationError;

    fn try_from(StructureValue(f, n): StructureValue) -> Result<Self, Self::Error> {
        Ok(match (f, n) {
            // Prefix of '+'
            (Functor(0), Arity(1)) => Self::PrefixOperation {
                operation: integer_operations::copy_signed,
            },
            // Addition
            (Functor(0), Arity(2)) => Self::InfixOperation {
                calculate_words_count: |a, b| TupleAddress::max(a, b) + 1,
                operation: integer_operations::add_signed,
            },
            // Negation
            (Functor(1), Arity(1)) => Self::PrefixOperation {
                operation: integer_operations::neg_signed,
            },
            // Subtraction
            (Functor(1), Arity(2)) => Self::InfixOperation {
                calculate_words_count: |a, b| TupleAddress::max(a, b) + 1,
                operation: integer_operations::sub_signed,
            },
            // Multiplication
            (Functor(2), Arity(2)) => Self::InfixOperation {
                calculate_words_count: |a, b| a + b + 1,
                operation: integer_operations::mul_signed,
            },
            // Integer Division
            (Functor(3 | 4), Arity(2)) => Self::DivMod(DivOrMod::Div),
            // Integer Modulo
            (Functor(5), Arity(2)) => Self::DivMod(DivOrMod::Mod),
            // Min
            (Functor(6), Arity(2)) => Self::MinMax {
                select_first_if: |(a, b)| a < b,
            },
            // Max
            (Functor(7), Arity(2)) => Self::MinMax {
                select_first_if: |(a, b)| a > b,
            },
            _ => return Err(ExpressionEvaluationError::BadStructure(f, n)),
        })
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
pub enum BadValueType {
    Expected {
        expected: ValueType,
        actual: ValueType,
    },
    ExpectedOneOf {
        expected: &'static [ValueType],
        actual: ValueType,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
pub enum ValueType {
    Reference,
    Structure,
    List,
    Constant,
    Integer,
    Environment,
    ChoicePoint,
    TrailVariable,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Address(NonZeroU16);

impl Address {
    pub fn new(n: NonZeroU16) -> Self {
        Self(n)
    }

    pub fn into_inner(self) -> NonZeroU16 {
        self.0
    }

    fn from_word(word: TupleWord) -> Option<Self> {
        NonZeroU16::new(word).map(Self)
    }

    fn into_word(self) -> u16 {
        self.0.get()
    }

    fn into_view(self) -> AddressView {
        AddressView(self.into_word())
    }
}

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

#[cfg(feature = "defmt-logging")]
impl defmt::Format for Address {
    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(fmt, "{:04X}", self.0)
    }
}

impl NoneRepresents for Address {
    const NONE_REPRESENTS: &'static str = "NULL";
}

pub struct AddressView(u16);

impl fmt::Debug for AddressView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04X}", self.0)
    }
}

#[cfg(feature = "defmt-logging")]
impl defmt::Format for AddressView {
    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(fmt, "{:04X}", self.0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct TupleAddress(u16);

impl fmt::Debug for TupleAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TupleAddress({})", self)
    }
}

impl fmt::Display for TupleAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04X}", self.0)
    }
}

#[cfg(feature = "defmt-logging")]
impl defmt::Format for TupleAddress {
    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(fmt, "{:04X}", self.0)
    }
}

impl core::ops::Add<u16> for TupleAddress {
    type Output = Self;

    fn add(self, rhs: u16) -> Self::Output {
        Self(self.0 + rhs)
    }
}

impl core::ops::Add<TupleAddress> for TupleAddress {
    type Output = Self;

    fn add(self, rhs: TupleAddress) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl core::ops::Add<Arity> for TupleAddress {
    type Output = Self;

    fn add(self, rhs: Arity) -> Self::Output {
        Self(self.0 + u16::from(rhs.0))
    }
}

impl core::ops::Add<Yn> for TupleAddress {
    type Output = Self;

    fn add(self, rhs: Yn) -> Self::Output {
        Self(self.0 + u16::from(rhs.yn))
    }
}

impl core::ops::Sub<TupleAddress> for TupleAddress {
    type Output = Self;

    fn sub(self, rhs: TupleAddress) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl TupleAddress {
    fn iter(self, terms_count: Arity) -> impl Iterator<Item = Self> + Clone {
        (self.0..).take(terms_count.0 as usize).map(Self)
    }

    fn into_view(self) -> TupleAddressView {
        TupleAddressView(self)
    }

    fn max(Self(lhs): Self, Self(rhs): Self) -> Self {
        Self(lhs.max(rhs))
    }
}

pub struct TupleAddressView(TupleAddress);

impl fmt::Debug for TupleAddressView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl fmt::Display for TupleAddressView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(feature = "defmt-logging")]
impl defmt::Format for TupleAddressView {
    fn format(&self, fmt: defmt::Formatter) {
        self.0.format(fmt)
    }
}

#[derive(Debug)]
enum MarkedState {
    NotMarked,
    IsMarked { next: Option<Address> },
}

impl MarkedState {
    fn clear(&mut self) -> Self {
        core::mem::replace(self, Self::NotMarked)
    }
}

#[derive(Debug)]
struct RegistryEntry {
    value_type: ValueType,
    tuple_address: TupleAddress,
    marked_state: MarkedState,
    is_unified_with: Option<Address>,
}

impl RegistryEntry {
    fn verify_is(&self, expected: ValueType) -> Result<&Self, BadValueType> {
        if self.value_type == expected {
            Ok(self)
        } else {
            Err(BadValueType::Expected {
                expected,
                actual: self.value_type.clone(),
            })
        }
    }
}

#[derive(Debug)]
struct NoRegistryEntryAt {
    address: Address,
}

struct BadRegistryValueType {
    address: Address,
    bad_value_type: BadValueType,
}

struct RegistryEntryRef<'m> {
    address: Address,
    entry: &'m RegistryEntry,
}

impl<'m> RegistryEntryRef<'m> {
    fn verify_is(self, expected: ValueType) -> Result<Self, BadRegistryValueType> {
        let address = self.address;
        self.entry
            .verify_is(expected)
            .map(|_| self)
            .map_err(|bad_value_type| BadRegistryValueType {
                address,
                bad_value_type,
            })
    }
}

impl<'m> core::ops::Deref for RegistryEntryRef<'m> {
    type Target = RegistryEntry;
    fn deref(&self) -> &Self::Target {
        self.entry
    }
}

struct RegistrySlotMut<'m> {
    address: Address,
    slot: &'m mut Option<RegistryEntry>,
}

impl<'m> RegistrySlotMut<'m> {
    fn entry(&mut self) -> Result<&mut RegistryEntry, NoRegistryEntryAt> {
        self.slot.as_mut().ok_or(NoRegistryEntryAt {
            address: self.address,
        })
    }

    fn clear(self) {
        *self.slot = None;
    }
}

struct Registry<'m>(&'m mut [Option<RegistryEntry>]);

impl<'m> Registry<'m> {
    fn new_registry_entry(&mut self) -> Option<(Address, &mut Option<RegistryEntry>)> {
        let (index, slot) = (1..)
            .zip(self.0.iter_mut())
            .find(|(_, entry)| entry.is_none())?;

        // Safety: index starts at 1
        let address = Address(unsafe { NonZeroU16::new_unchecked(index) });

        Some((address, slot))
    }

    fn add_item_to_be_scanned(
        &mut self,
        address: Address,
        next_list_head: &mut Option<Address>,
    ) -> Result<(), NoRegistryEntryAt> {
        let registry_entry = self.get_mut(address)?;

        if let MarkedState::IsMarked { .. } = registry_entry.marked_state {
            return Ok(());
        }

        log_trace!("Marking {}, next is {:?}", address, next_list_head);

        registry_entry.marked_state = MarkedState::IsMarked {
            next: next_list_head.replace(address),
        };

        Ok(())
    }

    fn add_maybe_item_to_be_scanned(
        &mut self,
        address: Option<Address>,
        next_list_head: &mut Option<Address>,
    ) -> Result<(), NoRegistryEntryAt> {
        if let Some(address) = address {
            self.add_item_to_be_scanned(address, next_list_head)?;
        }

        Ok(())
    }

    fn index_of_address(address: Address) -> usize {
        usize::from(NonZeroU16::get(address.0) - 1)
    }

    fn slot_mut(&mut self, address: Address) -> Result<RegistrySlotMut, NoRegistryEntryAt> {
        self.0
            .get_mut(Self::index_of_address(address))
            .map(|slot| RegistrySlotMut { address, slot })
            .ok_or(NoRegistryEntryAt { address })
    }

    fn get(&self, address: Address) -> Result<RegistryEntryRef, NoRegistryEntryAt> {
        self.0
            .get(Self::index_of_address(address))
            .and_then(Option::as_ref)
            .ok_or(NoRegistryEntryAt { address })
            .map(|entry| RegistryEntryRef { address, entry })
    }

    fn get_mut(&mut self, address: Address) -> Result<&mut RegistryEntry, NoRegistryEntryAt> {
        self.0
            .get_mut(Self::index_of_address(address))
            .and_then(Option::as_mut)
            .ok_or(NoRegistryEntryAt { address })
    }

    fn get_unified_with(&mut self, address: Address) -> Result<Address, NoRegistryEntryAt> {
        match self.get(address)?.is_unified_with {
            None => Ok(address),
            Some(unified_with) if unified_with == address => Ok(address),
            Some(unified_with) => {
                let unified_with = self.get_unified_with(unified_with)?;
                self.get_mut(address)?.is_unified_with = Some(unified_with);

                Ok(unified_with)
            }
        }
    }

    fn is_already_unified_with(
        &mut self,
        a1: Address,
        a2: Address,
    ) -> Result<bool, NoRegistryEntryAt> {
        let d1 = self.get_unified_with(a1)?;
        let d2 = self.get_unified_with(a2)?;

        Ok(match d1.into_word().cmp(&d2.into_word()) {
            core::cmp::Ordering::Less => {
                self.get_mut(a2)?.is_unified_with = Some(d1);
                self.get_mut(d2)?.is_unified_with = Some(d1);
                false
            }
            core::cmp::Ordering::Equal => true,
            core::cmp::Ordering::Greater => {
                self.get_mut(a1)?.is_unified_with = Some(d2);
                self.get_mut(d1)?.is_unified_with = Some(d2);
                false
            }
        })
    }

    fn clear_unification(&mut self) {
        for entry in self.0.iter_mut().filter_map(Option::as_mut) {
            entry.is_unified_with = None;
        }
    }
}

type TupleWord = u16;

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
pub enum TupleEntryError {
    MustNotBeZero,
}

trait TupleEntry: Sized {
    fn decode(word: TupleWord) -> Result<Self, TupleEntryError>;
    fn encode(self) -> TupleWord;
}

impl TupleEntry for Address {
    fn decode(word: TupleWord) -> Result<Self, TupleEntryError> {
        Address::from_word(word).ok_or(TupleEntryError::MustNotBeZero)
    }

    fn encode(self) -> TupleWord {
        self.into_word()
    }
}

impl TupleEntry for Option<Address> {
    fn decode(word: TupleWord) -> Result<Self, TupleEntryError> {
        Ok(Address::from_word(word))
    }

    fn encode(self) -> TupleWord {
        self.map_or(0, Address::into_word)
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
pub enum TupleMemoryError {
    AddressOutOfRange {
        address: TupleAddressView,
        size: TupleAddressView,
    },
    BadEntry {
        address: TupleAddressView,
        inner: TupleEntryError,
    },
}

impl TupleMemoryError {
    fn bad_entry(address: TupleAddress) -> impl FnOnce(TupleEntryError) -> Self {
        move |inner| Self::BadEntry {
            address: address.into_view(),
            inner,
        }
    }
}

trait TupleEntries: Sized {
    fn decode(
        tuple_memory: &TupleMemory,
        address: TupleAddress,
    ) -> Result<(Self, TupleAddress), TupleMemoryError>;

    fn encode(
        self,
        tuple_memory: &mut TupleMemory,
        address: TupleAddress,
    ) -> Result<TupleAddress, TupleMemoryError>;
}

struct Then<A: TupleEntry, B: TupleEntries>(A, B);

impl<A: TupleEntry, B: TupleEntries> TupleEntries for Then<A, B> {
    fn decode(
        tuple_memory: &TupleMemory,
        address: TupleAddress,
    ) -> Result<(Self, TupleAddress), TupleMemoryError> {
        let a =
            A::decode(tuple_memory.load(address)?).map_err(TupleMemoryError::bad_entry(address))?;

        let (b, address) = B::decode(tuple_memory, address + 1)?;
        Ok((Then(a, b), address))
    }

    fn encode(
        self,
        tuple_memory: &mut TupleMemory,
        address: TupleAddress,
    ) -> Result<TupleAddress, TupleMemoryError> {
        let Self(a, b) = self;
        *tuple_memory.get_mut(address)? = a.encode();
        b.encode(tuple_memory, address + 1)
    }
}

#[repr(transparent)]
struct DirectAccess<T: Tuple> {
    tuple: T,
}

impl<T: Tuple> TupleEntries for DirectAccess<T> {
    fn decode(
        tuple_memory: &TupleMemory,
        address: TupleAddress,
    ) -> Result<(Self, TupleAddress), TupleMemoryError> {
        // Safety: Tuples are all safe to load
        unsafe { tuple_memory.load_block(address) }
    }

    fn encode(
        self,
        tuple_memory: &mut TupleMemory,
        address: TupleAddress,
    ) -> Result<TupleAddress, TupleMemoryError> {
        // Safety: Tuples are all safe to store
        unsafe { tuple_memory.store_block(address, self) }
    }
}

trait TupleMemoryIndex {
    type Output: ?Sized;

    fn end(&self) -> TupleAddress;

    fn get(self, slice: &[TupleWord]) -> Option<&Self::Output>;
    fn get_mut(self, slice: &mut [TupleWord]) -> Option<&mut Self::Output>;
}

impl TupleMemoryIndex for TupleAddress {
    type Output = TupleWord;

    fn end(&self) -> TupleAddress {
        *self
    }

    fn get(self, slice: &[TupleWord]) -> Option<&Self::Output> {
        slice.get(usize::from(self.0))
    }

    fn get_mut(self, slice: &mut [TupleWord]) -> Option<&mut Self::Output> {
        slice.get_mut(usize::from(self.0))
    }
}

fn tuple_address_range_to_usize_range(
    range: core::ops::Range<TupleAddress>,
) -> core::ops::Range<usize> {
    usize::from(range.start.0)..usize::from(range.end.0)
}

impl TupleMemoryIndex for core::ops::Range<TupleAddress> {
    type Output = [TupleWord];

    fn end(&self) -> TupleAddress {
        self.end - TupleAddress(1)
    }

    fn get(self, slice: &[TupleWord]) -> Option<&Self::Output> {
        slice.get(tuple_address_range_to_usize_range(self))
    }

    fn get_mut(self, slice: &mut [TupleWord]) -> Option<&mut Self::Output> {
        slice.get_mut(tuple_address_range_to_usize_range(self))
    }
}

#[derive(Clone, Copy)]
pub struct TermsList<'a>(&'a [TupleWord]);

impl<'a> fmt::Debug for TermsList<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}]", CommaSeparated(self.into_iter()))
    }
}

#[cfg(feature = "defmt-logging")]
impl<'a> defmt::Format for TermsList<'a> {
    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(fmt, "[{}]", CommaSeparated(self.into_iter()))
    }
}

impl<'a, C> minicbor::Encode<C> for TermsList<'a> {
    fn encode<W: minicbor::encode::Write>(
        &self,
        e: &mut minicbor::Encoder<W>,
        _ctx: &mut C,
    ) -> Result<(), minicbor::encode::Error<W::Error>> {
        e.array(self.0.len() as u64)?;

        self.into_iter()
            .try_for_each(|address| e.encode(address.map(Address::into_inner))?.ok())
    }
}

impl<'a> TermsList<'a> {
    pub fn into_iter(
        self,
    ) -> impl Clone + ExactSizeIterator + FusedIterator + Iterator<Item = Option<Address>> + 'a
    {
        self.0.iter().copied().map(Address::from_word)
    }

    pub fn take<const N: usize>(self) -> [Option<Address>; N] {
        let mut terms = [None; N];
        for (slot, term) in terms.iter_mut().zip(self.into_iter()) {
            *slot = term;
        }
        terms
    }
}

#[derive(Debug)]
pub enum Solution<'a> {
    SingleSolution(TermsList<'a>),
    MultipleSolutions(TermsList<'a>),
}

impl<'a> Solution<'a> {
    pub fn into_comms(self) -> comms::Solution<TermsList<'a>> {
        match self {
            Self::SingleSolution(registers) => comms::Solution::SingleSolution(registers),
            Self::MultipleSolutions(registers) => comms::Solution::MultipleSolutions(registers),
        }
    }
}

struct TupleMemory<'m>(&'m mut [TupleWord]);

impl<'m> TupleMemory<'m> {
    fn get<I: TupleMemoryIndex>(&self, index: I) -> Result<&I::Output, TupleMemoryError> {
        let end = index.end();
        let size = self.0.len();
        index
            .get(self.0)
            .ok_or(TupleMemoryError::AddressOutOfRange {
                address: TupleAddressView(end),
                size: TupleAddressView(TupleAddress(size as u16)),
            })
    }

    fn get_mut<I: TupleMemoryIndex>(
        &mut self,
        index: I,
    ) -> Result<&mut I::Output, TupleMemoryError> {
        let end = index.end();
        let size = self.0.len();
        index
            .get_mut(self.0)
            .ok_or(TupleMemoryError::AddressOutOfRange {
                address: TupleAddressView(end),
                size: TupleAddressView(TupleAddress(size as u16)),
            })
    }

    fn load(&self, address: TupleAddress) -> Result<TupleWord, TupleMemoryError> {
        self.get(address).copied()
    }

    fn store(&mut self, address: TupleAddress, value: TupleWord) -> Result<(), TupleMemoryError> {
        self.get_mut(address).map(|slot| *slot = value)
    }

    const fn block_size<T>() -> TupleAddress {
        TupleAddress((core::mem::size_of::<T>() / core::mem::size_of::<TupleWord>()) as u16)
    }

    fn block_range<T>(start: TupleAddress) -> core::ops::Range<TupleAddress> {
        core::ops::Range {
            start,
            end: start + Self::block_size::<T>(),
        }
    }

    unsafe fn load_block<T>(
        &self,
        address: TupleAddress,
    ) -> Result<(T, TupleAddress), TupleMemoryError> {
        let entry = self
            .get(Self::block_range::<T>(address))?
            .as_ptr()
            .cast::<T>();

        Ok((
            core::ptr::read_unaligned(entry),
            address + Self::block_size::<T>(),
        ))
    }

    unsafe fn store_block<T>(
        &mut self,
        address: TupleAddress,
        block: T,
    ) -> Result<TupleAddress, TupleMemoryError> {
        let entry = self
            .get_mut(Self::block_range::<T>(address))?
            .as_mut_ptr()
            .cast::<T>();

        core::ptr::write_unaligned(entry, block);

        Ok(address + Self::block_size::<T>())
    }

    fn load_terms(
        &self,
        terms: Option<core::ops::Range<TupleAddress>>,
    ) -> Result<TermsList, TupleMemoryError> {
        Ok(TermsList(match terms {
            Some(terms) => self.get(terms)?,
            None => &[],
        }))
    }

    fn copy_within(&mut self, source: core::ops::Range<TupleAddress>, destination: TupleAddress) {
        let core::ops::Range {
            start: TupleAddress(start),
            end: TupleAddress(end),
        } = source;
        let TupleAddress(destination) = destination;

        self.0.copy_within(
            usize::from(start)..usize::from(end),
            usize::from(destination),
        );
    }

    // The caller must ensure that the data doesn't overlap with any other data
    unsafe fn get_integer_output(
        &mut self,
        output: IntegerEvaluationOutputData,
    ) -> integer_operations::UnsignedOutput {
        integer_operations::UnsignedOutput::new(self.0.as_mut_ptr(), output)
    }

    // The caller must ensure that the data doesn't overlap with any other data
    unsafe fn get_integer_input(
        &self,
        (s1, w1): (IntegerSign, IntegerWordsSlice),
    ) -> integer_operations::SignedInput {
        integer_operations::SignedInput::new(
            s1,
            self.get(w1.into_address_range()).unwrap_unchecked(),
        )
    }

    // The caller must ensure that all ranges are valid
    unsafe fn get_integer_input_input(
        &self,
        (s1, w1): (IntegerSign, IntegerWordsSlice),
        (s2, w2): (IntegerSign, IntegerWordsSlice),
    ) -> (
        integer_operations::SignedInput,
        integer_operations::SignedInput,
    ) {
        (
            integer_operations::SignedInput::new(
                s1,
                self.get(w1.into_address_range()).unwrap_unchecked(),
            ),
            integer_operations::SignedInput::new(
                s2,
                self.get(w2.into_address_range()).unwrap_unchecked(),
            ),
        )
    }

    // The caller must ensure that w0 does not overlap with either w1 or w2, and that all ranges are valid
    unsafe fn get_integer_output_input(
        &mut self,
        w0: IntegerEvaluationOutputData,
        w1: (IntegerSign, IntegerWordsSlice),
    ) -> (
        integer_operations::UnsignedOutput,
        integer_operations::SignedInput,
    ) {
        let w0 = integer_operations::UnsignedOutput::new(self.0.as_mut_ptr(), w0);
        let w1 = self.get_integer_input(w1);
        (w0, w1)
    }

    // The caller must ensure that w0 does not overlap with either w1 or w2, and that all ranges are valid
    unsafe fn get_integer_output_input_input(
        &mut self,
        w0: IntegerEvaluationOutputData,
        w1: (IntegerSign, IntegerWordsSlice),
        w2: (IntegerSign, IntegerWordsSlice),
    ) -> (
        integer_operations::UnsignedOutput,
        integer_operations::SignedInput,
        integer_operations::SignedInput,
    ) {
        let w0 = integer_operations::UnsignedOutput::new(self.0.as_mut_ptr(), w0);
        let (w1, w2) = self.get_integer_input_input(w1, w2);
        (w0, w1, w2)
    }

    // The caller must ensure that o0 and o1 don't overlap does not overlap with either each other or i1 or i2, and that all ranges are valid
    unsafe fn get_integer_output_output_input_input(
        &mut self,
        o0: IntegerEvaluationOutputData,
        o1: IntegerEvaluationOutputData,
        i0: (IntegerSign, IntegerWordsSlice),
        i1: (IntegerSign, IntegerWordsSlice),
    ) -> (
        integer_operations::UnsignedOutput,
        integer_operations::UnsignedOutput,
        integer_operations::SignedInput,
        integer_operations::SignedInput,
    ) {
        let o0 = integer_operations::UnsignedOutput::new(self.0.as_mut_ptr(), o0);
        let o1 = integer_operations::UnsignedOutput::new(self.0.as_mut_ptr(), o1);
        let (i0, i1) = self.get_integer_input_input(i0, i1);
        (o0, o1, i0, i1)
    }
}

struct NoData;

trait BaseTupleInitialData<Data> {
    fn encode(&self, tuple_memory: &mut [TupleWord]);
}

impl BaseTupleInitialData<NoData> for NoData {
    fn encode(&self, _tuple_memory: &mut [TupleWord]) {}
}

trait BaseTupleData {
    fn from_range(data_start: TupleAddress, data_usage: TupleAddress) -> Self;
}

impl BaseTupleData for NoData {
    fn from_range(_data_start: TupleAddress, _data_usage: TupleAddress) -> Self {
        Self
    }
}

struct TermsSlice {
    terms_start: TupleAddress,
    terms_count: Arity,
}

struct NoTerms;

struct FillWithNone;

trait BaseTupleInitialTerms<Terms> {
    fn encode(&self, tuple_memory: &mut [TupleWord]);
}

impl BaseTupleInitialTerms<NoTerms> for NoTerms {
    fn encode(&self, _tuple_memory: &mut [TupleWord]) {}
}

impl BaseTupleInitialTerms<TermsSlice> for FillWithNone {
    fn encode(&self, tuple_memory: &mut [TupleWord]) {
        for word in tuple_memory.iter_mut() {
            *word = None.encode();
        }
    }
}

impl<'a> BaseTupleInitialTerms<TermsSlice> for &'a [Option<Address>] {
    fn encode(&self, tuple_memory: &mut [TupleWord]) {
        for (word, value) in tuple_memory
            .iter_mut()
            .zip(self.iter().copied().chain(core::iter::repeat(None)))
        {
            *word = value.encode();
        }
    }
}

trait BaseTupleTerms {
    fn from_range(terms_start: TupleAddress, terms_count: Arity) -> Self;
    fn into_address_range(self) -> Option<core::ops::Range<TupleAddress>>;
}

impl BaseTupleTerms for NoTerms {
    fn from_range(_terms_start: TupleAddress, _terms_count: Arity) -> Self {
        Self
    }

    fn into_address_range(self) -> Option<core::ops::Range<TupleAddress>> {
        None
    }
}

impl BaseTupleTerms for TermsSlice {
    fn from_range(terms_start: TupleAddress, terms_count: Arity) -> Self {
        Self {
            terms_start,
            terms_count,
        }
    }

    fn into_address_range(self) -> Option<core::ops::Range<TupleAddress>> {
        let TermsSlice {
            terms_start: first_term,
            terms_count,
        } = self;
        Some(core::ops::Range {
            start: first_term,
            end: first_term + terms_count,
        })
    }
}

trait BaseTuple: Sized {
    const VALUE_TYPE: ValueType;
    type InitialData<'a>: BaseTupleInitialData<Self::Data>;
    type Data: BaseTupleData;
    type InitialTerms<'a>: BaseTupleInitialTerms<Self::Terms>;
    type Terms: BaseTupleTerms;
}

trait TupleDataInfo {
    fn data_size(&self) -> TupleAddress;
    fn data_usage(&self) -> TupleAddress {
        self.data_size()
    }

    fn trim_data(&mut self) -> bool {
        false
    }
}

impl<T: BaseTuple<Data = NoData>> TupleDataInfo for T {
    fn data_size(&self) -> TupleAddress {
        TupleAddress(0)
    }
}

trait TupleTermsInfo {
    fn terms_count(&self) -> Arity;
    fn terms_size(&self) -> Arity {
        Self::terms_count(self)
    }

    fn trim_terms(&mut self) -> bool {
        false
    }
}

impl<T: BaseTuple<Terms = NoTerms>> TupleTermsInfo for T {
    fn terms_count(&self) -> Arity {
        Arity(0)
    }
}

trait NonEmptyData {}

impl NonEmptyData for IntegerWordsSlice {}

trait NonEmptyTerms {}

impl NonEmptyTerms for TermsSlice {}

trait AssociatedDataAndTerms {
    type Data;
    type Terms;
}

impl<D, T> AssociatedDataAndTerms for (D, T) {
    type Data = D;
    type Terms = T;
}

trait NextFreeSpaceInfo: AssociatedDataAndTerms {
    fn next_free_space<
        T: BaseTuple<Data = Self::Data, Terms = Self::Terms> + TupleDataInfo + TupleTermsInfo,
    >(
        head_end: TupleAddress,
        tuple: &T,
    ) -> TupleAddress;
}

impl NextFreeSpaceInfo for (NoData, NoTerms) {
    fn next_free_space<
        T: BaseTuple<Data = Self::Data, Terms = Self::Terms> + TupleDataInfo + TupleTermsInfo,
    >(
        head_end: TupleAddress,
        _tuple: &T,
    ) -> TupleAddress {
        head_end
    }
}

impl<Terms: NonEmptyTerms> NextFreeSpaceInfo for (NoData, Terms) {
    fn next_free_space<
        T: BaseTuple<Data = Self::Data, Terms = Self::Terms> + TupleDataInfo + TupleTermsInfo,
    >(
        head_end: TupleAddress,
        tuple: &T,
    ) -> TupleAddress {
        head_end + tuple.terms_count()
    }
}

impl<Data: NonEmptyData> NextFreeSpaceInfo for (Data, NoTerms) {
    fn next_free_space<
        T: BaseTuple<Data = Self::Data, Terms = Self::Terms> + TupleDataInfo + TupleTermsInfo,
    >(
        head_end: TupleAddress,
        tuple: &T,
    ) -> TupleAddress {
        head_end + tuple.data_usage()
    }
}

struct AddressWithTuple<T: Tuple> {
    registry_address: Address,
    tuple: DirectAccess<T>,
}

impl<T: Tuple> TupleEntries for AddressWithTuple<T> {
    fn decode(
        tuple_memory: &TupleMemory,
        address: TupleAddress,
    ) -> Result<(Self, TupleAddress), TupleMemoryError> {
        TupleEntries::decode(tuple_memory, address).map(
            |(Then(registry_address, tuple), first_term)| {
                (
                    Self {
                        registry_address,
                        tuple,
                    },
                    first_term,
                )
            },
        )
    }

    fn encode(
        self,
        tuple_memory: &mut TupleMemory,
        address: TupleAddress,
    ) -> Result<TupleAddress, TupleMemoryError> {
        let Self {
            registry_address,
            tuple,
        } = self;
        Then(registry_address, tuple).encode(tuple_memory, address)
    }
}

struct TupleMetadata<T: Tuple> {
    registry_address: Address,
    data: T::Data,
    terms: T::Terms,
    next_free_space: TupleAddress,
    next_tuple: TupleAddress,
}

struct TupleLayout<T: Tuple> {
    registry_address: Address,
    tuple_address: TupleAddress,
    data: T::Data,
    next_tuple: TupleAddress,
}

impl<T: Tuple> TupleLayout<T> {
    fn registry_address(self) -> Address {
        self.registry_address
    }
}

trait TupleNextFreeSpaceInfo {
    fn next_free_space(&self, head_end: TupleAddress) -> TupleAddress;
}

impl<T: BaseTuple + TupleDataInfo + TupleTermsInfo> TupleNextFreeSpaceInfo for T
where
    (T::Data, T::Terms): NextFreeSpaceInfo<Data = T::Data, Terms = T::Terms>,
{
    fn next_free_space(&self, head_end: TupleAddress) -> TupleAddress {
        <(T::Data, T::Terms) as NextFreeSpaceInfo>::next_free_space(head_end, self)
    }
}

struct TupleEndInfo {
    next_free_space: TupleAddress,
    next_tuple: TupleAddress,
}

trait Tuple: fmt::Debug + BaseTuple + TupleDataInfo + TupleTermsInfo + TupleNextFreeSpaceInfo {
    fn decode(
        tuple_memory: &TupleMemory,
        tuple_address: TupleAddress,
    ) -> Result<(Self, TupleMetadata<Self>), TupleMemoryError> {
        let (
            AddressWithTuple {
                registry_address,
                tuple: DirectAccess { tuple },
            },
            head_end,
        ) = AddressWithTuple::<Self>::decode(tuple_memory, tuple_address)?;

        let data_size = tuple.data_size();
        let data_usage = tuple.data_usage();
        let data_start = head_end;
        let data_end = data_start + data_size;
        let data = <Self::Data>::from_range(data_start, data_usage);

        let terms_size = tuple.terms_size();
        let terms_count = tuple.terms_count();
        let terms_start = data_end;
        let terms_end = terms_start + terms_size;
        let terms = <Self::Terms>::from_range(terms_start, terms_count);

        let next_free_space = tuple.next_free_space(head_end);
        let next_tuple = terms_end;

        Ok((
            tuple,
            TupleMetadata {
                registry_address,
                terms,
                data,
                next_free_space,
                next_tuple,
            },
        ))
    }

    fn encode_head(
        self,
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
    ) -> Result<TupleAddress, TupleMemoryError> {
        AddressWithTuple::<Self> {
            registry_address,
            tuple: DirectAccess { tuple: self },
        }
        .encode(tuple_memory, tuple_address)
    }

    fn encode(
        self,
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
        terms: Self::InitialTerms<'_>,
        data: Self::InitialData<'_>,
    ) -> Result<TupleLayout<Self>, TupleMemoryError> {
        let data_size = self.data_size();
        let data_usage = self.data_usage();
        let terms_count = self.terms_count();
        let terms_size = self.terms_size();

        let head_end = self.encode_head(registry_address, tuple_memory, tuple_address)?;

        let data_start = head_end;
        let data_end = data_start + data_size;
        data.encode(tuple_memory.get_mut(data_start..(data_start + data_usage))?);

        let terms_start = data_end;
        let terms_end = terms_start + terms_size;
        terms.encode(tuple_memory.get_mut(terms_start..(terms_start + terms_count))?);

        let next_tuple = terms_end;

        Ok(TupleLayout {
            registry_address,
            tuple_address,
            data: <Self::Data>::from_range(data_start, data_usage),
            next_tuple,
        })
    }

    fn decode_and_verify_address(
        tuple_memory: &TupleMemory,
        address: Address,
        tuple_address: TupleAddress,
    ) -> Result<(Self, TupleMetadata<Self>), MemoryError> {
        let (tuple, metadata) = Self::decode(tuple_memory, tuple_address)?;

        if address == metadata.registry_address {
            Ok((tuple, metadata))
        } else {
            Err(MemoryError::TupleDoesNotReferToRegistryAddress {
                registry_address: address,
                tuple_address: tuple_address.into_view(),
                tuple_registry_address: metadata.registry_address,
            })
        }
    }

    fn decode_and_trim(
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
    ) -> Result<TupleEndInfo, TupleMemoryError> {
        let (
            mut tuple,
            TupleMetadata {
                next_free_space,
                next_tuple,
                ..
            },
        ) = Self::decode(tuple_memory, tuple_address)?;

        let data_has_been_trimmed = tuple.trim_data();
        let terms_have_been_trimmed = tuple.trim_terms();

        if data_has_been_trimmed || terms_have_been_trimmed {
            tuple.encode_head(registry_address, tuple_memory, tuple_address)?;
        }

        Ok(TupleEndInfo {
            next_free_space,
            next_tuple,
        })
    }
}

impl<T: fmt::Debug + BaseTuple + TupleDataInfo + TupleTermsInfo + TupleNextFreeSpaceInfo> Tuple
    for T
{
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
struct ReferenceValue(Address);

impl BaseTuple for ReferenceValue {
    const VALUE_TYPE: ValueType = ValueType::Reference;
    type InitialData<'a> = NoData;
    type Data = NoData;
    type InitialTerms<'a> = NoTerms;
    type Terms = NoTerms;
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
struct StructureValue(Functor, Arity);

impl TupleTermsInfo for StructureValue {
    fn terms_count(&self) -> Arity {
        let &Self(_, arity) = self;
        arity
    }
}

impl BaseTuple for StructureValue {
    const VALUE_TYPE: ValueType = ValueType::Structure;
    type InitialData<'a> = NoData;
    type Data = NoData;
    type InitialTerms<'a> = FillWithNone;
    type Terms = TermsSlice;
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
struct ListValue;

impl TupleTermsInfo for ListValue {
    fn terms_count(&self) -> Arity {
        Arity(2)
    }
}

impl BaseTuple for ListValue {
    const VALUE_TYPE: ValueType = ValueType::List;
    type InitialData<'a> = NoData;
    type Data = NoData;
    type InitialTerms<'a> = FillWithNone;
    type Terms = TermsSlice;
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
struct ConstantValue(Constant);

impl BaseTuple for ConstantValue {
    const VALUE_TYPE: ValueType = ValueType::Constant;
    type InitialData<'a> = NoData;
    type Data = NoData;
    type InitialTerms<'a> = NoTerms;
    type Terms = NoTerms;
}

struct IntegerWords<'a> {
    words: &'a [[u8; 4]],
}

impl<'a> BaseTupleInitialData<IntegerWordsSlice> for IntegerWords<'a> {
    fn encode(&self, tuple_memory: &mut [TupleWord]) {
        let words = self
            .words
            .iter()
            .flat_map(|&[a, b, c, d]| [u16::from_le_bytes([a, b]), u16::from_le_bytes([c, d])]);

        for (slot, word) in tuple_memory.iter_mut().zip(words) {
            *slot = word;
        }
    }
}

#[derive(Clone, Copy)]
struct IntegerWordsSlice {
    data_start: TupleAddress,
    words_count: TupleAddress,
}

impl BaseTupleData for IntegerWordsSlice {
    fn from_range(data_start: TupleAddress, data_usage: TupleAddress) -> Self {
        Self {
            data_start,
            words_count: data_usage,
        }
    }
}

impl IntegerWordsSlice {
    fn into_address_range(self) -> core::ops::Range<TupleAddress> {
        core::ops::Range {
            start: self.data_start,
            end: self.data_start + self.words_count,
        }
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
struct IntegerValue {
    sign: IntegerSign,
    words_count: TupleAddress,
    words_usage: TupleAddress,
}

impl IntegerValue {
    fn words_count<T: fmt::Debug>(words: &[T]) -> TupleAddress {
        TupleAddress(
            (words.len() * (core::mem::size_of::<T>() / core::mem::size_of::<TupleWord>())) as u16,
        )
    }
}

impl TupleDataInfo for IntegerValue {
    fn data_size(&self) -> TupleAddress {
        self.words_count
    }

    fn data_usage(&self) -> TupleAddress {
        self.words_usage
    }

    fn trim_data(&mut self) -> bool {
        self.words_count
            .neq_assign(self.words_usage, "Trimming Integer")
    }
}

impl BaseTuple for IntegerValue {
    const VALUE_TYPE: ValueType = ValueType::Integer;
    type InitialData<'a> = IntegerWords<'a>;
    type Data = IntegerWordsSlice;
    type InitialTerms<'a> = NoTerms;
    type Terms = NoTerms;
}

#[derive(Clone, Copy)]
#[repr(transparent)]
struct IntegerEvaluationOutputData(IntegerWordsSlice);

impl BaseTupleData for IntegerEvaluationOutputData {
    fn from_range(data_start: TupleAddress, data_usage: TupleAddress) -> Self {
        Self(IntegerWordsSlice::from_range(data_start, data_usage))
    }
}

impl NonEmptyData for IntegerEvaluationOutputData {}

struct FillWithZero;

impl BaseTupleInitialData<IntegerEvaluationOutputData> for FillWithZero {
    fn encode(&self, tuple_memory: &mut [TupleWord]) {
        for slot in tuple_memory {
            *slot = 0;
        }
    }
}

pub struct IntegerEvaluationOutputLayout {
    registry_address: Address,
    tuple_address: TupleAddress,
    data: IntegerEvaluationOutputData,
}

impl IntegerEvaluationOutputLayout {
    fn new(
        TupleLayout {
            registry_address,
            tuple_address,
            data,
            ..
        }: TupleLayout<IntegerEvaluationOutput>,
    ) -> Self {
        Self {
            registry_address,
            tuple_address,
            data,
        }
    }

    pub fn registry_address(&self) -> Address {
        self.registry_address
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
#[repr(transparent)]
struct IntegerEvaluationOutput(IntegerValue);

impl TupleDataInfo for IntegerEvaluationOutput {
    fn data_size(&self) -> TupleAddress {
        self.0.data_size()
    }

    fn data_usage(&self) -> TupleAddress {
        self.0.data_usage()
    }

    fn trim_data(&mut self) -> bool {
        self.0.trim_data()
    }
}

impl BaseTuple for IntegerEvaluationOutput {
    const VALUE_TYPE: ValueType = IntegerValue::VALUE_TYPE;
    type InitialData<'a> = FillWithZero;
    type Data = IntegerEvaluationOutputData;
    type InitialTerms<'a> = <IntegerValue as BaseTuple>::InitialTerms<'a>;
    type Terms = <IntegerValue as BaseTuple>::Terms;
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
struct Environment {
    continuation_environment: Option<Address>,
    continuation_point: Option<ProgramCounter>,
    number_of_active_permanent_variables: Arity,
    number_of_permanent_variables: Arity,
}

impl TupleTermsInfo for Environment {
    fn terms_count(&self) -> Arity {
        self.number_of_active_permanent_variables
    }

    fn terms_size(&self) -> Arity {
        self.number_of_permanent_variables
    }

    fn trim_terms(&mut self) -> bool {
        self.number_of_permanent_variables.neq_assign(
            self.number_of_active_permanent_variables,
            "Trimming Environment",
        )
    }
}

impl BaseTuple for Environment {
    const VALUE_TYPE: ValueType = ValueType::Environment;
    type InitialData<'a> = NoData;
    type Data = NoData;
    type InitialTerms<'a> = &'a [Option<Address>];
    type Terms = TermsSlice;
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
struct ChoicePoint {
    number_of_saved_registers: Arity,
    current_environment: Option<Address>,
    continuation_point: Option<ProgramCounter>,
    next_choice_point: Option<Address>,
    next_clause: ProgramCounter,
    cut_register: Option<Address>,
}

impl TupleTermsInfo for ChoicePoint {
    fn terms_count(&self) -> Arity {
        self.number_of_saved_registers
    }
}

impl BaseTuple for ChoicePoint {
    const VALUE_TYPE: ValueType = ValueType::ChoicePoint;
    type InitialData<'a> = NoData;
    type Data = NoData;
    type InitialTerms<'a> = &'a [Option<Address>];
    type Terms = TermsSlice;
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
struct TrailVariable {
    variable: Address,
    next_trail_item: Option<Address>,
}

impl BaseTuple for TrailVariable {
    const VALUE_TYPE: ValueType = ValueType::TrailVariable;
    type InitialData<'a> = NoData;
    type Data = NoData;
    type InitialTerms<'a> = NoTerms;
    type Terms = NoTerms;
}

#[derive(Debug)]
pub enum ValueHead {
    Structure(Functor, Arity),
    List,
    Constant(Constant),
    Integer { sign: basic_types::IntegerSign },
}

#[derive(Debug)]
pub enum ReferenceOrValueHead {
    Reference(Address),
    Value(ValueHead),
}

#[derive(Copy, Clone)]
pub struct IntegerLeBytes<'a>(&'a [TupleWord]);

impl<'a> fmt::Debug for IntegerLeBytes<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for word in self.0 {
            write!(f, "{:04x}", word)?;
        }

        Ok(())
    }
}

#[cfg(feature = "defmt-logging")]
impl<'a> defmt::Format for IntegerLeBytes<'a> {
    fn format(&self, fmt: defmt::Formatter) {
        for word in self.0 {
            defmt::write!(fmt, "{:04x}", word);
        }
    }
}

impl<'a> PartialEq for IntegerLeBytes<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other.into_iter())
    }
}

impl<'a, C> minicbor::Encode<C> for IntegerLeBytes<'a> {
    fn encode<W: minicbor::encode::Write>(
        &self,
        e: &mut minicbor::Encoder<W>,
        _ctx: &mut C,
    ) -> Result<(), minicbor::encode::Error<W::Error>> {
        e.begin_array()?;

        for byte in self.into_iter() {
            e.u8(byte)?;
        }

        e.end()?.ok()
    }
}

impl<'a> IntegerLeBytes<'a> {
    pub fn into_iter(self) -> impl Clone + Iterator<Item = u8> + 'a {
        self.0.iter().copied().flat_map(u16::to_le_bytes)
    }

    pub fn equals(&self, le_bytes: impl Iterator<Item = u8>) -> bool {
        let mut i1 = self.into_iter();
        let mut i2 = le_bytes;

        loop {
            return match (i1.next(), i2.next()) {
                (Some(b1), Some(b2)) => {
                    if b1 == b2 {
                        continue;
                    } else {
                        false
                    }
                }
                (None, None) => true,
                (Some(b), None) => b == 0 && i1.all(|b| b == 0),
                (None, Some(b)) => b == 0 && i2.all(|b| b == 0),
            };
        }
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
pub enum Value<'a> {
    Structure(Functor, Arity, TermsList<'a>),
    List(Option<Address>, Option<Address>),
    Constant(Constant),
    Integer {
        sign: basic_types::IntegerSign,
        le_bytes: IntegerLeBytes<'a>,
    },
}

impl<'a> Value<'a> {
    pub fn head(self) -> ValueHead {
        match self {
            Value::Structure(f, n, _) => ValueHead::Structure(f, n),
            Value::List(_, _) => ValueHead::List,
            Value::Constant(c) => ValueHead::Constant(c),
            Value::Integer { sign, .. } => ValueHead::Integer { sign },
        }
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
pub enum ReferenceOrValue<'a> {
    Reference(Address),
    Value(Value<'a>),
}

impl<'a> ReferenceOrValue<'a> {
    pub fn head(self) -> ReferenceOrValueHead {
        match self {
            ReferenceOrValue::Reference(reference) => ReferenceOrValueHead::Reference(reference),
            ReferenceOrValue::Value(value) => ReferenceOrValueHead::Value(value.head()),
        }
    }

    pub fn into_comms(self) -> comms::Value<TermsList<'a>, IntegerLeBytes<'a>> {
        match self {
            ReferenceOrValue::Reference(reference) => {
                comms::Value::Reference(reference.into_inner())
            }
            ReferenceOrValue::Value(Value::Structure(Functor(f), _, terms)) => {
                comms::Value::Structure(f, terms)
            }
            ReferenceOrValue::Value(Value::List(head, tail)) => {
                comms::Value::List(head.map(Address::into_inner), tail.map(Address::into_inner))
            }
            ReferenceOrValue::Value(Value::Constant(Constant(c))) => comms::Value::Constant(c),
            ReferenceOrValue::Value(Value::Integer { sign, le_bytes }) => comms::Value::Integer {
                sign: sign.into_comms(),
                le_bytes,
            },
        }
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
pub enum OutOfMemory {
    OutOfRegistryEntries,
    OutOfTupleSpace,
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
pub enum MemoryError {
    NoRegistryEntryAt {
        address: AddressView,
    },
    BadRegistryValueType {
        address: Address,
        bad_value_type: BadValueType,
    },
    TupleDoesNotReferToRegistryAddress {
        registry_address: Address,
        tuple_address: TupleAddressView,
        tuple_registry_address: Address,
    },
    TupleMemory(TupleMemoryError),
    NotAFreeVariable {
        reference: Address,
        address: Address,
    },
    NoEnvironment,
    ContinuationEnvironmentRemaining {
        continuation_environment: Address,
    },
    TrimmingTooManyVariables {
        number_of_trimmed_permanent_variables: Arity,
        number_of_active_permanent_variables: Arity,
    },
    NoChoicePoint,
    UnmarkedEntryInScanQueue {
        item_to_scan: Address,
    },
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl From<NoRegistryEntryAt> for MemoryError {
    fn from(NoRegistryEntryAt { address }: NoRegistryEntryAt) -> Self {
        Self::NoRegistryEntryAt {
            address: address.into_view(),
        }
    }
}

impl From<BadRegistryValueType> for MemoryError {
    fn from(
        BadRegistryValueType {
            address,
            bad_value_type,
        }: BadRegistryValueType,
    ) -> Self {
        Self::BadRegistryValueType {
            address,
            bad_value_type,
        }
    }
}

impl From<TupleMemoryError> for MemoryError {
    fn from(inner: TupleMemoryError) -> Self {
        Self::TupleMemory(inner)
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
pub enum PermanentVariableError {
    IndexOutOfRange {
        yn: Yn,
        permanent_variable_count: Arity,
    },
    NoValue {
        yn: Yn,
    },
    MemoryError {
        yn: Yn,
        inner: MemoryError,
    },
}

impl PermanentVariableError {
    fn memory_error<E>(yn: Yn) -> impl FnOnce(E) -> Self
    where
        MemoryError: From<E>,
    {
        move |inner| Self::MemoryError {
            yn,
            inner: MemoryError::from(inner),
        }
    }
}

#[derive(Debug)]
pub enum CutError {
    PermanentVariable(PermanentVariableError),
    Memory(MemoryError),
}

impl From<PermanentVariableError> for CutError {
    fn from(inner: PermanentVariableError) -> Self {
        Self::PermanentVariable(inner)
    }
}

impl From<MemoryError> for CutError {
    fn from(inner: MemoryError) -> Self {
        Self::Memory(inner)
    }
}

impl From<NoRegistryEntryAt> for CutError {
    fn from(inner: NoRegistryEntryAt) -> Self {
        Self::Memory(inner.into())
    }
}

impl From<BadRegistryValueType> for CutError {
    fn from(inner: BadRegistryValueType) -> Self {
        Self::Memory(inner.into())
    }
}

impl From<TupleMemoryError> for CutError {
    fn from(inner: TupleMemoryError) -> Self {
        Self::Memory(inner.into())
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
pub enum ExpressionEvaluationError {
    UnboundVariable(Address),
    NotAValidValue(Address, ValueType),
    BadStructure(Functor, Arity),
    Memory(MemoryError),
}

enum ExpressionEvaluationOrOutOfMemory {
    ExpressionEvaluationError(ExpressionEvaluationError),
    OutOfMemory(OutOfMemory),
}

impl From<ExpressionEvaluationError> for ExpressionEvaluationOrOutOfMemory {
    fn from(inner: ExpressionEvaluationError) -> Self {
        Self::ExpressionEvaluationError(inner)
    }
}

impl From<OutOfMemory> for ExpressionEvaluationOrOutOfMemory {
    fn from(inner: OutOfMemory) -> Self {
        Self::OutOfMemory(inner)
    }
}

impl<T> From<T> for ExpressionEvaluationOrOutOfMemory
where
    MemoryError: From<T>,
{
    fn from(inner: T) -> Self {
        Self::ExpressionEvaluationError(ExpressionEvaluationError::Memory(inner.into()))
    }
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
struct ScanningState {
    current_items_to_scan: Address,
    next_items_to_scan: Option<Address>,
}

struct ScanningResult {
    current_items_to_scan: Option<Address>,
    next_items_to_scan: Option<Address>,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
struct SweepingState {
    source: TupleAddress,
    destination: TupleAddress,
    all_tuples_were_marked: bool,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
enum GarbageCollectionState {
    Suspended,
    Starting,
    Scanning(ScanningState),
    Sweeping(SweepingState),
}

#[derive(Clone, Copy)]
pub enum GarbageCollectionIsRunning {
    Suspected,
    Running,
}

#[cfg(feature = "defmt-logging")]
trait DebugNewTuple: fmt::Debug + defmt::Format {}

#[cfg(feature = "defmt-logging")]
impl<T: fmt::Debug + defmt::Format> DebugNewTuple for T {}

#[cfg(not(feature = "defmt-logging"))]
trait DebugNewTuple: fmt::Debug {}

#[cfg(not(feature = "defmt-logging"))]
impl<T: fmt::Debug> DebugNewTuple for T {}

pub struct Heap<'m> {
    registry: Registry<'m>,
    tuple_memory: TupleMemory<'m>,
    tuple_memory_end: TupleAddress,
    current_environment: Option<Address>,
    latest_choice_point: Option<Address>,
    trail_top: Option<Address>,
    cut_register: Option<Address>,
    garbage_collection_state: GarbageCollectionState,
}

impl<'m> Heap<'m> {
    pub fn init(memory: &'m mut [u32]) -> Self {
        let memory_size = memory.len();
        let (registry, tuple_memory) = memory.split_at_mut(memory_size / 2);
        let (_, registry, _) = unsafe { registry.align_to_mut() };
        let (_, tuple_memory, _) = unsafe { tuple_memory.align_to_mut() };

        for entry in registry.iter_mut() {
            unsafe { core::ptr::write(entry, None) };
        }

        Self {
            registry: Registry(registry),
            tuple_memory: TupleMemory(tuple_memory),
            tuple_memory_end: TupleAddress(0),
            current_environment: None,
            latest_choice_point: None,
            trail_top: None,
            cut_register: None,
            garbage_collection_state: GarbageCollectionState::Suspended,
        }
    }

    fn new_value<T: Tuple + DebugNewTuple, D>(
        &mut self,
        factory: impl FnOnce(Address) -> T,
        terms: T::InitialTerms<'_>,
        data: T::InitialData<'_>,
        required_info: fn(TupleLayout<T>) -> D,
    ) -> Result<Result<D, OutOfMemory>, MemoryError> {
        let Some((address, registry_entry)) = self.registry.new_registry_entry() else {
            return Ok(Err(OutOfMemory::OutOfRegistryEntries));
        };

        let tuple_address = self.tuple_memory_end;

        let value = factory(address);

        log_trace!("New value at {}: {:?}", tuple_address, value);

        let Ok(tuple_layout) = value.encode(address, &mut self.tuple_memory, tuple_address, terms, data) else {
            return Ok(Err(OutOfMemory::OutOfTupleSpace));
        };

        self.tuple_memory_end = tuple_layout.next_tuple;

        log_trace!("h = {}", self.tuple_memory_end);

        *registry_entry = Some(RegistryEntry {
            value_type: T::VALUE_TYPE,
            tuple_address,
            marked_state: MarkedState::NotMarked,
            is_unified_with: None,
        });

        self.mark_moved_value(Some(address))?;

        Ok(Ok(required_info(tuple_layout)))
    }

    pub fn new_variable(&mut self) -> Result<Result<Address, OutOfMemory>, MemoryError> {
        self.new_value(
            ReferenceValue,
            NoTerms,
            NoData,
            TupleLayout::registry_address,
        )
    }

    pub fn new_structure(
        &mut self,
        f: Functor,
        n: Arity,
    ) -> Result<Result<Address, OutOfMemory>, MemoryError> {
        self.new_value(
            |_| StructureValue(f, n),
            FillWithNone,
            NoData,
            TupleLayout::registry_address,
        )
    }

    pub fn new_list(&mut self) -> Result<Result<Address, OutOfMemory>, MemoryError> {
        self.new_value(
            |_| ListValue,
            FillWithNone,
            NoData,
            TupleLayout::registry_address,
        )
    }

    pub fn new_constant(
        &mut self,
        c: Constant,
    ) -> Result<Result<Address, OutOfMemory>, MemoryError> {
        self.new_value(
            |_| ConstantValue(c),
            NoTerms,
            NoData,
            TupleLayout::registry_address,
        )
    }

    pub fn new_integer(
        &mut self,
        LongInteger { sign, words }: LongInteger<'_>,
    ) -> Result<Result<Address, OutOfMemory>, MemoryError> {
        self.new_value(
            |_| {
                let words_count = IntegerValue::words_count(words);

                IntegerValue {
                    sign,
                    words_count,
                    words_usage: words_count,
                }
            },
            NoTerms,
            IntegerWords { words },
            TupleLayout::registry_address,
        )
    }

    fn new_integer_output(
        &mut self,
        words_count: TupleAddress,
    ) -> Result<Result<IntegerEvaluationOutputLayout, OutOfMemory>, MemoryError> {
        self.new_value(
            |_| {
                IntegerEvaluationOutput(IntegerValue {
                    sign: IntegerSign::Zero,
                    words_count,
                    words_usage: words_count,
                })
            },
            NoTerms,
            FillWithZero,
            IntegerEvaluationOutputLayout::new,
        )
    }

    pub fn new_system_call_integer_output(
        &mut self,
    ) -> Result<Result<IntegerEvaluationOutputLayout, OutOfMemory>, MemoryError> {
        let bytes_count = core::mem::size_of::<u128>();
        let word_size = core::mem::size_of::<TupleWord>();
        let words_count = TupleAddress(
            (bytes_count / word_size) as u16 + u16::from((bytes_count % word_size) > 0),
        );

        self.new_integer_output(words_count)
    }

    pub fn get_value(
        &self,
        mut address: Address,
    ) -> Result<(Address, ReferenceOrValue), MemoryError> {
        loop {
            log_trace!("Looking up memory at {}", address);
            let registry_entry = self.registry.get(address)?;

            let (address, value) = match &registry_entry.value_type {
                ValueType::Reference => {
                    let (ReferenceValue(reference_address), metadata) =
                        ReferenceValue::decode_and_verify_address(
                            &self.tuple_memory,
                            address,
                            registry_entry.tuple_address,
                        )?;

                    if reference_address != address {
                        address = reference_address;
                        continue;
                    }

                    (
                        metadata.registry_address,
                        ReferenceOrValue::Reference(reference_address),
                    )
                }
                ValueType::Structure => {
                    let (StructureValue(f, n), metadata) =
                        StructureValue::decode_and_verify_address(
                            &self.tuple_memory,
                            address,
                            registry_entry.tuple_address,
                        )?;

                    (
                        metadata.registry_address,
                        ReferenceOrValue::Value(Value::Structure(
                            f,
                            n,
                            self.tuple_memory
                                .load_terms(metadata.terms.into_address_range())?,
                        )),
                    )
                }
                ValueType::List => {
                    let (ListValue, metadata) = ListValue::decode_and_verify_address(
                        &self.tuple_memory,
                        address,
                        registry_entry.tuple_address,
                    )?;

                    let mut terms = self
                        .tuple_memory
                        .load_terms(metadata.terms.into_address_range())?
                        .into_iter();

                    let head = terms.next().flatten();
                    let tail = terms.next().flatten();

                    (
                        metadata.registry_address,
                        ReferenceOrValue::Value(Value::List(head, tail)),
                    )
                }
                ValueType::Constant => {
                    let (ConstantValue(c), metadata) = ConstantValue::decode_and_verify_address(
                        &self.tuple_memory,
                        address,
                        registry_entry.tuple_address,
                    )?;

                    (
                        metadata.registry_address,
                        ReferenceOrValue::Value(Value::Constant(c)),
                    )
                }
                ValueType::Integer => {
                    let (integer, metadata) = IntegerValue::decode_and_verify_address(
                        &self.tuple_memory,
                        address,
                        registry_entry.tuple_address,
                    )?;

                    (
                        metadata.registry_address,
                        ReferenceOrValue::Value(Value::Integer {
                            sign: integer.sign,
                            le_bytes: IntegerLeBytes(
                                self.tuple_memory.get(metadata.data.into_address_range())?,
                            ),
                        }),
                    )
                }
                ValueType::Environment | ValueType::ChoicePoint | ValueType::TrailVariable => {
                    return Err(MemoryError::BadRegistryValueType {
                        address,
                        bad_value_type: BadValueType::ExpectedOneOf {
                            expected: &[
                                ValueType::Reference,
                                ValueType::Structure,
                                ValueType::List,
                                ValueType::Constant,
                                ValueType::Integer,
                            ],
                            actual: registry_entry.value_type.clone(),
                        },
                    })
                }
            };

            log_trace!("Value: {:?}", value);

            break Ok((address, value));
        }
    }

    fn structure_term_addresses(&self, address: Address) -> Result<TermsSlice, MemoryError> {
        let registry_entry = self.registry.get(address)?;

        Ok(match registry_entry.value_type {
            ValueType::Structure => {
                let (StructureValue(..), metadata) = StructureValue::decode_and_verify_address(
                    &self.tuple_memory,
                    address,
                    registry_entry.tuple_address,
                )?;
                metadata.terms
            }
            ValueType::List => {
                let (ListValue, metadata) = ListValue::decode_and_verify_address(
                    &self.tuple_memory,
                    address,
                    registry_entry.tuple_address,
                )?;
                metadata.terms
            }
            _ => {
                return Err(MemoryError::BadRegistryValueType {
                    address,
                    bad_value_type: BadValueType::ExpectedOneOf {
                        expected: &[ValueType::Structure, ValueType::List],
                        actual: registry_entry.value_type.clone(),
                    },
                })
            }
        })
    }

    fn get_environment(
        &self,
    ) -> Result<(Environment, TupleAddress, TupleMetadata<Environment>), MemoryError> {
        let current_environment = self.current_environment.ok_or(MemoryError::NoEnvironment)?;

        let entry = self
            .registry
            .get(current_environment)?
            .verify_is(ValueType::Environment)?;

        let (environment, metadata) = Environment::decode_and_verify_address(
            &self.tuple_memory,
            current_environment,
            entry.tuple_address,
        )?;

        Ok((environment, entry.tuple_address, metadata))
    }

    fn get_permanent_variable_address(
        &self,
        yn: Yn,
    ) -> Result<TupleAddress, PermanentVariableError> {
        let (environment, _, metadata) = self
            .get_environment()
            .map_err(PermanentVariableError::memory_error(yn))?;

        if yn.yn >= environment.number_of_active_permanent_variables.0 {
            return Err(PermanentVariableError::IndexOutOfRange {
                yn,
                permanent_variable_count: environment.number_of_active_permanent_variables,
            });
        }

        Ok(metadata.terms.terms_start + yn)
    }

    fn try_load_permanent_variable(
        &self,
        yn: Yn,
    ) -> Result<Option<Address>, PermanentVariableError> {
        let term_address = self.get_permanent_variable_address(yn)?;

        Option::<Address>::decode(
            self.tuple_memory
                .load(term_address)
                .map_err(PermanentVariableError::memory_error(yn))?,
        )
        .map_err(TupleMemoryError::bad_entry(term_address))
        .map_err(PermanentVariableError::memory_error(yn))
    }

    pub fn load_permanent_variable(&self, yn: Yn) -> Result<Address, PermanentVariableError> {
        self.try_load_permanent_variable(yn)?
            .ok_or(PermanentVariableError::NoValue { yn })
    }

    fn store_maybe_permanent_variable(
        &mut self,
        yn: Yn,
        address: Option<Address>,
    ) -> Result<(), PermanentVariableError> {
        let term_address = self.get_permanent_variable_address(yn)?;

        self.mark_moved_value(address)
            .map_err(PermanentVariableError::memory_error(yn))?;

        self.tuple_memory
            .store(term_address, address.encode())
            .map_err(PermanentVariableError::memory_error(yn))
    }

    pub fn store_permanent_variable(
        &mut self,
        yn: Yn,
        address: Address,
    ) -> Result<(), PermanentVariableError> {
        self.store_maybe_permanent_variable(yn, Some(address))
    }

    fn verify_is_reference(
        &self,
        address: Address,
    ) -> Result<(ReferenceValue, TupleAddress), MemoryError> {
        let entry = self
            .registry
            .get(address)?
            .verify_is(ValueType::Reference)?;

        let (reference, _) = ReferenceValue::decode_and_verify_address(
            &self.tuple_memory,
            address,
            entry.tuple_address,
        )?;

        Ok((reference, entry.tuple_address))
    }

    fn verify_is_free_variable(&self, address: Address) -> Result<TupleAddress, MemoryError> {
        let (ReferenceValue(reference), tuple_address) = self.verify_is_reference(address)?;

        if reference == address {
            Ok(tuple_address)
        } else {
            Err(MemoryError::NotAFreeVariable { address, reference })
        }
    }

    pub fn bind_variable_to_value(
        &mut self,
        variable_address: Address,
        value_address: Address,
    ) -> Result<Result<(), OutOfMemory>, MemoryError> {
        log_trace!(
            "Binding memory {} to value at {}",
            variable_address,
            value_address
        );

        let tuple_address = self.verify_is_free_variable(variable_address)?;

        self.mark_moved_value(Some(value_address))?;

        match self.add_variable_to_trail(variable_address, tuple_address) {
            Ok(Ok(())) => {
                ReferenceValue(value_address).encode(
                    variable_address,
                    &mut self.tuple_memory,
                    tuple_address,
                    NoTerms,
                    NoData,
                )?;

                Ok(Ok(()))
            }
            err => err,
        }
    }

    pub fn bind_variables(
        &mut self,
        a1: Address,
        a2: Address,
    ) -> Result<Result<(), OutOfMemory>, MemoryError> {
        let t1 = self.verify_is_free_variable(a1)?;
        let t2 = self.verify_is_free_variable(a2)?;

        if t1 < t2 {
            self.bind_variable_to_value(a2, a1)
        } else {
            self.bind_variable_to_value(a1, a2)
        }
    }

    pub fn unify(&mut self, a1: Address, a2: Address) -> Result<(), UnificationError> {
        let result = self.do_unify(a1, a2);
        self.registry.clear_unification();
        result
    }

    fn do_unify(&mut self, a1: Address, a2: Address) -> Result<(), UnificationError> {
        log_trace!("Unifying {} and {}", a1, a2);
        let (a1, v1) = self.get_value(a1)?;
        let (a2, v2) = self.get_value(a2)?;
        log_trace!("Resolved to {:?} @ {} and {:?} @ {}", v1, a1, v2, a2);

        let pairs = ((a1, v1), (a2, v2));

        if a1 == a2 {
            Ok(())
        } else {
            match pairs {
                ((a1, ReferenceOrValue::Reference(_)), (a2, ReferenceOrValue::Reference(_))) => {
                    self.bind_variables(a1, a2)??;
                    Ok(())
                }
                (
                    (variable_address, ReferenceOrValue::Reference(_)),
                    (value_address, ReferenceOrValue::Value(_)),
                )
                | (
                    (value_address, ReferenceOrValue::Value(_)),
                    (variable_address, ReferenceOrValue::Reference(_)),
                ) => {
                    self.bind_variable_to_value(variable_address, value_address)??;
                    Ok(())
                }
                (
                    (a1, ReferenceOrValue::Value(Value::Structure(f1, n1, _))),
                    (a2, ReferenceOrValue::Value(Value::Structure(f2, n2, _))),
                ) => {
                    if self.registry.is_already_unified_with(a1, a2)? {
                        Ok(())
                    } else if f1 == f2 && n1 == n2 {
                        let mut terms_1 = StructureIterationState::structure_reader(a1);
                        let mut terms_2 = StructureIterationState::structure_reader(a2);
                        for _ in 0..n1.0 {
                            let a1 = terms_1.read_next(self)?;
                            let a2 = terms_2.read_next(self)?;
                            self.do_unify(a1, a2)?
                        }

                        Ok(())
                    } else {
                        Err(UnificationError::UnificationFailure)
                    }
                }
                (
                    (a1, ReferenceOrValue::Value(Value::List(_, _))),
                    (a2, ReferenceOrValue::Value(Value::List(_, _))),
                ) => {
                    if self.registry.is_already_unified_with(a1, a2)? {
                        Ok(())
                    } else {
                        let mut terms_1 = StructureIterationState::structure_reader(a1);
                        let mut terms_2 = StructureIterationState::structure_reader(a2);
                        for _ in 0..2 {
                            let a1 = terms_1.read_next(self)?;
                            let a2 = terms_2.read_next(self)?;
                            self.do_unify(a1, a2)?
                        }

                        Ok(())
                    }
                }
                (
                    (_, ReferenceOrValue::Value(Value::Constant(c1))),
                    (_, ReferenceOrValue::Value(Value::Constant(c2))),
                ) => {
                    if c1 == c2 {
                        Ok(())
                    } else {
                        Err(UnificationError::UnificationFailure)
                    }
                }
                (
                    (
                        _,
                        ReferenceOrValue::Value(Value::Integer {
                            sign: s1,
                            le_bytes: b1,
                            ..
                        }),
                    ),
                    (
                        _,
                        ReferenceOrValue::Value(Value::Integer {
                            sign: s2,
                            le_bytes: b2,
                        }),
                    ),
                ) => {
                    if (s1, b1) == (s2, b2) {
                        Ok(())
                    } else {
                        Err(UnificationError::UnificationFailure)
                    }
                }

                ((_, ReferenceOrValue::Value(_)), (_, ReferenceOrValue::Value(_))) => {
                    Err(UnificationError::UnificationFailure)
                }
            }
        }
    }

    pub fn allocate(
        &mut self,
        number_of_permanent_variables: Arity,
        continuation_point: Option<ProgramCounter>,
        saved_registers: &[Option<Address>],
    ) -> Result<Result<(), OutOfMemory>, MemoryError> {
        let continuation_environment = self.current_environment;

        self.mark_moved_value(self.current_environment)?;

        for &address in saved_registers {
            self.mark_moved_value(address)?;
        }

        Ok(self
            .new_value(
                |_| Environment {
                    continuation_environment,
                    continuation_point,
                    number_of_active_permanent_variables: number_of_permanent_variables,
                    number_of_permanent_variables,
                },
                saved_registers,
                NoData,
                TupleLayout::registry_address,
            )?
            .map(|new_environment| {
                self.current_environment = Some(new_environment);
            }))
    }

    pub fn trim(&mut self, n: Arity) -> Result<(), MemoryError> {
        let (mut environment, tuple_address, metadata) = self.get_environment()?;
        if n > environment.number_of_active_permanent_variables {
            return Err(MemoryError::TrimmingTooManyVariables {
                number_of_trimmed_permanent_variables: environment
                    .number_of_active_permanent_variables,
                number_of_active_permanent_variables: n,
            });
        }

        if let Some((_, latest_choice_point, _)) = self.get_latest_choice_point()? {
            log_trace!("latest_choice_point: {}", latest_choice_point);
            log_trace!("current_environment: {}", tuple_address);

            if latest_choice_point > tuple_address {
                log_trace!("Not trimming conditional environment");
                return Ok(());
            }
        }

        environment.number_of_active_permanent_variables -= n;

        environment.encode_head(
            metadata.registry_address,
            &mut self.tuple_memory,
            tuple_address,
        )?;

        self.resume_garbage_collection();

        Ok(())
    }

    pub fn deallocate(
        &mut self,
        continuation_point: &mut Option<ProgramCounter>,
    ) -> Result<(), MemoryError> {
        let (
            Environment {
                continuation_environment,
                continuation_point: previous_continuation_point,
                number_of_active_permanent_variables: _,
                number_of_permanent_variables: _,
            },
            _,
            _,
        ) = self.get_environment()?;

        self.current_environment = continuation_environment;
        *continuation_point = previous_continuation_point;

        log_trace!("E => {}", OptionDisplay(self.current_environment));
        log_trace!("CP => {}", OptionDisplay(*continuation_point));

        self.resume_garbage_collection();

        Ok(())
    }

    pub fn new_choice_point(
        &mut self,
        next_clause: ProgramCounter,
        continuation_point: Option<ProgramCounter>,
        saved_registers: &[Option<Address>],
    ) -> Result<Result<(), OutOfMemory>, MemoryError> {
        let number_of_saved_registers = Arity(saved_registers.len() as u8);
        let current_environment = self.current_environment;
        let next_choice_point = self.latest_choice_point;
        let cut_register = self.cut_register;

        for address in saved_registers.iter().copied().chain([
            self.current_environment,
            self.latest_choice_point,
            self.cut_register,
        ]) {
            self.mark_moved_value(address)?;
        }

        Ok(self
            .new_value(
                |_| ChoicePoint {
                    number_of_saved_registers,
                    current_environment,
                    continuation_point,
                    next_choice_point,
                    next_clause,
                    cut_register,
                },
                saved_registers,
                NoData,
                TupleLayout::registry_address,
            )?
            .map(|new_choice_point| {
                self.latest_choice_point = Some(new_choice_point);
            }))
    }

    fn get_latest_choice_point(
        &self,
    ) -> Result<Option<(ChoicePoint, TupleAddress, TupleMetadata<ChoicePoint>)>, MemoryError> {
        let Some(latest_choice_point) = self.latest_choice_point else {
            return Ok(None)
        };

        let entry = self
            .registry
            .get(latest_choice_point)?
            .verify_is(ValueType::ChoicePoint)?;

        let (choice_point, metadata) = ChoicePoint::decode_and_verify_address(
            &self.tuple_memory,
            latest_choice_point,
            entry.tuple_address,
        )?;

        Ok(Some((choice_point, entry.tuple_address, metadata)))
    }

    pub fn backtrack(
        &mut self,
        pc: &mut Option<ProgramCounter>,
    ) -> Result<(), super::ExecutionFailure<'static>> {
        self.get_latest_choice_point()?
            .ok_or(super::ExecutionFailure::Failed)
            .map(|(choice_point, _, _)| *pc = Some(choice_point.next_clause))
    }

    fn wind_back_to_choice_point(
        &mut self,
        registers: &mut [Option<Address>],
        continuation_point: &mut Option<ProgramCounter>,
    ) -> Result<(ChoicePoint, Address, TupleAddress, Option<Address>), MemoryError> {
        let (choice_point, tuple_address, metadata) = self
            .get_latest_choice_point()?
            .ok_or(MemoryError::NoChoicePoint)?;

        for (register, value_address) in registers.iter_mut().zip(
            metadata
                .terms
                .terms_start
                .iter(choice_point.number_of_saved_registers),
        ) {
            let address = Address::from_word(self.tuple_memory.load(value_address)?);
            *register = address;

            self.mark_moved_value(address)?;
        }

        self.current_environment = choice_point.current_environment;
        *continuation_point = choice_point.continuation_point;

        self.unwind_trail(tuple_address)?;

        let next_choice_point = choice_point.next_choice_point;

        log_trace!("E => {}", OptionDisplay(self.current_environment));
        log_trace!("CP => {}", OptionDisplay(*continuation_point));

        self.resume_garbage_collection();

        Ok((
            choice_point,
            metadata.registry_address,
            tuple_address,
            next_choice_point,
        ))
    }

    pub fn retry_choice_point(
        &mut self,
        registers: &mut [Option<Address>],
        continuation_point: &mut Option<ProgramCounter>,
        next_clause: ProgramCounter,
    ) -> Result<(), MemoryError> {
        let (mut choice_point, registry_address, tuple_address, _metadata) =
            self.wind_back_to_choice_point(registers, continuation_point)?;

        choice_point.next_clause = next_clause;

        choice_point.encode_head(registry_address, &mut self.tuple_memory, tuple_address)?;

        Ok(())
    }

    pub fn remove_choice_point(
        &mut self,
        registers: &mut [Option<Address>],
        continuation_point: &mut Option<ProgramCounter>,
    ) -> Result<(), MemoryError> {
        let (_choice_point, _registry_address, _tuple_address, next_choice_point) =
            self.wind_back_to_choice_point(registers, continuation_point)?;

        self.latest_choice_point = next_choice_point;

        Ok(())
    }

    fn add_variable_to_trail(
        &mut self,
        variable: Address,
        tuple_address: TupleAddress,
    ) -> Result<Result<(), OutOfMemory>, MemoryError> {
        let is_unconditional = self
            .get_latest_choice_point()?
            .map_or(true, |(_, latest_choice_point, _)| {
                tuple_address > latest_choice_point
            });

        if is_unconditional {
            log_trace!("Not trailing unconditional variable");
            return Ok(Ok(()));
        }

        self.mark_moved_value(Some(variable))?;
        self.mark_moved_value(self.trail_top)?;

        let next_trail_item = self.trail_top;
        Ok(self
            .new_value(
                |_| TrailVariable {
                    variable,
                    next_trail_item,
                },
                NoTerms,
                NoData,
                TupleLayout::registry_address,
            )?
            .map(|new_trail_item| {
                self.trail_top = Some(new_trail_item);
            }))
    }

    pub fn save_trail_top(&self) -> Option<Address> {
        self.trail_top
    }

    pub fn restore_saved_trail(
        &mut self,
        saved_trail_top: Option<Address>,
    ) -> Result<(), MemoryError> {
        let boundary = if let Some(saved_trail_top) = saved_trail_top {
            let entry = self
                .registry
                .get(saved_trail_top)?
                .verify_is(ValueType::TrailVariable)?;

            entry.tuple_address
        } else {
            TupleAddress(0)
        };

        self.unwind_trail(boundary)
    }

    fn unwind_trail(&mut self, boundary: TupleAddress) -> Result<(), MemoryError> {
        log_trace!("Unwinding Trail");
        while let Some(trail_top) = self.trail_top {
            let entry = self
                .registry
                .get(trail_top)?
                .verify_is(ValueType::TrailVariable)?;

            if entry.tuple_address <= boundary {
                break;
            }

            log_trace!("Unwinding Trail item @ {}", trail_top);

            let (trail_item, _) = TrailVariable::decode_and_verify_address(
                &self.tuple_memory,
                trail_top,
                entry.tuple_address,
            )?;

            log_trace!("Resetting Reference @ {}", trail_item.variable);

            let (_, item_tuple_address) = self.verify_is_reference(trail_item.variable)?;

            ReferenceValue(trail_item.variable).encode(
                trail_item.variable,
                &mut self.tuple_memory,
                item_tuple_address,
                NoTerms,
                NoData,
            )?;

            self.trail_top = trail_item.next_trail_item;
        }

        log_trace!("Finished Unwinding Trail");

        Ok(())
    }

    pub fn update_cut_register(&mut self) {
        log_trace!("B0 => {}", OptionDisplay(self.latest_choice_point));
        self.cut_register = self.latest_choice_point;
    }

    fn do_cut(&mut self, cut_register: Option<Address>) -> Result<(), CutError> {
        log_trace!("Cutting at {}", OptionDisplay(cut_register));

        let new_choice_point = match (self.latest_choice_point, cut_register) {
            (None | Some(_), None) => None,
            (None, Some(_)) => {
                log_trace!("Not cutting");
                return Ok(());
            }
            (Some(latest_choice_point), Some(cut_register)) => {
                log_trace!("Tidying trace");

                let latest_choice_point_entry = self
                    .registry
                    .get(latest_choice_point)?
                    .verify_is(ValueType::ChoicePoint)?;

                let cut_register_entry = self
                    .registry
                    .get(cut_register)?
                    .verify_is(ValueType::ChoicePoint)?;

                if latest_choice_point_entry.tuple_address <= cut_register_entry.tuple_address {
                    log_trace!("Not cutting");
                    return Ok(());
                }

                Some(cut_register)
            }
        };

        log_trace!(
            "Latest Choice Point: {} => {}",
            OptionDisplay(self.latest_choice_point),
            OptionDisplay(new_choice_point)
        );

        self.latest_choice_point = new_choice_point;

        Ok(())
    }

    pub fn neck_cut(&mut self) -> Result<(), CutError> {
        self.do_cut(self.cut_register)
    }

    pub fn get_level(&mut self, yn: Yn) -> Result<(), PermanentVariableError> {
        self.store_maybe_permanent_variable(yn, self.cut_register)
    }

    pub fn cut(&mut self, yn: Yn) -> Result<(), CutError> {
        self.do_cut(self.try_load_permanent_variable(yn)?)
    }

    fn do_evaluate(
        &mut self,
        address: Option<Address>,
    ) -> Result<(Address, IntegerSign, IntegerWordsSlice), ExpressionEvaluationOrOutOfMemory> {
        let mut address = address.ok_or(MemoryError::NoRegistryEntryAt {
            address: AddressView(0),
        })?;

        loop {
            log_trace!("Looking up memory at {}", address);
            let registry_entry = self.registry.get(address)?;

            return Ok(match &registry_entry.value_type {
                ValueType::Reference => {
                    let (ReferenceValue(reference_address), _) =
                        ReferenceValue::decode_and_verify_address(
                            &self.tuple_memory,
                            address,
                            registry_entry.tuple_address,
                        )?;

                    if reference_address == address {
                        return Err(ExpressionEvaluationError::UnboundVariable(address).into());
                    }

                    address = reference_address;
                    continue;
                }
                ValueType::Integer => {
                    let (IntegerValue { sign, .. }, metadata) =
                        IntegerValue::decode_and_verify_address(
                            &self.tuple_memory,
                            address,
                            registry_entry.tuple_address,
                        )?;

                    (address, sign, metadata.data)
                }
                ValueType::Structure => {
                    let (structure, metadata) = StructureValue::decode_and_verify_address(
                        &self.tuple_memory,
                        address,
                        registry_entry.tuple_address,
                    )?;

                    match SpecialFunctor::try_from(structure)? {
                        SpecialFunctor::PrefixOperation { operation } => {
                            let [a1] = self
                                .tuple_memory
                                .load_terms(metadata.terms.into_address_range())?
                                .take();

                            let (_, s1, w1) = self.do_evaluate(a1)?;

                            let w0_words_count = w1.words_count + 1;

                            let o0 = self.new_integer_output(w0_words_count)??;

                            // Safety: w0 corresponds to a newly created integer output, so has a unique address
                            let (s0, w0_words_usage) = unsafe {
                                operation(
                                    self.tuple_memory
                                        .get_integer_output_input(o0.data, (s1, w1)),
                                )
                            };

                            IntegerValue {
                                sign: s0,
                                words_count: w0_words_count,
                                words_usage: TupleAddress(w0_words_usage),
                            }
                            .encode_head(
                                o0.registry_address,
                                &mut self.tuple_memory,
                                o0.tuple_address,
                            )?;

                            (o0.registry_address, s0, o0.data.0)
                        }
                        SpecialFunctor::InfixOperation {
                            calculate_words_count,
                            operation,
                        } => {
                            let [a1, a2] = self
                                .tuple_memory
                                .load_terms(metadata.terms.into_address_range())?
                                .take();

                            let (_, s1, w1) = self.do_evaluate(a1)?;
                            let (_, s2, w2) = self.do_evaluate(a2)?;

                            let w0_words_count =
                                calculate_words_count(w1.words_count, w2.words_count);

                            let o0 = self.new_integer_output(w0_words_count)??;

                            // Safety: w0 corresponds to a newly created integer output, so has a unique address
                            let (s0, w0_words_usage) = unsafe {
                                operation(self.tuple_memory.get_integer_output_input_input(
                                    o0.data,
                                    (s1, w1),
                                    (s2, w2),
                                ))
                            };

                            IntegerValue {
                                sign: s0,
                                words_count: w0_words_count,
                                words_usage: TupleAddress(w0_words_usage),
                            }
                            .encode_head(
                                o0.registry_address,
                                &mut self.tuple_memory,
                                o0.tuple_address,
                            )?;

                            (o0.registry_address, s0, o0.data.0)
                        }
                        SpecialFunctor::DivMod(select) => {
                            let [a0, a1] = self
                                .tuple_memory
                                .load_terms(metadata.terms.into_address_range())?
                                .take();

                            let (_, s0, w0) = self.do_evaluate(a0)?;
                            let (_, s1, w1) = self.do_evaluate(a1)?;

                            let words_count = w0.words_count;

                            let od = self.new_integer_output(words_count)??;
                            let om = self.new_integer_output(words_count)??;

                            // Safety: wd and wm corresponds to a newly created integer output, so has a unique address
                            let ((sd, ud), (sm, um)) = unsafe {
                                integer_operations::div_mod_signed(
                                    self.tuple_memory.get_integer_output_output_input_input(
                                        od.data,
                                        om.data,
                                        (s0, w0),
                                        (s1, w1),
                                    ),
                                )
                            };

                            for (sign, words_usage, registry_address, tuple_address) in [
                                (sd, TupleAddress(ud), od.registry_address, od.tuple_address),
                                (sm, TupleAddress(um), om.registry_address, om.tuple_address),
                            ] {
                                IntegerValue {
                                    sign,
                                    words_count,
                                    words_usage,
                                }
                                .encode_head(
                                    registry_address,
                                    &mut self.tuple_memory,
                                    tuple_address,
                                )?;
                            }

                            match select {
                                DivOrMod::Div => (od.registry_address, sd, od.data.0),
                                DivOrMod::Mod => (om.registry_address, sm, om.data.0),
                            }
                        }
                        SpecialFunctor::MinMax { select_first_if } => {
                            let [a1, a2] = self
                                .tuple_memory
                                .load_terms(metadata.terms.into_address_range())?
                                .take();

                            let (a1, s1, w1) = self.do_evaluate(a1)?;
                            let (a2, s2, w2) = self.do_evaluate(a2)?;

                            if select_first_if(unsafe {
                                self.tuple_memory
                                    .get_integer_input_input((s1, w1), (s2, w2))
                            }) {
                                (a1, s1, w1)
                            } else {
                                (a2, s2, w2)
                            }
                        }
                    }
                }
                value_type => {
                    return Err(
                        ExpressionEvaluationOrOutOfMemory::ExpressionEvaluationError(
                            ExpressionEvaluationError::NotAValidValue(address, value_type.clone()),
                        ),
                    )
                }
            });
        }
    }

    pub fn evaluate(
        &mut self,
        address: Address,
    ) -> Result<Result<Address, OutOfMemory>, ExpressionEvaluationError> {
        match self.do_evaluate(Some(address)) {
            Ok((address, _, _)) => Ok(Ok(address)),
            Err(ExpressionEvaluationOrOutOfMemory::ExpressionEvaluationError(err)) => Err(err),
            Err(ExpressionEvaluationOrOutOfMemory::OutOfMemory(err)) => Ok(Err(err)),
        }
    }

    pub fn compare(
        &mut self,
        a1: Address,
        a2: Address,
    ) -> Result<Result<core::cmp::Ordering, OutOfMemory>, ExpressionEvaluationError> {
        let (_, s1, w1) = match self.do_evaluate(Some(a1)) {
            Ok(a1) => a1,
            Err(ExpressionEvaluationOrOutOfMemory::ExpressionEvaluationError(err)) => {
                return Err(err)
            }
            Err(ExpressionEvaluationOrOutOfMemory::OutOfMemory(err)) => return Ok(Err(err)),
        };

        let (_, s2, w2) = match self.do_evaluate(Some(a2)) {
            Ok(a1) => a1,
            Err(ExpressionEvaluationOrOutOfMemory::ExpressionEvaluationError(err)) => {
                return Err(err)
            }
            Err(ExpressionEvaluationOrOutOfMemory::OutOfMemory(err)) => return Ok(Err(err)),
        };

        // Safety: The ranges are returned from do_evaluate so are valid
        let (w1, w2) = unsafe {
            self.tuple_memory
                .get_integer_input_input((s1, w1), (s2, w2))
        };

        Ok(Ok(w1.cmp(&w2)))
    }

    pub fn write_system_call_integer(
        &mut self,
        integer: IntegerEvaluationOutputLayout,
        sign: IntegerSign,
        le_bytes: &[u8],
    ) -> Result<(), MemoryError> {
        fn write<T>((dest, src): (&mut T, T)) {
            *dest = src;
        }

        let mut unsigned_output = unsafe { self.tuple_memory.get_integer_output(integer.data) };

        let mut le_bytes = le_bytes.iter().copied();

        let mut infinite_le_bytes = le_bytes.by_ref().chain(core::iter::repeat(0));

        unsigned_output
            .words_mut()
            .zip(core::iter::from_fn(|| {
                let mut buffer = [0_u8; core::mem::size_of::<TupleWord>()];
                buffer
                    .iter_mut()
                    .zip(infinite_le_bytes.by_ref())
                    .for_each(write);
                Some(TupleWord::from_le_bytes(buffer))
            }))
            .for_each(write);

        assert!(le_bytes.all(|n| n == 0));

        IntegerValue {
            sign,
            words_count: integer.data.0.words_count,
            words_usage: TupleAddress(unsigned_output.usage()),
        }
        .encode_head(
            integer.registry_address,
            &mut self.tuple_memory,
            integer.tuple_address,
        )?;

        Ok(())
    }

    fn start_garbage_collection(
        &mut self,
        registers: &[Option<Address>],
    ) -> Result<GarbageCollectionState, NoRegistryEntryAt> {
        let mut list_head = None;

        for (index, slot) in (1..).zip(self.registry.0.iter_mut()) {
            if let Some(entry) = slot {
                if let MarkedState::IsMarked { .. } = &mut entry.marked_state {
                    panic!(
                        "{} is marked",
                        Address(unsafe { NonZeroU16::new_unchecked(index) })
                    );
                }
            }
        }

        for &address in registers {
            self.registry
                .add_maybe_item_to_be_scanned(address, &mut list_head)?;
        }

        {
            let Heap {
                registry: _,
                tuple_memory: _,
                tuple_memory_end: _,
                current_environment,
                latest_choice_point,
                trail_top,
                cut_register,
                garbage_collection_state: _,
            } = *self;

            for address in [
                current_environment,
                latest_choice_point,
                trail_top,
                cut_register,
            ] {
                self.registry
                    .add_maybe_item_to_be_scanned(address, &mut list_head)?;
            }
        }

        log_trace!("{:?}", list_head);

        Ok(if let Some(current_items_to_scan) = list_head {
            GarbageCollectionState::Scanning(ScanningState {
                current_items_to_scan,
                next_items_to_scan: None,
            })
        } else {
            GarbageCollectionState::Suspended
        })
    }

    fn scan_item(&mut self, state: ScanningState) -> Result<ScanningResult, MemoryError> {
        let ScanningState {
            current_items_to_scan: item_to_scan,
            mut next_items_to_scan,
        } = state;

        let next_list_head = &mut next_items_to_scan;

        let registry_entry = self.registry.get(item_to_scan)?;

        let MarkedState::IsMarked { next } = registry_entry.marked_state else {
            return Err(MemoryError::UnmarkedEntryInScanQueue { item_to_scan });
        };

        let term_address_range = match registry_entry.value_type {
            ValueType::Reference => {
                let (ReferenceValue(reference_address), metadata) =
                    ReferenceValue::decode_and_verify_address(
                        &self.tuple_memory,
                        item_to_scan,
                        registry_entry.tuple_address,
                    )?;

                self.registry
                    .add_item_to_be_scanned(reference_address, next_list_head)?;

                metadata.terms.into_address_range()
            }
            ValueType::Structure => {
                let (StructureValue(..), metadata) = StructureValue::decode_and_verify_address(
                    &self.tuple_memory,
                    item_to_scan,
                    registry_entry.tuple_address,
                )?;

                metadata.terms.into_address_range()
            }
            ValueType::List => {
                let (ListValue, metadata) = ListValue::decode_and_verify_address(
                    &self.tuple_memory,
                    item_to_scan,
                    registry_entry.tuple_address,
                )?;

                metadata.terms.into_address_range()
            }
            ValueType::Constant => {
                let (ConstantValue(..), metadata) = ConstantValue::decode_and_verify_address(
                    &self.tuple_memory,
                    item_to_scan,
                    registry_entry.tuple_address,
                )?;

                metadata.terms.into_address_range()
            }
            ValueType::Integer => {
                let (IntegerValue { .. }, metadata) = IntegerValue::decode_and_verify_address(
                    &self.tuple_memory,
                    item_to_scan,
                    registry_entry.tuple_address,
                )?;

                metadata.terms.into_address_range()
            }
            ValueType::Environment => {
                let (
                    Environment {
                        continuation_environment,
                        continuation_point: _,
                        number_of_active_permanent_variables: _,
                        number_of_permanent_variables: _,
                    },
                    metadata,
                ) = Environment::decode_and_verify_address(
                    &self.tuple_memory,
                    item_to_scan,
                    registry_entry.tuple_address,
                )?;

                self.registry
                    .add_maybe_item_to_be_scanned(continuation_environment, next_list_head)?;

                metadata.terms.into_address_range()
            }
            ValueType::ChoicePoint => {
                let (
                    ChoicePoint {
                        number_of_saved_registers: _,
                        current_environment,
                        continuation_point: _,
                        next_choice_point,
                        next_clause: _,
                        cut_register,
                    },
                    metadata,
                ) = ChoicePoint::decode_and_verify_address(
                    &self.tuple_memory,
                    item_to_scan,
                    registry_entry.tuple_address,
                )?;

                self.registry
                    .add_maybe_item_to_be_scanned(current_environment, next_list_head)?;
                self.registry
                    .add_maybe_item_to_be_scanned(next_choice_point, next_list_head)?;
                self.registry
                    .add_maybe_item_to_be_scanned(cut_register, next_list_head)?;

                metadata.terms.into_address_range()
            }
            ValueType::TrailVariable => {
                let (
                    TrailVariable {
                        variable,
                        next_trail_item,
                    },
                    metadata,
                ) = TrailVariable::decode_and_verify_address(
                    &self.tuple_memory,
                    item_to_scan,
                    registry_entry.tuple_address,
                )?;

                self.registry
                    .add_item_to_be_scanned(variable, next_list_head)?;
                self.registry
                    .add_maybe_item_to_be_scanned(next_trail_item, next_list_head)?;

                metadata.terms.into_address_range()
            }
        };

        for term_address in self
            .tuple_memory
            .load_terms(term_address_range)?
            .into_iter()
        {
            self.registry
                .add_maybe_item_to_be_scanned(term_address, next_list_head)?;
        }

        Ok(ScanningResult {
            current_items_to_scan: next,
            next_items_to_scan,
        })
    }

    fn sweep_item(&mut self, state: SweepingState) -> Result<SweepingState, MemoryError> {
        let SweepingState {
            source,
            destination,
            all_tuples_were_marked,
        } = state;

        let registry_address = Address::decode(self.tuple_memory.load(source)?)
            .map_err(TupleMemoryError::bad_entry(source))?;
        let mut registry_slot = self.registry.slot_mut(registry_address)?;
        let registry_entry = registry_slot.entry()?;

        let TupleEndInfo {
            next_free_space,
            next_tuple,
        } = match registry_entry.value_type {
            ValueType::Reference => {
                ReferenceValue::decode_and_trim(registry_address, &mut self.tuple_memory, source)?
            }
            ValueType::Structure => {
                StructureValue::decode_and_trim(registry_address, &mut self.tuple_memory, source)?
            }
            ValueType::List => {
                ListValue::decode_and_trim(registry_address, &mut self.tuple_memory, source)?
            }
            ValueType::Constant => {
                ConstantValue::decode_and_trim(registry_address, &mut self.tuple_memory, source)?
            }
            ValueType::Integer => {
                IntegerValue::decode_and_trim(registry_address, &mut self.tuple_memory, source)?
            }
            ValueType::Environment => {
                Environment::decode_and_trim(registry_address, &mut self.tuple_memory, source)?
            }
            ValueType::ChoicePoint => {
                ChoicePoint::decode_and_trim(registry_address, &mut self.tuple_memory, source)?
            }
            ValueType::TrailVariable => {
                TrailVariable::decode_and_trim(registry_address, &mut self.tuple_memory, source)?
            }
        };

        Ok(match registry_entry.marked_state.clear() {
            MarkedState::NotMarked => {
                log_trace!("Freeing {}", registry_address);

                registry_slot.clear();

                SweepingState {
                    source: next_tuple,
                    destination,
                    all_tuples_were_marked: false,
                }
            }
            MarkedState::IsMarked { .. } => {
                log_trace!("Keeping {}", registry_address);

                self.tuple_memory
                    .copy_within(source..next_free_space, destination);

                registry_entry.tuple_address = destination;

                SweepingState {
                    source: next_tuple,
                    destination: destination + (next_free_space - source),
                    all_tuples_were_marked,
                }
            }
        })
    }

    fn mark_moved_value(&mut self, address: Option<Address>) -> Result<(), NoRegistryEntryAt> {
        let Some(address) = address else { return Ok(()); };

        log_trace!("{} has moved", address);

        let registry_entry = self.registry.get_mut(address)?;

        if let MarkedState::IsMarked { .. } = registry_entry.marked_state {
            return Ok(());
        }

        registry_entry.marked_state = match &mut self.garbage_collection_state {
            GarbageCollectionState::Suspended | GarbageCollectionState::Starting => {
                MarkedState::NotMarked
            }
            GarbageCollectionState::Scanning(ScanningState {
                next_items_to_scan, ..
            }) => MarkedState::IsMarked {
                next: next_items_to_scan.replace(address),
            },
            GarbageCollectionState::Sweeping(SweepingState { source, .. }) => {
                if registry_entry.tuple_address < *source {
                    MarkedState::NotMarked
                } else {
                    MarkedState::IsMarked { next: None }
                }
            }
        };

        Ok(())
    }

    pub fn run_garbage_collection(
        &mut self,
        registers: &[Option<Address>],
    ) -> Result<GarbageCollectionIsRunning, MemoryError> {
        log_trace!("{:?}", self.garbage_collection_state);

        let new_garbage_collection_state = match self.garbage_collection_state {
            GarbageCollectionState::Suspended => return Ok(GarbageCollectionIsRunning::Suspected),
            GarbageCollectionState::Starting => self.start_garbage_collection(registers)?,
            GarbageCollectionState::Scanning(state) => match self.scan_item(state)? {
                ScanningResult {
                    current_items_to_scan: Some(current_items_to_scan),
                    next_items_to_scan,
                } => GarbageCollectionState::Scanning(ScanningState {
                    current_items_to_scan,
                    next_items_to_scan,
                }),
                ScanningResult {
                    current_items_to_scan: None,
                    next_items_to_scan: Some(current_items_to_scan),
                } => GarbageCollectionState::Scanning(ScanningState {
                    current_items_to_scan,
                    next_items_to_scan: None,
                }),
                ScanningResult {
                    current_items_to_scan: None,
                    next_items_to_scan: None,
                } => GarbageCollectionState::Sweeping(SweepingState {
                    source: TupleAddress(0),
                    destination: TupleAddress(0),
                    all_tuples_were_marked: true,
                }),
            },
            GarbageCollectionState::Sweeping(state) => {
                if state.source == self.tuple_memory_end {
                    self.tuple_memory_end = state.destination;
                    if state.all_tuples_were_marked {
                        GarbageCollectionState::Suspended
                    } else {
                        GarbageCollectionState::Starting
                    }
                } else {
                    GarbageCollectionState::Sweeping(self.sweep_item(state)?)
                }
            }
        };

        self.garbage_collection_state = new_garbage_collection_state;

        Ok(GarbageCollectionIsRunning::Running)
    }

    pub fn resume_garbage_collection(&mut self) {
        if let GarbageCollectionState::Suspended = self.garbage_collection_state {
            log_trace!("Resuming Garbage Collection");
            self.garbage_collection_state = GarbageCollectionState::Starting;
        }
    }

    pub fn query_has_multiple_solutions(&self) -> bool {
        self.latest_choice_point.is_some()
    }

    pub fn solution(&self) -> Result<TermsList, MemoryError> {
        let (environment, _, metadata) = self.get_environment()?;
        if let Some(continuation_environment) = environment.continuation_environment {
            return Err(MemoryError::ContinuationEnvironmentRemaining {
                continuation_environment,
            });
        }

        Ok(self
            .tuple_memory
            .load_terms(metadata.terms.into_address_range())?)
    }
}
