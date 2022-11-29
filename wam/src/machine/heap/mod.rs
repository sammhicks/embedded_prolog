use core::{fmt, num::NonZeroU16};

use crate::{log_trace, serializable::SerializableWrapper};

use super::basic_types::{
    Arity, Constant, Functor, NoneRepresents, OptionDisplay, ProgramCounter, Yn,
};

pub mod structure_iteration;

#[derive(Debug)]
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
pub enum ValueType {
    Reference,
    Structure,
    List,
    Constant,
    Environment,
    ChoicePoint,
    TrailVariable,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Address(NonZeroU16);

impl Address {
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

impl NoneRepresents for Address {
    const NONE_REPRESENTS: &'static str = "NULL";
}

impl SerializableWrapper for Option<Address> {
    type Inner = u16;

    fn from_inner(inner: Self::Inner) -> Self {
        Address::from_word(inner)
    }

    fn into_inner(self) -> Self::Inner {
        self.map_or(0, Address::into_word)
    }
}

pub struct AddressView(u16);

impl fmt::Debug for AddressView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04X}", self.0)
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

        log_trace!("Marking {address}, next is {next_list_head:?}");

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
}

type TupleWord = u16;

#[derive(Debug)]
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
        Ok(Self::from_inner(word))
    }

    fn encode(self) -> TupleWord {
        self.into_inner()
    }
}

#[derive(Debug)]
pub enum TupleMemoryError {
    AddressOutOfRange {
        address: usize,
        size: usize,
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
        tuple_memory.store(address, a.encode())?;
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

struct TupleMemory<'m>(&'m mut [TupleWord]);

impl<'m> TupleMemory<'m> {
    fn load(&self, address: TupleAddress) -> Result<TupleWord, TupleMemoryError> {
        let address = address.0 as usize;
        let size = self.0.len();
        self.0
            .get(address)
            .copied()
            .ok_or(TupleMemoryError::AddressOutOfRange { address, size })
    }

    fn store(&mut self, address: TupleAddress, value: TupleWord) -> Result<(), TupleMemoryError> {
        let address = address.0 as usize;
        let size = self.0.len();

        let entry = self
            .0
            .get_mut(address)
            .ok_or(TupleMemoryError::AddressOutOfRange { address, size })?;

        *entry = value;

        Ok(())
    }

    const fn block_size<T>() -> usize {
        core::mem::size_of::<T>() / core::mem::size_of::<TupleWord>()
    }

    const fn block_range<T>(start: usize) -> core::ops::Range<usize> {
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
            .0
            .get(Self::block_range::<T>(address.0.into()))
            .ok_or(TupleMemoryError::AddressOutOfRange {
                address: address.0.into(),
                size: self.0.len(),
            })?
            .as_ptr()
            .cast::<T>();

        Ok((
            core::ptr::read_unaligned(entry),
            address + Self::block_size::<T>() as u16,
        ))
    }

    unsafe fn store_block<T>(
        &mut self,
        address: TupleAddress,
        block: T,
    ) -> Result<TupleAddress, TupleMemoryError> {
        let memory_size = self.0.len();

        let entry = self
            .0
            .get_mut(Self::block_range::<T>(address.0.into()))
            .ok_or(TupleMemoryError::AddressOutOfRange {
                address: address.0.into(),
                size: memory_size,
            })?
            .as_mut_ptr()
            .cast();

        core::ptr::write_unaligned(entry, block);

        Ok(address + Self::block_size::<T>() as u16)
    }

    fn load_terms(
        &self,
        terms: core::ops::Range<TupleAddress>,
    ) -> Result<impl Iterator<Item = Option<Address>> + '_, TupleMemoryError> {
        let terms = core::ops::Range {
            start: usize::from(terms.start.0),
            end: usize::from(terms.end.0),
        };

        let end = terms.end.saturating_sub(1);
        let size = self.0.len();

        let words = self
            .0
            .get(terms)
            .ok_or(TupleMemoryError::AddressOutOfRange { address: end, size })?;

        Ok(words.iter().copied().map(Address::from_word))
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
}

struct TermsSlice {
    first_term: TupleAddress,
    terms_count: Arity,
}

impl TermsSlice {
    fn into_address_range(self) -> core::ops::Range<TupleAddress> {
        let TermsSlice {
            first_term,
            terms_count,
        } = self;
        core::ops::Range {
            start: first_term,
            end: first_term + terms_count,
        }
    }
}

trait IntoMaybeAddressRange {
    fn into_maybe_address_range(self) -> Option<core::ops::Range<TupleAddress>>;
}

impl IntoMaybeAddressRange for NoTerms {
    fn into_maybe_address_range(self) -> Option<core::ops::Range<TupleAddress>> {
        None
    }
}

impl IntoMaybeAddressRange for TermsSlice {
    fn into_maybe_address_range(self) -> Option<core::ops::Range<TupleAddress>> {
        Some(self.into_address_range())
    }
}

struct NoTerms;

struct FillWithNone;

trait BaseTupleInitialTerms<T> {
    fn encode(
        &self,
        tuple_memory: &mut TupleMemory,
        first_term: TupleAddress,
        terms_count: Arity,
    ) -> Result<(), TupleMemoryError>;
}

impl BaseTupleInitialTerms<NoTerms> for NoTerms {
    fn encode(
        &self,
        _: &mut TupleMemory,
        _: TupleAddress,
        _: Arity,
    ) -> Result<(), TupleMemoryError> {
        Ok(())
    }
}

impl BaseTupleInitialTerms<TermsSlice> for FillWithNone {
    fn encode(
        &self,
        tuple_memory: &mut TupleMemory,
        first_term: TupleAddress,
        terms_count: Arity,
    ) -> Result<(), TupleMemoryError> {
        for term_address in first_term.iter(terms_count) {
            tuple_memory.store(term_address, None::<Address>.encode())?;
        }

        Ok(())
    }
}

impl<'a> BaseTupleInitialTerms<TermsSlice> for &'a [Option<Address>] {
    fn encode(
        &self,
        tuple_memory: &mut TupleMemory,
        first_term: TupleAddress,
        terms_count: Arity,
    ) -> Result<(), TupleMemoryError> {
        for (tuple_address, value) in first_term
            .iter(terms_count)
            .zip(self.iter().copied().chain(core::iter::repeat(None)))
        {
            tuple_memory.store(tuple_address, value.encode())?;
        }

        Ok(())
    }
}

trait BaseTupleTerms {
    fn from_range(first_term: TupleAddress, terms_count: Arity) -> Self;
}

impl BaseTupleTerms for NoTerms {
    fn from_range(_: TupleAddress, _: Arity) -> Self {
        Self
    }
}

impl BaseTupleTerms for TermsSlice {
    fn from_range(first_term: TupleAddress, terms_count: Arity) -> Self {
        Self {
            first_term,
            terms_count,
        }
    }
}

trait BaseTuple: Sized {
    const VALUE_TYPE: ValueType;
    type InitialTerms<'a>: BaseTupleInitialTerms<Self::Terms>;
    type Terms: BaseTupleTerms;

    /// The number of terms
    fn terms_count(&self) -> Arity;

    /// The amount of space the terms take. In most cases, this is the same as `terms_count()`
    fn terms_size(&self) -> Arity {
        self.terms_count()
    }
}

struct AddressWithTuple<T: BaseTuple> {
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

struct TupleEndInfo {
    next_free_space: TupleAddress,
    next_tuple: TupleAddress,
}

struct TupleMetadata<T: Tuple> {
    registry_address: Address,
    terms: T::Terms,
    next_free_space: TupleAddress,
    next_tuple: TupleAddress,
}

impl<T: Tuple> TupleMetadata<T> {
    fn end_info(&self) -> TupleEndInfo {
        let &Self {
            next_free_space,
            next_tuple,
            ..
        } = self;

        TupleEndInfo {
            next_free_space,
            next_tuple,
        }
    }
}

trait Tuple: BaseTuple {
    fn decode(
        tuple_memory: &TupleMemory,
        tuple_address: TupleAddress,
    ) -> Result<(Self, TupleMetadata<Self>), TupleMemoryError> {
        let (
            AddressWithTuple {
                registry_address,
                tuple: DirectAccess { tuple },
            },
            first_term,
        ) = AddressWithTuple::<Self>::decode(tuple_memory, tuple_address)?;

        let terms_count = tuple.terms_count();
        let terms_size = tuple.terms_size();

        Ok((
            tuple,
            TupleMetadata {
                registry_address,
                terms: <Self::Terms as BaseTupleTerms>::from_range(first_term, terms_count),
                next_free_space: first_term + terms_count,
                next_tuple: first_term + terms_size,
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
    ) -> Result<TupleAddress, TupleMemoryError> {
        let terms_count = self.terms_count();
        let terms_size = self.terms_size();
        let first_term = self.encode_head(registry_address, tuple_memory, tuple_address)?;
        terms.encode(tuple_memory, first_term, terms_count)?;
        Ok(first_term + terms_size)
    }
}

impl<T: BaseTuple> Tuple for T {}

#[derive(Debug)]
struct ReferenceValue(Address);

impl BaseTuple for ReferenceValue {
    const VALUE_TYPE: ValueType = ValueType::Reference;
    type InitialTerms<'a> = NoTerms;
    type Terms = NoTerms;

    fn terms_count(&self) -> Arity {
        Arity(0)
    }
}

#[derive(Debug)]
struct StructureValue(Functor, Arity);

impl BaseTuple for StructureValue {
    const VALUE_TYPE: ValueType = ValueType::Structure;
    type InitialTerms<'a> = FillWithNone;
    type Terms = TermsSlice;

    fn terms_count(&self) -> Arity {
        let &Self(_, arity) = self;
        arity
    }
}

#[derive(Debug)]
struct ListValue;

impl BaseTuple for ListValue {
    const VALUE_TYPE: ValueType = ValueType::List;
    type InitialTerms<'a> = FillWithNone;
    type Terms = TermsSlice;

    fn terms_count(&self) -> Arity {
        Arity(2)
    }
}

#[derive(Debug)]
struct ConstantValue(Constant);

impl BaseTuple for ConstantValue {
    const VALUE_TYPE: ValueType = ValueType::Constant;
    type InitialTerms<'a> = NoTerms;
    type Terms = NoTerms;

    fn terms_count(&self) -> Arity {
        Arity(0)
    }
}

#[derive(Debug)]
struct Environment {
    continuation_environment: Option<Address>,
    continuation_point: Option<ProgramCounter>,
    number_of_active_permanent_variables: Arity,
    number_of_permanent_variables: Arity,
}

impl BaseTuple for Environment {
    const VALUE_TYPE: ValueType = ValueType::Environment;
    type InitialTerms<'a> = &'a [Option<Address>];
    type Terms = TermsSlice;

    fn terms_count(&self) -> Arity {
        self.number_of_active_permanent_variables
    }

    fn terms_size(&self) -> Arity {
        self.number_of_permanent_variables
    }
}

#[derive(Debug)]
struct ChoicePoint {
    number_of_saved_registers: Arity,
    current_environment: Option<Address>,
    continuation_point: Option<ProgramCounter>,
    next_choice_point: Option<Address>,
    next_clause: ProgramCounter,
    cut_register: Option<Address>,
}

impl BaseTuple for ChoicePoint {
    const VALUE_TYPE: ValueType = ValueType::ChoicePoint;
    type InitialTerms<'a> = &'a [Option<Address>];
    type Terms = TermsSlice;

    fn terms_count(&self) -> Arity {
        self.number_of_saved_registers
    }
}

#[derive(Debug)]
struct TrailVariable {
    variable: Address,
    next_trail_item: Option<Address>,
}

impl BaseTuple for TrailVariable {
    const VALUE_TYPE: ValueType = ValueType::TrailVariable;
    type InitialTerms<'a> = NoTerms;
    type Terms = NoTerms;

    fn terms_count(&self) -> Arity {
        Arity(0)
    }
}

#[derive(Debug)]
pub enum Value {
    Structure(Functor, Arity),
    List,
    Constant(Constant),
}

#[derive(Debug)]
pub enum ReferenceOrValue {
    Reference(Address),
    Value(Value),
}

#[derive(Debug)]
pub enum OutOfMemory {
    OutOfRegistryEntries,
    OutOfTupleSpace,
}

#[derive(Debug)]
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

trait DecodeAndVerify: Tuple {
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
}

impl<T: Tuple> DecodeAndVerify for T {}

#[derive(Debug)]
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

#[derive(Debug, Clone, Copy)]
struct ScanningState {
    current_items_to_scan: Address,
    next_items_to_scan: Option<Address>,
}

struct ScanningResult {
    current_items_to_scan: Option<Address>,
    next_items_to_scan: Option<Address>,
}

#[derive(Debug, Clone, Copy)]
struct SweepingState {
    source: TupleAddress,
    destination: TupleAddress,
    all_tuples_were_marked: bool,
}

#[derive(Debug, Clone, Copy)]
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

    fn new_value<T: Tuple + core::fmt::Debug>(
        &mut self,
        factory: impl FnOnce(Address) -> T,
        terms: T::InitialTerms<'_>,
    ) -> Result<Result<Address, OutOfMemory>, MemoryError> {
        let Some((address, registry_entry)) = self.registry.new_registry_entry() else {
            return Ok(Err(OutOfMemory::OutOfRegistryEntries));
        };

        let tuple_address = self.tuple_memory_end;

        let value = factory(address);

        log_trace!("New value at {}: {:?}", tuple_address, value);

        let Ok(memory_end) = value.encode(address, &mut self.tuple_memory, tuple_address, terms) else {
            return Ok(Err(OutOfMemory::OutOfTupleSpace));
        };

        self.tuple_memory_end = memory_end;

        log_trace!("h = {}", self.tuple_memory_end);

        *registry_entry = Some(RegistryEntry {
            value_type: T::VALUE_TYPE,
            tuple_address,
            marked_state: MarkedState::NotMarked,
        });

        self.mark_moved_value(Some(address))?;

        Ok(Ok(address))
    }

    pub fn new_variable(&mut self) -> Result<Result<Address, OutOfMemory>, MemoryError> {
        self.new_value(ReferenceValue, NoTerms)
    }

    pub fn new_structure(
        &mut self,
        f: Functor,
        n: Arity,
    ) -> Result<Result<Address, OutOfMemory>, MemoryError> {
        self.new_value(|_| StructureValue(f, n), FillWithNone)
    }

    pub fn new_list(&mut self) -> Result<Result<Address, OutOfMemory>, MemoryError> {
        self.new_value(|_| ListValue, FillWithNone)
    }

    pub fn new_constant(
        &mut self,
        c: Constant,
    ) -> Result<Result<Address, OutOfMemory>, MemoryError> {
        self.new_value(|_| ConstantValue(c), NoTerms)
    }

    pub fn get_maybe_value(
        &self,
        address: Option<Address>,
    ) -> Result<
        (
            Address,
            ReferenceOrValue,
            impl Iterator<Item = Option<Address>> + '_,
        ),
        MemoryError,
    > {
        self.get_value(address.ok_or(MemoryError::NoRegistryEntryAt {
            address: AddressView(address.into_inner()),
        })?)
    }

    pub fn get_value(
        &self,
        mut address: Address,
    ) -> Result<
        (
            Address,
            ReferenceOrValue,
            impl Iterator<Item = Option<Address>> + '_,
        ),
        MemoryError,
    > {
        loop {
            log_trace!("Looking up memory at {}", address);
            let registry_entry = self.registry.get(address)?;

            let (address, value, terms) = match &registry_entry.value_type {
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
                        metadata.terms.into_maybe_address_range(),
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
                        ReferenceOrValue::Value(Value::Structure(f, n)),
                        metadata.terms.into_maybe_address_range(),
                    )
                }
                ValueType::List => {
                    let (ListValue, metadata) = ListValue::decode_and_verify_address(
                        &self.tuple_memory,
                        address,
                        registry_entry.tuple_address,
                    )?;

                    (
                        metadata.registry_address,
                        ReferenceOrValue::Value(Value::List),
                        metadata.terms.into_maybe_address_range(),
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
                        metadata.terms.into_maybe_address_range(),
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
                            ],
                            actual: registry_entry.value_type.clone(),
                        },
                    })
                }
            };

            log_trace!("Value: {:?}", value);

            let terms = terms
                .map(|terms| self.tuple_memory.load_terms(terms))
                .transpose()?
                .into_iter()
                .flatten();

            break Ok((address, value, terms));
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

        Ok(metadata.terms.first_term + yn)
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
                .first_term
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

                metadata.terms.into_maybe_address_range()
            }
            ValueType::Structure => {
                let (StructureValue(..), metadata) = StructureValue::decode_and_verify_address(
                    &self.tuple_memory,
                    item_to_scan,
                    registry_entry.tuple_address,
                )?;

                metadata.terms.into_maybe_address_range()
            }
            ValueType::List => {
                let (ListValue, metadata) = ListValue::decode_and_verify_address(
                    &self.tuple_memory,
                    item_to_scan,
                    registry_entry.tuple_address,
                )?;

                metadata.terms.into_maybe_address_range()
            }
            ValueType::Constant => {
                let (ConstantValue(..), metadata) = ConstantValue::decode_and_verify_address(
                    &self.tuple_memory,
                    item_to_scan,
                    registry_entry.tuple_address,
                )?;

                metadata.terms.into_maybe_address_range()
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

                metadata.terms.into_maybe_address_range()
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

                metadata.terms.into_maybe_address_range()
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

                metadata.terms.into_maybe_address_range()
            }
        };

        if let Some(term_address_range) = term_address_range {
            for term_address in self.tuple_memory.load_terms(term_address_range)? {
                self.registry
                    .add_maybe_item_to_be_scanned(term_address, next_list_head)?;
            }
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
            ValueType::Reference => ReferenceValue::decode(&self.tuple_memory, source)?
                .1
                .end_info(),
            ValueType::Structure => StructureValue::decode(&self.tuple_memory, source)?
                .1
                .end_info(),
            ValueType::List => ListValue::decode(&self.tuple_memory, source)?.1.end_info(),
            ValueType::Constant => ConstantValue::decode(&self.tuple_memory, source)?
                .1
                .end_info(),
            ValueType::Environment => {
                let (mut environment, metadata) = Environment::decode(&self.tuple_memory, source)?;

                if environment.number_of_permanent_variables
                    != environment.number_of_active_permanent_variables
                {
                    log_trace!(
                        "Trimming environment space: {} => {}",
                        environment.number_of_permanent_variables,
                        environment.number_of_active_permanent_variables
                    );

                    environment.number_of_permanent_variables =
                        environment.number_of_active_permanent_variables;

                    environment.encode_head(
                        metadata.registry_address,
                        &mut self.tuple_memory,
                        source,
                    )?;
                }

                metadata.end_info()
            }
            ValueType::ChoicePoint => ChoicePoint::decode(&self.tuple_memory, source)?
                .1
                .end_info(),
            ValueType::TrailVariable => TrailVariable::decode(&self.tuple_memory, source)?
                .1
                .end_info(),
        };

        Ok(match registry_entry.marked_state.clear() {
            MarkedState::NotMarked => {
                log_trace!("Freeing {registry_address}");

                registry_slot.clear();

                SweepingState {
                    source: next_tuple,
                    destination,
                    all_tuples_were_marked: false,
                }
            }
            MarkedState::IsMarked { .. } => {
                log_trace!("Keeping {registry_address}");

                self.tuple_memory
                    .copy_within(source..next_tuple, destination);

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

    pub fn solution_registers(
        &self,
    ) -> Result<impl Iterator<Item = Option<Address>> + '_, MemoryError> {
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
