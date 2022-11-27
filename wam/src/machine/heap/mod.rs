use core::{fmt, num::NonZeroU16};

use crate::{log_trace, serializable::SerializableWrapper};

use super::basic_types::{
    Arity, Constant, Functor, NoneRepresents, OptionDisplay, ProgramCounter, Yn,
};

pub mod structure_iteration;

#[derive(Debug)]
enum ValueType {
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
        Self(self.0 + rhs.0 as u16)
    }
}

impl core::ops::Add<Yn> for TupleAddress {
    type Output = Self;

    fn add(self, rhs: Yn) -> Self::Output {
        Self(self.0 + rhs.yn as u16)
    }
}

impl core::ops::Sub<TupleAddress> for TupleAddress {
    type Output = Self;

    fn sub(self, rhs: TupleAddress) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl TupleAddress {
    fn iter(self, arity: Arity) -> impl Iterator<Item = Self> + Clone {
        (self.0..).take(arity.0 as usize).map(Self)
    }
}

#[derive(Debug)]
enum MarkedState {
    NotMarked,
    IsMarked { next: Option<Address> },
}

#[derive(Debug)]
struct NoRegistryEntryAt {
    address: Address,
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

#[derive(Debug)]
struct RegistryEntry {
    value_type: ValueType,
    tuple_address: TupleAddress,
    marked_state: MarkedState,
}

struct Registry<'m>(&'m mut [Option<RegistryEntry>]);

impl<'m> Registry<'m> {
    fn new_registry_entry(&mut self) -> Option<(Address, &mut Option<RegistryEntry>)> {
        let (index, slot) = (1..)
            .zip(self.0.iter_mut())
            .find(|(_, entry)| entry.is_none())?;

        let address = Address(NonZeroU16::new(index).unwrap());

        Some((address, slot))
    }

    fn add_item_to_be_scanned(&mut self, address: Address, next_list_head: &mut Option<Address>) {
        let registry_entry = self.get_mut(address).unwrap();

        if let MarkedState::IsMarked { .. } = registry_entry.marked_state {
            return;
        }

        log_trace!("Marking {address}, next is {next_list_head:?}");

        registry_entry.marked_state = MarkedState::IsMarked {
            next: core::mem::replace(next_list_head, Some(address)),
        };
    }

    fn add_maybe_item_to_be_scanned(
        &mut self,
        address: Option<Address>,
        next_list_head: &mut Option<Address>,
    ) {
        if let Some(address) = address {
            self.add_item_to_be_scanned(address, next_list_head)
        }
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

    fn get(&self, address: Address) -> Result<&RegistryEntry, NoRegistryEntryAt> {
        self.0
            .get(Self::index_of_address(address))
            .and_then(Option::as_ref)
            .ok_or(NoRegistryEntryAt { address })
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
        address: u16,
        inner: TupleEntryError,
    },
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
            A::decode(tuple_memory.load(address)?).map_err(|inner| TupleMemoryError::BadEntry {
                address: address.0,
                inner,
            })?;

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
        unsafe { tuple_memory.load_block(address) }
    }

    fn encode(
        self,
        tuple_memory: &mut TupleMemory,
        address: TupleAddress,
    ) -> Result<TupleAddress, TupleMemoryError> {
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
        std::mem::size_of::<T>() / std::mem::size_of::<TupleWord>()
    }

    const fn block_range<T>(start: usize) -> std::ops::Range<usize> {
        std::ops::Range {
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
            .as_ptr() as *const T;

        Ok((
            unsafe { core::ptr::read_unaligned(entry) },
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
            .as_mut_ptr() as *mut T;

        unsafe { core::ptr::write_unaligned(entry, block) }

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

struct AddressSlice {
    first_term: TupleAddress,
    arity: Arity,
}

impl AddressSlice {
    fn into_address_range(self) -> core::ops::Range<TupleAddress> {
        let AddressSlice { first_term, arity } = self;
        core::ops::Range {
            start: first_term,
            end: first_term + arity,
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

impl IntoMaybeAddressRange for AddressSlice {
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
        arity: Arity,
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

impl BaseTupleInitialTerms<AddressSlice> for FillWithNone {
    fn encode(
        &self,
        tuple_memory: &mut TupleMemory,
        first_term: TupleAddress,
        arity: Arity,
    ) -> Result<(), TupleMemoryError> {
        for term_address in first_term.iter(arity) {
            tuple_memory.store(term_address, None::<Address>.encode())?;
        }

        Ok(())
    }
}

impl<'a> BaseTupleInitialTerms<AddressSlice> for &'a [Option<Address>] {
    fn encode(
        &self,
        tuple_memory: &mut TupleMemory,
        first_term: TupleAddress,
        arity: Arity,
    ) -> Result<(), TupleMemoryError> {
        for (tuple_address, value) in first_term
            .iter(arity)
            .zip(self.iter().copied().chain(core::iter::repeat(None)))
        {
            tuple_memory.store(tuple_address, value.encode())?;
        }

        Ok(())
    }
}

trait BaseTupleTerms {
    fn from_range(first_term: TupleAddress, arity: Arity) -> Self;
}

impl BaseTupleTerms for NoTerms {
    fn from_range(_: TupleAddress, Arity(arity): Arity) -> Self {
        assert_eq!(arity, 0);
        Self
    }
}

impl BaseTupleTerms for AddressSlice {
    fn from_range(first_term: TupleAddress, arity: Arity) -> Self {
        Self { first_term, arity }
    }
}

trait BaseTuple: Sized {
    const VALUE_TYPE: ValueType;
    type InitialTerms<'a>: BaseTupleInitialTerms<Self::Terms>;
    type Terms: BaseTupleTerms;

    fn arity(&self) -> Arity;
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

struct TupleMetadata<T: Tuple> {
    registry_address: Address,
    terms: T::Terms,
    next_tuple: TupleAddress,
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

        let arity = tuple.arity();

        Ok((
            tuple,
            TupleMetadata {
                registry_address,
                terms: <Self::Terms as BaseTupleTerms>::from_range(first_term, arity),
                next_tuple: first_term + arity,
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
        let arity = self.arity();
        let first_term = self.encode_head(registry_address, tuple_memory, tuple_address)?;
        terms.encode(tuple_memory, first_term, arity)?;
        Ok(first_term + arity)
    }
}

impl<T: BaseTuple> Tuple for T {}

#[derive(Debug)]
struct ReferenceValue(Address);

impl BaseTuple for ReferenceValue {
    const VALUE_TYPE: ValueType = ValueType::Reference;
    type InitialTerms<'a> = NoTerms;
    type Terms = NoTerms;

    fn arity(&self) -> Arity {
        Arity(0)
    }
}

#[derive(Debug)]
struct StructureValue(Functor, Arity);

impl BaseTuple for StructureValue {
    const VALUE_TYPE: ValueType = ValueType::Structure;
    type InitialTerms<'a> = FillWithNone;
    type Terms = AddressSlice;

    fn arity(&self) -> Arity {
        let &Self(_, arity) = self;
        arity
    }
}

#[derive(Debug)]
struct ListValue;

impl BaseTuple for ListValue {
    const VALUE_TYPE: ValueType = ValueType::List;
    type InitialTerms<'a> = FillWithNone;
    type Terms = AddressSlice;

    fn arity(&self) -> Arity {
        Arity(2)
    }
}

#[derive(Debug)]
struct ConstantValue(Constant);

impl BaseTuple for ConstantValue {
    const VALUE_TYPE: ValueType = ValueType::Constant;
    type InitialTerms<'a> = NoTerms;
    type Terms = NoTerms;

    fn arity(&self) -> Arity {
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
    type Terms = AddressSlice;

    fn arity(&self) -> Arity {
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
    trail_top: Option<Address>,
    cut_register: Option<Address>,
}

impl BaseTuple for ChoicePoint {
    const VALUE_TYPE: ValueType = ValueType::ChoicePoint;
    type InitialTerms<'a> = &'a [Option<Address>];
    type Terms = AddressSlice;

    fn arity(&self) -> Arity {
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

    fn arity(&self) -> Arity {
        Arity(0)
    }
}

#[derive(Debug)]
pub enum Value {
    Reference(Address),
    Structure(Functor, Arity),
    List,
    Constant(Constant),
}

#[derive(Debug)]
pub enum PermanentVariableError {
    IndexOutOfRange {
        yn: Yn,
        permanent_variable_count: Arity,
    },
    NoValue {
        yn: Yn,
    },
    TupleMemoryError {
        yn: Yn,
        inner: TupleMemoryError,
    },
}

#[derive(Debug)]
pub enum MemoryError {
    NoRegistryEntryAt { address: u16 },
    TupleMemory(TupleMemoryError),
    OutOfRegistryEntries,
    OutOfTupleSpace,
}

impl From<NoRegistryEntryAt> for MemoryError {
    fn from(NoRegistryEntryAt { address }: NoRegistryEntryAt) -> Self {
        Self::NoRegistryEntryAt {
            address: address.into_word(),
        }
    }
}

impl From<TupleMemoryError> for MemoryError {
    fn from(inner: TupleMemoryError) -> Self {
        Self::TupleMemory(inner)
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

impl From<TupleMemoryError> for CutError {
    fn from(inner: TupleMemoryError) -> Self {
        Self::Memory(MemoryError::TupleMemory(inner))
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
    ) -> Result<Address, MemoryError> {
        let (address, registry_entry) = self
            .registry
            .new_registry_entry()
            .ok_or(MemoryError::OutOfRegistryEntries)?;

        let tuple_address = self.tuple_memory_end;

        let value = factory(address);

        log_trace!("New value at {}: {:?}", tuple_address, value);

        self.tuple_memory_end = value
            .encode(address, &mut self.tuple_memory, tuple_address, terms)
            .map_err(|_| MemoryError::OutOfTupleSpace)?;

        log_trace!("h = {}", self.tuple_memory_end);

        *registry_entry = Some(RegistryEntry {
            value_type: T::VALUE_TYPE,
            tuple_address,
            marked_state: MarkedState::NotMarked,
        });

        self.mark_moved_value(address);

        Ok(address)
    }

    pub fn new_variable(&mut self) -> Result<Address, MemoryError> {
        self.new_value(ReferenceValue, NoTerms)
    }

    pub fn new_structure(&mut self, f: Functor, n: Arity) -> Result<Address, MemoryError> {
        self.new_value(|_| StructureValue(f, n), FillWithNone)
    }

    pub fn new_list(&mut self) -> Result<Address, MemoryError> {
        self.new_value(|_| ListValue, FillWithNone)
    }

    pub fn new_constant(&mut self, c: Constant) -> Result<Address, MemoryError> {
        self.new_value(|_| ConstantValue(c), NoTerms)
    }

    pub fn get_value(
        &self,
        mut address: Address,
    ) -> Result<(Address, Value, impl Iterator<Item = Option<Address>> + '_), MemoryError> {
        loop {
            log_trace!("Looking up memory at {}", address);
            let registry_entry = self.registry.get(address).unwrap();

            assert_eq!(
                Some(address),
                Address::from_word(self.tuple_memory.load(registry_entry.tuple_address)?)
            );

            let (address, value, terms) = match &registry_entry.value_type {
                ValueType::Reference => {
                    let (ReferenceValue(reference_address), metadata) =
                        ReferenceValue::decode(&self.tuple_memory, registry_entry.tuple_address)?;

                    if reference_address != address {
                        address = reference_address;
                        continue;
                    }

                    (
                        metadata.registry_address,
                        Value::Reference(reference_address),
                        metadata.terms.into_maybe_address_range(),
                    )
                }
                ValueType::Structure => {
                    let (StructureValue(f, n), metadata) =
                        StructureValue::decode(&self.tuple_memory, registry_entry.tuple_address)?;

                    (
                        metadata.registry_address,
                        Value::Structure(f, n),
                        metadata.terms.into_maybe_address_range(),
                    )
                }
                ValueType::List => {
                    let (ListValue, metadata) =
                        ListValue::decode(&self.tuple_memory, registry_entry.tuple_address)?;
                    (
                        metadata.registry_address,
                        Value::List,
                        metadata.terms.into_maybe_address_range(),
                    )
                }
                ValueType::Constant => {
                    let (ConstantValue(c), metadata) =
                        ConstantValue::decode(&self.tuple_memory, registry_entry.tuple_address)?;
                    (
                        metadata.registry_address,
                        Value::Constant(c),
                        metadata.terms.into_maybe_address_range(),
                    )
                }
                ValueType::Environment | ValueType::ChoicePoint | ValueType::TrailVariable => {
                    panic!("Expected value, found {:?}", registry_entry.value_type);
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

    fn structure_term_addresses(&self, address: Address) -> AddressSlice {
        let registry_entry = self.registry.get(address).unwrap();

        match registry_entry.value_type {
            ValueType::Structure => {
                let (StructureValue(..), metadata) =
                    StructureValue::decode(&self.tuple_memory, registry_entry.tuple_address)
                        .unwrap();
                assert_eq!(address, metadata.registry_address);
                metadata.terms
            }
            ValueType::List => {
                let (ListValue, metadata) =
                    ListValue::decode(&self.tuple_memory, registry_entry.tuple_address).unwrap();
                assert_eq!(address, metadata.registry_address);
                metadata.terms
            }
            _ => panic!("Invalid value type: {:?}", registry_entry.value_type),
        }
    }

    fn get_environment(&self) -> (Environment, TupleAddress, TupleMetadata<Environment>) {
        let current_environment = self.current_environment.expect("No Environment");

        let entry = self.registry.get(current_environment).unwrap();

        assert!(matches!(&entry.value_type, ValueType::Environment));

        let (environment, metadata) =
            Environment::decode(&self.tuple_memory, entry.tuple_address).unwrap();
        assert_eq!(metadata.registry_address, current_environment);

        (environment, entry.tuple_address, metadata)
    }

    fn get_permanent_variable_address(
        &self,
        yn: Yn,
    ) -> Result<TupleAddress, PermanentVariableError> {
        let (environment, _, metadata) = self.get_environment();

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

        Ok(Option::<Address>::decode(
            self.tuple_memory
                .load(term_address)
                .map_err(|inner| PermanentVariableError::TupleMemoryError { yn, inner })?,
        )
        .unwrap())
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

        if let Some(address) = address {
            self.mark_moved_value(address);
        }

        self.tuple_memory
            .store(term_address, address.encode())
            .map_err(|inner| PermanentVariableError::TupleMemoryError { inner, yn })
    }

    pub fn store_permanent_variable(
        &mut self,
        yn: Yn,
        address: Address,
    ) -> Result<(), PermanentVariableError> {
        self.store_maybe_permanent_variable(yn, Some(address))
    }

    fn verify_is_reference(&self, address: Address) -> (ReferenceValue, TupleAddress) {
        let entry = self.registry.get(address).unwrap();

        assert!(matches!(&entry.value_type, ValueType::Reference));
        let (reference, metadata) =
            ReferenceValue::decode(&self.tuple_memory, entry.tuple_address).unwrap();

        assert_eq!(metadata.registry_address, address);

        (reference, entry.tuple_address)
    }

    fn verify_is_free_variable(&self, address: Address) -> TupleAddress {
        let (reference, tuple_address) = self.verify_is_reference(address);

        assert_eq!(reference.0, address);

        tuple_address
    }

    pub fn bind_variable_to_value(
        &mut self,
        variable_address: Address,
        value_address: Address,
    ) -> Result<(), MemoryError> {
        log_trace!(
            "Binding memory {} to value at {}",
            variable_address,
            value_address
        );

        let tuple_address = self.verify_is_free_variable(variable_address);

        ReferenceValue(value_address).encode(
            variable_address,
            &mut self.tuple_memory,
            tuple_address,
            NoTerms,
        )?;

        self.mark_moved_value(value_address);

        self.add_variable_to_trail(variable_address, tuple_address)
    }

    pub fn bind_variables(&mut self, a1: Address, a2: Address) -> Result<(), MemoryError> {
        let t1 = self.verify_is_free_variable(a1);
        let t2 = self.verify_is_free_variable(a2);

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
        terms: &[Option<Address>],
    ) -> Result<(), MemoryError> {
        let continuation_environment = self.current_environment;

        let new_environment = self.new_value(
            |_| Environment {
                continuation_environment,
                continuation_point,
                number_of_active_permanent_variables: number_of_permanent_variables,
                number_of_permanent_variables,
            },
            terms,
        )?;
        self.current_environment = Some(new_environment);

        Ok(())
    }

    pub fn trim(&mut self, n: Arity) -> Result<(), MemoryError> {
        let (mut environment, tuple_address, metadata) = self.get_environment();
        assert!(environment.number_of_active_permanent_variables >= n);

        if let Some((_, latest_choice_point, _)) = self.get_latest_choice_point() {
            log_trace!("latest_choice_point: {}", latest_choice_point);
            log_trace!("current_environment: {}", tuple_address);

            if latest_choice_point > tuple_address {
                log_trace!("Not trimming conditional environment");
                return Ok(());
            }
        }

        environment.number_of_active_permanent_variables =
            Arity(environment.number_of_active_permanent_variables.0 - n.0);

        environment.encode_head(
            metadata.registry_address,
            &mut self.tuple_memory,
            tuple_address,
        )?;

        self.resume_garbage_collection();

        Ok(())
    }

    pub fn deallocate(&mut self, continuation_point: &mut Option<ProgramCounter>) {
        let Environment {
            continuation_environment,
            continuation_point: previous_continuation_point,
            number_of_active_permanent_variables: _,
            number_of_permanent_variables: _,
        } = self.get_environment().0;

        self.current_environment = continuation_environment;
        *continuation_point = previous_continuation_point;

        log_trace!("E => {}", OptionDisplay(self.current_environment));
        log_trace!("CP => {}", OptionDisplay(*continuation_point));

        self.resume_garbage_collection();
    }

    pub fn new_choice_point(
        &mut self,
        next_clause: ProgramCounter,
        continuation_point: Option<ProgramCounter>,
        saved_registers: &[Option<Address>],
    ) -> Result<(), MemoryError> {
        let number_of_saved_registers = Arity(saved_registers.len() as u8);
        let current_environment = self.current_environment;
        let next_choice_point = self.latest_choice_point;
        let trail_top = self.trail_top;
        let cut_register = self.cut_register;

        let new_choice_point = self.new_value(
            |_| ChoicePoint {
                number_of_saved_registers,
                current_environment,
                continuation_point,
                next_choice_point,
                next_clause,
                trail_top,
                cut_register,
            },
            saved_registers,
        )?;

        self.latest_choice_point = Some(new_choice_point);

        Ok(())
    }

    fn get_latest_choice_point(
        &self,
    ) -> Option<(ChoicePoint, TupleAddress, TupleMetadata<ChoicePoint>)> {
        let latest_choice_point = self.latest_choice_point?;

        let entry = self.registry.get(latest_choice_point).unwrap();

        assert!(matches!(&entry.value_type, ValueType::ChoicePoint));

        let (choice_point, metadata) =
            ChoicePoint::decode(&self.tuple_memory, entry.tuple_address).unwrap();
        assert_eq!(metadata.registry_address, latest_choice_point);

        Some((choice_point, entry.tuple_address, metadata))
    }

    pub fn backtrack(
        &mut self,
        pc: &mut Option<ProgramCounter>,
    ) -> Result<(), super::ExecutionFailure<'static>> {
        self.get_latest_choice_point()
            .ok_or(super::ExecutionFailure::Failed)
            .map(|(choice_point, _, _)| *pc = Some(choice_point.next_clause))
    }

    fn wind_back_to_choice_point(
        &mut self,
        registers: &mut [Option<Address>],
        continuation_point: &mut Option<ProgramCounter>,
    ) -> (ChoicePoint, Address, TupleAddress, Option<Address>) {
        let (choice_point, tuple_address, metadata) =
            self.get_latest_choice_point().expect("No Choice Point");

        for (register, value_address) in registers.iter_mut().zip(
            metadata
                .terms
                .first_term
                .iter(choice_point.number_of_saved_registers),
        ) {
            *register = Address::from_word(self.tuple_memory.load(value_address).unwrap());
        }

        self.current_environment = choice_point.current_environment;
        *continuation_point = choice_point.continuation_point;

        self.unwind_trail(tuple_address);

        self.trail_top = choice_point.trail_top;
        let next_choice_point = choice_point.next_choice_point;

        log_trace!("E => {}", OptionDisplay(self.current_environment));
        log_trace!("CP => {}", OptionDisplay(*continuation_point));

        self.resume_garbage_collection();

        (
            choice_point,
            metadata.registry_address,
            tuple_address,
            next_choice_point,
        )
    }

    pub fn retry_choice_point(
        &mut self,
        registers: &mut [Option<Address>],
        continuation_point: &mut Option<ProgramCounter>,
        next_clause: ProgramCounter,
    ) {
        let (mut choice_point, registry_address, tuple_address, _metadata) =
            self.wind_back_to_choice_point(registers, continuation_point);

        choice_point.next_clause = next_clause;

        choice_point
            .encode_head(registry_address, &mut self.tuple_memory, tuple_address)
            .unwrap();
    }

    pub fn remove_choice_point(
        &mut self,
        registers: &mut [Option<Address>],
        continuation_point: &mut Option<ProgramCounter>,
    ) {
        let (_choice_point, _registry_address, _tuple_address, next_choice_point) =
            self.wind_back_to_choice_point(registers, continuation_point);

        self.latest_choice_point = next_choice_point;
    }

    fn add_variable_to_trail(
        &mut self,
        variable: Address,
        tuple_address: TupleAddress,
    ) -> Result<(), MemoryError> {
        let is_unconditional = self
            .get_latest_choice_point()
            .map_or(true, |(_, latest_choice_point, _)| {
                tuple_address > latest_choice_point
            });

        if is_unconditional {
            log_trace!("Not trailing unconditional variable");
            return Ok(());
        }

        let next_trail_item = self.trail_top;
        let new_trail_item = self.new_value(
            |_| TrailVariable {
                variable,
                next_trail_item,
            },
            NoTerms,
        )?;
        self.trail_top = Some(new_trail_item);

        Ok(())
    }

    fn unwind_trail(&mut self, boundary: TupleAddress) {
        log_trace!("Unwinding Trail");
        while let Some(trail_top) = self.trail_top {
            let entry = self.registry.get(trail_top).unwrap();
            assert!(matches!(&entry.value_type, ValueType::TrailVariable));

            if entry.tuple_address < boundary {
                break;
            }

            log_trace!("Unwinding Trail item @ {}", trail_top);

            let (trail_item, metadata) =
                TrailVariable::decode(&self.tuple_memory, entry.tuple_address).unwrap();

            assert_eq!(metadata.registry_address, trail_top);

            log_trace!("Resetting Reference @ {}", trail_item.variable);

            let item_tuple_address = self.verify_is_reference(trail_item.variable).1;

            ReferenceValue(trail_item.variable)
                .encode(
                    trail_item.variable,
                    &mut self.tuple_memory,
                    item_tuple_address,
                    NoTerms,
                )
                .unwrap();

            self.trail_top = trail_item.next_trail_item;
        }

        log_trace!("Finished Unwinding Trail");
    }

    pub fn update_cut_register(&mut self) {
        log_trace!("B0 => {}", OptionDisplay(self.latest_choice_point));
        self.cut_register = self.latest_choice_point;
    }

    fn do_cut(&mut self, cut_register: Option<Address>) -> Result<(), CutError> {
        log_trace!("Cutting at {}", OptionDisplay(cut_register));

        let new_choice_point = match (self.latest_choice_point, cut_register) {
            (None, None) => None,
            (Some(_), None) => None,
            (None, Some(_)) => {
                log_trace!("Not cutting");
                return Ok(());
            }
            (Some(latest_choice_point), Some(cut_register)) => {
                log_trace!("Tidying trace");

                let latest_choice_point_entry = self.registry.get(latest_choice_point).unwrap();

                let cut_register_entry = self.registry.get(cut_register).unwrap();

                assert!(matches!(
                    latest_choice_point_entry.value_type,
                    ValueType::ChoicePoint
                ));

                assert!(matches!(
                    cut_register_entry.value_type,
                    ValueType::ChoicePoint
                ));

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
    ) -> GarbageCollectionState {
        let mut list_head = None;

        for &address in registers.iter().filter_map(Option::as_ref) {
            self.registry
                .add_item_to_be_scanned(address, &mut list_head);
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
                    .add_maybe_item_to_be_scanned(address, &mut list_head);
            }
        }

        log_trace!("{:?}", list_head);

        if let Some(current_items_to_scan) = list_head {
            GarbageCollectionState::Scanning(ScanningState {
                current_items_to_scan,
                next_items_to_scan: None,
            })
        } else {
            GarbageCollectionState::Suspended
        }
    }

    fn scan_item(&mut self, state: ScanningState) -> Result<ScanningResult, TupleMemoryError> {
        let ScanningState {
            current_items_to_scan: item_to_scan,
            mut next_items_to_scan,
        } = state;

        let next_list_head = &mut next_items_to_scan;

        let registry_entry = self.registry.get_mut(item_to_scan).unwrap();

        let MarkedState::IsMarked { next: current_items_to_scan } = registry_entry.marked_state else {
            panic!("Entry {} is in scan queue but is not marked", item_to_scan);
        };

        assert_eq!(
            Some(item_to_scan),
            Address::from_word(self.tuple_memory.load(registry_entry.tuple_address)?)
        );

        let term_address_range = match registry_entry.value_type {
            ValueType::Reference => {
                let (ReferenceValue(reference_address), metadata) =
                    ReferenceValue::decode(&self.tuple_memory, registry_entry.tuple_address)?;

                self.registry
                    .add_item_to_be_scanned(reference_address, next_list_head);

                metadata.terms.into_maybe_address_range()
            }
            ValueType::Structure => {
                let (StructureValue(..), metadata) =
                    StructureValue::decode(&self.tuple_memory, registry_entry.tuple_address)?;

                metadata.terms.into_maybe_address_range()
            }
            ValueType::List => {
                let (ListValue, metadata) =
                    ListValue::decode(&self.tuple_memory, registry_entry.tuple_address)?;

                metadata.terms.into_maybe_address_range()
            }
            ValueType::Constant => {
                let (ConstantValue(..), metadata) =
                    ConstantValue::decode(&self.tuple_memory, registry_entry.tuple_address)?;

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
                ) = Environment::decode(&self.tuple_memory, registry_entry.tuple_address)?;

                self.registry
                    .add_maybe_item_to_be_scanned(continuation_environment, next_list_head);

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
                        trail_top,
                        cut_register,
                    },
                    metadata,
                ) = ChoicePoint::decode(&self.tuple_memory, registry_entry.tuple_address)?;

                self.registry
                    .add_maybe_item_to_be_scanned(current_environment, next_list_head);
                self.registry
                    .add_maybe_item_to_be_scanned(next_choice_point, next_list_head);
                self.registry
                    .add_maybe_item_to_be_scanned(trail_top, next_list_head);
                self.registry
                    .add_maybe_item_to_be_scanned(cut_register, next_list_head);

                metadata.terms.into_maybe_address_range()
            }
            ValueType::TrailVariable => {
                let (
                    TrailVariable {
                        variable,
                        next_trail_item,
                    },
                    metadata,
                ) = TrailVariable::decode(&self.tuple_memory, registry_entry.tuple_address)?;

                self.registry
                    .add_item_to_be_scanned(variable, next_list_head);
                self.registry
                    .add_maybe_item_to_be_scanned(next_trail_item, next_list_head);

                metadata.terms.into_maybe_address_range()
            }
        };

        if let Some(term_address_range) = term_address_range {
            for term_address in self.tuple_memory.load_terms(term_address_range)?.flatten() {
                self.registry
                    .add_item_to_be_scanned(term_address, next_list_head);
            }
        }

        Ok(ScanningResult {
            current_items_to_scan,
            next_items_to_scan,
        })
    }

    fn sweep_item(&mut self, state: SweepingState) -> Result<SweepingState, TupleMemoryError> {
        let SweepingState {
            source,
            destination,
            all_tuples_were_marked,
        } = state;

        let registry_address =
            Address::from_word(self.tuple_memory.load(source)?).expect("No Entry");
        let mut registry_slot = self.registry.slot_mut(registry_address).unwrap();
        let registry_entry = registry_slot.entry().unwrap();

        let next_tuple = match registry_entry.value_type {
            ValueType::Reference => {
                ReferenceValue::decode(&self.tuple_memory, source)?
                    .1
                    .next_tuple
            }
            ValueType::Structure => {
                StructureValue::decode(&self.tuple_memory, source)?
                    .1
                    .next_tuple
            }
            ValueType::List => ListValue::decode(&self.tuple_memory, source)?.1.next_tuple,
            ValueType::Constant => {
                ConstantValue::decode(&self.tuple_memory, source)?
                    .1
                    .next_tuple
            }
            ValueType::Environment => {
                Environment::decode(&self.tuple_memory, source)?
                    .1
                    .next_tuple
            }
            ValueType::ChoicePoint => {
                ChoicePoint::decode(&self.tuple_memory, source)?
                    .1
                    .next_tuple
            }
            ValueType::TrailVariable => {
                TrailVariable::decode(&self.tuple_memory, source)?
                    .1
                    .next_tuple
            }
        };

        Ok(
            match core::mem::replace(&mut registry_entry.marked_state, MarkedState::NotMarked) {
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
                        destination: destination + (next_tuple - source),
                        all_tuples_were_marked,
                    }
                }
            },
        )
    }

    fn mark_moved_value(&mut self, address: Address) {
        let registry_entry = self.registry.get_mut(address).unwrap();

        if let MarkedState::IsMarked { .. } = registry_entry.marked_state {
            return;
        }

        registry_entry.marked_state = match &mut self.garbage_collection_state {
            GarbageCollectionState::Suspended | GarbageCollectionState::Starting => {
                MarkedState::NotMarked
            }
            GarbageCollectionState::Scanning(ScanningState {
                next_items_to_scan, ..
            }) => MarkedState::IsMarked {
                next: core::mem::replace(next_items_to_scan, Some(address)),
            },
            GarbageCollectionState::Sweeping(_) => MarkedState::IsMarked { next: None },
        };
    }

    pub fn run_garbage_collection(
        &mut self,
        registers: &[Option<Address>],
    ) -> Result<GarbageCollectionIsRunning, TupleMemoryError> {
        log_trace!("{:?}", self.garbage_collection_state);

        let new_garbage_collection_state = match self.garbage_collection_state {
            GarbageCollectionState::Suspended => return Ok(GarbageCollectionIsRunning::Suspected),
            GarbageCollectionState::Starting => self.start_garbage_collection(registers),
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
    ) -> Result<impl Iterator<Item = Option<Address>> + '_, TupleMemoryError> {
        let (environment, _, metadata) = self.get_environment();
        assert!(environment.continuation_environment.is_none());

        self.tuple_memory
            .load_terms(metadata.terms.into_address_range())
    }
}
