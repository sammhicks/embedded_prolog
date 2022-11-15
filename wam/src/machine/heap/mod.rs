use core::fmt;

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
pub struct Address(u16);

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

impl Address {
    const NO_ADDRESS: u16 = u16::MAX;

    /// The "None" value of an address. Unsafe because the memory representation is also a valid value of Address
    pub const unsafe fn none() -> Self {
        Self(Self::NO_ADDRESS)
    }
}

impl SerializableWrapper for Address {
    type Inner = u16;

    fn from_inner(inner: Self::Inner) -> Self {
        Self(inner)
    }

    fn into_inner(self) -> Self::Inner {
        self.0
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

impl std::ops::Add<u16> for TupleAddress {
    type Output = Self;

    fn add(self, rhs: u16) -> Self::Output {
        Self(self.0 + rhs)
    }
}

impl std::ops::Add<Arity> for TupleAddress {
    type Output = Self;

    fn add(self, rhs: Arity) -> Self::Output {
        Self(self.0 + rhs.0 as u16)
    }
}

impl std::ops::Add<Yn> for TupleAddress {
    type Output = Self;

    fn add(self, rhs: Yn) -> Self::Output {
        Self(self.0 + rhs.yn as u16)
    }
}

impl TupleAddress {
    fn iter(self, arity: Arity) -> impl Iterator<Item = Self> + Clone {
        (self.0..).take(arity.0 as usize).map(Self)
    }
}

#[derive(Debug)]
struct RegistryEntry {
    value_type: ValueType,
    tuple_address: TupleAddress,
}

struct Registry<'m>(&'m mut [Option<RegistryEntry>]);

impl<'m> Registry<'m> {
    fn new_registry_entry(&mut self) -> Option<(Address, &mut Option<RegistryEntry>)> {
        let (index, slot) = self
            .0
            .iter_mut()
            .enumerate()
            .find(|(_, entry)| entry.is_none())?;

        let address = Address(core::convert::TryInto::try_into(index).expect("Invalid index"));

        Some((address, slot))
    }
}

impl<'m> std::ops::Index<Address> for Registry<'m> {
    type Output = Option<RegistryEntry>;

    fn index(&self, index: Address) -> &Self::Output {
        &self.0[index.0 as usize]
    }
}

impl<'m> std::ops::IndexMut<Address> for Registry<'m> {
    fn index_mut(&mut self, index: Address) -> &mut Self::Output {
        &mut self.0[index.0 as usize]
    }
}

type TupleWord = u16;
type TupleHalfWord = u8;

#[derive(Debug)]
pub enum TupleEntryError {
    BadHalfWord {
        expected: TupleHalfWord,
        actual: TupleHalfWord,
    },
    BadWord {
        expected: TupleWord,
        actual: TupleWord,
    },
}

trait TupleHalfEntry: Sized {
    fn decode(half_word: TupleHalfWord) -> Result<Self, TupleEntryError>;
    fn encode(self) -> TupleHalfWord;
}

impl<T: SerializableWrapper<Inner = TupleHalfWord>> TupleHalfEntry for T {
    fn decode(half_word: TupleHalfWord) -> Result<Self, TupleEntryError> {
        Ok(Self::from_inner(half_word))
    }

    fn encode(self) -> TupleHalfWord {
        self.into_inner()
    }
}

struct MustBeZero;

trait TupleEntry: Sized {
    fn decode(word: TupleWord) -> Result<Self, TupleEntryError>;
    fn encode(self) -> TupleWord;
}

impl TupleEntry for Option<Address> {
    fn decode(word: TupleWord) -> Result<Self, TupleEntryError> {
        Ok(if word == Address::NO_ADDRESS {
            None
        } else {
            Some(Address(word))
        })
    }

    fn encode(self) -> TupleWord {
        self.map_or(Address::NO_ADDRESS, SerializableWrapper::into_inner)
    }
}

impl TupleEntry for Option<ProgramCounter> {
    fn decode(word: TupleWord) -> Result<Self, TupleEntryError> {
        Ok(if word == ProgramCounter::END_OF_PROGRAM {
            None
        } else {
            Some(ProgramCounter(word))
        })
    }

    fn encode(self) -> TupleWord {
        self.map_or(Address::NO_ADDRESS, SerializableWrapper::into_inner)
    }
}

impl<A: TupleHalfEntry, B: TupleHalfEntry> TupleEntry for (A, B) {
    fn decode(word: TupleWord) -> Result<Self, TupleEntryError> {
        let [a, b] = word.to_be_bytes();
        Ok((A::decode(a)?, B::decode(b)?))
    }

    fn encode(self) -> TupleWord {
        let (a, b) = self;
        TupleWord::from_be_bytes([a.encode(), b.encode()])
    }
}

impl<T: SerializableWrapper<Inner = u16>> TupleEntry for T {
    fn decode(word: TupleWord) -> Result<Self, TupleEntryError> {
        Ok(Self::from_inner(word))
    }

    fn encode(self) -> TupleWord {
        self.into_inner()
    }
}

impl TupleHalfEntry for MustBeZero {
    fn decode(half_word: TupleHalfWord) -> Result<Self, TupleEntryError> {
        if half_word == 0 {
            Ok(Self)
        } else {
            Err(TupleEntryError::BadHalfWord {
                expected: 0,
                actual: half_word,
            })
        }
    }

    fn encode(self) -> TupleHalfWord {
        0
    }
}

impl TupleEntry for MustBeZero {
    fn decode(word: TupleWord) -> Result<Self, TupleEntryError> {
        if word == 0 {
            Ok(Self)
        } else {
            Err(TupleEntryError::BadWord {
                expected: 0,
                actual: word,
            })
        }
    }

    fn encode(self) -> TupleWord {
        0
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

struct SingleEntry<T: TupleEntry>(T);

impl<T: TupleEntry> TupleEntries for SingleEntry<T> {
    fn decode(
        tuple_memory: &TupleMemory,
        address: TupleAddress,
    ) -> Result<(Self, TupleAddress), TupleMemoryError> {
        T::decode(tuple_memory.load(address)?)
            .map(|entry| (SingleEntry(entry), address + 1))
            .map_err(|inner| TupleMemoryError::BadEntry {
                address: address.0,
                inner,
            })
    }

    fn encode(
        self,
        tuple_memory: &mut TupleMemory,
        address: TupleAddress,
    ) -> Result<TupleAddress, TupleMemoryError> {
        tuple_memory.store(address, self.0.encode())?;
        Ok(address + 1)
    }
}

impl<A: TupleEntry, B: TupleEntries> TupleEntries for (A, B) {
    fn decode(
        tuple_memory: &TupleMemory,
        address: TupleAddress,
    ) -> Result<(Self, TupleAddress), TupleMemoryError> {
        let (SingleEntry(a), address) = SingleEntry::decode(tuple_memory, address)?;
        let (b, address) = B::decode(tuple_memory, address)?;
        Ok(((a, b), address))
    }

    fn encode(
        self,
        tuple_memory: &mut TupleMemory,
        address: TupleAddress,
    ) -> Result<TupleAddress, TupleMemoryError> {
        let (a, b) = self;
        let address = SingleEntry(a).encode(tuple_memory, address)?;
        b.encode(tuple_memory, address)
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

    fn load_terms(
        &self,
        terms: core::ops::Range<usize>,
    ) -> Result<impl Iterator<Item = Address> + '_, TupleMemoryError> {
        let end = terms.end.saturating_sub(1);
        let size = self.0.len();

        let words = self
            .0
            .get(terms)
            .ok_or(TupleMemoryError::AddressOutOfRange { address: end, size })?;

        Ok(words.iter().copied().map(Address))
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

    fn decode<T: TupleEntries>(
        &self,
        address: TupleAddress,
    ) -> Result<(T, TupleAddress), TupleMemoryError> {
        T::decode(self, address)
    }

    fn encode(
        &mut self,
        address: TupleAddress,
        entry: impl TupleEntries,
    ) -> Result<TupleAddress, TupleMemoryError> {
        entry.encode(self, address)
    }
}

struct NoTerms;

struct TermSlice {
    first_term: TupleAddress,
    arity: Arity,
}

impl TermSlice {
    fn into_address_range(self) -> core::ops::Range<usize> {
        let TermSlice { first_term, arity } = self;
        let start = first_term.0 as usize;
        let end = (first_term + arity).0 as usize;
        start..end
    }
}

trait IntoMaybeAddressRange {
    fn into_maybe_address_range(self) -> Option<core::ops::Range<usize>>;
}

impl IntoMaybeAddressRange for NoTerms {
    fn into_maybe_address_range(self) -> Option<core::ops::Range<usize>> {
        None
    }
}

impl IntoMaybeAddressRange for TermSlice {
    fn into_maybe_address_range(self) -> Option<core::ops::Range<usize>> {
        Some(self.into_address_range())
    }
}

struct TupleMetadata<T: Tuple> {
    registry_address: Address,
    terms: T::Terms,
    next_tuple: TupleAddress,
}

trait Tuple: Sized {
    const VALUE_TYPE: ValueType;
    type InitialTerms: ?Sized;
    type Terms;

    fn decode(
        tuple_memory: &TupleMemory,
        tuple_address: TupleAddress,
    ) -> Result<(Self, TupleMetadata<Self>), TupleMemoryError>;

    fn encode(
        &self,
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
        terms: &Self::InitialTerms,
    ) -> Result<TupleAddress, TupleMemoryError>;
}

#[derive(Debug)]
struct ReferenceValue(Address);

impl Tuple for ReferenceValue {
    const VALUE_TYPE: ValueType = ValueType::Reference;
    type InitialTerms = ();
    type Terms = NoTerms;

    fn decode(
        tuple_memory: &TupleMemory,
        tuple_address: TupleAddress,
    ) -> Result<(Self, TupleMetadata<Self>), TupleMemoryError> {
        let ((registry_address, SingleEntry(r)), next_tuple) =
            tuple_memory.decode(tuple_address)?;

        Ok((
            Self(r),
            TupleMetadata {
                registry_address,
                terms: NoTerms,
                next_tuple,
            },
        ))
    }

    fn encode(
        &self,
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
        _: &(),
    ) -> Result<TupleAddress, TupleMemoryError> {
        let Self(r) = *self;

        tuple_memory.encode(tuple_address, (registry_address, SingleEntry(r)))
    }
}

#[derive(Debug)]
struct StructureValue(Functor, Arity);

impl Tuple for StructureValue {
    const VALUE_TYPE: ValueType = ValueType::Structure;
    type InitialTerms = ();
    type Terms = TermSlice;

    fn decode(
        tuple_memory: &TupleMemory,
        tuple_address: TupleAddress,
    ) -> Result<(Self, TupleMetadata<Self>), TupleMemoryError> {
        let ((registry_address, (f, SingleEntry((MustBeZero, arity)))), first_term) =
            tuple_memory.decode(tuple_address)?;

        Ok((
            Self(f, arity),
            TupleMetadata {
                registry_address,
                terms: TermSlice { first_term, arity },
                next_tuple: first_term + arity,
            },
        ))
    }

    fn encode(
        &self,
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
        _: &(),
    ) -> Result<TupleAddress, TupleMemoryError> {
        let Self(f, n) = *self;

        let first_term = tuple_memory.encode(
            tuple_address,
            (registry_address, (f, SingleEntry((MustBeZero, n)))),
        )?;

        for term_address in first_term.iter(n) {
            tuple_memory.store(term_address, Address::NO_ADDRESS)?;
        }

        Ok(first_term + n)
    }
}

#[derive(Debug)]
struct ListValue;

impl ListValue {
    const ARITY: Arity = Arity(2);
}

impl Tuple for ListValue {
    const VALUE_TYPE: ValueType = ValueType::List;
    type InitialTerms = ();
    type Terms = TermSlice;

    fn decode(
        tuple_memory: &TupleMemory,
        tuple_address: TupleAddress,
    ) -> Result<(Self, TupleMetadata<Self>), TupleMemoryError> {
        let (SingleEntry(registry_address), first_term) = tuple_memory.decode(tuple_address)?;

        Ok((
            Self,
            TupleMetadata {
                registry_address,
                terms: TermSlice {
                    first_term,
                    arity: Self::ARITY,
                },
                next_tuple: first_term + Self::ARITY,
            },
        ))
    }

    fn encode(
        &self,
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
        _: &(),
    ) -> Result<TupleAddress, TupleMemoryError> {
        let first_term = tuple_memory.encode(tuple_address, SingleEntry(registry_address))?;

        for term_address in first_term.iter(Self::ARITY) {
            tuple_memory.store(term_address, Address::NO_ADDRESS)?;
        }

        Ok(first_term + Self::ARITY)
    }
}

#[derive(Debug)]
struct ConstantValue(Constant);

impl Tuple for ConstantValue {
    const VALUE_TYPE: ValueType = ValueType::Constant;
    type InitialTerms = ();
    type Terms = NoTerms;

    fn decode(
        tuple_memory: &TupleMemory,
        tuple_address: TupleAddress,
    ) -> Result<(Self, TupleMetadata<Self>), TupleMemoryError> {
        let ((registry_address, SingleEntry(c)), next_tuple) =
            tuple_memory.decode(tuple_address)?;

        Ok((
            Self(c),
            TupleMetadata {
                registry_address,
                terms: NoTerms,
                next_tuple,
            },
        ))
    }

    fn encode(
        &self,
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
        _: &(),
    ) -> Result<TupleAddress, TupleMemoryError> {
        let Self(c) = *self;

        tuple_memory.encode(tuple_address, (registry_address, SingleEntry(c)))
    }
}

#[derive(Debug)]
struct Environment {
    continuation_environment: Option<Address>,
    continuation_point: Option<ProgramCounter>,
    number_of_active_permanent_variables: Arity,
    number_of_permanent_variables: Arity,
}

impl Environment {
    fn encode_head(
        &self,
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
    ) -> Result<TupleAddress, TupleMemoryError> {
        let Self {
            continuation_environment,
            continuation_point,
            number_of_active_permanent_variables,
            number_of_permanent_variables,
        } = *self;

        tuple_memory.encode(
            tuple_address,
            (
                registry_address,
                (
                    continuation_environment,
                    (
                        continuation_point,
                        SingleEntry((
                            number_of_active_permanent_variables,
                            number_of_permanent_variables,
                        )),
                    ),
                ),
            ),
        )
    }
}

impl Tuple for Environment {
    const VALUE_TYPE: ValueType = ValueType::Environment;
    type InitialTerms = [Address];
    type Terms = TermSlice;

    fn decode(
        tuple_memory: &TupleMemory,
        tuple_address: TupleAddress,
    ) -> Result<(Self, TupleMetadata<Self>), TupleMemoryError> {
        let (
            (
                registry_address,
                (
                    continuation_environment,
                    (
                        continuation_point,
                        SingleEntry((
                            number_of_active_permanent_variables,
                            number_of_permanent_variables,
                        )),
                    ),
                ),
            ),
            first_term,
        ) = tuple_memory.decode(tuple_address)?;

        Ok((
            Self {
                continuation_environment,
                continuation_point,
                number_of_active_permanent_variables,
                number_of_permanent_variables,
            },
            TupleMetadata {
                registry_address,
                terms: TermSlice {
                    first_term,
                    arity: number_of_permanent_variables,
                },
                next_tuple: first_term + number_of_permanent_variables,
            },
        ))
    }

    fn encode(
        &self,
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
        terms: &Self::InitialTerms,
    ) -> Result<TupleAddress, TupleMemoryError> {
        let first_term = self.encode_head(registry_address, tuple_memory, tuple_address)?;

        for (tuple_address, value) in first_term.iter(self.number_of_permanent_variables).zip(
            terms
                .iter()
                .copied()
                .chain(core::iter::repeat(unsafe { Address::none() })),
        ) {
            tuple_memory.store(tuple_address, value.0)?;
        }

        Ok(first_term + self.number_of_active_permanent_variables)
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

impl ChoicePoint {
    fn encode_head(
        &self,
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
    ) -> Result<TupleAddress, TupleMemoryError> {
        let Self {
            number_of_saved_registers,
            current_environment,
            continuation_point,
            next_choice_point,
            next_clause,
            trail_top,
            cut_register,
        } = *self;

        tuple_memory.encode(
            tuple_address,
            (
                registry_address,
                (
                    (MustBeZero, number_of_saved_registers),
                    (
                        current_environment,
                        (
                            continuation_point,
                            (
                                next_choice_point,
                                (next_clause, (trail_top, SingleEntry(cut_register))),
                            ),
                        ),
                    ),
                ),
            ),
        )
    }
}

impl Tuple for ChoicePoint {
    const VALUE_TYPE: ValueType = ValueType::ChoicePoint;
    type InitialTerms = [Address];
    type Terms = TermSlice;

    fn decode(
        tuple_memory: &TupleMemory,
        tuple_address: TupleAddress,
    ) -> Result<(Self, TupleMetadata<Self>), TupleMemoryError> {
        let (
            (
                registry_address,
                (
                    (MustBeZero, number_of_saved_registers),
                    (
                        current_environment,
                        (
                            continuation_point,
                            (
                                next_choice_point,
                                (next_clause, (trail_top, SingleEntry(cut_register))),
                            ),
                        ),
                    ),
                ),
            ),
            first_term,
        ) = tuple_memory.decode(tuple_address)?;

        let arity = number_of_saved_registers;
        let next_tuple = first_term + arity;

        Ok((
            Self {
                number_of_saved_registers,
                current_environment,
                continuation_point,
                next_choice_point,
                next_clause,
                trail_top,
                cut_register,
            },
            TupleMetadata {
                registry_address,
                terms: TermSlice { first_term, arity },
                next_tuple,
            },
        ))
    }

    fn encode(
        &self,
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
        saved_registers: &[Address],
    ) -> Result<TupleAddress, TupleMemoryError> {
        assert_eq!(
            self.number_of_saved_registers.0 as usize,
            saved_registers.len()
        );

        let first_term = self.encode_head(registry_address, tuple_memory, tuple_address)?;

        for (term_address, &value) in first_term
            .iter(self.number_of_saved_registers)
            .zip(saved_registers)
        {
            tuple_memory.store(term_address, value.0)?;
        }

        Ok(first_term + self.number_of_saved_registers)
    }
}

#[derive(Debug)]
struct TrailVariable {
    variable: Address,
    next_trail_item: Option<Address>,
}

impl Tuple for TrailVariable {
    const VALUE_TYPE: ValueType = ValueType::TrailVariable;
    type InitialTerms = ();
    type Terms = NoTerms;

    fn decode(
        tuple_memory: &TupleMemory,
        tuple_address: TupleAddress,
    ) -> Result<(Self, TupleMetadata<Self>), TupleMemoryError> {
        let ((registry_address, (variable, SingleEntry(next_trail_item))), next_tuple) =
            tuple_memory.decode(tuple_address)?;

        Ok((
            Self {
                variable,
                next_trail_item,
            },
            TupleMetadata {
                registry_address,
                terms: NoTerms,
                next_tuple,
            },
        ))
    }

    fn encode(
        &self,
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
        _: &(),
    ) -> Result<TupleAddress, TupleMemoryError> {
        let Self {
            variable,
            next_trail_item,
        } = *self;

        tuple_memory.encode(
            tuple_address,
            (registry_address, (variable, SingleEntry(next_trail_item))),
        )
    }
}

#[derive(Debug)]
pub enum Value {
    Reference(Address),
    Structure(Functor, Arity),
    List,
    Constant(Constant),
}

impl Value {
    fn reference(ReferenceValue(address): ReferenceValue) -> Self {
        Self::Reference(address)
    }

    fn structure(StructureValue(f, n): StructureValue) -> Self {
        Self::Structure(f, n)
    }

    fn list(_: ListValue) -> Self {
        Self::List
    }

    fn constant(ConstantValue(c): ConstantValue) -> Self {
        Self::Constant(c)
    }
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
    TupleMemory(TupleMemoryError),
    OutOfRegistryEntries,
    OutOfTupleSpace,
}

impl From<TupleMemoryError> for MemoryError {
    fn from(inner: TupleMemoryError) -> Self {
        Self::TupleMemory(inner)
    }
}

pub struct Heap<'m> {
    registry: Registry<'m>,
    tuple_memory: TupleMemory<'m>,
    tuple_memory_end: TupleAddress,
    current_environment: Option<Address>,
    latest_choice_point: Option<Address>,
    trail_top: Option<Address>,
    cut_register: Option<Address>,
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
        }
    }

    fn new_value<T: Tuple + core::fmt::Debug>(
        &mut self,
        factory: impl FnOnce(Address) -> T,
        terms: &T::InitialTerms,
    ) -> Result<Address, MemoryError> {
        let (address, registry_entry) = self
            .registry
            .new_registry_entry()
            .ok_or(MemoryError::OutOfRegistryEntries)?;

        let tuple_address = self.tuple_memory_end;

        let value = factory(address);

        crate::log_trace!("New value at {}: {:?}", tuple_address, value);

        self.tuple_memory_end = value
            .encode(address, &mut self.tuple_memory, tuple_address, terms)
            .map_err(|_| MemoryError::OutOfTupleSpace)?;

        crate::log_trace!("h = {}", self.tuple_memory_end);

        *registry_entry = Some(RegistryEntry {
            value_type: T::VALUE_TYPE,
            tuple_address,
        });

        Ok(address)
    }

    pub fn new_variable(&mut self) -> Result<Address, MemoryError> {
        self.new_value(ReferenceValue, &())
    }

    pub fn new_structure(&mut self, f: Functor, n: Arity) -> Result<Address, MemoryError> {
        self.new_value(|_| StructureValue(f, n), &())
    }

    pub fn new_list(&mut self) -> Result<Address, MemoryError> {
        self.new_value(|_| ListValue, &())
    }

    pub fn new_constant(&mut self, c: Constant) -> Result<Address, MemoryError> {
        self.new_value(|_| ConstantValue(c), &())
    }

    pub fn get_value(
        &self,
        mut address: Address,
    ) -> Result<(Address, Value, impl Iterator<Item = Address> + '_), MemoryError> {
        loop {
            crate::log_trace!("Looking up memory at {}", address);
            let registry_entry = self.registry[address]
                .as_ref()
                .expect("Uninitialized Entry");

            assert_eq!(
                address,
                Address(self.tuple_memory.load(registry_entry.tuple_address)?)
            );

            let (address, value, terms) = match &registry_entry.value_type {
                ValueType::Reference => {
                    let (value, metadata) =
                        ReferenceValue::decode(&self.tuple_memory, registry_entry.tuple_address)?;

                    let ReferenceValue(reference_address) = value;

                    if reference_address != address {
                        address = reference_address;
                        continue;
                    }

                    (
                        metadata.registry_address,
                        Value::reference(value),
                        metadata.terms.into_maybe_address_range(),
                    )
                }
                ValueType::Structure => {
                    let (value, metadata) =
                        StructureValue::decode(&self.tuple_memory, registry_entry.tuple_address)?;

                    (
                        metadata.registry_address,
                        Value::structure(value),
                        metadata.terms.into_maybe_address_range(),
                    )
                }
                ValueType::List => {
                    let (value, metadata) =
                        ListValue::decode(&self.tuple_memory, registry_entry.tuple_address)?;
                    (
                        metadata.registry_address,
                        Value::list(value),
                        metadata.terms.into_maybe_address_range(),
                    )
                }
                ValueType::Constant => {
                    let (value, metadata) =
                        ConstantValue::decode(&self.tuple_memory, registry_entry.tuple_address)?;
                    (
                        metadata.registry_address,
                        Value::constant(value),
                        metadata.terms.into_maybe_address_range(),
                    )
                }
                ValueType::Environment | ValueType::ChoicePoint | ValueType::TrailVariable => {
                    panic!("Expected value, found {:?}", registry_entry.value_type);
                }
            };

            crate::log_trace!("Value: {:?}", value);

            let terms = terms
                .map(|terms| self.tuple_memory.load_terms(terms))
                .transpose()?
                .into_iter()
                .flatten();

            break Ok((address, value, terms));
        }
    }

    fn structure_term_addresses(&self, address: Address) -> TermSlice {
        let registry_entry = self.registry[address]
            .as_ref()
            .expect("Structure not found");

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
        let current_environment = self
            .current_environment
            .unwrap_or_else(|| panic!("No Environment"));

        let entry = self.registry[current_environment]
            .as_ref()
            .expect("No Entry");
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

    unsafe fn load_permanent_variable_unchecked(
        &self,
        yn: Yn,
    ) -> Result<Address, PermanentVariableError> {
        let term_address = self.get_permanent_variable_address(yn)?;

        Ok(Address(self.tuple_memory.load(term_address).map_err(
            |inner| PermanentVariableError::TupleMemoryError { yn, inner },
        )?))
    }

    pub fn load_permanent_variable(&self, yn: Yn) -> Result<Address, PermanentVariableError> {
        let address = unsafe { self.load_permanent_variable_unchecked(yn)? };
        if address == unsafe { Address::none() } {
            return Err(PermanentVariableError::NoValue { yn });
        }

        Ok(address)
    }

    pub fn store_permanent_variable(
        &mut self,
        yn: Yn,
        address: Address,
    ) -> Result<(), PermanentVariableError> {
        let term_address = self.get_permanent_variable_address(yn)?;

        self.tuple_memory
            .store(term_address, address.0)
            .map_err(|inner| PermanentVariableError::TupleMemoryError { inner, yn })
    }

    fn verify_is_reference(&self, address: Address) -> (ReferenceValue, TupleAddress) {
        let entry = self.registry[address].as_ref().expect("No Entry");

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
        crate::log_trace!(
            "Binding memory {} to value at {}",
            variable_address,
            value_address
        );

        let tuple_address = self.verify_is_free_variable(variable_address);

        ReferenceValue(value_address).encode(
            variable_address,
            &mut self.tuple_memory,
            tuple_address,
            &(),
        )?;

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
        terms: &[Address],
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

        crate::log_trace!("E => {}", OptionDisplay(self.current_environment));
        crate::log_trace!("CP => {}", OptionDisplay(*continuation_point));
    }

    pub fn new_choice_point(
        &mut self,
        next_clause: ProgramCounter,
        continuation_point: Option<ProgramCounter>,
        saved_registers: &[Address],
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

        let entry = self.registry[latest_choice_point]
            .as_ref()
            .expect("No Entry");
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
        registers: &mut [Address],
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
            *register = Address(self.tuple_memory.load(value_address).unwrap());
        }

        self.current_environment = choice_point.current_environment;
        *continuation_point = choice_point.continuation_point;

        self.unwind_trail(tuple_address);

        self.trail_top = choice_point.trail_top;
        let next_choice_point = choice_point.next_choice_point;

        crate::log_trace!("E => {}", OptionDisplay(self.current_environment));
        crate::log_trace!("CP => {}", OptionDisplay(*continuation_point));

        (
            choice_point,
            metadata.registry_address,
            tuple_address,
            next_choice_point,
        )
    }

    pub fn retry_choice_point(
        &mut self,
        registers: &mut [Address],
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
        registers: &mut [Address],
        continuation_point: &mut Option<ProgramCounter>,
    ) {
        let next_choice_point = self
            .wind_back_to_choice_point(registers, continuation_point)
            .3;

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
            &(),
        )?;
        self.trail_top = Some(new_trail_item);

        Ok(())
    }

    fn unwind_trail(&mut self, boundary: TupleAddress) {
        log_trace!("Unwinding Trail");
        while let Some(trail_top) = self.trail_top {
            let entry = self.registry[trail_top].as_ref().expect("No Entry");
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
                    &(),
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

    fn do_cut(&mut self, cut_register: Option<Address>) {
        log_trace!("Cutting at {}", OptionDisplay(cut_register));

        let new_choice_point = match (self.latest_choice_point, cut_register) {
            (None, None) => None,
            (Some(_), None) => None,
            (None, Some(_)) => {
                log_trace!("Not cutting");
                return;
            }
            (Some(latest_choice_point), Some(cut_register)) => {
                let latest_choice_point_entry = self.registry[latest_choice_point]
                    .as_ref()
                    .unwrap_or_else(|| panic!("No choice point entry"));

                let cut_register_entry = self.registry[cut_register]
                    .as_ref()
                    .unwrap_or_else(|| panic!("No cut register entry"));

                assert!(matches!(
                    latest_choice_point_entry.value_type,
                    ValueType::ChoicePoint
                ));

                assert!(matches!(
                    cut_register_entry.value_type,
                    ValueType::ChoicePoint
                ));

                if latest_choice_point_entry.tuple_address > cut_register_entry.tuple_address {
                    Some(cut_register)
                } else {
                    log_trace!("Not cutting");
                    return;
                }
            }
        };

        log_trace!(
            "Latest Choice Point: {} => {}",
            OptionDisplay(self.latest_choice_point),
            OptionDisplay(new_choice_point)
        );

        self.latest_choice_point = new_choice_point;
    }

    pub fn neck_cut(&mut self) {
        self.do_cut(self.cut_register)
    }

    pub fn get_level(&mut self, yn: Yn) -> Result<(), PermanentVariableError> {
        self.store_permanent_variable(yn, self.cut_register.unwrap_or(unsafe { Address::none() }))
    }

    pub fn cut(&mut self, yn: Yn) -> Result<(), PermanentVariableError> {
        let address = unsafe { self.load_permanent_variable_unchecked(yn)? };

        self.do_cut(if address == unsafe { Address::none() } {
            None
        } else {
            Some(address)
        });

        Ok(())
    }

    pub fn query_has_multiple_solutions(&self) -> bool {
        self.latest_choice_point.is_some()
    }

    pub fn solution_registers(
        &self,
    ) -> Result<impl Iterator<Item = Address> + '_, TupleMemoryError> {
        let (environment, _, metadata) = self.get_environment();
        assert!(environment.continuation_environment.is_none());

        self.tuple_memory
            .load_terms(metadata.terms.into_address_range())
    }
}
