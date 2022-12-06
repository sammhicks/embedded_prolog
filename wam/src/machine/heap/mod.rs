use core::{fmt, num::NonZeroU16};

use crate::{log_trace, serializable::SerializableWrapper};

use super::basic_types::{
    self, Arity, Constant, Functor, LongInteger, NoneRepresents, OptionDisplay, ProgramCounter, Yn,
};

mod integer_operations;
pub mod structure_iteration;

type IntegerSign = core::cmp::Ordering;

enum SpecialFunctor {
    BinaryOperation {
        calculate_words_count: fn(TupleAddress, TupleAddress) -> TupleAddress,
        operation: fn(
            (
                integer_operations::UnsignedOutput,
                integer_operations::SignedInput,
                integer_operations::SignedInput,
            ),
        ) -> IntegerSign,
    },
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
            // Addition
            (Functor(0), Arity(2)) => Self::BinaryOperation {
                calculate_words_count: |a, b| TupleAddress::max(a, b) + 1,
                operation: integer_operations::add_signed,
            },
            // Subtraction
            (Functor(1), Arity(2)) => Self::BinaryOperation {
                calculate_words_count: |a, b| TupleAddress::max(a, b) + 1,
                operation: integer_operations::sub_signed,
            },
            // Multiplication
            (Functor(2), Arity(2)) => Self::BinaryOperation {
                calculate_words_count: |a, b| a + b + 1,
                operation: integer_operations::mul_signed,
            },
            // Min
            (Functor(4), Arity(2)) => Self::MinMax {
                select_first_if: |(a, b)| a < b,
            },
            // Max
            (Functor(5), Arity(2)) => Self::MinMax {
                select_first_if: |(a, b)| a > b,
            },
            _ => return Err(ExpressionEvaluationError::BadStructure(f, n)),
        })
    }
}

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
    Integer,
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
    ) -> Result<
        impl core::iter::FusedIterator + Iterator<Item = Option<Address>> + '_,
        TupleMemoryError,
    > {
        Ok(match terms {
            Some(terms) => self.get(terms)?,
            None => &[],
        }
        .iter()
        .map(|&word| Address::from_word(word)))
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
    unsafe fn get_integer_output_input_input(
        &mut self,
        w0: IntegerWordsSlice,
        w1: (IntegerSign, IntegerWordsSlice),
        w2: (IntegerSign, IntegerWordsSlice),
    ) -> (
        integer_operations::UnsignedOutput,
        integer_operations::SignedInput,
        integer_operations::SignedInput,
    ) {
        let w0 = integer_operations::UnsignedOutput::new(core::slice::from_raw_parts_mut(
            self.0.as_mut_ptr().offset(w0.data_start.0 as isize),
            w0.words_count.0.into(),
        ));
        let (w1, w2) = self.get_integer_input_input(w1, w2);
        (w0, w1, w2)
    }
}

struct NoData;

impl NoData {
    fn into_iter<I: Iterator<Item = u8>>(self) -> DataIterator<I> {
        DataIterator::NoData
    }
}

trait BaseTupleInitialData<Data> {
    fn encode(&self, tuple_memory: &mut [TupleWord]);
}

impl BaseTupleInitialData<NoData> for NoData {
    fn encode(&self, _tuple_memory: &mut [TupleWord]) {}
}

trait BaseTupleData {
    fn from_range<T: Tuple<Data = Self>>(tuple: &T, data_start: TupleAddress) -> Self;
}

impl BaseTupleData for NoData {
    fn from_range<T: Tuple<Data = Self>>(_: &T, _: TupleAddress) -> Self {
        Self
    }
}

struct TermsSlice {
    first_term: TupleAddress,
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
    fn from_range<T: Tuple<Terms = Self>>(tuple: &T, first_term: TupleAddress) -> Self;
    fn into_address_range(self) -> Option<core::ops::Range<TupleAddress>>;
}

impl BaseTupleTerms for NoTerms {
    fn from_range<T: Tuple<Terms = Self>>(_: &T, _: TupleAddress) -> Self {
        Self
    }

    fn into_address_range(self) -> Option<core::ops::Range<TupleAddress>> {
        None
    }
}

impl BaseTupleTerms for TermsSlice {
    fn from_range<T: Tuple<Terms = Self>>(tuple: &T, first_term: TupleAddress) -> Self {
        Self {
            first_term,
            terms_count: tuple.terms_count(),
        }
    }

    fn into_address_range(self) -> Option<core::ops::Range<TupleAddress>> {
        let TermsSlice {
            first_term,
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
}

impl<T: BaseTuple<Terms = NoTerms>> TupleTermsInfo for T {
    fn terms_count(&self) -> Arity {
        Arity(0)
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

struct TupleEndInfo {
    next_free_space: TupleAddress,
    next_tuple: TupleAddress,
}

struct TupleMetadata<T: Tuple> {
    registry_address: Address,
    data: T::Data,
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

trait Tuple: BaseTuple + TupleDataInfo + TupleTermsInfo {
    fn decode(
        tuple_memory: &TupleMemory,
        tuple_address: TupleAddress,
    ) -> Result<(Self, TupleMetadata<Self>), TupleMemoryError> {
        let (
            AddressWithTuple {
                registry_address,
                tuple: DirectAccess { tuple },
            },
            data_start,
        ) = AddressWithTuple::<Self>::decode(tuple_memory, tuple_address)?;

        let data_size = tuple.data_size();
        let data = <Self::Data>::from_range(&tuple, data_start);

        let terms_count = tuple.terms_count();
        let terms_size = tuple.terms_size();
        let terms_start = data_start + data_size;
        let terms = <Self::Terms>::from_range(&tuple, terms_start);

        Ok((
            tuple,
            TupleMetadata {
                registry_address,
                terms,
                data,
                next_free_space: terms_start + terms_count,
                next_tuple: terms_start + terms_size,
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
    ) -> Result<TupleAddress, TupleMemoryError> {
        let terms_count = self.terms_count();
        let terms_size = self.terms_size();
        let data_size = self.data_size();

        let terms_start = self.encode_head(registry_address, tuple_memory, tuple_address)?;
        let terms_end = terms_start + terms_count;
        terms.encode(tuple_memory.get_mut(terms_start..terms_end)?);

        let data_start = terms_start + terms_size;
        let data_end = data_start + data_size;
        data.encode(tuple_memory.get_mut(data_start..data_end)?);
        Ok(data_end)
    }
}

impl<T: BaseTuple + TupleTermsInfo + TupleDataInfo> Tuple for T {}

#[derive(Debug)]
struct ReferenceValue(Address);

impl BaseTuple for ReferenceValue {
    const VALUE_TYPE: ValueType = ValueType::Reference;
    type InitialData<'a> = NoData;
    type Data = NoData;
    type InitialTerms<'a> = NoTerms;
    type Terms = NoTerms;
}

#[derive(Debug)]
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
struct ConstantValue(Constant);

impl BaseTuple for ConstantValue {
    const VALUE_TYPE: ValueType = ValueType::Constant;
    type InitialData<'a> = NoData;
    type Data = NoData;
    type InitialTerms<'a> = NoTerms;
    type Terms = NoTerms;
}

struct FillWithZero;

impl BaseTupleInitialData<IntegerWordsSlice> for FillWithZero {
    fn encode(&self, tuple_memory: &mut [TupleWord]) {
        for slot in tuple_memory {
            *slot = 0;
        }
    }
}

struct IntegerWords<'a> {
    words: &'a [u32],
}

impl<'a> BaseTupleInitialData<IntegerWordsSlice> for IntegerWords<'a> {
    fn encode(&self, tuple_memory: &mut [TupleWord]) {
        let words = self.words.iter().rev().flat_map(|&word| {
            let [a, b, c, d] = word.to_le_bytes();
            [u16::from_le_bytes([a, b]), u16::from_le_bytes([c, d])]
        });

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
    fn from_range<T: Tuple<Data = Self>>(tuple: &T, data_start: TupleAddress) -> Self {
        Self {
            data_start,
            words_count: tuple.data_size(),
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
struct IntegerValue {
    sign: IntegerSign,
    words_count: TupleAddress,
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
}

impl BaseTuple for IntegerValue {
    const VALUE_TYPE: ValueType = ValueType::Integer;
    type InitialData<'a> = IntegerWords<'a>;
    type Data = IntegerWordsSlice;
    type InitialTerms<'a> = NoTerms;
    type Terms = NoTerms;
}

#[derive(Debug)]
#[repr(transparent)]
struct IntegerEvaluationOutput(IntegerValue);

impl TupleDataInfo for IntegerEvaluationOutput {
    fn data_size(&self) -> TupleAddress {
        self.0.data_size()
    }
}

impl BaseTuple for IntegerEvaluationOutput {
    const VALUE_TYPE: ValueType = IntegerValue::VALUE_TYPE;
    type InitialData<'a> = FillWithZero;
    type Data = <IntegerValue as BaseTuple>::Data;
    type InitialTerms<'a> = <IntegerValue as BaseTuple>::InitialTerms<'a>;
    type Terms = <IntegerValue as BaseTuple>::Terms;
}

#[derive(Debug)]
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
}

impl BaseTuple for Environment {
    const VALUE_TYPE: ValueType = ValueType::Environment;
    type InitialData<'a> = NoData;
    type Data = NoData;
    type InitialTerms<'a> = &'a [Option<Address>];
    type Terms = TermsSlice;
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
pub enum Value {
    Structure(Functor, Arity),
    List,
    Constant(Constant),
    Integer {
        sign: basic_types::IntegerSign,
        bytes_count: u32,
    },
}

#[derive(Debug)]
pub enum ReferenceOrValue {
    Reference(Address),
    Value(Value),
}

#[derive(Clone)]
enum DataIterator<I: Iterator<Item = u8>> {
    NoData,
    Integer(I),
}

impl<I: Iterator<Item = u8>> Iterator for DataIterator<I> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::NoData => None,
            Self::Integer(i) => i.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::NoData => (0, Some(0)),
            Self::Integer(i) => i.size_hint(),
        }
    }
}

impl<I: DoubleEndedIterator + Iterator<Item = u8>> DoubleEndedIterator for DataIterator<I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            Self::NoData => None,
            Self::Integer(i) => i.next_back(),
        }
    }
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

#[derive(Debug)]
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
        data: T::InitialData<'_>,
    ) -> Result<Result<Address, OutOfMemory>, MemoryError> {
        let Some((address, registry_entry)) = self.registry.new_registry_entry() else {
            return Ok(Err(OutOfMemory::OutOfRegistryEntries));
        };

        let tuple_address = self.tuple_memory_end;

        let value = factory(address);

        log_trace!("New value at {}: {:?}", tuple_address, value);

        let Ok(memory_end) = value.encode(address, &mut self.tuple_memory, tuple_address, terms, data) else {
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
        self.new_value(ReferenceValue, NoTerms, NoData)
    }

    pub fn new_structure(
        &mut self,
        f: Functor,
        n: Arity,
    ) -> Result<Result<Address, OutOfMemory>, MemoryError> {
        self.new_value(|_| StructureValue(f, n), FillWithNone, NoData)
    }

    pub fn new_list(&mut self) -> Result<Result<Address, OutOfMemory>, MemoryError> {
        self.new_value(|_| ListValue, FillWithNone, NoData)
    }

    pub fn new_constant(
        &mut self,
        c: Constant,
    ) -> Result<Result<Address, OutOfMemory>, MemoryError> {
        self.new_value(|_| ConstantValue(c), NoTerms, NoData)
    }

    pub fn new_integer(
        &mut self,
        LongInteger { sign, words }: LongInteger<'_>,
    ) -> Result<Result<Address, OutOfMemory>, MemoryError> {
        self.new_value(
            |_| {
                if words.iter().copied().all(|word| word == 0) {
                    IntegerValue {
                        sign: IntegerSign::Equal,
                        words_count: TupleAddress(0),
                    }
                } else {
                    IntegerValue {
                        sign: match sign {
                            basic_types::IntegerSign::Positive => IntegerSign::Greater,
                            basic_types::IntegerSign::Negative => IntegerSign::Less,
                        },
                        words_count: IntegerValue::words_count(words),
                    }
                }
            },
            NoTerms,
            IntegerWords { words },
        )
    }

    fn new_integer_output(
        &mut self,
        words_count: TupleAddress,
    ) -> Result<Result<Address, OutOfMemory>, MemoryError> {
        self.new_value(
            |_| {
                IntegerEvaluationOutput(IntegerValue {
                    sign: IntegerSign::Equal,
                    words_count,
                })
            },
            NoTerms,
            FillWithZero,
        )
    }

    pub fn get_maybe_value(
        &self,
        address: Option<Address>,
    ) -> Result<
        (
            Address,
            ReferenceOrValue,
            impl Iterator<Item = u8> + '_,
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
            impl DoubleEndedIterator + Iterator<Item = u8> + '_,
            impl Iterator<Item = Option<Address>> + '_,
        ),
        MemoryError,
    > {
        loop {
            log_trace!("Looking up memory at {}", address);
            let registry_entry = self.registry.get(address)?;

            let (address, value, data, terms) = match &registry_entry.value_type {
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
                        metadata.data.into_iter(),
                        metadata.terms.into_address_range(),
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
                        metadata.data.into_iter(),
                        metadata.terms.into_address_range(),
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
                        metadata.data.into_iter(),
                        metadata.terms.into_address_range(),
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
                        metadata.data.into_iter(),
                        metadata.terms.into_address_range(),
                    )
                }
                ValueType::Integer => {
                    let (IntegerValue { sign, words_count }, metadata) =
                        IntegerValue::decode_and_verify_address(
                            &self.tuple_memory,
                            address,
                            registry_entry.tuple_address,
                        )?;

                    let sign = match sign {
                        IntegerSign::Greater | IntegerSign::Equal => {
                            basic_types::IntegerSign::Positive
                        }
                        IntegerSign::Less => basic_types::IntegerSign::Negative,
                    };

                    let bytes_count =
                        u32::from(words_count.0) * (core::mem::size_of::<TupleWord>() as u32);

                    (
                        metadata.registry_address,
                        ReferenceOrValue::Value(Value::Integer { sign, bytes_count }),
                        DataIterator::Integer(
                            self.tuple_memory
                                .get(metadata.data.into_address_range())?
                                .iter()
                                .rev()
                                .copied()
                                .flat_map(u16::to_be_bytes),
                        ),
                        metadata.terms.into_address_range(),
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

            break Ok((address, value, data, self.tuple_memory.load_terms(terms)?));
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
                NoData,
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
            address: AddressView(address.into_inner()),
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
                        SpecialFunctor::BinaryOperation {
                            calculate_words_count,
                            operation,
                        } => {
                            let mut terms = self
                                .tuple_memory
                                .load_terms(metadata.terms.into_address_range())?;

                            let a1 = terms.next().flatten();
                            let a2 = terms.next().flatten();

                            drop(terms);

                            let (_, s1, w1) = self.do_evaluate(a1)?;
                            let (_, s2, w2) = self.do_evaluate(a2)?;

                            let w0_words_count =
                                calculate_words_count(w1.words_count, w2.words_count);

                            let a0 = self.new_integer_output(w0_words_count)??;
                            let a0_entry = self.registry.get(a0)?;
                            let (IntegerValue { .. }, a0_metadata) =
                                IntegerValue::decode_and_verify_address(
                                    &self.tuple_memory,
                                    a0,
                                    a0_entry.tuple_address,
                                )?;

                            let w0 = a0_metadata.data;

                            // Safety: w0 corresponds to a newly created integer output, so has a unique address
                            let s0 = unsafe {
                                operation(self.tuple_memory.get_integer_output_input_input(
                                    w0,
                                    (s1, w1),
                                    (s2, w2),
                                ))
                            };

                            IntegerValue {
                                sign: s0,
                                words_count: w0_words_count,
                            }
                            .encode_head(
                                a0,
                                &mut self.tuple_memory,
                                a0_entry.tuple_address,
                            )?;

                            (a0, s0, w0)
                        }
                        SpecialFunctor::MinMax { select_first_if } => {
                            let mut terms = self
                                .tuple_memory
                                .load_terms(metadata.terms.into_address_range())?;

                            let a1 = terms.next().flatten();
                            let a2 = terms.next().flatten();

                            drop(terms);

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

        // let (address, value, data, terms) = self.get_value(address)?;

        // let f = match value {
        //     ReferenceOrValue::Reference(_) => {
        //         return Err(ExpressionEvaluationError::UnboundVariable(address))
        //     }
        //     ReferenceOrValue::Value(Value::Integer {sign,..}) => return Ok(Ok((address, sign, ))),
        //     ReferenceOrValue::Value(value @ (Value::List | Value::Constant(_))) => {
        //         return Err(ExpressionEvaluationError::NotAValidValue(address, value))
        //     }
        //     ReferenceOrValue::Value(Value::Structure(f, n)) => {
        //         SpecialFunctorWithArity::try_from((f, n))?
        //     }
        // };

        // match (f, n) {
        //     (SpecialFunctor::Add, _) =>
        // }
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

        for term_address in self.tuple_memory.load_terms(term_address_range)? {
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
            ValueType::Integer => IntegerValue::decode(&self.tuple_memory, source)?
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
