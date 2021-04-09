use core::{fmt, panic};

use crate::{
    log_trace,
    serializable::{Serializable, SerializableWrapper},
};

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
    Trail,
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

impl From<u32> for Address {
    fn from(word: u32) -> Self {
        Self(std::convert::TryInto::try_into(word).unwrap_or_else(|_| panic!("Bad word: {}", word)))
    }
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

impl Serializable for Option<Address> {
    type Bytes = <Address as Serializable>::Bytes;

    fn from_be_bytes(bytes: Self::Bytes) -> Self {
        let address = Address::from_be_bytes(bytes);
        if address.0 == Address::NO_ADDRESS {
            None
        } else {
            Some(address)
        }
    }

    fn into_be_bytes(self) -> Self::Bytes {
        self.unwrap_or(Address(Address::NO_ADDRESS)).into_be_bytes()
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
    const NULL: Self = Self(u16::MAX);

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
    fn new_registry_entry(&mut self) -> (Address, &mut Option<RegistryEntry>) {
        let (index, slot) = self
            .0
            .iter_mut()
            .enumerate()
            .find(|(_, entry)| entry.is_none())
            .expect("No more registry entries");

        let address = Address(core::convert::TryInto::try_into(index).expect("Invalid index"));

        (address, slot)
    }
}

impl<'m> std::ops::Index<Address> for Registry<'m> {
    type Output = Option<RegistryEntry>;

    fn index(&self, index: Address) -> &Self::Output {
        &self.0[index.0 as usize]
    }
}

struct TupleMemory<'m>(&'m mut [u32]);

impl<'m> std::ops::Index<TupleAddress> for TupleMemory<'m> {
    type Output = u32;

    fn index(&self, index: TupleAddress) -> &Self::Output {
        &self.0[index.0 as usize]
    }
}

impl<'m> std::ops::IndexMut<TupleAddress> for TupleMemory<'m> {
    fn index_mut(&mut self, index: TupleAddress) -> &mut Self::Output {
        &mut self.0[index.0 as usize]
    }
}

struct TupleMetadata {
    registry_address: Address,
    first_term: TupleAddress,
    arity: Arity,
    next_tuple: TupleAddress,
}

trait Tuple: Sized {
    const VALUE_TYPE: ValueType;
    type Terms: ?Sized;

    fn decode(tuple_memory: &TupleMemory, tuple_address: TupleAddress) -> (Self, TupleMetadata);

    fn encode(
        &self,
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
        terms: &Self::Terms,
    ) -> TupleAddress;
}

#[derive(Debug)]
struct ReferenceValue(Address);

impl Tuple for ReferenceValue {
    const VALUE_TYPE: ValueType = ValueType::Reference;
    type Terms = ();

    fn decode(tuple_memory: &TupleMemory, tuple_address: TupleAddress) -> (Self, TupleMetadata) {
        let (registry_address, r) =
            Serializable::from_be_bytes(tuple_memory[tuple_address].to_be_bytes());

        (
            Self(r),
            TupleMetadata {
                registry_address,
                first_term: TupleAddress::NULL,
                arity: Arity::ZERO,
                next_tuple: tuple_address + 1,
            },
        )
    }

    fn encode(
        &self,
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
        _: &(),
    ) -> TupleAddress {
        let Self(r) = *self;

        let [a1, a0] = registry_address.into_be_bytes();
        let [r1, r0] = r.into_be_bytes();

        tuple_memory[tuple_address] = u32::from_be_bytes([a1, a0, r1, r0]);

        tuple_address + 1
    }
}

#[derive(Debug)]
struct StructureValue(Functor, Arity);

impl Tuple for StructureValue {
    const VALUE_TYPE: ValueType = ValueType::Structure;
    type Terms = ();

    fn decode(tuple_memory: &TupleMemory, tuple_address: TupleAddress) -> (Self, TupleMetadata) {
        let [a1, a0, f1, f0] = tuple_memory[tuple_address].to_be_bytes();
        let [z2, z1, z0, n] = tuple_memory[tuple_address + 1].to_be_bytes();
        assert_eq!(z2, 0);
        assert_eq!(z1, 0);
        assert_eq!(z0, 0);

        let arity = Arity(n);
        let first_term = tuple_address + 2;
        (
            Self(Functor::from_be_bytes([f1, f0]), arity),
            TupleMetadata {
                registry_address: Address::from_be_bytes([a1, a0]),
                first_term,
                arity,
                next_tuple: first_term + arity,
            },
        )
    }

    fn encode(
        &self,
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
        _: &(),
    ) -> TupleAddress {
        let Self(f, n) = *self;

        let [a1, a0] = registry_address.into_be_bytes();
        let [f1, f0] = f.into_be_bytes();

        tuple_memory[tuple_address] = u32::from_be_bytes([a1, a0, f1, f0]);
        tuple_memory[tuple_address + 1] = u32::from_be_bytes([0, 0, 0, n.0]);

        let first_term: TupleAddress = tuple_address + 2;

        for term_address in first_term.iter(n) {
            tuple_memory[term_address] = u32::MAX;
        }

        first_term + n
    }
}

#[derive(Debug)]
struct ListValue;

impl ListValue {
    const ARITY: Arity = Arity(2);
}

impl Tuple for ListValue {
    const VALUE_TYPE: ValueType = ValueType::List;
    type Terms = ();

    fn decode(tuple_memory: &TupleMemory, tuple_address: TupleAddress) -> (Self, TupleMetadata) {
        let [a1, a0, z1, z0] = tuple_memory[tuple_address].to_be_bytes();
        assert_eq!(z1, 0);
        assert_eq!(z0, 0);

        let first_term = tuple_address + 1;
        (
            Self,
            TupleMetadata {
                registry_address: Address::from_be_bytes([a1, a0]),
                first_term,
                arity: Self::ARITY,
                next_tuple: first_term + Self::ARITY,
            },
        )
    }

    fn encode(
        &self,
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
        _: &(),
    ) -> TupleAddress {
        let [a1, a0] = registry_address.into_be_bytes();

        tuple_memory[tuple_address] = u32::from_be_bytes([a1, a0, 0, 0]);
        tuple_memory[tuple_address + 1] = u32::MAX;
        tuple_memory[tuple_address + 2] = u32::MAX;

        tuple_address + 3
    }
}

#[derive(Debug)]
struct ConstantValue(Constant);

impl Tuple for ConstantValue {
    const VALUE_TYPE: ValueType = ValueType::Constant;
    type Terms = ();

    fn decode(tuple_memory: &TupleMemory, tuple_address: TupleAddress) -> (Self, TupleMetadata) {
        let (registry_address, c) =
            Serializable::from_be_bytes(tuple_memory[tuple_address].to_be_bytes());
        (
            Self(c),
            TupleMetadata {
                registry_address,
                first_term: TupleAddress::NULL,
                arity: Arity(0),
                next_tuple: tuple_address + 1,
            },
        )
    }

    fn encode(
        &self,
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
        _: &(),
    ) -> TupleAddress {
        let Self(c) = *self;

        tuple_memory[tuple_address] = u32::from_be_bytes((registry_address, c).into_be_bytes());

        tuple_address + 1
    }
}

struct FirstEnvironmentWord {
    registry_address: Address,
    continuation_environment: Option<Address>,
}

impl FirstEnvironmentWord {
    fn decode(tuple_memory: &TupleMemory, tuple_address: TupleAddress) -> Self {
        let (registry_address, continuation_environment) =
            Serializable::from_be_bytes(tuple_memory[tuple_address].to_be_bytes());

        Self {
            registry_address,
            continuation_environment,
        }
    }

    fn encode(self, tuple_memory: &mut TupleMemory, tuple_address: TupleAddress) {
        let Self {
            registry_address,
            continuation_environment,
        } = self;

        tuple_memory[tuple_address] =
            u32::from_be_bytes((registry_address, continuation_environment).into_be_bytes());
    }
}

struct SecondEnvironmentWord {
    continuation_point: Option<ProgramCounter>,
    number_of_active_permanent_variables: Arity,
    number_of_permanent_variables: Arity,
}

impl SecondEnvironmentWord {
    fn decode(tuple_memory: &TupleMemory, tuple_address: TupleAddress) -> Self {
        let [cp1, cp0, na, np] = tuple_memory[tuple_address + 1].to_be_bytes();

        Self {
            continuation_point: Option::<ProgramCounter>::from_be_bytes([cp1, cp0]),
            number_of_active_permanent_variables: Arity::from_be_bytes([na]),
            number_of_permanent_variables: Arity::from_be_bytes([np]),
        }
    }

    fn encode(self, tuple_memory: &mut TupleMemory, tuple_address: TupleAddress) {
        let Self {
            continuation_point,
            number_of_active_permanent_variables,
            number_of_permanent_variables,
        } = self;

        let [cp1, cp0] = continuation_point.into_be_bytes();
        let [na] = number_of_active_permanent_variables.into_be_bytes();
        let [np] = number_of_permanent_variables.into_be_bytes();

        tuple_memory[tuple_address + 1] = u32::from_be_bytes([cp1, cp0, na, np]);
    }
}

#[derive(Debug)]
struct Environment {
    continuation_environment: Option<Address>,
    continuation_point: Option<ProgramCounter>,
    number_of_active_permanent_variables: Arity,
    number_of_permanent_variables: Arity,
}

impl Tuple for Environment {
    const VALUE_TYPE: ValueType = ValueType::Environment;
    type Terms = [Address];

    fn decode(tuple_memory: &TupleMemory, tuple_address: TupleAddress) -> (Self, TupleMetadata) {
        let FirstEnvironmentWord {
            registry_address,
            continuation_environment,
        } = FirstEnvironmentWord::decode(tuple_memory, tuple_address);

        let SecondEnvironmentWord {
            continuation_point,
            number_of_active_permanent_variables,
            number_of_permanent_variables,
        } = SecondEnvironmentWord::decode(tuple_memory, tuple_address);

        let first_term = tuple_address + 2;
        (
            Self {
                continuation_environment,
                continuation_point,
                number_of_active_permanent_variables,
                number_of_permanent_variables,
            },
            TupleMetadata {
                registry_address,
                first_term,
                arity: number_of_active_permanent_variables,
                next_tuple: first_term + number_of_permanent_variables,
            },
        )
    }

    fn encode(
        &self,
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
        terms: &[Address],
    ) -> TupleAddress {
        let Self {
            continuation_environment,
            continuation_point,
            number_of_active_permanent_variables,
            number_of_permanent_variables,
        } = *self;

        assert!(terms.is_empty() || (terms.len() == self.number_of_permanent_variables.0 as usize));

        FirstEnvironmentWord {
            registry_address,
            continuation_environment,
        }
        .encode(tuple_memory, tuple_address);

        SecondEnvironmentWord {
            continuation_point,
            number_of_active_permanent_variables,
            number_of_permanent_variables,
        }
        .encode(tuple_memory, tuple_address);

        let first_term_address: TupleAddress = tuple_address + 2;

        for (tuple_address, value) in first_term_address
            .iter(self.number_of_permanent_variables)
            .zip(
                terms
                    .iter()
                    .copied()
                    .chain(core::iter::repeat(unsafe { Address::none() })),
            )
        {
            tuple_memory[tuple_address] = value.0 as u32;
        }

        first_term_address + self.number_of_active_permanent_variables
    }
}

struct FirstChoicePointWord {
    registry_address: Address,
    number_of_saved_registers: Arity,
}

impl FirstChoicePointWord {
    fn decode(tuple_memory: &TupleMemory, tuple_address: TupleAddress) -> Self {
        let [a1, a0, z, n] = tuple_memory[tuple_address].to_be_bytes();

        assert_eq!(z, 0);

        Self {
            registry_address: Address::from_be_bytes([a1, a0]),
            number_of_saved_registers: Arity::from_be_bytes([n]),
        }
    }

    fn encode(self, tuple_memory: &mut TupleMemory, tuple_address: TupleAddress) {
        let Self {
            registry_address,
            number_of_saved_registers,
        } = self;

        let [a1, a0] = registry_address.into_be_bytes();
        let [n] = number_of_saved_registers.into_be_bytes();

        tuple_memory[tuple_address] = u32::from_be_bytes([a1, a0, 0, n]);
    }
}

struct SecondChoicePointWord {
    current_environment: Option<Address>,
    continuation_point: Option<ProgramCounter>,
}

impl SecondChoicePointWord {
    fn decode(tuple_memory: &TupleMemory, tuple_address: TupleAddress) -> Self {
        let (current_environment, continuation_point) =
            Serializable::from_be_bytes(tuple_memory[tuple_address + 1].to_be_bytes());

        Self {
            current_environment,
            continuation_point,
        }
    }

    fn encode(self, tuple_memory: &mut TupleMemory, tuple_address: TupleAddress) {
        let Self {
            current_environment,
            continuation_point,
        } = self;

        tuple_memory[tuple_address + 1] =
            u32::from_be_bytes((current_environment, continuation_point).into_be_bytes());
    }
}

struct ThirdChoicePointWord {
    next_choice_point: Option<Address>,
    next_clause: ProgramCounter,
}

impl ThirdChoicePointWord {
    fn decode(tuple_memory: &TupleMemory, tuple_address: TupleAddress) -> Self {
        let (next_choice_point, next_clause) =
            Serializable::from_be_bytes(tuple_memory[tuple_address + 2].to_be_bytes());

        Self {
            next_choice_point,
            next_clause,
        }
    }

    fn encode(self, tuple_memory: &mut TupleMemory, tuple_address: TupleAddress) {
        let Self {
            next_choice_point,
            next_clause,
        } = self;

        tuple_memory[tuple_address + 2] =
            u32::from_be_bytes((next_choice_point, next_clause).into_be_bytes());
    }
}

struct FourthChoicePointWord {
    trail_top: Option<Address>,
    cut_register: Option<Address>,
}

impl FourthChoicePointWord {
    fn decode(tuple_memory: &TupleMemory, tuple_address: TupleAddress) -> Self {
        let (trail_top, cut_register) =
            Serializable::from_be_bytes(tuple_memory[tuple_address + 3].to_be_bytes());

        Self {
            trail_top,
            cut_register,
        }
    }

    fn encode(self, tuple_memory: &mut TupleMemory, tuple_address: TupleAddress) {
        let Self {
            trail_top,
            cut_register,
        } = self;

        tuple_memory[tuple_address + 3] =
            u32::from_be_bytes((trail_top, cut_register).into_be_bytes());
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

impl Tuple for ChoicePoint {
    const VALUE_TYPE: ValueType = ValueType::ChoicePoint;

    type Terms = [Address];

    fn decode(tuple_memory: &TupleMemory, tuple_address: TupleAddress) -> (Self, TupleMetadata) {
        let FirstChoicePointWord {
            registry_address,
            number_of_saved_registers,
        } = FirstChoicePointWord::decode(tuple_memory, tuple_address);

        let SecondChoicePointWord {
            current_environment,
            continuation_point,
        } = SecondChoicePointWord::decode(tuple_memory, tuple_address);

        let ThirdChoicePointWord {
            next_choice_point,
            next_clause,
        } = ThirdChoicePointWord::decode(tuple_memory, tuple_address);

        let FourthChoicePointWord {
            trail_top,
            cut_register,
        } = FourthChoicePointWord::decode(tuple_memory, tuple_address);

        let arity = number_of_saved_registers;
        let first_term = tuple_address + 4;
        let next_tuple = first_term + arity;
        (
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
                first_term,
                arity,
                next_tuple,
            },
        )
    }

    fn encode(
        &self,
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
        saved_registers: &[Address],
    ) -> TupleAddress {
        let Self {
            number_of_saved_registers,
            current_environment,
            continuation_point,
            next_choice_point,
            next_clause,
            trail_top,
            cut_register,
        } = *self;

        assert_eq!(number_of_saved_registers.0 as usize, saved_registers.len());

        FirstChoicePointWord {
            registry_address,
            number_of_saved_registers,
        }
        .encode(tuple_memory, tuple_address);

        SecondChoicePointWord {
            current_environment,
            continuation_point,
        }
        .encode(tuple_memory, tuple_address);

        ThirdChoicePointWord {
            next_choice_point,
            next_clause,
        }
        .encode(tuple_memory, tuple_address);

        FourthChoicePointWord {
            trail_top,
            cut_register,
        }
        .encode(tuple_memory, tuple_address);

        let first_term = tuple_address + 4;

        for (term_address, &value) in first_term
            .iter(number_of_saved_registers)
            .zip(saved_registers)
        {
            tuple_memory[term_address] = value.0 as u32;
        }

        first_term + number_of_saved_registers
    }
}

#[derive(Debug)]
struct TrailItem {
    item: Address,
    next_trail_item: Option<Address>,
}

impl Tuple for TrailItem {
    const VALUE_TYPE: ValueType = ValueType::Trail;
    type Terms = ();

    fn decode(tuple_memory: &TupleMemory, tuple_address: TupleAddress) -> (Self, TupleMetadata) {
        let [a1, a0, z1, z0] = tuple_memory[tuple_address].to_be_bytes();
        let [i1, i0, n1, n0] = tuple_memory[tuple_address + 1].to_be_bytes();
        assert_eq!(z1, 0);
        assert_eq!(z0, 0);

        (
            Self {
                item: Address::from_be_bytes([i1, i0]),
                next_trail_item: Option::<Address>::from_be_bytes([n1, n0]),
            },
            TupleMetadata {
                registry_address: Address::from_be_bytes([a1, a0]),
                first_term: TupleAddress::NULL,
                arity: Arity::ZERO,
                next_tuple: tuple_address + 2,
            },
        )
    }

    fn encode(
        &self,
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
        _: &(),
    ) -> TupleAddress {
        let [a1, a0] = registry_address.into_be_bytes();
        let [i1, i0] = self.item.into_be_bytes();
        let [n1, n0] = self.next_trail_item.into_be_bytes();

        tuple_memory[tuple_address] = u32::from_be_bytes([a1, a0, 0, 0]);
        tuple_memory[tuple_address + 1] = u32::from_be_bytes([i1, i0, n1, n0]);

        tuple_address + 2
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

pub struct Heap<'m> {
    registry: Registry<'m>,
    tuple_memory: TupleMemory<'m>,
    tuple_end: TupleAddress,
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

        for entry in registry.iter_mut() {
            unsafe { core::ptr::write(entry, None) };
        }

        Self {
            registry: Registry(registry),
            tuple_memory: TupleMemory(tuple_memory),
            tuple_end: TupleAddress(0),
            current_environment: None,
            latest_choice_point: None,
            trail_top: None,
            cut_register: None,
        }
    }

    fn new_value<T: Tuple + core::fmt::Debug>(
        &mut self,
        factory: impl FnOnce(Address) -> T,
        terms: &T::Terms,
    ) -> Address {
        let (address, registry_entry) = self.registry.new_registry_entry();

        let tuple_address = self.tuple_end;

        let value = factory(address);

        crate::log_trace!("New value at {}: {:?}", tuple_address, value);

        self.tuple_end = value.encode(address, &mut self.tuple_memory, tuple_address, terms);

        crate::log_trace!("h = {}", self.tuple_end);

        *registry_entry = Some(RegistryEntry {
            value_type: T::VALUE_TYPE,
            tuple_address,
        });

        address
    }

    pub fn new_variable(&mut self) -> Address {
        self.new_value(ReferenceValue, &())
    }

    pub fn new_structure(&mut self, f: Functor, n: Arity) -> Address {
        self.new_value(|_| StructureValue(f, n), &())
    }

    pub fn new_list(&mut self) -> Address {
        self.new_value(|_| ListValue, &())
    }

    pub fn new_constant(&mut self, c: Constant) -> Address {
        self.new_value(|_| ConstantValue(c), &())
    }

    pub fn get_value(
        &self,
        mut address: Address,
    ) -> (Address, Value, impl Iterator<Item = Address> + '_) {
        loop {
            crate::log_trace!("Looking up memory at {}", address);
            let entry = self.registry[address]
                .as_ref()
                .expect("Uninitialized Entry");

            let (value, metadata) = match &entry.value_type {
                ValueType::Reference => {
                    let (value, metadata) =
                        ReferenceValue::decode(&self.tuple_memory, entry.tuple_address);

                    if value.0 != address {
                        address = value.0;
                        continue;
                    }

                    (Value::reference(value), metadata)
                }
                ValueType::Structure => {
                    let (value, metadata) =
                        StructureValue::decode(&self.tuple_memory, entry.tuple_address);
                    (Value::structure(value), metadata)
                }
                ValueType::List => {
                    let (value, metadata) =
                        ListValue::decode(&self.tuple_memory, entry.tuple_address);
                    (Value::list(value), metadata)
                }
                ValueType::Constant => {
                    let (value, metadata) =
                        ConstantValue::decode(&self.tuple_memory, entry.tuple_address);
                    (Value::constant(value), metadata)
                }
                ValueType::Environment | ValueType::ChoicePoint | ValueType::Trail => {
                    panic!("Expected value, found {:?}", entry.value_type);
                }
            };

            crate::log_trace!("Value: {:?}", value);

            break (
                metadata.registry_address,
                value,
                metadata
                    .first_term
                    .iter(metadata.arity)
                    .map(move |tuple_address| Address::from(self.tuple_memory[tuple_address])),
            );
        }
    }

    fn structure_term_addresses(&self, address: Address) -> (TupleAddress, Arity) {
        let registry_entry = self.registry[address]
            .as_ref()
            .expect("Structure not found");

        match registry_entry.value_type {
            ValueType::Structure => {
                let (StructureValue(_f, n), metadata) =
                    StructureValue::decode(&self.tuple_memory, registry_entry.tuple_address);
                assert_eq!(address, metadata.registry_address);
                (metadata.first_term, n)
            }
            ValueType::List => {
                let (ListValue, metadata) =
                    ListValue::decode(&self.tuple_memory, registry_entry.tuple_address);
                assert_eq!(address, metadata.registry_address);
                (metadata.first_term, ListValue::ARITY)
            }
            _ => panic!("Invalid value type: {:?}", registry_entry.value_type),
        }
    }

    fn get_environment(&self) -> (Environment, TupleAddress, TupleMetadata) {
        let current_environment = self
            .current_environment
            .unwrap_or_else(|| panic!("No Environment"));

        let entry = self.registry[current_environment]
            .as_ref()
            .expect("No Entry");
        assert!(matches!(&entry.value_type, ValueType::Environment));

        let (environment, metadata) = Environment::decode(&self.tuple_memory, entry.tuple_address);
        assert_eq!(metadata.registry_address, current_environment);

        (environment, entry.tuple_address, metadata)
    }

    fn get_permanent_variable_address(&self, yn: Yn) -> TupleAddress {
        let (environment, _, metadata) = self.get_environment();

        assert!(yn.yn < environment.number_of_active_permanent_variables.0);

        metadata.first_term + yn
    }

    unsafe fn load_permanent_variable_unchecked(&self, yn: Yn) -> Address {
        let term_address = self.get_permanent_variable_address(yn);

        Address::from(self.tuple_memory[term_address])
    }

    pub fn load_permanent_variable(&self, yn: Yn) -> Address {
        let address = unsafe { self.load_permanent_variable_unchecked(yn) };
        if address == unsafe { Address::none() } {
            panic!("No permanent variable @ {}", yn.yn);
        }

        address
    }

    pub fn store_permanent_variable(&mut self, yn: Yn, address: Address) {
        let term_address = self.get_permanent_variable_address(yn);

        self.tuple_memory[term_address] = address.0 as u32;
    }

    fn verify_is_reference(&self, address: Address) -> (ReferenceValue, TupleAddress) {
        let entry = self.registry[address].as_ref().expect("No Entry");

        assert!(matches!(&entry.value_type, ValueType::Reference));
        let (reference, metadata) = ReferenceValue::decode(&self.tuple_memory, entry.tuple_address);

        assert_eq!(metadata.registry_address, address);

        (reference, entry.tuple_address)
    }

    fn verify_is_free_variable(&self, address: Address) -> TupleAddress {
        let (reference, tuple_address) = self.verify_is_reference(address);

        assert_eq!(reference.0, address);

        tuple_address
    }

    pub fn bind_variable_to_value(&mut self, variable_address: Address, value_address: Address) {
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
        );

        self.add_variable_to_trail(variable_address);
    }

    pub fn bind_variables(&mut self, a1: Address, a2: Address) {
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
    ) {
        let continuation_environment = self.current_environment;

        let new_environment = self.new_value(
            |_| Environment {
                continuation_environment,
                continuation_point,
                number_of_active_permanent_variables: number_of_permanent_variables,
                number_of_permanent_variables,
            },
            terms,
        );
        self.current_environment = Some(new_environment);
    }

    pub fn trim(&mut self, n: Arity) {
        let (
            Environment {
                continuation_point,
                number_of_active_permanent_variables,
                number_of_permanent_variables,
                ..
            },
            tuple_address,
            _,
        ) = self.get_environment();
        assert!(number_of_active_permanent_variables >= n);

        if let Some(latest_choice_point) = self.latest_choice_point {
            let latest_choice_point = self.registry[latest_choice_point]
                .as_ref()
                .expect("No Entry");

            log_trace!("latest_choice_point: {}", latest_choice_point.tuple_address);
            log_trace!("current_environment: {}", tuple_address);

            if latest_choice_point.tuple_address > tuple_address {
                log_trace!("Not trimming conditional environment");
                return;
            }
        }

        let number_of_active_permanent_variables =
            Arity(number_of_active_permanent_variables.0 - n.0);

        SecondEnvironmentWord {
            continuation_point,
            number_of_active_permanent_variables,
            number_of_permanent_variables,
        }
        .encode(&mut self.tuple_memory, tuple_address)
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
    ) {
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
        );

        self.latest_choice_point = Some(new_choice_point);
    }

    fn get_latest_choice_point(&self) -> (ChoicePoint, TupleAddress, TupleMetadata) {
        let latest_choice_point = self
            .latest_choice_point
            .unwrap_or_else(|| panic!("No choice point"));

        let entry = self.registry[latest_choice_point]
            .as_ref()
            .expect("No Entry");
        assert!(matches!(&entry.value_type, ValueType::ChoicePoint));

        let (choice_point, metadata) = ChoicePoint::decode(&self.tuple_memory, entry.tuple_address);
        assert_eq!(metadata.registry_address, latest_choice_point);

        (choice_point, entry.tuple_address, metadata)
    }

    pub fn backtrack(
        &mut self,
        pc: &mut Option<ProgramCounter>,
    ) -> Result<(), super::ExecutionFailure> {
        self.latest_choice_point
            .ok_or(super::ExecutionFailure::Failed)
            .map(|_| {
                *pc = Some(self.get_latest_choice_point().0.next_clause);
            })
    }

    fn wind_back_to_choice_point(
        &mut self,
        registers: &mut [Address],
        continuation_point: &mut Option<ProgramCounter>,
    ) -> (ChoicePoint, TupleAddress, Option<Address>) {
        let (choice_point, tuple_address, metadata) = self.get_latest_choice_point();

        for (register, value_address) in registers.iter_mut().zip(
            metadata
                .first_term
                .iter(choice_point.number_of_saved_registers),
        ) {
            *register = Address::from(self.tuple_memory[value_address]);
        }

        self.current_environment = choice_point.current_environment;
        *continuation_point = choice_point.continuation_point;

        self.unwind_trail(tuple_address);

        self.trail_top = choice_point.trail_top;
        let next_choice_point = choice_point.next_choice_point;

        crate::log_trace!("E => {}", OptionDisplay(self.current_environment));
        crate::log_trace!("CP => {}", OptionDisplay(*continuation_point));

        (choice_point, tuple_address, next_choice_point)
    }

    pub fn retry_choice_point(
        &mut self,
        registers: &mut [Address],
        continuation_point: &mut Option<ProgramCounter>,
        next_clause: ProgramCounter,
    ) {
        let (
            ChoicePoint {
                next_choice_point, ..
            },
            tuple_address,
            _metadata,
        ) = self.wind_back_to_choice_point(registers, continuation_point);

        ThirdChoicePointWord {
            next_choice_point,
            next_clause,
        }
        .encode(&mut self.tuple_memory, tuple_address);

        // HB <- H
    }

    pub fn remove_choice_point(
        &mut self,
        registers: &mut [Address],
        continuation_point: &mut Option<ProgramCounter>,
    ) {
        let next_choice_point = self
            .wind_back_to_choice_point(registers, continuation_point)
            .2;

        self.latest_choice_point = next_choice_point;
    }

    fn add_variable_to_trail(&mut self, address: Address) {
        let next_trail_item = self.trail_top;
        let new_trail_item = self.new_value(
            |_| TrailItem {
                item: address,
                next_trail_item,
            },
            &(),
        );
        self.trail_top = Some(new_trail_item);
    }

    fn unwind_trail(&mut self, boundary: TupleAddress) {
        log_trace!("Unwinding Trail");
        while let Some(trail_top) = self.trail_top {
            let entry = self.registry[trail_top].as_ref().expect("No Entry");
            assert!(matches!(&entry.value_type, ValueType::Trail));

            if entry.tuple_address < boundary {
                break;
            }

            log_trace!("Unwinding Trail item @ {}", trail_top);

            let (trail_item, metadata) = TrailItem::decode(&self.tuple_memory, entry.tuple_address);

            assert_eq!(metadata.registry_address, trail_top);

            log_trace!("Resetting Reference @ {}", trail_item.item);

            let item_tuple_address = self.verify_is_reference(trail_item.item).1;

            ReferenceValue(trail_item.item).encode(
                trail_item.item,
                &mut self.tuple_memory,
                item_tuple_address,
                &(),
            );

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

    pub fn get_level(&mut self, yn: Yn) {
        self.store_permanent_variable(yn, self.cut_register.unwrap_or(unsafe { Address::none() }));
    }

    pub fn cut(&mut self, yn: Yn) {
        let address = unsafe { self.load_permanent_variable_unchecked(yn) };

        self.do_cut(if address == unsafe { Address::none() } {
            None
        } else {
            Some(address)
        });
    }

    pub fn query_has_multiple_solutions(&self) -> bool {
        self.latest_choice_point.is_some()
    }

    pub fn solution_registers(&self) -> impl Iterator<Item = Address> + '_ {
        let (environment, _, metadata) = self.get_environment();
        assert!(environment.continuation_environment.is_none());

        metadata
            .first_term
            .iter(environment.number_of_permanent_variables)
            .map(move |tuple_address| Address::from(self.tuple_memory[tuple_address]))
    }
}
