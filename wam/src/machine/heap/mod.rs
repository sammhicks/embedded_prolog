use core::{fmt, panic};

use crate::serializable::{Serializable, SerializableWrapper};

use super::basic_types::{Arity, Constant, Functor, ProgramCounter, Yn};

pub mod structure_iteration;

#[derive(Debug)]
enum ValueType {
    Reference,
    Structure,
    List,
    Constant,
    Environment,
}

#[derive(Clone, Copy, PartialEq, Eq)]
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

impl From<u32> for Address {
    fn from(word: u32) -> Self {
        Self(std::convert::TryInto::try_into(word).unwrap_or_else(|_| panic!("Bad word: {}", word)))
    }
}

impl Address {
    pub const NULL: Self = Self(u16::MAX);
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
    const NULL: Self = Self(u16::MAX);
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
        let [a1, a0, r1, r0] = tuple_memory[tuple_address].to_be_bytes();

        (
            Self(Address::from_be_bytes([r1, r0])),
            TupleMetadata {
                registry_address: Address::from_be_bytes([a1, a0]),
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
        let Self(r) = self;

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
        let Self(f, n) = self;

        let [a1, a0] = registry_address.into_be_bytes();
        let [f1, f0] = f.into_be_bytes();

        tuple_memory[tuple_address] = u32::from_be_bytes([a1, a0, f1, f0]);
        tuple_memory[tuple_address + 1] = u32::from_be_bytes([0, 0, 0, n.0]);

        let first_term: TupleAddress = tuple_address + 2;

        for term_address in (first_term.0..).take(n.0 as usize).map(TupleAddress) {
            tuple_memory[term_address] = u32::MAX;
        }

        first_term + *n
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
        let [a1, a0, c1, c0] = tuple_memory[tuple_address].to_be_bytes();
        (
            Self(Constant::from_be_bytes([c1, c0])),
            TupleMetadata {
                registry_address: Address::from_be_bytes([a1, a0]),
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
        let Self(c) = self;

        let [a1, a0] = registry_address.into_be_bytes();
        let [c1, c0] = c.into_be_bytes();

        tuple_memory[tuple_address] = u32::from_be_bytes([a1, a0, c1, c0]);

        tuple_address + 1
    }
}

struct FirstEnvironmentWord {
    registry_address: Address,
    continuation_environment: Address,
}

impl FirstEnvironmentWord {
    fn decode(tuple_memory: &TupleMemory, tuple_address: TupleAddress) -> Self {
        let [a1, a0, ce1, ce0] = tuple_memory[tuple_address].to_be_bytes();
        Self {
            registry_address: Address::from_be_bytes([a1, a0]),
            continuation_environment: Address::from_be_bytes([ce1, ce0]),
        }
    }

    fn encode(
        &self,
        registry_address: Address,
        tuple_memory: &mut TupleMemory,
        tuple_address: TupleAddress,
    ) {
        let [a1, a0] = registry_address.into_be_bytes();
        let [ce1, ce0] = self.continuation_environment.into_be_bytes();

        tuple_memory[tuple_address] = u32::from_be_bytes([a1, a0, ce1, ce0]);
    }
}

struct SecondEnvironmentWord {
    continuation_point: ProgramCounter,
    number_of_active_permanent_variables: Arity,
    number_of_permanent_variables: Arity,
}

impl SecondEnvironmentWord {
    fn decode(tuple_memory: &TupleMemory, tuple_address: TupleAddress) -> Self {
        let [cp1, cp0, na, np] = tuple_memory[tuple_address + 1].to_be_bytes();

        Self {
            continuation_point: ProgramCounter::from_be_bytes([cp1, cp0]),
            number_of_active_permanent_variables: Arity(na),
            number_of_permanent_variables: Arity(np),
        }
    }

    fn encode(&self, tuple_memory: &mut TupleMemory, tuple_address: TupleAddress) {
        let [cp1, cp0] = self.continuation_point.into_be_bytes();
        let na = self.number_of_active_permanent_variables.0;
        let np = self.number_of_permanent_variables.0;

        tuple_memory[tuple_address + 1] = u32::from_be_bytes([cp1, cp0, na, np]);
    }
}

#[derive(Debug)]
struct Environment {
    continuation_environment: Address,
    continuation_point: ProgramCounter,
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
        assert!(terms.is_empty() || (terms.len() == self.number_of_permanent_variables.0 as usize));

        FirstEnvironmentWord {
            registry_address,
            continuation_environment: self.continuation_environment,
        }
        .encode(registry_address, tuple_memory, tuple_address);

        SecondEnvironmentWord {
            continuation_point: self.continuation_point,
            number_of_active_permanent_variables: self.number_of_active_permanent_variables,
            number_of_permanent_variables: self.number_of_permanent_variables,
        }
        .encode(tuple_memory, tuple_address);

        let first_term_address: TupleAddress = tuple_address + 2;

        for (tuple_address, value) in (first_term_address.0..)
            .map(TupleAddress)
            .zip(
                terms
                    .iter()
                    .copied()
                    .chain(core::iter::repeat(Address::NULL)),
            )
            .take(self.number_of_permanent_variables.0 as usize)
        {
            tuple_memory[tuple_address] = value.0 as u32;
        }

        first_term_address + self.number_of_active_permanent_variables
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
    current_environment: Address,
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
            current_environment: Address::NULL,
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

            let (value, metadata) = match entry.value_type {
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
                ValueType::Environment => {
                    panic!("Expected value, found environment")
                }
            };

            crate::log_trace!("Value: {:?}", value);

            break (
                metadata.registry_address,
                value,
                (metadata.first_term.0..)
                    .take(metadata.arity.0 as usize)
                    .map(move |tuple_address| {
                        Address::from(self.tuple_memory[TupleAddress(tuple_address)])
                    }),
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
            ValueType::Environment => {
                let (environment, metadata) =
                    Environment::decode(&self.tuple_memory, registry_entry.tuple_address);
                assert_eq!(address, metadata.registry_address);
                (
                    metadata.first_term,
                    environment.number_of_active_permanent_variables,
                )
            }
            _ => panic!("Invalid value type: {:?}", registry_entry.value_type),
        }
    }

    fn get_environment(&self) -> (Environment, TupleAddress, TupleMetadata) {
        let entry = self.registry[self.current_environment]
            .as_ref()
            .expect("No Entry");
        assert!(matches!(&entry.value_type, ValueType::Environment));

        let (environment, metadata) = Environment::decode(&self.tuple_memory, entry.tuple_address);
        assert_eq!(metadata.registry_address, self.current_environment);

        (environment, entry.tuple_address, metadata)
    }

    fn get_permanent_variable_address(&self, yn: Yn) -> TupleAddress {
        let (environment, _, metadata) = self.get_environment();

        assert!(yn.yn < environment.number_of_active_permanent_variables.0);

        metadata.first_term + yn
    }

    pub fn load_permanent_variable(&self, yn: Yn) -> Address {
        let term_address = self.get_permanent_variable_address(yn);

        Address::from(self.tuple_memory[term_address])
    }

    pub fn store_permanent_variable(&mut self, yn: Yn, address: Address) {
        let term_address = self.get_permanent_variable_address(yn);

        self.tuple_memory[term_address] = address.0 as u32;
    }

    fn verify_is_free_variable(&self, address: Address) -> TupleAddress {
        let entry = self.registry[address].as_ref().expect("No Entry");

        assert!(matches!(&entry.value_type, ValueType::Reference));
        let (reference, metadata) = ReferenceValue::decode(&self.tuple_memory, entry.tuple_address);

        assert_eq!(metadata.registry_address, address);
        assert_eq!(reference.0, address);

        entry.tuple_address
    }

    pub fn bind_variable_to_value(&mut self, variable_address: Address, value_address: Address) {
        crate::log_trace!(
            "Binding memory {} to value at {}",
            variable_address,
            value_address
        );

        let tuple_address = self.verify_is_free_variable(variable_address);

        // TODO: add to trail
        ReferenceValue(value_address).encode(
            variable_address,
            &mut self.tuple_memory,
            tuple_address,
            &(),
        );
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
        continuation_point: ProgramCounter,
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
        self.current_environment = new_environment;
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

        let number_of_active_permanent_variables =
            Arity(number_of_active_permanent_variables.0 - n.0);

        SecondEnvironmentWord {
            continuation_point,
            number_of_active_permanent_variables,
            number_of_permanent_variables,
        }
        .encode(&mut self.tuple_memory, tuple_address)
    }

    pub fn deallocate(&mut self) -> ProgramCounter {
        let Environment {
            continuation_environment,
            continuation_point,
            ..
        } = self.get_environment().0;

        self.current_environment = continuation_environment;

        crate::log_trace!("E => {}", self.current_environment);

        continuation_point
    }

    pub fn solution_registers(&self) -> impl Iterator<Item = Address> + '_ {
        let (environment, _, metadata) = self.get_environment();
        (metadata.first_term.0..)
            .map(TupleAddress)
            .take(environment.number_of_permanent_variables.0 as usize)
            .map(move |tuple_address| Address::from(self.tuple_memory[tuple_address]))
    }
}
