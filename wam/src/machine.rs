use core::fmt;

use crate::{
    log_debug, log_trace,
    serializable::{Serializable, SerializableWrapper},
};

#[derive(Clone, Copy, Debug)]
pub struct Xn {
    xn: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct Yn {
    yn: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct Ai {
    ai: u8,
}

#[derive(Clone, Copy)]
pub struct RegisterIndex(u8);

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
pub struct Functor(u16);

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
pub struct Arity(u8);

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
    const ZERO: Self = Self(0);
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Constant(u16);

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
pub struct ProgramCounter(u16);

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
    const NULL: Self = Self(u16::MAX);

    fn offset(self, offset: u16) -> Self {
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

mod heap {
    use core::{fmt, panic};

    use super::{Arity, Constant, Functor, ProgramCounter, Serializable, SerializableWrapper, Yn};

    #[derive(Debug)]
    enum ValueType {
        Reference,
        Structure,
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
            Self(
                std::convert::TryInto::try_into(word)
                    .unwrap_or_else(|_| panic!("Bad word: {}", word)),
            )
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
    pub struct TupleAddress(u16);

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
    pub struct RegistryEntry {
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

    // impl<'m> std::ops::IndexMut<TupleAddress> for TupleMemory<'m> {
    //     fn index_mut(&mut self, index: TupleAddress) -> &mut Self::Output {
    //         &mut self.0[index.0 as usize]
    //     }
    // }

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

        fn decode(tuple_memory: &TupleMemory, tuple_address: TupleAddress)
            -> (Self, TupleMetadata);

        fn encode(
            &self,
            registry_address: Address,
            tuple_memory: &mut TupleMemory,
            tuple_address: TupleAddress,
            terms: &Self::Terms,
        ) -> TupleAddress;
    }

    #[derive(Debug)]
    pub struct ReferenceValue(Address);

    impl Tuple for ReferenceValue {
        const VALUE_TYPE: ValueType = ValueType::Reference;
        type Terms = ();

        fn decode(
            tuple_memory: &TupleMemory,
            tuple_address: TupleAddress,
        ) -> (Self, TupleMetadata) {
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
    pub struct StructureValue(Functor, Arity);

    impl Tuple for StructureValue {
        const VALUE_TYPE: ValueType = ValueType::Structure;
        type Terms = ();

        fn decode(
            tuple_memory: &TupleMemory,
            tuple_address: TupleAddress,
        ) -> (Self, TupleMetadata) {
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
    struct ConstantValue(Constant);

    impl Tuple for ConstantValue {
        const VALUE_TYPE: ValueType = ValueType::Constant;
        type Terms = ();

        fn decode(
            tuple_memory: &TupleMemory,
            tuple_address: TupleAddress,
        ) -> (Self, TupleMetadata) {
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
    pub struct Environment {
        continuation_environment: Address,
        continuation_point: ProgramCounter,
        number_of_active_permanent_variables: Arity,
        number_of_permanent_variables: Arity,
    }

    impl Tuple for Environment {
        const VALUE_TYPE: ValueType = ValueType::Environment;
        type Terms = [Address];

        fn decode(
            tuple_memory: &TupleMemory,
            tuple_address: TupleAddress,
        ) -> (Self, TupleMetadata) {
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
            assert!(
                terms.is_empty() || (terms.len() == self.number_of_permanent_variables.0 as usize)
            );

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
        Constant(Constant),
    }

    impl Value {
        fn reference(ReferenceValue(address): ReferenceValue) -> Self {
            Self::Reference(address)
        }

        fn structure(StructureValue(f, n): StructureValue) -> Self {
            Self::Structure(f, n)
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

        fn structure_term_addresses(&self, structure_address: Address) -> (TupleAddress, Arity) {
            let registry_entry = self.registry[structure_address]
                .as_ref()
                .expect("Structure not found");

            match registry_entry.value_type {
                ValueType::Structure => {
                    let (StructureValue(_f, n), metadata) =
                        StructureValue::decode(&self.tuple_memory, registry_entry.tuple_address);
                    assert_eq!(structure_address, metadata.registry_address);
                    (metadata.first_term, n)
                }
                ValueType::Environment => {
                    let (environment, metadata) =
                        Environment::decode(&self.tuple_memory, registry_entry.tuple_address);
                    assert_eq!(structure_address, metadata.registry_address);
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

            let (environment, metadata) =
                Environment::decode(&self.tuple_memory, entry.tuple_address);
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
            let (reference, metadata) =
                ReferenceValue::decode(&self.tuple_memory, entry.tuple_address);

            assert_eq!(metadata.registry_address, address);
            assert_eq!(reference.0, address);

            entry.tuple_address
        }

        pub fn bind_variable_to_value(
            &mut self,
            variable_address: Address,
            value_address: Address,
        ) {
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

    pub mod structure_iteration {
        use super::{Address, Arity, Heap, TupleAddress};

        pub enum ReadWriteMode {
            Read,
            Write,
        }

        struct InnerState {
            read_write_mode: ReadWriteMode,
            structure_address: Address,
            index: Arity,
        }

        #[derive(Default)]
        pub struct StructureIterationState(Option<InnerState>);

        impl StructureIterationState {
            pub fn read_write_mode(&self) -> &ReadWriteMode {
                &self
                    .0
                    .as_ref()
                    .expect("Not iterating over state!")
                    .read_write_mode
            }

            pub fn structure_reader(structure_address: Address) -> Self {
                Self(Some(InnerState {
                    read_write_mode: ReadWriteMode::Read,
                    structure_address,
                    index: Arity::ZERO,
                }))
            }

            pub fn start_reading_structure(&mut self, structure_address: Address) {
                assert!(matches!(&self.0, None));
                *self = Self::structure_reader(structure_address);
            }

            pub fn start_writing(&mut self, structure_address: Address) {
                assert!(matches!(&self.0, None));
                *self = Self(Some(InnerState {
                    read_write_mode: ReadWriteMode::Write,
                    structure_address,
                    index: Arity::ZERO,
                }));
            }

            fn check_done(&mut self, index: Arity, arity: Arity) {
                if index == arity {
                    crate::log_trace!("Finished iterating over structure");
                    self.0 = None;
                }
            }

            fn with_next<'m, T, H>(
                &mut self,
                heap: H,
                action: impl FnOnce(H, TupleAddress) -> T,
            ) -> T
            where
                H: core::ops::Deref<Target = Heap<'m>>,
            {
                let inner_state = self.0.as_mut().expect("Not reading or writing");

                let (first_term, arity) =
                    heap.structure_term_addresses(inner_state.structure_address);

                if inner_state.index == arity {
                    panic!("No more terms");
                }

                let term_address = first_term + inner_state.index;

                inner_state.index.0 += 1;

                let result = action(heap, term_address);

                let index = inner_state.index;

                self.check_done(index, arity);

                result
            }

            pub fn read_next(&mut self, heap: &Heap) -> Address {
                self.with_next(heap, |heap, address| {
                    Address::from(heap.tuple_memory[address])
                })
            }

            pub fn write_next(&mut self, heap: &mut Heap, address: Address) {
                self.with_next(heap, |heap, term_address| {
                    crate::log_trace!("Writing {} to {}", address, term_address);
                    heap.tuple_memory[term_address] = address.0 as u32;
                })
            }

            pub fn skip(&mut self, heap: &Heap, n: Arity) {
                let inner_state = self.0.as_mut().expect("Not reading or writing");

                let arity = heap
                    .structure_term_addresses(inner_state.structure_address)
                    .1;

                inner_state.index.0 += n.0;

                if inner_state.index > arity {
                    panic!("No more terms to read");
                }

                let index = inner_state.index;

                self.check_done(index, arity);
            }
        }
    }
}

use heap::{
    structure_iteration::{ReadWriteMode, StructureIterationState},
    Heap,
};
pub use heap::{Address, Value};

struct BadMemoryWord(u32);

impl fmt::Debug for BadMemoryWord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:08X}", self.0)
    }
}

pub struct BadMemoryRange<'m>(&'m [u32]);

impl fmt::Debug for BadMemoryRange<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for &word in self.0 {
            list.entry(&BadMemoryWord(word));
        }
        list.finish()
    }
}

#[derive(Debug)]
pub enum Error<'m> {
    BadInstruction(ProgramCounter, BadMemoryRange<'m>),
}

#[derive(Debug)]
enum Instruction {
    PutVariableXn { ai: Ai, xn: Xn },
    PutVariableYn { ai: Ai, yn: Yn },
    PutValueXn { ai: Ai, xn: Xn },
    PutValueYn { ai: Ai, yn: Yn },
    PutStructure { ai: Ai, f: Functor, n: Arity },
    PutConstant { ai: Ai, c: Constant },
    PutVoid { ai: Ai, n: Arity },
    GetVariableXn { ai: Ai, xn: Xn },
    GetVariableYn { ai: Ai, yn: Yn },
    GetValueXn { ai: Ai, xn: Xn },
    GetValueYn { ai: Ai, yn: Yn },
    GetStructure { ai: Ai, f: Functor, n: Arity },
    GetConstant { ai: Ai, c: Constant },
    SetVariableXn { xn: Xn },
    SetVariableYn { yn: Yn },
    SetValueXn { xn: Xn },
    SetValueYn { yn: Yn },
    SetConstant { c: Constant },
    SetVoid { n: Arity },
    UnifyVariableXn { xn: Xn },
    UnifyVariableYn { yn: Yn },
    UnifyValueXn { xn: Xn },
    UnifyValueYn { yn: Yn },
    UnifyConstant { c: Constant },
    UnifyVoid { n: Arity },
    Allocate { n: Arity },
    Trim { n: Arity },
    Deallocate,
    Call { p: ProgramCounter, n: Arity },
    Execute { p: ProgramCounter, n: Arity },
    Proceed,
}

impl Instruction {
    fn decode_structure_long_functor(memory: &[u32], pc: ProgramCounter) -> Result<Functor, Error> {
        if let [0, 0, f1, f0] = memory[pc.offset(1).0 as usize].to_be_bytes() {
            Ok(Functor(u16::from_be_bytes([f1, f0])))
        } else {
            Err(Error::BadInstruction(
                pc,
                BadMemoryRange(&memory[pc.0 as usize..pc.offset(2).0 as usize]),
            ))
        }
    }

    fn decode(memory: &[u32], pc: ProgramCounter) -> Result<(ProgramCounter, Self), Error> {
        Ok(match memory[pc.0 as usize].to_be_bytes() {
            [0x00, ai, 0, xn] => (
                pc.offset(1),
                Instruction::PutVariableXn {
                    ai: Ai { ai },
                    xn: Xn { xn },
                },
            ),
            [0x01, ai, 0, yn] => (
                pc.offset(1),
                Instruction::PutVariableYn {
                    ai: Ai { ai },
                    yn: Yn { yn },
                },
            ),
            [0x02, ai, 0, xn] => (
                pc.offset(1),
                Instruction::PutValueXn {
                    ai: Ai { ai },
                    xn: Xn { xn },
                },
            ),
            [0x03, ai, 0, yn] => (
                pc.offset(1),
                Instruction::PutValueYn {
                    ai: Ai { ai },
                    yn: Yn { yn },
                },
            ),
            [0x04, ai, f, n] => (
                pc.offset(1),
                Instruction::PutStructure {
                    ai: Ai { ai },
                    f: Functor(f as u16),
                    n: Arity(n),
                },
            ),
            [0x05, ai, 0, n] => (
                pc.offset(2),
                Instruction::PutStructure {
                    ai: Ai { ai },
                    f: Self::decode_structure_long_functor(memory, pc)?,
                    n: Arity(n),
                },
            ),
            [0x07, ai, c1, c0] => (
                pc.offset(1),
                Instruction::PutConstant {
                    ai: Ai { ai },
                    c: Constant::from_be_bytes([c1, c0]),
                },
            ),
            [0x0a, ai, 0, n] => (
                pc.offset(1),
                Instruction::PutVoid {
                    ai: Ai { ai },
                    n: Arity(n),
                },
            ),
            [0x10, ai, 0, xn] => (
                pc.offset(1),
                Instruction::GetVariableXn {
                    ai: Ai { ai },
                    xn: Xn { xn },
                },
            ),
            [0x11, ai, 0, yn] => (
                pc.offset(1),
                Instruction::GetVariableYn {
                    ai: Ai { ai },
                    yn: Yn { yn },
                },
            ),
            [0x12, ai, 0, xn] => (
                pc.offset(1),
                Instruction::GetValueXn {
                    ai: Ai { ai },
                    xn: Xn { xn },
                },
            ),
            [0x13, ai, 0, yn] => (
                pc.offset(1),
                Instruction::GetValueYn {
                    ai: Ai { ai },
                    yn: Yn { yn },
                },
            ),
            [0x14, ai, f, n] => (
                pc.offset(1),
                Instruction::GetStructure {
                    ai: Ai { ai },
                    f: Functor(f as u16),
                    n: Arity(n),
                },
            ),
            [0x15, ai, 0, n] => (
                pc.offset(2),
                Instruction::GetStructure {
                    ai: Ai { ai },
                    f: Self::decode_structure_long_functor(memory, pc)?,
                    n: Arity(n),
                },
            ),
            [0x17, ai, c1, c0] => (
                pc.offset(1),
                Instruction::GetConstant {
                    ai: Ai { ai },
                    c: Constant::from_be_bytes([c1, c0]),
                },
            ),
            [0x20, 0, 0, xn] => (pc.offset(1), Instruction::SetVariableXn { xn: Xn { xn } }),
            [0x21, 0, 0, yn] => (pc.offset(1), Instruction::SetVariableYn { yn: Yn { yn } }),
            [0x22, 0, 0, xn] => (pc.offset(1), Instruction::SetValueXn { xn: Xn { xn } }),
            [0x23, 0, 0, yn] => (pc.offset(1), Instruction::SetValueYn { yn: Yn { yn } }),
            [0x27, 0, c1, c0] => (
                pc.offset(1),
                Instruction::SetConstant {
                    c: Constant::from_be_bytes([c1, c0]),
                },
            ),
            [0x2a, 0, 0, n] => (pc.offset(1), Instruction::SetVoid { n: Arity(n) }),
            [0x30, 0, 0, xn] => (pc.offset(1), Instruction::UnifyVariableXn { xn: Xn { xn } }),
            [0x31, 0, 0, yn] => (pc.offset(1), Instruction::UnifyVariableYn { yn: Yn { yn } }),
            [0x32, 0, 0, xn] => (pc.offset(1), Instruction::UnifyValueXn { xn: Xn { xn } }),
            [0x33, 0, 0, yn] => (pc.offset(1), Instruction::UnifyValueYn { yn: Yn { yn } }),
            [0x37, 0, c1, c0] => (
                pc.offset(1),
                Instruction::UnifyConstant {
                    c: Constant::from_be_bytes([c1, c0]),
                },
            ),
            [0x3a, 0, 0, n] => (pc.offset(1), Instruction::UnifyVoid { n: Arity(n) }),
            [0x40, 0, 0, n] => (pc.offset(1), Instruction::Allocate { n: Arity(n) }),
            [0x41, 0, 0, n] => (pc.offset(1), Instruction::Trim { n: Arity(n) }),
            [0x42, 0, 0, 0] => (pc.offset(1), Instruction::Deallocate),
            [0x43, n, p1, p0] => (
                pc.offset(1),
                Instruction::Call {
                    p: ProgramCounter::from_be_bytes([p1, p0]),
                    n: Arity(n),
                },
            ),
            [0x44, n, p1, p0] => (
                pc.offset(1),
                Instruction::Execute {
                    p: ProgramCounter::from_be_bytes([p1, p0]),
                    n: Arity(n),
                },
            ),
            [0x45, 0, 0, 0] => (pc.offset(1), Instruction::Proceed),
            _ => {
                return Err(Error::BadInstruction(
                    pc,
                    BadMemoryRange(core::slice::from_ref(&memory[pc.0 as usize])),
                ))
            }
        })
    }
}

#[derive(Debug)]
struct RegisterBlock([Address; 32]);

impl RegisterBlock {
    fn load(&self, index: impl Into<RegisterIndex>) -> Address {
        let index = index.into().0 as usize;
        log_trace!("Loading Register {}", index);
        *self.0.get(index).unwrap_or_else(|| {
            panic!(
                "Register Index {} out of range ({})",
                index,
                (&self.0[..]).len()
            );
        })
    }

    fn store(&mut self, index: impl Into<RegisterIndex>, address: Address) {
        let index = index.into().0 as usize;
        log_trace!("Storing {} in Register {}", address, index);
        let len = (&self.0[..]).len();
        *self.0.get_mut(index).unwrap_or_else(|| {
            panic!("Register Index {} out of range ({})", index, len);
        }) = address;
    }

    fn clear_above(&mut self, n: Arity) {
        for register in &mut self.0[(n.0 as usize)..] {
            *register = Address::NULL;
        }
    }

    fn query_registers(&self, n: Arity) -> &[Address] {
        &self.0[0..n.0 as usize]
    }
}

#[derive(Debug)]
enum CurrentlyExecuting {
    Query,
    Program,
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum ExecutionFailure {
    Failed = b'F',
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum ExecutionSuccess {
    SingleAnswer = b'A',
    MultipleAnswers = b'C',
}

#[derive(Debug)]
struct UnificationFailure;

pub struct MachineMemory<'m> {
    pub program: &'m [u32],
    pub query: &'m [u32],
    pub memory: &'m mut [u32],
}

pub struct Machine<'m> {
    currently_executing: CurrentlyExecuting,
    structure_iteration_state: StructureIterationState,
    pc: ProgramCounter,
    cp: ProgramCounter,
    argument_count: Arity,
    registers: RegisterBlock,
    program: &'m [u32],
    query: &'m [u32],
    memory: Heap<'m>,
}

impl<'m> Machine<'m> {
    pub fn run(
        MachineMemory {
            program,
            query,
            memory,
        }: MachineMemory<'m>,
    ) -> Result<(Self, ExecutionSuccess), ExecutionFailure> {
        let mut machine = Self {
            currently_executing: CurrentlyExecuting::Query,
            structure_iteration_state: StructureIterationState::default(),
            pc: ProgramCounter(0),
            cp: ProgramCounter::NULL,
            argument_count: Arity::ZERO,
            registers: RegisterBlock([Address::NULL; 32]),
            program,
            query,
            memory: Heap::init(memory),
        };

        let execution_result = machine.continue_execution();
        execution_result.map(|success| (machine, success))
    }

    fn continue_execution(&mut self) -> Result<ExecutionSuccess, ExecutionFailure> {
        loop {
            if self.pc == ProgramCounter::NULL {
                return Ok(ExecutionSuccess::SingleAnswer);
            }

            let (new_pc, instruction) = match &self.currently_executing {
                CurrentlyExecuting::Query => Instruction::decode(self.query, self.pc).unwrap(),
                CurrentlyExecuting::Program => Instruction::decode(self.program, self.pc).unwrap(),
            };

            log_debug!(
                "{:?} Instruction @{} : {:?}",
                self.currently_executing,
                self.pc,
                instruction
            );

            self.pc = new_pc;

            self.execute_instruction(instruction)?;

            log_trace!("");
        }
    }

    fn execute_instruction(&mut self, instruction: Instruction) -> Result<(), ExecutionFailure> {
        match instruction {
            Instruction::PutVariableXn { ai, xn } => {
                let address = self.new_variable();
                self.registers.store(ai, address);
                self.registers.store(xn, address);
                Ok(())
            }
            Instruction::PutVariableYn { ai, yn } => {
                let address = self.new_variable();
                self.registers.store(ai, address);
                self.memory.store_permanent_variable(yn, address);
                Ok(())
            }
            Instruction::PutValueXn { ai, xn } => {
                let address = self.registers.load(xn);
                self.registers.store(ai, address);
                Ok(())
            }
            Instruction::PutValueYn { ai, yn } => {
                let address = self.memory.load_permanent_variable(yn);
                self.registers.store(ai, address);
                Ok(())
            }
            Instruction::PutStructure { ai, f, n } => {
                let address = self.new_structure(f, n);
                self.registers.store(ai, address);
                Ok(())
            }
            Instruction::PutConstant { ai, c } => {
                let address = self.new_constant(c);
                self.registers.store(ai, address);
                Ok(())
            }
            Instruction::PutVoid { ai, n } => {
                for ai in (ai.ai..).take(n.0 as usize).map(|ai| Ai { ai }) {
                    let address = self.new_variable();
                    self.registers.store(ai, address);
                }
                Ok(())
            }
            Instruction::GetVariableXn { ai, xn } => {
                self.registers.store(xn, self.registers.load(ai));
                Ok(())
            }
            Instruction::GetVariableYn { ai, yn } => {
                let address = self.registers.load(ai);
                self.memory.store_permanent_variable(yn, address);
                Ok(())
            }
            Instruction::GetValueXn { ai, xn } => self
                .unify(self.registers.load(xn), self.registers.load(ai))
                .or_else(|UnificationFailure| self.backtrack()),
            Instruction::GetValueYn { ai, yn } => self
                .unify(
                    self.memory.load_permanent_variable(yn),
                    self.registers.load(ai),
                )
                .or_else(|UnificationFailure| self.backtrack()),
            Instruction::GetStructure { ai, f, n } => {
                let (address, value) = self.get_register_value(ai);
                match value {
                    Value::Reference(variable_address) => {
                        log_trace!("Writing structure {}/{}", f, n);

                        let value_address = self.new_structure(f, n);

                        self.memory
                            .bind_variable_to_value(variable_address, value_address);
                    }
                    Value::Structure(found_f, found_n) => {
                        if f == found_f && n == found_n {
                            log_trace!("Reading structure {}/{}", f, n);

                            self.structure_iteration_state
                                .start_reading_structure(address);
                        } else {
                            self.backtrack()?;
                        }
                    }
                    Value::Constant(_) => self.backtrack()?,
                }
                Ok(())
            }
            Instruction::GetConstant { ai, c } => match self.get_register_value(ai).1 {
                Value::Reference(variable_address) => {
                    let value_address = self.new_constant(c);
                    self.memory
                        .bind_variable_to_value(variable_address, value_address);
                    Ok(())
                }
                Value::Constant(rc) if rc == c => Ok(()),
                _ => self.backtrack(),
            },
            Instruction::SetVariableXn { xn } => {
                let address = self.new_variable();
                self.registers.store(xn, address);
                self.structure_iteration_state
                    .write_next(&mut self.memory, address);
                Ok(())
            }
            Instruction::SetVariableYn { yn } => {
                let address = self.new_variable();
                self.memory.store_permanent_variable(yn, address);
                self.structure_iteration_state
                    .write_next(&mut self.memory, address);
                Ok(())
            }
            Instruction::SetValueXn { xn } => {
                let address = self.registers.load(xn);
                self.structure_iteration_state
                    .write_next(&mut self.memory, address);
                Ok(())
            }
            Instruction::SetValueYn { yn } => {
                let address = self.memory.load_permanent_variable(yn);
                self.structure_iteration_state
                    .write_next(&mut self.memory, address);
                Ok(())
            }
            Instruction::SetConstant { c } => {
                let address = self.new_constant(c);
                self.structure_iteration_state
                    .write_next(&mut self.memory, address);
                Ok(())
            }
            Instruction::SetVoid { n } => {
                for _ in 0..n.into_inner() {
                    let address = self.new_variable();
                    self.structure_iteration_state
                        .write_next(&mut self.memory, address);
                }
                Ok(())
            }
            Instruction::UnifyVariableXn { xn } => {
                self.unify_variable(xn, |this, xn, address| this.registers.store(xn, address));
                Ok(())
            }
            Instruction::UnifyVariableYn { yn } => {
                self.unify_variable(yn, |this, yn, address| {
                    this.memory.store_permanent_variable(yn, address);
                });
                Ok(())
            }
            Instruction::UnifyValueXn { xn } => {
                self.unify_value(xn, |this, xn| this.registers.load(xn))
            }
            Instruction::UnifyValueYn { yn } => {
                self.unify_value(yn, |this, yn| this.memory.load_permanent_variable(yn))
            }
            Instruction::UnifyConstant { c } => {
                match self.structure_iteration_state.read_write_mode() {
                    ReadWriteMode::Read => {
                        let term_address = self.structure_iteration_state.read_next(&self.memory);

                        let value = self.memory.get_value(term_address).1;

                        match value {
                            Value::Reference(variable_address) => {
                                let value_address = self.new_constant(c);
                                self.memory
                                    .bind_variable_to_value(variable_address, value_address);
                                // trail(address)
                                Ok(())
                            }
                            Value::Constant(c1) => {
                                if c == c1 {
                                    Ok(())
                                } else {
                                    self.backtrack()
                                }
                            }
                            Value::Structure(_, _) => self.backtrack(),
                        }
                    }
                    ReadWriteMode::Write => {
                        let address = self.new_constant(c);
                        self.structure_iteration_state
                            .write_next(&mut self.memory, address);

                        Ok(())
                    }
                }
            }
            Instruction::UnifyVoid { n } => {
                match self.structure_iteration_state.read_write_mode() {
                    ReadWriteMode::Read => {
                        self.structure_iteration_state.skip(&self.memory, n);
                        Ok(())
                    }
                    ReadWriteMode::Write => {
                        for _ in 0..n.0 {
                            let address = self.new_variable();
                            self.structure_iteration_state
                                .write_next(&mut self.memory, address);
                        }
                        Ok(())
                    }
                }
            }
            Instruction::Allocate { n } => {
                self.memory.allocate(n, self.cp, &[]);
                Ok(())
            }
            Instruction::Trim { n } => {
                self.memory.trim(n);
                Ok(())
            }
            Instruction::Deallocate => {
                self.cp = self.memory.deallocate();

                log_trace!("CP => {}", self.cp);

                Ok(())
            }
            Instruction::Call { p, n } => {
                match self.currently_executing {
                    CurrentlyExecuting::Query => {
                        self.currently_executing = CurrentlyExecuting::Program;

                        let registers = self.registers.query_registers(n);

                        for &value in registers {
                            log_trace!("Saved Register Value: {}", value);
                        }

                        self.memory.allocate(n, ProgramCounter::NULL, registers);

                        self.cp = ProgramCounter::NULL;
                    }
                    CurrentlyExecuting::Program => {
                        self.cp = self.pc;
                    }
                }
                self.pc = p;
                self.argument_count = n;

                self.registers.clear_above(n);

                // TODO - Set cut point

                Ok(())
            }
            Instruction::Execute { p, n } => {
                self.pc = p;
                self.argument_count = n;

                self.registers.clear_above(n);

                // TODO - Set cut point

                Ok(())
            }
            Instruction::Proceed => {
                log_trace!("Proceeding to {}", self.cp);
                self.pc = self.cp;
                Ok(())
            }
        }
    }

    fn get_register_value(&self, index: Ai) -> (Address, Value) {
        let (address, value, _) = self.memory.get_value(self.registers.load(index));
        (address, value)
    }

    pub fn solution_registers(&self) -> impl Iterator<Item = Address> + '_ {
        self.memory.solution_registers()
    }

    pub fn lookup_memory(
        &self,
        address: Address,
    ) -> (Address, Value, impl Iterator<Item = Address> + '_) {
        self.memory.get_value(address)
    }

    fn new_variable(&mut self) -> Address {
        self.memory.new_variable()
    }

    fn new_structure(&mut self, f: Functor, n: Arity) -> Address {
        let structure_address = self.memory.new_structure(f, n);

        self.structure_iteration_state
            .start_writing(structure_address);

        structure_address
    }

    fn new_constant(&mut self, c: Constant) -> Address {
        self.memory.new_constant(c)
    }

    fn unify_variable<I>(&mut self, index: I, store: impl FnOnce(&mut Self, I, Address)) {
        match self.structure_iteration_state.read_write_mode() {
            ReadWriteMode::Read => {
                let term_address = self.structure_iteration_state.read_next(&self.memory);

                store(self, index, term_address);
            }
            ReadWriteMode::Write => {
                let address = self.new_variable();
                store(self, index, address);
                self.structure_iteration_state
                    .write_next(&mut self.memory, address);
            }
        }
    }

    fn unify_value<I>(
        &mut self,
        index: I,
        load: impl FnOnce(&mut Self, I) -> Address,
    ) -> Result<(), ExecutionFailure> {
        match self.structure_iteration_state.read_write_mode() {
            ReadWriteMode::Read => {
                let term_address = self.structure_iteration_state.read_next(&self.memory);

                let register_address = load(self, index);

                self.unify(register_address, term_address)
                    .or_else(|UnificationFailure| self.backtrack())
            }
            ReadWriteMode::Write => {
                let address = load(self, index);
                // TODO - Do we need a new variable?
                // self.new_variable_with_value(address);
                self.structure_iteration_state
                    .write_next(&mut self.memory, address);
                Ok(())
            }
        }
    }

    fn unify(&mut self, a1: Address, a2: Address) -> Result<(), UnificationFailure> {
        log_trace!("Unifying {} and {}", a1, a2);
        let (a1, v1, _) = self.memory.get_value(a1);
        let (a2, v2, _) = self.memory.get_value(a2);
        log_trace!("Resolved to {:?} @ {} and {:?} @ {}", v1, a1, v2, a2);
        match (a1, v1, a2, v2) {
            (a1, Value::Reference(_), a2, Value::Reference(_)) => {
                self.memory.bind_variables(a1, a2);
                Ok(())
            }
            (a1, Value::Reference(_), a2, value) => {
                assert!(!matches!(value, Value::Reference(_)));
                self.memory.bind_variable_to_value(a1, a2);
                Ok(())
            }
            (a1, value, a2, Value::Reference(_)) => {
                assert!(!matches!(value, Value::Reference(_)));
                self.memory.bind_variable_to_value(a2, a1);
                Ok(())
            }
            (a1, Value::Structure(f1, n1), a2, Value::Structure(f2, n2)) => {
                if f1 == f2 && n1 == n2 {
                    let mut terms_1 = StructureIterationState::structure_reader(a1);
                    let mut terms_2 = StructureIterationState::structure_reader(a2);
                    for _ in 0..n1.0 {
                        let a1 = terms_1.read_next(&self.memory);
                        let a2 = terms_2.read_next(&self.memory);
                        self.unify(a1, a2)?;
                    }

                    Ok(())
                } else {
                    Err(UnificationFailure)
                }
            }
            (_, Value::Constant(c1), _, Value::Constant(c2)) => {
                if c1 == c2 {
                    Ok(())
                } else {
                    Err(UnificationFailure)
                }
            }
            _ => Err(UnificationFailure),
        }
    }

    fn backtrack(&mut self) -> Result<(), ExecutionFailure> {
        log_debug!("BACKTRACK!");
        Err(ExecutionFailure::Failed)
    }
}
