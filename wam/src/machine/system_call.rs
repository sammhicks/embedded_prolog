use core::{fmt, marker::PhantomData};

use comms::HexNewType;

use crate::{log_info, log_trace};

use super::{
    basic_types::{Arity, IntegerSign, Xn},
    heap::{
        structure_iteration, IntegerEvaluationOutputLayout, MemoryError, OutOfMemory,
        UnificationError,
    },
    Heap, MaybeFree, RegisterBlock,
};

#[derive(Clone, Copy)]
pub struct Value {
    sign: IntegerSign,
    size: u128,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.sign, self.size)
    }
}

trait FromValue: Sized {
    fn from_value(value: Value) -> Option<Self>;
}

impl<T> FromValue for T
where
    T: TryFrom<u128, Error = core::num::TryFromIntError>
        + TryFrom<i128, Error = core::num::TryFromIntError>,
{
    fn from_value(Value { sign, size }: Value) -> Option<Self> {
        if let IntegerSign::Negative = sign {
            Self::try_from(size.checked_neg()?).ok()
        } else {
            Self::try_from(size).ok()
        }
    }
}

pub trait IntoValue: Sized {
    fn into_value(self) -> Value;
}

macro_rules! impl_unsigned_number_into_value {
    ($($t:ty),*) => {
        $(
            impl IntoValue for $t {
                fn into_value(self) -> Value {
                    u128::try_from(self)
                        .map(|size| Value {
                            sign: size.cmp(&0).into(),
                            size,
                        })
                        .unwrap()
                }
            }
        )*
    };
}

impl_unsigned_number_into_value!(u8, u16, u32, u64, u128);

macro_rules! impl_signed_number_into_value {
    ($($t:ty),*) => {
        $(
            impl IntoValue for $t {
                fn into_value(self) -> Value {
                    i128::try_from(self)
                        .map(|size| Value {
                            sign: size.cmp(&0).into(),
                            size: size.unsigned_abs(),
                        })
                        .unwrap()
                }
            }
        )*
    };
}

impl_signed_number_into_value!(i8, i16, i32, i64, i128);

#[allow(dead_code)]
#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
pub struct SystemCallIndexOutOfRange {
    system_call_index: SystemCallIndex,
    system_call_count: usize,
}

pub enum SystemCallError {
    UnificationFailure,
    RegisterBlockError(super::RegisterBlockError),
    MemoryError(MemoryError),
    StructureIteration(structure_iteration::Error),
    OutOfMemory(OutOfMemory),
    SystemCallIndexOutOfRange(SystemCallIndexOutOfRange),
}

impl From<super::RegisterBlockError> for SystemCallError {
    fn from(inner: super::RegisterBlockError) -> Self {
        Self::RegisterBlockError(inner)
    }
}

impl From<MemoryError> for SystemCallError {
    fn from(inner: MemoryError) -> Self {
        Self::MemoryError(inner)
    }
}

impl From<OutOfMemory> for SystemCallError {
    fn from(inner: OutOfMemory) -> Self {
        Self::OutOfMemory(inner)
    }
}

impl From<UnificationError> for SystemCallError {
    fn from(inner: UnificationError) -> Self {
        match inner {
            UnificationError::UnificationFailure => Self::UnificationFailure,
            UnificationError::OutOfMemory(oom) => Self::OutOfMemory(oom),
            UnificationError::MemoryError(error) => Self::MemoryError(error),
            UnificationError::StructureIteration(error) => Self::StructureIteration(error),
        }
    }
}

pub struct Storage<T: IntoValue> {
    register_index: Xn,
    integer: IntegerEvaluationOutputLayout,

    _data: PhantomData<fn() -> T>,
}

impl<T: IntoValue> Storage<T> {
    fn write(self, machine: &mut Machine, value: T) -> Result<(), SystemCallError> {
        let value = value.into_value();

        let value_address = self.integer.registry_address();

        machine.memory.write_system_call_integer(
            self.integer,
            value.sign,
            &value.size.to_le_bytes(),
        )?;

        machine
            .memory
            .unify(machine.registers.load(self.register_index)?, value_address)?;

        Ok(())
    }
}

pub struct Machine<'me, 'memory> {
    pub registers: &'me mut RegisterBlock,
    pub memory: &'me mut Heap<'memory>,
}

pub struct MachineState<'me, 'memory> {
    machine: Machine<'me, 'memory>,
    current_register: Xn,
}

impl<'me, 'memory> MachineState<'me, 'memory> {
    fn new(machine: Machine<'me, 'memory>) -> Self {
        Self {
            machine,
            current_register: Xn { xn: 0 },
        }
    }

    fn next(&mut self) -> Xn {
        let index = self.current_register;
        self.current_register.xn += 1;
        index
    }

    fn read<T: FromValue>(&mut self) -> Result<T, SystemCallError> {
        fn write<T>((dest, src): (&mut T, T)) {
            *dest = src;
        }

        let register_index = self.next();

        log_trace!("Reading from {:?}", register_index);

        let (_address, value) = self
            .machine
            .memory
            .get_value(self.machine.registers.load(register_index).unwrap())?;

        let MaybeFree::BoundTo(super::Value::Integer { sign, le_bytes, .. }) = value else { return Err(SystemCallError::UnificationFailure) };

        let mut le_bytes = le_bytes.into_iter();

        let mut buffer = [0_u8; core::mem::size_of::<u128>()];

        buffer.iter_mut().zip(le_bytes.by_ref()).for_each(write);

        if le_bytes.any(|b| b == 0) {
            return Err(SystemCallError::UnificationFailure);
        }

        let size = u128::from_le_bytes(buffer);

        T::from_value(Value { sign, size }).ok_or(SystemCallError::UnificationFailure)
    }

    fn request_storage<T: IntoValue>(&mut self) -> Result<Storage<T>, SystemCallError> {
        let register_index = self.next();
        log_trace!("Requesting storage into {:?}", register_index);

        let integer = self.machine.memory.new_system_call_integer_output()??;

        Ok(Storage {
            register_index,
            integer,
            _data: PhantomData,
        })
    }
}

macro_rules! count {
    () => {
        0
    };

    ($name:ident $(, $rest:ident)*) => {
        1 + count!($($rest),*)
    };
}

pub trait MachineReadable: Sized {
    const SIZE: Arity;

    fn read(machine: &mut MachineState) -> Result<Self, SystemCallError>;
}

impl MachineReadable for () {
    const SIZE: Arity = Arity(0);

    fn read(_machine: &mut MachineState) -> Result<Self, SystemCallError> {
        Ok(())
    }
}

macro_rules! implement_machine_readable {
    () => {};
    ($name:ident $(, $names:ident)*) => {
        impl <$name: FromValue $(,$names: FromValue)*> MachineReadable for ($name, $($names,)*) {
            const SIZE: Arity = Arity(count!($name $(,$names)*));

            #[allow(non_snake_case)]
            fn read(machine: &mut MachineState) -> Result<Self, SystemCallError> {
                let $name = machine.read()?;
                $(
                    let $names = machine.read()?;
                )*
                Ok(($name, $($names,)*))
            }
        }

        implement_machine_readable!($($names),*);
    };
}

pub struct WithMutState<I>(I);

impl<I: MachineReadable> MachineReadable for WithState<I> {
    const SIZE: Arity = I::SIZE;

    fn read(machine: &mut MachineState) -> Result<Self, SystemCallError> {
        I::read(machine).map(WithState)
    }
}

pub struct WithState<I>(I);

impl<I: MachineReadable> MachineReadable for WithMutState<I> {
    const SIZE: Arity = I::SIZE;

    fn read(machine: &mut MachineState) -> Result<Self, SystemCallError> {
        I::read(machine).map(WithMutState)
    }
}

pub trait MachineWritable {
    type Storage;

    const SIZE: Arity;

    fn request_storage(machine: &mut MachineState) -> Result<Self::Storage, SystemCallError>;
    fn write(
        self,
        machine: &mut MachineState,
        storage: Self::Storage,
    ) -> Result<(), SystemCallError>;
}

impl<T: IntoValue> MachineWritable for T {
    type Storage = Storage<T>;

    const SIZE: Arity = Arity(1);

    fn request_storage(machine: &mut MachineState) -> Result<Self::Storage, SystemCallError> {
        machine.request_storage()
    }

    fn write(
        self,
        machine: &mut MachineState,
        storage: Self::Storage,
    ) -> Result<(), SystemCallError> {
        storage.write(&mut machine.machine, self)
    }
}

impl MachineWritable for () {
    type Storage = ();

    const SIZE: Arity = Arity(0);

    fn request_storage(_machine: &mut MachineState) -> Result<Self::Storage, SystemCallError> {
        Ok(())
    }

    fn write(self, _machine: &mut MachineState, (): Self::Storage) -> Result<(), SystemCallError> {
        Ok(())
    }
}

macro_rules! implement_machine_writable {
    () => {};
    ($storage:ident:$name:ident $(, $storages:ident:$names:ident)*) => {
        impl <$name: IntoValue $(,$names: IntoValue)*> MachineWritable for ($name, $($names,)*) {
            type Storage = (Storage<$name>, $(Storage<$names>,)*);

            const SIZE: Arity = Arity(count!($name $(,$names)*));

            #[allow(non_snake_case)]
            fn request_storage(machine: &mut MachineState) -> Result<Self::Storage, SystemCallError> {
                let $name = machine.request_storage()?;
                $(
                    let $names = machine.request_storage()?;
                )*
                Ok(($name, $($names,)*))
            }

            #[allow(non_snake_case)]
            fn write(self, machine: &mut MachineState, storage: Self::Storage) -> Result<(), SystemCallError> {
                let ($name, $($names,)*) = self;
                let ($storage, $($storages,)*) = storage;
                $name.write(machine, $storage)?;
                $(
                    $names.write(machine, $storages)?;
                )*
                Ok(())
            }
        }

        implement_machine_writable!($($storages:$names),*);
    }
}

pub struct Handler<F, I: MachineReadable, O: MachineWritable, S> {
    name: &'static str,
    f: F,
    _data: PhantomData<fn(&S, &I, &O)>,
}

pub trait ExecuteHandler<I: MachineReadable, O: MachineWritable, S> {
    fn execute(&self, state: &mut S, input: I) -> O;
}

impl<F: Fn() -> O, O: MachineWritable, S> ExecuteHandler<(), O, S> for Handler<F, (), O, S> {
    fn execute(&self, _state: &mut S, (): ()) -> O {
        (self.f)()
    }
}

impl<F: Fn(&S) -> O, O: MachineWritable, S> ExecuteHandler<WithState<()>, O, S>
    for Handler<F, WithState<()>, O, S>
{
    fn execute(&self, state: &mut S, WithState(()): WithState<()>) -> O {
        (self.f)(state)
    }
}

impl<F: Fn(&mut S) -> O, O: MachineWritable, S> ExecuteHandler<WithMutState<()>, O, S>
    for Handler<F, WithMutState<()>, O, S>
{
    fn execute(&self, state: &mut S, WithMutState(()): WithMutState<()>) -> O {
        (self.f)(state)
    }
}

macro_rules! implement_execute_handler {
    () => {};
    ($name:ident $(, $names:ident)*) => {
        impl<F: Fn($name $(, $names)*) -> O, $name: FromValue $(, $names: FromValue)*, O: MachineWritable, S> ExecuteHandler<($name, $($names,)*), O, S>
            for Handler<F, ($name, $($names,)*), O, S>
        {
            #[allow(non_snake_case)]
            fn execute(&self, _state: &mut S, input: ($name, $($names,)*)) -> O {
                let ($name, $($names,)*) = input;
                (self.f)($name $(, $names)*)
            }
        }

        impl<F: Fn(&S, $name $(, $names)*) -> O, $name: FromValue $(, $names: FromValue)*, O: MachineWritable, S> ExecuteHandler<WithState<($name, $($names,)*)>, O, S>
            for Handler<F, WithState<($name, $($names,)*)>, O, S>
        {
            #[allow(non_snake_case)]
            fn execute(&self, state: &mut S, input: WithState<($name, $($names,)*)>) -> O {
                let WithState(($name, $($names,)*)) = input;
                (self.f)(state, $name $(, $names)*)
            }
        }

        impl<F: Fn(&mut S, $name $(, $names)*) -> O, $name: FromValue $(, $names: FromValue)*, O: MachineWritable, S> ExecuteHandler<WithMutState<($name, $($names,)*)>, O, S>
            for Handler<F, WithMutState<($name, $($names,)*)>, O, S>
        {
            #[allow(non_snake_case)]
            fn execute(&self, state: &mut S, input: WithMutState<($name, $($names,)*)>) -> O {
                let WithMutState(($name, $($names,)*)) = input;
                (self.f)(state, $name $(, $names)*)
            }
        }

        implement_execute_handler!($($names),*);
    };
}

macro_rules! implement_traits {
    ($($storage:ident:$name:ident,)*) => {
        implement_machine_readable!($($name),*);
        implement_machine_writable!($($storage:$name),*);
        implement_execute_handler!($($name),*);
    };
}

implement_traits!(
    S01: T01,
    S02: T02,
    S03: T03,
    S04: T04,
    S05: T05,
    S06: T06,
    S07: T07,
    S08: T08,
    S09: T09,
    S10: T10,
    S11: T11,
    S12: T12,
    S13: T13,
    S14: T14,
    S15: T15,
    S16: T16,
);

impl<F, I: MachineReadable, O: MachineWritable, S> Handler<F, I, O, S>
where
    Handler<F, I, O, S>: ExecuteHandler<I, O, S>,
{
    const ARITY: Arity = Arity(I::SIZE.0 + O::SIZE.0);
}

pub trait SystemCall {
    type State;

    fn name(&self) -> &'static str;
    fn arity(&self) -> Arity;

    fn system_call<'me, 'memory>(
        &self,
        machine: Machine<'me, 'memory>,
        state: &mut Self::State,
    ) -> Result<(), SystemCallError>;
}

impl<F, I: MachineReadable, O: MachineWritable, S> SystemCall for Handler<F, I, O, S>
where
    Handler<F, I, O, S>: ExecuteHandler<I, O, S>,
{
    type State = S;

    fn name(&self) -> &'static str {
        self.name
    }

    fn arity(&self) -> Arity {
        Self::ARITY
    }

    fn system_call<'me, 'memory>(
        &self,
        machine: Machine<'me, 'memory>,
        state: &mut Self::State,
    ) -> Result<(), SystemCallError> {
        let mut machine = MachineState::new(machine);

        let input = I::read(&mut machine)?;
        let storage = O::request_storage(&mut machine)?;
        let output = self.execute(state, input);
        output.write(&mut machine, storage)?;

        Ok(())
    }
}

#[derive(Clone, Copy, HexNewType)]
#[cfg_attr(feature = "defmt-logging", derive(comms::HexDefmt))]
pub struct SystemCallIndex(pub u8);

pub trait SystemCalls {
    fn count(&self) -> u16;

    fn for_each_call<F: FnMut(&str, Arity) -> Result<(), E>, E>(&self, f: F) -> Result<(), E>;

    fn execute(&mut self, machine: Machine, i: SystemCallIndex) -> Result<(), SystemCallError>;
}

pub struct SystemCallEncoder<'a, Calls: SystemCalls>(pub &'a Calls);

impl<'a, Calls: SystemCalls, C> minicbor::Encode<C> for SystemCallEncoder<'a, Calls> {
    fn encode<W: minicbor::encode::Write>(
        &self,
        e: &mut minicbor::Encoder<W>,
        ctx: &mut C,
    ) -> Result<(), minicbor::encode::Error<W::Error>> {
        let Self(calls) = self;
        e.array(calls.count().into())?;

        calls.for_each_call(|name, Arity(arity)| (name, arity).encode(e, ctx))?;

        Ok(())
    }
}

struct SystemCallSet<'a, State> {
    system_calls: &'a [&'a dyn SystemCall<State = State>],
    state: State,
}

impl<'a, State> SystemCalls for SystemCallSet<'a, State> {
    fn count(&self) -> u16 {
        self.system_calls.len() as u16
    }

    fn for_each_call<F: FnMut(&str, Arity) -> Result<(), E>, E>(&self, mut f: F) -> Result<(), E> {
        for system_call in self.system_calls {
            f(system_call.name(), system_call.arity())?;
        }

        Ok(())
    }

    fn execute(&mut self, machine: Machine, i: SystemCallIndex) -> Result<(), SystemCallError> {
        self.system_calls
            .get(usize::from(i.0))
            .ok_or(SystemCallError::SystemCallIndexOutOfRange(
                SystemCallIndexOutOfRange {
                    system_call_index: i,
                    system_call_count: self.system_calls.len(),
                },
            ))
            .and_then(|system_call| {
                log_info!("Calling {}/{}", system_call.name(), system_call.arity());
                system_call.system_call(machine, &mut self.state)
            })
    }
}

pub fn system_calls<'a, State>(
    system_calls: &'a [&'a dyn SystemCall<State = State>],
    state: State,
) -> impl SystemCalls + '_ {
    SystemCallSet {
        system_calls,
        state,
    }
}

pub const fn system_call_handler<F, I: MachineReadable, O: MachineWritable, S>(
    name: &'static str,
    f: F,
) -> Handler<F, I, O, S>
where
    Handler<F, I, O, S>: ExecuteHandler<I, O, S>,
{
    Handler {
        name,
        f,
        _data: PhantomData,
    }
}
