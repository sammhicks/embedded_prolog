use super::{Address, Arity, Heap, TupleAddress};
use crate::{log_trace, machine::heap::TupleEntry};

#[derive(Clone)]
pub enum ReadWriteMode {
    Read,
    Write,
}

#[derive(Clone)]
struct InnerState {
    read_write_mode: ReadWriteMode,
    address: Address,
    index: Arity,
}

#[derive(Debug)]
pub enum Error {
    NotActive,
    CurrentlyActive,
    NoMoreTerms,
    NoTermAt { index: Arity },
    Memory(super::MemoryError),
}

impl From<super::MemoryError> for Error {
    fn from(memory_error: super::MemoryError) -> Self {
        Self::Memory(memory_error)
    }
}

impl From<super::TupleMemoryError> for Error {
    fn from(error: super::TupleMemoryError) -> Self {
        Self::Memory(error.into())
    }
}

impl From<super::NoRegistryEntryAt> for Error {
    fn from(error: super::NoRegistryEntryAt) -> Self {
        Self::Memory(error.into())
    }
}

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Clone)]
pub struct State(Option<InnerState>);

impl State {
    pub fn new() -> Self {
        Self(None)
    }

    pub fn verify_not_active(&self) -> Result<()> {
        if self.0.is_none() {
            Ok(())
        } else {
            Err(Error::CurrentlyActive)
        }
    }

    pub fn read_write_mode(&self) -> Result<&ReadWriteMode> {
        Ok(&self.0.as_ref().ok_or(Error::NotActive)?.read_write_mode)
    }

    pub fn structure_reader(address: Address) -> Self {
        Self(Some(InnerState {
            read_write_mode: ReadWriteMode::Read,
            address,
            index: Arity::ZERO,
        }))
    }

    pub fn reset(&mut self) {
        self.0 = None;
    }

    pub fn start_reading(&mut self, address: Address) -> Result<()> {
        self.verify_not_active()?;
        *self = Self::structure_reader(address);
        Ok(())
    }

    pub fn start_writing(&mut self, address: Address) -> Result<()> {
        self.verify_not_active()?;
        *self = Self(Some(InnerState {
            read_write_mode: ReadWriteMode::Write,
            address,
            index: Arity::ZERO,
        }));
        Ok(())
    }

    fn check_done(&mut self, index: Arity, terms_count: Arity) {
        if index == terms_count {
            log_trace!("Finished iterating over structure");
            self.0 = None;
        }
    }

    fn with_next<'m, T, H>(
        &mut self,
        heap: H,
        action: impl FnOnce(H, Arity, TupleAddress) -> core::result::Result<T, Error>,
    ) -> Result<T>
    where
        H: core::ops::Deref<Target = Heap<'m>>,
    {
        let inner_state = self.0.as_mut().ok_or(Error::NotActive)?;

        let super::TermsSlice {
            first_term,
            terms_count,
        } = heap.structure_term_addresses(inner_state.address)?;

        if inner_state.index == terms_count {
            return Err(Error::NoMoreTerms);
        }

        let term_address = first_term + inner_state.index;

        inner_state.index.0 += 1;

        let result = action(heap, inner_state.index, term_address)?;

        let index = inner_state.index;

        self.check_done(index, terms_count);

        Ok(result)
    }

    pub fn read_next(&mut self, heap: &Heap) -> Result<Address> {
        self.with_next(heap, |heap, index, address| {
            Address::from_word(heap.tuple_memory.load(address)?).ok_or(Error::NoTermAt { index })
        })
    }

    pub fn write_next(&mut self, heap: &mut Heap, address: Address) -> Result<()> {
        self.with_next(heap, |heap, index, term_address| {
            log_trace!("Writing {} to {}", address, index);
            heap.tuple_memory.store(term_address, address.encode())?;
            heap.mark_moved_value(Some(address))?;
            Ok(())
        })
    }

    pub fn skip(&mut self, heap: &Heap, n: Arity) -> Result<()> {
        let inner_state = self.0.as_mut().ok_or(Error::NotActive)?;

        let super::TermsSlice { terms_count, .. } =
            heap.structure_term_addresses(inner_state.address)?;

        inner_state.index.0 += n.0;

        if inner_state.index > terms_count {
            return Err(Error::NoMoreTerms);
        }

        let index = inner_state.index;

        self.check_done(index, terms_count);

        Ok(())
    }
}
