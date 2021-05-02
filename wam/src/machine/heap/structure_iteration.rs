use super::{Address, Arity, Heap, TupleAddress};

pub enum ReadWriteMode {
    Read,
    Write,
}

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
}

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Default)]
pub struct StructureIterationState(Option<InnerState>);

impl StructureIterationState {
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
    ) -> Result<T>
    where
        H: core::ops::Deref<Target = Heap<'m>>,
    {
        let inner_state = self.0.as_mut().ok_or(Error::NotActive)?;

        let super::TermSlice { first_term, arity } =
            heap.structure_term_addresses(inner_state.address);

        if inner_state.index == arity {
            return Err(Error::NoMoreTerms);
        }

        let term_address = first_term + inner_state.index;

        inner_state.index.0 += 1;

        let result = action(heap, term_address);

        let index = inner_state.index;

        self.check_done(index, arity);

        Ok(result)
    }

    pub fn read_next(&mut self, heap: &Heap) -> Result<Address> {
        self.with_next(heap, |heap, address| {
            Address(heap.tuple_memory.load(address).unwrap())
        })
    }

    pub fn write_next(&mut self, heap: &mut Heap, address: Address) -> Result<()> {
        self.with_next(heap, |heap, term_address| {
            crate::log_trace!("Writing {} to {}", address, term_address);
            heap.tuple_memory.store(term_address, address.0).unwrap();
        })
    }

    pub fn skip(&mut self, heap: &Heap, n: Arity) -> Result<()> {
        let inner_state = self.0.as_mut().ok_or(Error::NotActive)?;

        let super::TermSlice { arity, .. } = heap.structure_term_addresses(inner_state.address);

        inner_state.index.0 += n.0;

        if inner_state.index > arity {
            return Err(Error::NoMoreTerms);
        }

        let index = inner_state.index;

        self.check_done(index, arity);

        Ok(())
    }
}
