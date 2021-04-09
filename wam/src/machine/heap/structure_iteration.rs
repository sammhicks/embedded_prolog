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

#[derive(Default)]
pub struct StructureIterationState(Option<InnerState>);

impl StructureIterationState {
    pub fn verify_not_active(&self) {
        assert!(self.0.is_none())
    }

    pub fn read_write_mode(&self) -> &ReadWriteMode {
        &self
            .0
            .as_ref()
            .expect("Not iterating over state!")
            .read_write_mode
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

    pub fn start_reading(&mut self, address: Address) {
        assert!(matches!(&self.0, None));
        *self = Self::structure_reader(address);
    }

    pub fn start_writing(&mut self, address: Address) {
        assert!(matches!(&self.0, None));
        *self = Self(Some(InnerState {
            read_write_mode: ReadWriteMode::Write,
            address,
            index: Arity::ZERO,
        }));
    }

    fn check_done(&mut self, index: Arity, arity: Arity) {
        if index == arity {
            crate::log_trace!("Finished iterating over structure");
            self.0 = None;
        }
    }

    fn with_next<'m, T, H>(&mut self, heap: H, action: impl FnOnce(H, TupleAddress) -> T) -> T
    where
        H: core::ops::Deref<Target = Heap<'m>>,
    {
        let inner_state = self.0.as_mut().expect("Not reading or writing");

        let (first_term, arity) = heap.structure_term_addresses(inner_state.address);

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

        let arity = heap.structure_term_addresses(inner_state.address).1;

        inner_state.index.0 += n.0;

        if inner_state.index > arity {
            panic!("No more terms to read");
        }

        let index = inner_state.index;

        self.check_done(index, arity);
    }
}
