#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]

use core::{fmt, num::NonZeroU16};

pub use minicbor;

use comms_derive::HexNewType;

#[derive(HexNewType, minicbor::Encode, minicbor::Decode)]
#[cfg_attr(feature = "defmt", derive(comms_derive::HexDefmt))]
#[cbor(transparent)]
pub struct Functor(#[n(0)] pub u16);

#[derive(HexNewType, minicbor::Encode, minicbor::Decode)]
#[cfg_attr(feature = "defmt", derive(comms_derive::HexDefmt))]
#[cbor(transparent)]
pub struct Arity(#[n(0)] pub u8);

#[derive(HexNewType, minicbor::Encode, minicbor::Decode)]
#[cfg_attr(feature = "defmt", derive(comms_derive::HexDefmt))]
#[cbor(transparent)]
pub struct Constant(#[n(0)] pub u16);

#[derive(HexNewType, minicbor::Encode, minicbor::Decode)]
#[cfg_attr(feature = "defmt", derive(comms_derive::HexDefmt))]
#[cbor(transparent)]
pub struct Address(#[n(0)] pub NonZeroU16);

#[derive(PartialEq, Eq)]
pub struct Hash(pub sha2::digest::Output<sha2::Sha256>);

impl Hash {
    pub fn new(bytes: &[u8]) -> Self {
        use sha2::Digest;
        Hash(sha2::Sha256::digest(bytes))
    }
}

impl fmt::UpperHex for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl fmt::Debug for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperHex::fmt(&self, f)
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for Hash {
    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(fmt, "{:02X}", self.0.as_slice())
    }
}

impl<C> minicbor::Encode<C> for Hash {
    fn encode<W: minicbor::encode::Write>(
        &self,
        e: &mut minicbor::Encoder<W>,
        _ctx: &mut C,
    ) -> Result<(), minicbor::encode::Error<W::Error>> {
        e.bytes(self.0.as_slice())?.ok()
    }
}

impl<'b, C> minicbor::Decode<'b, C> for Hash {
    fn decode(
        d: &mut minicbor::Decoder<'b>,
        _ctx: &mut C,
    ) -> Result<Self, minicbor::decode::Error> {
        d.bytes().and_then(|bytes| {
            <[u8; 32]>::try_from(bytes)
                .map(|bytes| Hash(bytes.into()))
                .map_err(|_| minicbor::decode::Error::message("Bad Hash Length"))
        })
    }
}

#[derive(Debug, minicbor::Encode, minicbor::Decode)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct CodeSubmission {
    #[n(0)]
    pub code_length: usize,
    #[n(1)]
    pub hash: Hash,
}

impl CodeSubmission {
    pub fn new(words: &[[u8; 4]]) -> Self {
        use sha2::digest::{Digest, FixedOutput};
        let mut hasher = sha2::Sha256::new();

        for word in words {
            hasher.update(word);
        }

        Self {
            code_length: words.len(),
            hash: Hash(hasher.finalize_fixed()),
        }
    }
}

#[derive(Debug, minicbor::Encode, minicbor::Decode)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum Command {
    #[n(0)]
    ReportStatus,
    #[n(1)]
    GetSystemCalls,
    #[n(2)]
    SubmitProgram {
        #[n(0)]
        code_submission: CodeSubmission,
    },
    #[n(3)]
    SubmitQuery {
        #[n(0)]
        code_submission: CodeSubmission,
    },
    #[n(4)]
    LookupMemory {
        #[n(0)]
        address: Address,
    },
    #[n(5)]
    NextSolution,
}

struct ErrorWriter<'a, W: minicbor::encode::Write> {
    encoder: &'a mut minicbor::Encoder<W>,
    state: Result<(), minicbor::encode::Error<W::Error>>,
}

impl<'a, W: minicbor::encode::Write> core::fmt::Write for ErrorWriter<'a, W> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        if self.state.is_ok() {
            self.encoder.str(s).map_err(|err| {
                self.state = Err(err);

                core::fmt::Error
            })?;

            Ok(())
        } else {
            Err(core::fmt::Error)
        }
    }
}

pub struct CommandResponse<T, E>(Result<T, E>);

impl<T, E> CommandResponse<T, E> {
    pub fn into_response(self) -> Result<T, E> {
        self.0
    }
}

impl<'b, C, T: minicbor::Decode<'b, C>, E: minicbor::Decode<'b, C>> minicbor::Decode<'b, C>
    for CommandResponse<T, E>
{
    fn decode(d: &mut minicbor::Decoder<'b>, ctx: &mut C) -> Result<Self, minicbor::decode::Error> {
        {
            let mut d = d.clone();

            let position = d.position();

            if Some(2) != d.array()? {
                return Err(
                    minicbor::decode::Error::message("expected enum (2-element array)")
                        .at(position),
                );
            }
            if d.u32()? == 0 {
                return Ok(Self(Err(d.decode_with::<C, E>(ctx)?)));
            }
        }

        Ok(Self(Ok(d.decode_with::<C, T>(ctx)?)))
    }
}

#[derive(Debug, minicbor::Encode, minicbor::Decode)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum ReportStatusResponse {
    #[n(1)]
    WaitingForProgram,
    #[n(2)]
    WaitingForQuery,
}

#[derive(minicbor::Encode, minicbor::Decode)]
pub enum GetSystemCallsResponse<Calls> {
    #[n(3)]
    SystemCalls(#[n(0)] Calls),
}

#[derive(minicbor::Encode, minicbor::Decode)]
pub enum SubmitProgramResponse {
    #[n(4)]
    Success,
}

#[derive(minicbor::Encode, minicbor::Decode)]
pub enum Solution<SolutionRegisters> {
    #[n(0)]
    SingleSolution(#[n(0)] SolutionRegisters),
    #[n(1)]
    MultipleSolutions(#[n(0)] SolutionRegisters),
}

#[derive(minicbor::Encode, minicbor::Decode)]
pub enum SubmitQueryResponse<SolutionRegisters> {
    #[n(5)]
    NoSolution,
    #[n(6)]
    Solution(#[n(0)] Solution<SolutionRegisters>),
}

#[derive(minicbor::Encode, minicbor::Decode)]
pub enum IntegerSign {
    #[n(0)]
    Negative,
    #[n(1)]
    Zero,
    #[n(2)]
    Positive,
}

#[derive(minicbor::Encode, minicbor::Decode)]
pub enum Value<Terms, IntegerLeBytes> {
    #[n(0)]
    FreeVariable,
    #[n(1)]
    Structure(#[n(0)] Functor, #[n(1)] Terms),
    #[n(2)]
    List(#[n(0)] Option<Address>, #[n(1)] Option<Address>),
    #[n(3)]
    Constant(#[n(0)] Constant),
    #[n(4)]
    Integer {
        #[n(0)]
        sign: IntegerSign,
        #[n(1)]
        le_bytes: IntegerLeBytes,
    },
}

#[derive(minicbor::Encode, minicbor::Decode)]
pub enum LookupMemoryResponse<Terms, IntegerLeBytes> {
    #[n(7)]
    MemoryValue {
        #[n(0)]
        address: Address,
        #[n(1)]
        value: Value<Terms, IntegerLeBytes>,
    },
}
