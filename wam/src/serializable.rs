pub trait Serializable {
    type Bytes: Default + AsRef<[u8]> + AsMut<[u8]>;
    fn from_be_bytes(bytes: Self::Bytes) -> Self;
    fn into_be_bytes(self) -> Self::Bytes;
}

impl Serializable for u8 {
    type Bytes = [u8; 1];

    fn from_be_bytes(bytes: Self::Bytes) -> Self {
        Self::from_be_bytes(bytes)
    }

    fn into_be_bytes(self) -> Self::Bytes {
        Self::to_be_bytes(self)
    }
}

impl Serializable for u16 {
    type Bytes = [u8; 2];

    fn from_be_bytes(bytes: Self::Bytes) -> Self {
        Self::from_be_bytes(bytes)
    }

    fn into_be_bytes(self) -> Self::Bytes {
        Self::to_be_bytes(self)
    }
}

impl Serializable for u32 {
    type Bytes = [u8; 4];

    fn from_be_bytes(bytes: Self::Bytes) -> Self {
        Self::from_be_bytes(bytes)
    }

    fn into_be_bytes(self) -> Self::Bytes {
        Self::to_be_bytes(self)
    }
}

pub trait SerializableWrapper {
    type Inner: Serializable;
    fn from_inner(inner: Self::Inner) -> Self;
    fn into_inner(self) -> Self::Inner;
}

impl<W> Serializable for W
where
    W: SerializableWrapper,
    W::Inner: Serializable,
{
    type Bytes = <W::Inner as Serializable>::Bytes;

    fn from_be_bytes(bytes: Self::Bytes) -> Self {
        Self::from_inner(W::Inner::from_be_bytes(bytes))
    }

    fn into_be_bytes(self) -> Self::Bytes {
        self.into_inner().into_be_bytes()
    }
}
