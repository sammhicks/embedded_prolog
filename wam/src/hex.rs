use core::fmt::{self, Formatter, LowerHex, UpperHex};

pub struct Hex<'a>(pub &'a [u8]);

impl<'a> LowerHex for Hex<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for b in self.0 {
            write!(f, "{:02x}", b)?;
        }
        Ok(())
    }
}

impl<'a> UpperHex for Hex<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for b in self.0 {
            write!(f, "{:02X}", b)?;
        }
        Ok(())
    }
}
