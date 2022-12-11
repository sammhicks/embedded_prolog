use core::cmp::Ordering;

type Word = u16;
type DoubleWord = u32;

type WordUsage = u16;

struct SplitDouble {
    lesser: Word,
    greater: Word,
}

impl SplitDouble {
    fn new(d: DoubleWord) -> Self {
        Self {
            lesser: d as Word,
            greater: (d >> (8 * core::mem::size_of::<Word>())) as Word,
        }
    }
}

type Sign = Ordering;

struct UnsignedInput<'a> {
    words: &'a [Word],
}

impl<'a> UnsignedInput<'a> {
    fn new(words: &'a [Word]) -> Self {
        words
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, n)| (*n != 0).then_some(i))
            .map_or(Self { words: &[] }, |last_nonzero| Self {
                words: &words[..=last_nonzero],
            })
    }

    fn words(&self) -> impl Iterator<Item = Word> + 'a {
        self.words.iter().copied()
    }

    fn double_words(&self) -> impl Iterator<Item = DoubleWord> + 'a {
        self.words().map(From::from)
    }

    fn infinite_words(&self) -> impl Iterator<Item = Word> + 'a {
        self.words().chain(core::iter::repeat(0))
    }
}

impl<'a> PartialEq for UnsignedInput<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl<'a> Eq for UnsignedInput<'a> {}

impl<'a> PartialOrd for UnsignedInput<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for UnsignedInput<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.infinite_words()
            .zip(other.infinite_words())
            .take(usize::max(self.words.len(), other.words.len()))
            .fold(None, |current_cmp, (my_word, other_word)| {
                match my_word.cmp(&other_word) {
                    Ordering::Equal => current_cmp,
                    cmp @ (Ordering::Less | Ordering::Greater) => Some(cmp),
                }
            })
            .unwrap_or(Ordering::Equal)
    }
}

pub struct SignedInput<'a> {
    sign: Sign,
    size: UnsignedInput<'a>,
}

impl<'a> SignedInput<'a> {
    pub fn new(sign: Sign, words: &'a [Word]) -> Self {
        Self {
            sign,
            size: UnsignedInput::new(words),
        }
    }

    fn split(self) -> (Sign, UnsignedInput<'a>) {
        let Self { sign, size } = self;
        (sign, size)
    }
}

impl<'a> PartialEq for SignedInput<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl<'a> Eq for SignedInput<'a> {}

impl<'a> PartialOrd for SignedInput<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for SignedInput<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.sign, other.sign) {
            (Ordering::Equal, Ordering::Equal) => Ordering::Equal,
            (Ordering::Less, Ordering::Equal | Ordering::Greater)
            | (Ordering::Equal, Ordering::Greater) => Ordering::Less,
            (Ordering::Greater, Ordering::Less | Ordering::Equal)
            | (Ordering::Equal, Ordering::Less) => Ordering::Greater,
            (Ordering::Greater, Ordering::Greater) => self.size.cmp(&other.size),
            (Ordering::Less, Ordering::Less) => self.size.cmp(&other.size).reverse(),
        }
    }
}

impl<'a> core::ops::Neg for SignedInput<'a> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.sign = self.sign.reverse();
        self
    }
}

pub struct UnsignedOutput<'a> {
    words: &'a mut [Word],
}

impl<'a> UnsignedOutput<'a> {
    pub(super) unsafe fn new(
        data_start: *mut Word,
        data: super::IntegerEvaluationOutputData,
    ) -> Self {
        Self {
            words: core::slice::from_raw_parts_mut(
                data_start.offset(data.0.data_start.0 as isize),
                data.0.words_count.0.into(),
            ),
        }
    }

    fn as_ref(&self) -> UnsignedInput {
        UnsignedInput { words: self.words }
    }

    fn reborrow(&mut self) -> UnsignedOutput {
        UnsignedOutput { words: self.words }
    }

    pub fn usage(&self) -> WordUsage {
        self.words
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, n)| (*n != 0).then_some((i + 1) as WordUsage))
            .unwrap_or(0)
    }

    pub fn words_mut(&mut self) -> impl Iterator<Item = &mut Word> {
        self.words.iter_mut()
    }

    fn zero(mut self) -> (Sign, WordUsage) {
        for word in self.words_mut() {
            *word = 0;
        }

        (Sign::Equal, 0)
    }

    fn copy_from(self, (sign, src): (Sign, UnsignedInput)) -> (Sign, WordUsage) {
        self.words[..src.words.len()].copy_from_slice(src.words);
        (sign, self.usage())
    }
}

fn carrying_add(r1: Word, r2: Word, carry: bool) -> (Word, bool) {
    let (a, b) = r1.overflowing_add(r2);
    let (c, d) = a.overflowing_add(carry.into());
    (c, b || d)
}

fn borrowing_sub(r1: Word, r2: Word, borrow: bool) -> (Word, bool) {
    let (a, b) = r1.overflowing_sub(r2);
    let (c, d) = a.overflowing_sub(borrow.into());
    (c, b || d)
}

fn add_unsigned(mut r0: UnsignedOutput, r1: UnsignedInput, r2: UnsignedInput) -> WordUsage {
    r0.words_mut()
        .zip(r1.infinite_words().zip(r2.infinite_words()))
        .fold(false, |carry, (r0, (r1, r2))| {
            let (value, carry) = carrying_add(r1, r2, carry);
            *r0 = value;
            carry
        });

    r0.usage()
}

pub fn add_signed((r0, r1, r2): (UnsignedOutput, SignedInput, SignedInput)) -> (Sign, WordUsage) {
    match (r1.split(), r2.split()) {
        ((Ordering::Equal, _), r2) => r0.copy_from(r2),
        (r1, (Ordering::Equal, _)) => r0.copy_from(r1),
        ((Ordering::Greater, r1), (Ordering::Greater, r2)) => {
            (Sign::Greater, add_unsigned(r0, r1, r2))
        }
        ((Ordering::Greater, r1), (Ordering::Less, r2)) => sub_unsigned(r0, r1, r2),
        ((Ordering::Less, r1), (Ordering::Greater, r2)) => sub_unsigned(r0, r2, r1),
        ((Ordering::Less, r1), (Ordering::Less, r2)) => (Sign::Less, add_unsigned(r0, r1, r2)),
    }
}

fn sub_unsigned(r0: UnsignedOutput, r1: UnsignedInput, r2: UnsignedInput) -> (Sign, WordUsage) {
    fn sub_unsigned_unchecked(
        mut r0: UnsignedOutput,
        r1: UnsignedInput,
        r2: UnsignedInput,
    ) -> WordUsage {
        r0.words_mut()
            .zip(r1.infinite_words().zip(r2.infinite_words()))
            .fold(false, |carry, (r0, (r1, r2))| {
                let (value, carry) = borrowing_sub(r1, r2, carry);
                *r0 = value;
                carry
            });

        r0.usage()
    }

    match r1.cmp(&r2) {
        Ordering::Equal => (Sign::Equal, 0),
        Ordering::Less => (Ordering::Less, sub_unsigned_unchecked(r0, r2, r1)),
        Ordering::Greater => (Ordering::Greater, sub_unsigned_unchecked(r0, r1, r2)),
    }
}

pub fn sub_signed((r0, r1, r2): (UnsignedOutput, SignedInput, SignedInput)) -> (Sign, WordUsage) {
    add_signed((r0, r1, -r2))
}

fn mul_unsigned(r0: UnsignedOutput, r1: UnsignedInput, r2: UnsignedInput) -> WordUsage {
    for (i1, r1) in r1.double_words().enumerate() {
        r2.double_words()
            .chain(core::iter::once(0))
            .enumerate()
            .fold(0, |carry, (i2, r2)| {
                let SplitDouble { lesser, greater } = SplitDouble::new(
                    r1 * r2 + DoubleWord::from(carry) + DoubleWord::from(r0.words[i1 + i2]),
                );
                r0.words[i1 + i2] = lesser;
                greater
            });
    }

    r0.usage()
}

pub fn mul_signed((r0, r1, r2): (UnsignedOutput, SignedInput, SignedInput)) -> (Sign, WordUsage) {
    let (s1, r1) = r1.split();
    let (s2, r2) = r2.split();

    if s1.is_eq() || s2.is_eq() {
        r0.zero()
    } else {
        (
            if s1 == s2 {
                Ordering::Greater
            } else {
                Ordering::Less
            },
            mul_unsigned(r0, r1, r2),
        )
    }
}

fn div_mod_unsigned(
    rd: UnsignedOutput,
    rm: UnsignedOutput,
    r1: UnsignedInput,
    r2: UnsignedInput,
) -> (WordUsage, WordUsage) {
    fn sub_unsigned_assign_unchecked(mut r0: UnsignedOutput, r1: &UnsignedInput) {
        r0.words_mut()
            .zip(r1.infinite_words())
            .fold(false, |carry, (r0, r1)| {
                let (value, carry) = borrowing_sub(*r0, r1, carry);
                *r0 = value;
                carry
            });
    }

    for rd in rd.words.iter_mut() {
        *rd = 0;
    }

    for (rm, r1) in rm.words.iter_mut().zip(r1.infinite_words()) {
        *rm = r1;
    }

    for i in (0..rm.words.len()).rev() {
        let mut rm = UnsignedOutput {
            words: &mut rm.words[i..],
        };

        while rm.as_ref() >= r2 {
            sub_unsigned_assign_unchecked(rm.reborrow(), &r2);
            rd.words[i] += 1;
        }
    }

    (rd.usage(), rm.usage())
}

pub fn div_mod_signed(
    (rd, rm, r1, r2): (UnsignedOutput, UnsignedOutput, SignedInput, SignedInput),
) -> ((Sign, WordUsage), (Sign, WordUsage)) {
    let (s1, r1) = r1.split();
    let (s2, r2) = r2.split();

    if s1.is_eq() || s2.is_eq() {
        (rd.zero(), rm.zero())
    } else {
        let (ud, um) = div_mod_unsigned(rd, rm, r1, r2);
        let sd = if s1 == s2 {
            Ordering::Greater
        } else {
            Ordering::Less
        };
        let sm = s1;
        ((sd, ud), (sm, um))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write<T>((dest, src): (&mut T, T)) {
        *dest = src;
    }

    type Words = [Word; 8];

    fn borrow_signed<const N: usize>((sign, words): &(Sign, [Word; N])) -> SignedInput {
        SignedInput::new(*sign, words)
    }

    fn read_unsigned(words: Words, usage: WordUsage) -> u128 {
        let mut buffer = [0_u8; 16];

        buffer
            .iter_mut()
            .zip(
                words
                    .iter()
                    .take(usize::from(usage))
                    .copied()
                    .flat_map(Word::to_le_bytes),
            )
            .for_each(write);

        u128::from_le_bytes(buffer)
    }

    fn write_unsigned(n: u128) -> Words {
        let mut words = Words::default();

        let mut le_bytes = IntoIterator::into_iter(n.to_le_bytes()).chain(core::iter::repeat(0));

        words
            .iter_mut()
            .zip(core::iter::from_fn(|| {
                let mut buffer = [0_u8; core::mem::size_of::<Word>()];
                buffer.iter_mut().zip(le_bytes.by_ref()).for_each(write);
                Some(Word::from_le_bytes(buffer))
            }))
            .for_each(write);

        words
    }

    fn read_signed(((sign, usage), words): ((Sign, WordUsage), Words)) -> i128 {
        let n = read_unsigned(words, usage) as i128;

        if let Sign::Less = sign {
            -n
        } else {
            n
        }
    }

    fn write_signed(n: i128) -> (Sign, Words) {
        (n.cmp(&0), write_unsigned(n.unsigned_abs()))
    }

    #[test]
    fn unsigned_addition() {
        for _ in 0..1000 {
            let r1 = u128::from(rand::random::<u64>());
            let r2 = u128::from(rand::random::<u64>());

            let mut words = Words::default();

            let usage = add_unsigned(
                UnsignedOutput { words: &mut words },
                UnsignedInput::new(&write_unsigned(r1)),
                UnsignedInput::new(&write_unsigned(r2)),
            );

            let expected = r1 + r2;
            let actual = read_unsigned(words, usage);

            assert_eq!(expected, actual, "{r1} + {r2} = {expected}");
        }
    }

    #[test]
    fn signed_addition() {
        for _ in 0..1000 {
            let r1 = i128::from(rand::random::<i64>());
            let r2 = i128::from(rand::random::<i64>());

            let mut words = Words::default();

            let sign = add_signed((
                UnsignedOutput { words: &mut words },
                borrow_signed(&write_signed(r1)),
                borrow_signed(&write_signed(r2)),
            ));

            let expected = r1 + r2;
            let actual = read_signed((sign, words));

            assert_eq!(expected, actual, "{r1} + {r2} = {expected}");
        }
    }

    #[test]
    fn unsigned_subtraction() {
        for _ in 0..1000 {
            let r1 = rand::random::<u64>();
            let r2 = rand::random::<u64>();

            let mut words = Words::default();

            let sign = sub_unsigned(
                UnsignedOutput { words: &mut words },
                UnsignedInput::new(&write_unsigned(r1.into())),
                UnsignedInput::new(&write_unsigned(r2.into())),
            );

            let expected = i128::from(r1) - i128::from(r2);
            let actual = read_signed((sign, words));

            assert_eq!(expected, actual, "{r1} - {r2} = {expected}");
        }
    }

    #[test]
    fn signed_subtraction() {
        for _ in 0..1000 {
            let r1 = i128::from(rand::random::<i64>());
            let r2 = i128::from(rand::random::<i64>());

            let mut words = Words::default();

            let sign = sub_signed((
                UnsignedOutput { words: &mut words },
                borrow_signed(&write_signed(r1)),
                borrow_signed(&write_signed(r2)),
            ));

            let expected = r1 - r2;
            let actual = read_signed((sign, words));

            assert_eq!(expected, actual, "{r1} - {r2} = {expected}");
        }
    }

    #[test]
    fn unsigned_multiplication() {
        for _ in 0..1000 {
            let r1 = u128::from(rand::random::<u64>());
            let r2 = u128::from(rand::random::<u64>());

            let mut words = Words::default();

            let usage = mul_unsigned(
                UnsignedOutput { words: &mut words },
                UnsignedInput::new(&write_unsigned(r1)),
                UnsignedInput::new(&write_unsigned(r2)),
            );

            let expected = r1 * r2;
            let actual = read_unsigned(words, usage);

            assert_eq!(expected, actual, "{r1} * {r2} = {expected}");
        }
    }

    #[test]
    fn signed_multiplication() {
        for _ in 0..1000 {
            let r1 = i128::from(rand::random::<i64>());
            let r2 = i128::from(rand::random::<i64>());

            let mut words = Words::default();

            let sign = mul_signed((
                UnsignedOutput { words: &mut words },
                borrow_signed(&write_signed(r1)),
                borrow_signed(&write_signed(r2)),
            ));

            let expected = r1 * r2;
            let actual = read_signed((sign, words));

            assert_eq!(expected, actual, "{r1} * {r2} = {expected}");
        }
    }

    #[test]
    fn unsigned_div_mod() {
        for _ in 0..1000 {
            let r1 = u128::from(rand::random::<u64>());
            let r2 = u128::from(rand::random::<u64>());

            let mut words_div = Words::default();
            let mut words_mod = Words::default();

            let (usage_div, usage_mod) = div_mod_unsigned(
                UnsignedOutput {
                    words: &mut words_div,
                },
                UnsignedOutput {
                    words: &mut words_mod,
                },
                UnsignedInput::new(&write_unsigned(r1)),
                UnsignedInput::new(&write_unsigned(r2)),
            );

            let expected_div = r1 / r2;
            let expected_mod = r1 % r2;

            let actual_div = read_unsigned(words_div, usage_div);
            let actual_mod = read_unsigned(words_mod, usage_mod);

            assert_eq!(expected_div, actual_div, "{r1} / {r2} = {expected_div}");
            assert_eq!(expected_mod, actual_mod, "{r1} % {r2} = {expected_mod}");
        }
    }

    #[test]
    fn signed_div_mod() {
        for _ in 0..1000 {
            let r1 = i128::from(rand::random::<i64>());
            let r2 = i128::from(rand::random::<i64>());

            let mut words_div = Words::default();
            let mut words_mod = Words::default();

            let (sign_div, sign_mod) = div_mod_signed((
                UnsignedOutput {
                    words: &mut words_div,
                },
                UnsignedOutput {
                    words: &mut words_mod,
                },
                borrow_signed(&write_signed(r1)),
                borrow_signed(&write_signed(r2)),
            ));

            let expected_div = r1 / r2;
            let expected_mod = r1 % r2;

            let actual_div = read_signed((sign_div, words_div));
            let actual_mod = read_signed((sign_mod, words_mod));

            assert_eq!(expected_div, actual_div, "{r1} / {r2} = {expected_div}");
            assert_eq!(expected_mod, actual_mod, "{r1} % {r2} = {expected_mod}");
        }
    }
}
