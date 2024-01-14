use std::ops;

use crate::field::utils::{extended_euclidean, multiplicative_inverse};

use std::fmt::{Debug, Formatter};

use super::base::I320;

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct FieldElement {
    pub value: u128,
    pub prime: u128,
}

// `PRIME` is the expression `1 + 407 * 2u128.pow(119)` evaluated
// see: https://github.com/aszepieniec/stark-anatomy/blob/76c375505a28e7f02f8803f77f8d7620d834071d/docs/basic-tools.md?plain=1#L113-L119
pub const PRIME: u128 = 270497897142230380135924736767050121217;

// Macro to define FieldElement constants
#[macro_export]
macro_rules! felt {
    ($value:expr) => {
        FieldElement {
            value: $value,
            prime: $crate::field::types::field::PRIME,
        }
    };
}

impl Debug for FieldElement {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.value())
    }
}

pub const ZERO: FieldElement = felt!(0);

pub const ONE: FieldElement = felt!(1);

impl FieldElement {
    pub fn new(num: u128) -> Self {
        Self {
            value: num % PRIME,
            prime: PRIME,
        }
    }

    pub fn pow(self, exponent: FieldElement) -> Self {
        let l = I320::from(self.value);
        let r = I320::from(exponent.value);
        let prime = I320::from(self.prime);

        let power = l.pow(r);
        let power = power % prime;
        let power: u128 = power.into();
        Self::new(power)
    }

    pub fn is_zero(&self) -> bool {
        self.value == 0
    }

    pub fn inverse(&self) -> Self {
        let r = I320::from(self.value);
        let prime = I320::from(self.prime);

        let inverse =
            multiplicative_inverse(r, prime).expect("Field Element has a multiplicative inverse");

        Self::new(inverse.into())
    }

    pub fn value(self) -> u128 {
        self.value
    }
}

impl std::ops::Neg for FieldElement {
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self.value == 0 {
            self
        } else {
            Self::new(self.prime - self.value)
        }
    }
}

impl ops::Add<FieldElement> for FieldElement {
    type Output = FieldElement;

    fn add(self, rhs: FieldElement) -> Self::Output {
        let l = I320::from(self.value);
        let r = I320::from(rhs.value);
        let prime = I320::from(self.prime);

        let sum = (l + r) % prime;
        Self::new(sum.into())
    }
}

impl std::ops::AddAssign for FieldElement {
    fn add_assign(&mut self, rhs: Self) {
        self.value =
            ((I320::from(self.value) + I320::from(rhs.value)) % I320::from(self.prime)).into();
    }
}

impl ops::Sub<FieldElement> for FieldElement {
    type Output = FieldElement;

    fn sub(self, rhs: FieldElement) -> Self::Output {
        let l = I320::from(self.value);
        let r = I320::from(rhs.value);
        let prime = I320::from(self.prime);

        let diff = l - r;
        let diff = (diff + prime) % prime;
        Self::new(diff.into())
    }
}

impl std::ops::SubAssign for FieldElement {
    fn sub_assign(&mut self, rhs: Self) {
        if self.value < rhs.value {
            self.value =
                ((I320::from(self.value) + I320::from(self.prime)) - I320::from(rhs.value)).into();
        } else {
            self.value = (I320::from(self.value) - I320::from(rhs.value)).into();
        }
    }
}

impl ops::Mul<FieldElement> for FieldElement {
    type Output = FieldElement;

    fn mul(self, rhs: FieldElement) -> Self::Output {
        let l = I320::from(self.value);
        let r = I320::from(rhs.value);
        let prime = I320::from(self.prime);

        let product = (l * r) % prime;
        Self::new(product.into())
    }
}

impl ops::Div<FieldElement> for FieldElement {
    type Output = FieldElement;

    fn div(self, rhs: FieldElement) -> Self::Output {
        if rhs.is_zero() {
            panic!("Cannot divide by zero");
        }

        let l = I320::from(self.value);
        let r = I320::from(rhs.value);
        let prime = I320::from(self.prime);

        let (_, inverse, _) = extended_euclidean(r, prime);
        let inverse = (inverse + prime) % prime;
        let quotient = (l * inverse) % prime;
        Self::new(quotient.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_inv_minus {
        ($value: expr, $minus: ident) => {
            #[test]
            fn $minus() {
                // Given
                let a = FieldElement::new(1);
                let b = FieldElement::new(PRIME - $value);

                // When
                let result = a / b;
                let result = result * b;

                // Then
                let expected = FieldElement::new(1);
                assert_eq!(expected.value, result.value);
            }
        };
    }

    #[test]
    fn test_pow() {
        // Given
        let base = FieldElement::new(2);
        let exponent = FieldElement::new(160);

        // When
        let result = base.pow(exponent);

        // Then
        let expected = 242584109230747146804944788495759879579u128;
        assert_eq!(expected, result.value);
    }

    #[test]
    fn test_add() {
        // Given

        let a = FieldElement::new(PRIME - 10);
        let b = FieldElement::new(12);

        // When
        let result = a + b;

        // Then
        let expected = FieldElement::new(2);
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_sub() {
        // Given
        let a = ZERO;
        let b = FieldElement::new(12);

        // When
        let result = a - b;

        // Then
        let expected = FieldElement::new(PRIME - 12);
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_mul() {
        // Given
        let a = FieldElement::new(u64::MAX as u128 - 2);
        let b = FieldElement::new(u64::MAX as u128 - 1);

        // When
        let result = a * b;

        // Then
        let expected = FieldElement::new(69784469778708083235216150296170332165);
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_div() {
        // Given
        let a = FieldElement::new(u64::MAX as u128 - 2);
        let b = FieldElement::new(u64::MAX as u128 - 1);

        // When
        let result = a / b;

        // Then
        let expected = FieldElement::new(263166645724356846472197722797662682189);
        assert_eq!(expected.value, result.value)
    }

    test_inv_minus!(1, test_inv_minus_1);
    test_inv_minus!(2, test_inv_minus_2);
    test_inv_minus!(3, test_inv_minus_3);
    test_inv_minus!(4, test_inv_minus_4);
    test_inv_minus!(5, test_inv_minus_5);
    test_inv_minus!(6, test_inv_minus_6);
    test_inv_minus!(7, test_inv_minus_7);
    test_inv_minus!(8, test_inv_minus_8);
    test_inv_minus!(9, test_inv_minus_9);
    test_inv_minus!(10, test_inv_minus_10);
    test_inv_minus!(11, test_inv_minus_11);
}
