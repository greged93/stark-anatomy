use std::ops;

use crate::field::utils::{extended_euclidean, multiplicative_inverse};

use super::base::I256;

#[derive(Copy, Clone)]
pub struct FieldElement {
    value: u128,
    prime: u128,
}

// `PRIME` is the expression `1 + 407 * 2u128.pow(119)` evaluated
// see: https://github.com/aszepieniec/stark-anatomy/blob/76c375505a28e7f02f8803f77f8d7620d834071d/docs/basic-tools.md?plain=1#L113-L119
pub const PRIME: u128 = 270497897142230380135924736767050121217;

// Macro to define FieldElement constants using the prime defined in `stark anatomy` tutorial
#[macro_export]
macro_rules! felt {
    ($value:expr) => {
        FieldElement {
            value: $value,
            prime: crate::field::types::field::PRIME,
        }
    };
}

// Define the constants using the macro
const ZERO: FieldElement = felt!(0);
const ONE: FieldElement = felt!(1);

impl FieldElement {
    pub fn new(num: u128) -> Self {
        Self {
            value: num % PRIME,
            prime: PRIME,
        }
    }

    pub fn pow(self, exponent: FieldElement) -> Self {
        let l = I256::from(self.value);
        let r = I256::from(exponent.value);
        let prime = I256::from(self.prime);

        let power = l.pow(r);
        let power = power % prime;
        let power: u128 = power.into();
        Self::new(power)
    }

    pub fn is_zero(&self) -> bool {
        self.value == 0
    }

    pub fn inverse(&self) -> Option<Self> {
        let r = I256::from(self.value);
        let prime = I256::from(self.prime);

        let inverse = multiplicative_inverse(r, prime)?;

        Some(Self::new(inverse.into()))
    }

    pub fn value(self) -> u128 {
        self.value
    }
}

impl ops::Add<FieldElement> for FieldElement {
    type Output = FieldElement;

    fn add(self, rhs: FieldElement) -> Self::Output {
        let l = I256::from(self.value);
        let r = I256::from(rhs.value);
        let prime = I256::from(self.prime);

        let sum = (l + r) % prime;
        Self::new(sum.into())
    }
}

impl ops::Sub<FieldElement> for FieldElement {
    type Output = FieldElement;

    fn sub(self, rhs: FieldElement) -> Self::Output {
        let l = I256::from(self.value);
        let r = I256::from(rhs.value);
        let prime = I256::from(self.prime);

        let diff = l - r;
        let diff = (diff + prime) % prime;
        Self::new(diff.into())
    }
}

impl ops::Mul<FieldElement> for FieldElement {
    type Output = FieldElement;

    fn mul(self, rhs: FieldElement) -> Self::Output {
        let l = I256::from(self.value);
        let r = I256::from(rhs.value);
        let prime = I256::from(self.prime);

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

        let l = I256::from(self.value);
        let r = I256::from(rhs.value);
        let prime = I256::from(self.prime);

        let (_, inverse, _) = extended_euclidean(r, prime);
        let quotient = (l * inverse) % prime;
        Self::new(quotient.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pow() {
        // Given
        let base = felt!(2);
        let exponent = felt!(160);

        // When
        let result = base.pow(exponent);

        // Then
        let expected = 242584109230747146804944788495759879579u128;
        assert_eq!(expected, result.value);
    }

    #[test]
    fn test_add() {
        // Given

        let a = felt!(PRIME - 10);
        let b = felt!(12);

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
        let b = felt!(12);

        // When
        let result = a - b;

        // Then
        let expected = felt!(PRIME - 12);
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_mul() {
        // Given
        let a = felt!(u64::MAX as u128 - 2);
        let b = felt!(u64::MAX as u128 - 1);

        // When
        let result = a * b;

        // Then
        let expected = felt!(69784469778708083235216150296170332165);
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_div() {
        // Given
        let a = felt!(u64::MAX as u128 - 2);
        let b = felt!(u64::MAX as u128 - 1);

        // When
        let result = a / b;

        // Then
        let expected = felt!(263166645724356846472197722797662682189);
        assert_eq!(expected.value, result.value)
    }

    #[test]
    fn test_inverse_of_negative_one() {
        // Given
        let a = felt!(PRIME - 1);
        // In modular arithmetic with respect to a prime, "prime - 1" is equivalent to "-1".
        // Explanation:
        // In modular arithmetic, two numbers are considered equivalent if their difference is a multiple of the modulus (in this case, PRIME).
        // For the number 1 and the number "prime - 1", their difference is PRIME itself, which is a multiple of PRIME.
        // Therefore, adding "prime - 1" (which is equivalent to subtracting 1) to any number modulo PRIME has the same effect as subtracting 1 from that number.
        // This makes "prime - 1" act as the additive inverse of 1, which is why it is equivalent to "-1".

        // When
        let inverse = a.inverse();

        // Then
        assert!(inverse.is_some()); // Ensure the inverse exists
        assert_eq!(PRIME - 1, inverse.unwrap().value());

        let product = a * inverse.unwrap();
        // Multiplying (PRIME - 1) * (PRIME - 1) can potentially result in a multiplication overflow
        // since the resulting value is just one less than PRIME squared. If not handled correctly,
        // this can produce erroneous results. In modular arithmetic, the correct result should be
        // (PRIME - 1) * (PRIME - 1) ≡ 1 (mod PRIME).
        assert_eq!(ONE.value, product.value);
    }

    #[test]
    fn test_inverse_of_one() {
        // Given
        let a = ONE;

        // When
        let inverse = a.inverse();

        // Then
        assert_eq!(ONE.value, inverse.unwrap().value); // The inverse of 1 mod any number is 1
    }

    #[test]
    fn test_inverse_near_half_prime() {
        // Given
        let a = felt!(PRIME / 2);

        // When
        let inverse = a.inverse();

        // Then
        assert!(inverse.is_some()); // Ensure the inverse exists

        // When multiplied by its inverse, it should return ONE modulo PRIME
        let product = a * inverse.unwrap();
        assert_eq!(ONE.value, product.value);
    }

    #[test]
    fn test_no_inverse_for_zero() {
        // Given
        let a = ZERO;

        // When
        let inverse = a.inverse();

        // Then
        assert!(inverse.is_none()); // 0 shouldn't have a multiplicative inverse in this field
    }
}
