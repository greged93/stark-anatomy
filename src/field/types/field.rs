use std::ops;

use crate::field::utils::extended_euclidean;

use super::base::I256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FieldElement {
    value: u128,
    prime: u128,
}

impl FieldElement {
    pub fn zero(prime: u128) -> Self {
        Self { value: 0, prime }
    }

    pub fn one(prime: u128) -> Self {
        Self { value: 1, prime }
    }

    pub fn new(num: u128, prime: u128) -> Self {
        Self {
            value: num % prime,
            prime,
        }
    }

    pub fn pow(self, exponent: FieldElement) -> Self {
        let l = I256::from(self.value);
        let r = I256::from(exponent.value);
        let prime = I256::from(self.prime);

        let power = l.pow(r);
        let power = power % prime;
        let power: u128 = power.into();
        Self::new(power, self.prime)
    }

    pub fn is_zero(&self) -> bool {
        self.value == 0
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
        Self::new(sum.into(), self.prime)
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
        Self::new(diff.into(), self.prime)
    }
}

impl ops::Mul<FieldElement> for FieldElement {
    type Output = FieldElement;

    fn mul(self, rhs: FieldElement) -> Self::Output {
        let l = I256::from(self.value);
        let r = I256::from(rhs.value);
        let prime = I256::from(self.prime);

        let product = (l * r) % prime;
        Self::new(product.into(), self.prime)
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
        Self::new(quotient.into(), self.prime)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const PRIME: u128 = 1 + 407 * 2u128.pow(119);

    #[test]
    fn test_pow() {
        // Given
        let base = FieldElement::new(2, PRIME);
        let exponent = FieldElement::new(160, PRIME);

        // When
        let result = base.pow(exponent);

        // Then
        let expected = 242584109230747146804944788495759879579u128;
        assert_eq!(expected, result.value);
    }

    #[test]
    fn test_add() {
        // Given
        let a = FieldElement::new(PRIME - 10, PRIME);
        let b = FieldElement::new(12, PRIME);

        // When
        let result = a + b;

        // Then
        let expected = FieldElement::new(2, PRIME);
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_sub() {
        // Given
        let a = FieldElement::new(0, PRIME);
        let b = FieldElement::new(12, PRIME);

        // When
        let result = a - b;

        // Then
        let expected = FieldElement::new(PRIME - 12, PRIME);
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_mul() {
        // Given
        let a = FieldElement::new(u64::MAX as u128 - 2, PRIME);
        let b = FieldElement::new(u64::MAX as u128 - 1, PRIME);

        // When
        let result = a * b;

        // Then
        let expected = FieldElement::new(69784469778708083235216150296170332165, PRIME);
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_div() {
        // Given
        let a = FieldElement::new(u64::MAX as u128 - 2, PRIME);
        let b = FieldElement::new(u64::MAX as u128 - 1, PRIME);

        // When
        let result = a / b;

        // Then
        let expected = FieldElement::new(263166645724356846472197722797662682189, PRIME);
        assert_eq!(expected.value, result.value)
    }
}
