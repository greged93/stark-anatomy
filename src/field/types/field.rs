use std::ops;

use crate::field::utils::extended_eucledian;

use super::base::I256;

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
}

impl ops::Add<FieldElement> for FieldElement {
    type Output = FieldElement;

    fn add(self, rhs: FieldElement) -> Self::Output {
        let l = I256::from(self.value);
        let r = I256::from(rhs.value);

        let sum = l + r;
        let sum: u128 = sum.into();
        Self::new(sum, self.prime)
    }
}

impl ops::Sub<FieldElement> for FieldElement {
    type Output = FieldElement;

    fn sub(self, rhs: FieldElement) -> Self::Output {
        let l = I256::from(self.value);
        let r = I256::from(rhs.value);
        let prime = I256::from(self.prime);

        let diff = l - r;
        let diff = diff % prime;
        let diff: u128 = diff.into();
        Self::new(diff, self.prime)
    }
}

impl ops::Mul<FieldElement> for FieldElement {
    type Output = FieldElement;

    fn mul(self, rhs: FieldElement) -> Self::Output {
        let l = I256::from(self.value);
        let r = I256::from(rhs.value);
        let prime = I256::from(self.prime);

        let product = l * r;
        let product = product % prime;
        let product: u128 = product.into();
        Self::new(product, self.prime)
    }
}

impl ops::Div<FieldElement> for FieldElement {
    type Output = FieldElement;

    fn div(self, rhs: FieldElement) -> Self::Output {
        let l = I256::from(self.value);
        let r = I256::from(rhs.value);
        let prime = I256::from(self.prime);

        let (inverse, _, _) = extended_eucledian(r, prime);
        let quotient = l * inverse;
        let quotient = quotient % prime;
        let quotient: u128 = quotient.into();
        Self::new(quotient, self.prime)
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
}
