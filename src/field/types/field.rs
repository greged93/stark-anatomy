use std::ops;

use crate::field::utils::extended_eucledian;

use super::base::I256;

pub struct FieldElement {
    value: u128,
    prime: u128,
}

impl FieldElement {
    pub fn new(num: u128, prime: u128) -> Self {
        Self {
            value: num % prime,
            prime,
        }
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
