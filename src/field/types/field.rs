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
