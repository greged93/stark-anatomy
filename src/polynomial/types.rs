use crate::field::types::field::FieldElement;

#[derive(Clone)]
pub struct Polynomial(pub Vec<FieldElement>);

// mostly following the impl from: https://github.com/lambdaclass/STARK101-rs/blob/main/stark101/src/polynomial.rs

const PRIME: u128 = 1 + 407 * 2u128.pow(119);

// univariate
impl Polynomial {

    pub fn new(coefficients: &[FieldElement]) -> Self {
        Polynomial(coefficients.into())
    }

    // degree is the length of the list minus the number of trailing zeroes
    pub fn degree(&self) -> isize {
        Self::trim_trailing_zeroes(&self.0).len() as isize - 1
    }

    fn trim_trailing_zeroes(p: &[FieldElement]) -> Vec<FieldElement> {

        Self::remove_trailing_elements(p, &FieldElement::zero(PRIME))

    }
    
    fn remove_trailing_elements(
        elements: &[FieldElement],
        element_to_remove: &FieldElement,
    ) -> Vec<FieldElement> {
        let it = elements
            .into_iter()
            .skip_while(|x| *x == element_to_remove)
            .map(Clone::clone);
        let mut v =  it.collect::<Vec<FieldElement>>();
        v.reverse();
        v

    }

    pub fn is_zero(&self) -> bool {
        Self::degree(&self) == -1
    }

    pub fn leading_coefficient(&self) -> FieldElement {
        let degree = Self::degree(self);
        if degree == -1 {
           FieldElement::zero(PRIME)
        } else {
            self.0.get(degree as usize)
        }
    }
    

}


#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_degree_zero() {
        // Given
        let zeroes = [FieldElement::zero(PRIME)];
        let zero_polynomial = Polynomial::new(&zeroes);

        // When
        let result = zero_polynomial.degree();

        // Then
        let expected = -1;
        assert_eq!(expected, result);

        assert!(zero_polynomial.is_zero(), "the laws of physics have changed, report to your local diety and repent for your sins");
    }

    #[test]    
    fn test_degree_one() {
        // Given
        let coefficients = [FieldElement::one(PRIME), FieldElement::zero(PRIME)];
        let zero_polynomial = Polynomial::new(&coefficients);

        // When
        let result = zero_polynomial.degree();

        // Then
        let expected = -1;
        assert_eq!(expected, result);
    }
}

