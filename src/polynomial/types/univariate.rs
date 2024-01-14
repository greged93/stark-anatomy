#[allow(unused_imports)]
use crate::felt;

use crate::field::types::field::{FieldElement, ONE, ZERO};

#[derive(Clone, Debug)]
pub struct Polynomial(pub Vec<FieldElement>);

// univariate
impl Polynomial {
    pub fn new(coefficients: &[FieldElement]) -> Self {
        Self(coefficients.into())
    }

    // degree is the length of the list minus the number of trailing zeroes
    pub fn degree(&self) -> isize {
        Self::canonical_form(&self.0).len() as isize - 1
    }

    pub fn leading_coefficient(&self) -> FieldElement {
        let degree = Self::degree(self);
        if degree == -1 {
            ZERO
        } else {
            *self.0.get(degree as usize).unwrap()
        }
    }

    pub fn is_zero(&self) -> bool {
        Self::degree(self) == -1
    }

    /// Removes trailing zeros from the polynomial coefficients.
    ///
    /// The vector of `FieldElement`s representing a polynomial is treated as follows:
    /// - The index 0 element of the vector corresponds to the constant term of the polynomial.
    /// - Each subsequent index corresponds to the coefficient of the next highest power of x.
    ///   For example, the coefficient of x^2 would be found at index 2.
    /// - Therefore, trailing zeros in the vector represent zero coefficients for the highest degrees
    ///   of the polynomial, which do not affect the polynomial's value or behavior.
    ///
    /// Given this representation, to remove trailing zeros, we need to examine and remove elements
    /// from the end of the vector. To achieve this efficiently, we reverse the vector, skip the zeros,
    /// and then reverse it again to restore its original order.
    fn canonical_form(p: &[FieldElement]) -> Vec<FieldElement> {
        // 1. Start by obtaining an iterator over the coefficients.
        p.iter()
            // 2. Reverse the iterator to start from the end of the coefficients.
            .rev()
            // 3. Skip all consecutive zeros from the end.
            .skip_while(|&&x| x == ZERO)
            // 4. Clone each item to convert from &&FieldElement to FieldElement.
            .cloned()
            // 5. Convert the iterator to a Vec<FieldElement>.
            .collect::<Vec<_>>()
            // 6. Convert the Vec<FieldElement> back into an iterator.
            .into_iter()
            // 7. Reverse the order back to the original ordering.
            .rev()
            // 8. Finally, collect the results into a Vec<FieldElement> to be returned.
            .collect()
    }

    /// Divides the current polynomial by another polynomial.
    ///
    /// The method employs polynomial long division. Given two polynomials `A(x)` and `B(x)`,
    /// with `deg(A) >= deg(B)`, this method computes the quotient `Q(x)` and the remainder `R(x)`
    /// such that `A(x) = B(x) * Q(x) + R(x)`, where `deg(R) < deg(B)`.
    ///
    /// # Arguments
    ///
    /// * `other`: The divisor polynomial.
    ///
    /// # Returns
    ///
    /// A tuple containing the quotient `Q(x)` and remainder `R(x)`.
    ///
    /// # Panics
    ///
    /// This method panics if `other` is a zero polynomial, since division by zero is undefined.
    pub fn qdiv(&self, other: impl Into<Self>) -> (Self, Self) {
        // Convert the input to a Polynomial and trim its trailing zeros.
        let other_poly: Self = other.into();
        let other_elems = Self::canonical_form(&other_poly.0);
        assert!(!other_elems.is_empty(), "Dividing by zero polynomial.");

        // Trim trailing zeros from the self polynomial.
        let self_elems = Self::canonical_form(&self.0);

        // If the dividend polynomial is a zero polynomial, the quotient and remainder are both zero.
        if self_elems.is_empty() {
            return (Self(vec![]), Self(vec![]));
        }

        // Begin polynomial long division.
        // Initialize the remainder (rem) as the dividend polynomial.
        let mut rem = self_elems;
        let mut degree_difference = rem.len() as isize - other_elems.len() as isize;

        // Initialize the quotient polynomial with enough zeros to match the potential degree.
        let mut quotient = if degree_difference > 0 {
            vec![ZERO].repeat((degree_difference + 1) as usize).to_vec()
        } else {
            vec![ZERO]
        };

        // Continue the division as long as the degree of the remainder is at least the degree of the divisor.
        while degree_difference >= 0 {
            // Compute the factor that makes the leading terms cancel each other out.
            let tmp = rem.last().unwrap().to_owned() * other_elems.last().unwrap().inverse();

            // Update the quotient polynomial with this factor.
            quotient[degree_difference as usize] += tmp;

            // Initialize the index of the last non-zero coefficient of the remainder.
            let mut last_non_zero = degree_difference - 1;

            // Subtract the divisor polynomial times the computed factor from the remainder.
            for (i, coef) in other_elems.iter().enumerate() {
                let k = i + degree_difference as usize;
                rem[k] -= tmp * *coef;

                // Update the index if the coefficient is non-zero.
                if rem[k] != ZERO {
                    last_non_zero = k as isize
                }
            }
            // Remove trailing zeros from the remainder.
            rem = rem.into_iter().take((last_non_zero + 1) as usize).collect();
            // Update the degree difference for the next iteration.
            degree_difference = rem.len() as isize - other_elems.len() as isize;
        }

        // Return the quotient and remainder after removing any potential trailing zeros from the quotient.
        (Self(Self::canonical_form(&quotient)), Self(rem))
    }

    /// `apply_pairwise` takes two lists of field elements (representing polynomial coefficients)
    /// and an operation function, then returns a list resulting from applying the operation elementwise.
    /// This method also takes an `id_element` which is used to fill in missing coefficients when the
    /// two polynomials don't have the same degree.
    fn apply_pairwise<F>(
        l1: &[FieldElement],
        l2: &[FieldElement],
        operation: F,
        id_element: FieldElement,
    ) -> Vec<FieldElement>
    where
        F: Fn(FieldElement, FieldElement) -> FieldElement,
    {
        // Determine the maximum length between the two lists.
        // This will be the length of the resulting list.
        let max_len = std::cmp::max(l1.len(), l2.len());
        let mut result = Vec::with_capacity(max_len);

        // Iterate through the lists. If an element doesn't exist in one of the lists (because the list is shorter),
        // use the identity element (id_element) as its value.
        for i in 0..max_len {
            let elem1 = l1.get(i).cloned().unwrap_or(id_element);
            let elem2 = l2.get(i).cloned().unwrap_or(id_element);
            result.push(operation(elem1, elem2));
        }

        // Ensure that the result doesn't have unnecessary trailing zeros.
        // This ensures that the polynomial's representation is as concise as possible.
        Self::canonical_form(&result)
    }

    /// Interpolate the polynomials given a set of y_values.
    /// - y_values: y coordinates of the points.
    /// - lagrange_polynomials: the polynomials obtained from calculate_lagrange_polynomials.
    ///
    /// Returns the interpolated polynomial.
    #[allow(dead_code)]
    fn interpolate_domain(domain: &[FieldElement], values: &[FieldElement]) -> Self {
        let x = Self::new(&[ZERO, ONE]);

        let mut accumulator = Self::new(&[ZERO]);

        for i in 0..domain.len() {
            let mut product = Self::new(&[values[i]]);

            for j in 0..domain.len() {
                if i == j {
                    continue;
                }

                let term = (x.clone() - Self::new(&[domain[j]]))
                    * Self::new(&[(domain[i] - domain[j]).inverse()]);

                product = product * term;
            }

            accumulator = accumulator + product;
        }

        accumulator
    }

    /// Evaluates the polynomial at a given point.
    pub fn evaluate(&self, point: &FieldElement) -> FieldElement {
        let mut xi = ONE;
        let mut value = ZERO;

        for c in &self.0 {
            value += (*c) * xi;
            xi = xi * *point;
        }

        value
    }

    /// Calculates the zerofier polynomial for a given domain of field elements.
    ///
    /// The zerofier polynomial is a polynomial that has roots at each element of the domain.
    /// Given a domain `[d1, d2, ..., dn]`, the zerofier polynomial is calculated as:
    /// `(x - d1) * (x - d2) * ... * (x - dn)`.
    ///
    /// # Arguments
    ///
    /// * `domain` - A reference to a vector of field elements representing the domain.
    ///
    /// # Returns
    ///
    /// The zerofier polynomial for the given domain.
    pub fn zerofier(domain: &[FieldElement]) -> Self {
        // Define a polynomial `x` as `x - 0`, which is equivalent to the variable `x`.
        let x = Self(vec![ZERO, ONE]);

        // Initialize the accumulator polynomial `acc` as `1`.
        let mut acc = Self(vec![ONE]);

        // Iterate over each element `d` in the domain.
        for &d in domain.iter() {
            // Multiply the accumulator polynomial `acc` by the polynomial `x - d`.
            acc = acc * (x.clone() - Self(vec![d]));
        }

        // Normalize the coefficients of the accumulator polynomial.
        Self(Self::canonical_form(&acc.0))
    }

    /// Scales the polynomial by a given factor.
    ///
    /// This function multiplies each coefficient of the polynomial by the corresponding power of the factor.
    /// Specifically, the `i`-th coefficient is multiplied by `factor^i`.
    ///
    /// # Arguments
    ///
    /// * `factor` - The field element by which to scale the polynomial.
    ///
    /// # Returns
    ///
    /// A new polynomial that is the result of scaling the original polynomial by the given factor.
    pub fn scale(&self, factor: FieldElement) -> Self {
        // Clone the coefficients of the original polynomial.
        let mut res_coeffs = self.0.clone();

        // Initialize a variable `f` to keep track of the current power of the factor.
        let mut f = ONE;

        // Iterate over each coefficient of the polynomial.
        for (_i, coeff) in res_coeffs.iter_mut().enumerate() {
            // Multiply the `i`-th coefficient by `f`, which is `factor^i`.
            *coeff = *coeff * f;

            // Update `f` to the next power of the factor.
            f = f * factor;
        }

        // Normalize the coefficients of the resulting polynomial.
        Self(Self::canonical_form(&res_coeffs))
    }
}

impl PartialEq for Polynomial {
    fn eq(&self, other: &Self) -> bool {
        let trimmed_self = Self::canonical_form(&self.0);
        let trimmed_other = Self::canonical_form(&other.0);
        trimmed_self == trimmed_other
    }
}

impl std::ops::Add for Polynomial {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self(Self::apply_pairwise(&self.0, &other.0, |x, y| x + y, ZERO))
    }
}

impl std::ops::Sub for Polynomial {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self(Self::apply_pairwise(&self.0, &other.0, |x, y| x - y, ZERO))
    }
}

impl std::ops::Neg for Polynomial {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(vec![]) - self
    }
}

impl std::ops::Mul for Polynomial {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        // Initialize a vector to store the coefficients of the resulting polynomial.
        // The length of the resulting polynomial is the sum of the lengths of the input polynomials minus 1.
        let mut result_coeffs = vec![ZERO; self.0.len() + other.0.len() - 1];

        // Iterate over the coefficients of the first polynomial.
        for i in 0..self.0.len() {
            // Skip the iteration if the coefficient is zero to optimize the calculation.
            if !self.0[i].is_zero() {
                // Iterate over the coefficients of the second polynomial.
                for j in 0..other.0.len() {
                    // Multiply the current coefficients of the first and second polynomials,
                    // and add the result to the corresponding coefficient in the resulting polynomial.
                    result_coeffs[i + j] += self.0[i] * other.0[j];
                }
            }
        }

        // Create a new polynomial with the resulting coefficients.
        // The `canonical_form` function is used to normalize the coefficients.
        Self(Self::canonical_form(&result_coeffs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO: FieldElement = felt!(2);
    const THREE: FieldElement = felt!(3);
    const FOUR: FieldElement = felt!(4);
    const FIVE: FieldElement = felt!(5);
    const SEVEN: FieldElement = felt!(7);

    #[test]
    fn test_degree_zero() {
        // Given
        let zeroes = [ZERO];
        let zero_polynomial = Polynomial::new(&zeroes);

        // When
        let result = zero_polynomial.degree();

        // Then
        assert_eq!(-1, result);

        assert!(
            zero_polynomial.is_zero(),
            "the laws of physics have changed, report to your local diety and repent for your sins"
        );
    }

    #[test]
    fn test_degree_one() {
        // Given
        let coefficients = [ZERO, ONE];
        let one_polynomial = Polynomial::new(&coefficients);

        // When
        let result = one_polynomial.degree();

        // Then
        assert_eq!(1, result);
    }

    #[test]
    fn test_canonical_form() {
        let vec_with_zeros = vec![ONE, TWO, ZERO, ZERO];
        let trimmed = Polynomial::canonical_form(&vec_with_zeros);
        assert_eq!(trimmed, vec![FieldElement::new(1), FieldElement::new(2)]);
    }

    #[test]
    fn test_basic_operations() {
        let a = Polynomial::new(&[ONE, TWO]);
        let b = Polynomial::new(&[TWO, ONE]);

        // Testing addition
        assert_eq!(a.clone() + b.clone(), Polynomial::new(&[THREE, THREE]));

        // Testing multiplication
        assert_eq!(a * b, Polynomial::new(&[TWO, FIVE, TWO]));
    }

    #[test]
    fn test_internal_methods() {
        let p = Polynomial::new(&[ZERO, ONE, ZERO, TWO, ZERO, ZERO]);

        // Testing canonical_form
        assert_eq!(
            Polynomial::canonical_form(&p.0),
            vec![ZERO, ONE, ZERO, TWO],
            "trim trailing zeroes failed"
        );

        // Testing apply_pairwise for addition
        let q = Polynomial::new(&[TWO, TWO]);
        let result = Polynomial::apply_pairwise(&p.0, &q.0, |x, y| x + y, ZERO);
        assert_eq!(result, vec![TWO, THREE, ZERO, TWO], "addition no worky");
    }

    #[test]
    fn test_distributivity_degree_1() {
        let a = Polynomial::new(&[ONE, TWO]);
        let b = Polynomial::new(&[TWO, ONE]);
        let c = Polynomial::new(&[ZERO, FIVE]);

        let lhs = a.clone() * (b.clone() + c.clone());
        let rhs = (a.clone() * b) + (a * c);

        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_distributivity_degree_3() {
        let a = Polynomial::new(&[ONE, TWO, FIVE, THREE]); // Degree 3 polynomial
        let b = Polynomial::new(&[TWO, ONE, SEVEN, THREE]); // Degree 3 polynomial
        let c = Polynomial::new(&[ZERO, FIVE, TWO, THREE]); // Degree 3 polynomial

        let lhs = a.clone() * (b.clone() + c.clone());
        let rhs = (a.clone() * b) + (a * c);

        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_distributivity() {
        let a = Polynomial::new(&[ONE, ZERO, FIVE, TWO]);
        let b = Polynomial::new(&[TWO, TWO, ONE]);
        let c = Polynomial::new(&[ZERO, FIVE, TWO, FIVE, FIVE, ONE]);

        let lhs = (b.clone() + c.clone()) * a.clone();
        let rhs = (a.clone() * b) + (a * c);

        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_polynomial_multiplication() {
        let a = Polynomial::new(&[ONE, ZERO, FIVE, TWO]);
        let b = Polynomial::new(&[TWO, TWO, ONE]);
        let expected_product = Polynomial::new(&[TWO, TWO, felt!(11), felt!(14), felt!(9), TWO]);

        let actual_product = a * b;

        assert_eq!(
            actual_product, expected_product,
            "Failed polynomial multiplication test"
        );
    }

    #[test]
    fn test_polynomial_subtraction() {
        let a = Polynomial::new(&[ONE, ZERO, FIVE, TWO]);
        let b = Polynomial::new(&[TWO, TWO, ONE]);
        let expected_difference = Polynomial::new(&[-ONE, -TWO, FOUR, TWO]); // Adjusted for correct size.

        let actual_difference = a - b;

        assert_eq!(
            actual_difference, expected_difference,
            "Failed polynomial subtraction test"
        );
    }

    #[test]
    fn test_polynomial_division_quotient() {
        let product = Polynomial::new(&[TWO, TWO, felt!(11), felt!(14), felt!(9), TWO]);
        let a = Polynomial::new(&[ONE, ZERO, FIVE, TWO]);
        let expected_quo = Polynomial::new(&[TWO, TWO, ONE]);

        let (quo, _rem) = product.qdiv(a);

        assert_eq!(quo, expected_quo, "Quotient did not match expected value");
    }

    #[test]
    fn test_division() {
        let a = Polynomial::new(&[ONE, ZERO, FIVE, TWO]);
        let b = Polynomial::new(&[TWO, TWO, ONE]);
        let c = Polynomial::new(&[ZERO, FIVE, TWO, FIVE, FIVE, ONE]);

        let (quo, rem) = (a.clone() * b.clone()).qdiv(a.clone());

        assert!(rem.is_zero(), "division a*b/a should have no remainder");
        assert_eq!(quo, b, "division a*b/a should have quotient of b");

        let (quo, rem) = (a.clone() * b.clone()).qdiv(b.clone());

        assert!(rem.is_zero(), "division a*b/b should have no remainder");
        assert_eq!(quo, a, "division a*b/b should have quotient of a");

        let (quo, rem) = (a.clone() * b.clone()).qdiv(c.clone());

        assert!(
            !rem.is_zero(),
            "division a*b/b should not be divisible by c"
        );

        assert!(quo * c + rem == a * b, "quo * c + rem == a*b");
    }

    #[test]
    fn test_division_non_termination() {
        let zero = Polynomial::new(&[ZERO]);
        let one = Polynomial::new(&[ONE]);

        // Dividend with leading zeros and a degree far larger than the divisor
        let a = Polynomial::new(&[ZERO, ZERO, ZERO, ONE, ZERO, FIVE, TWO, THREE, felt!(4)]);

        let divisors = vec![
            Polynomial::new(&[TWO, TWO, ONE]), // Original divisor
            Polynomial::new(&[ZERO, ONE]),     // A small degree divisor
            Polynomial::new(&[FIVE, felt!(4), THREE, TWO, ONE]), // Degree same as dividend's effective degree
            Polynomial::new(&[ONE, ZERO, ZERO, ZERO, ZERO]),     // Mostly zeros
        ];

        for b in divisors {
            let (quo, rem) = a.clone().qdiv(b.clone());

            // If the division completes, check if the results are valid
            assert_eq!(
                a,
                (quo.clone() * b.clone()) + rem.clone(),
                "a should be equal to quo * {:?} + rem",
                b
            );

            // Additional checks
            assert!(quo != zero, "Quotient shouldn't be zero for these cases");
            assert!(rem != one, "Remainder shouldn't be one for these cases");
        }
    }

    #[test]
    fn test_interpolate() {
        let values = vec![FIVE, TWO, TWO, ONE, FIVE, ZERO];
        let domain: Vec<FieldElement> = (0..=5).map(|i| felt!(i)).collect();
        dbg!(&domain);
        // call into interpolate_domain
        let poly = Polynomial::interpolate_domain(&domain, &values);

        // Check that the polynomial evaluates correctly on each point in the domain
        for i in 0..domain.len() {
            assert_eq!(
                poly.evaluate(&domain[i]),
                values[i],
                "polynomial {:?} fails at domain={:?} at i={} ",
                poly,
                &domain[i],
                i
            );
        }

        // Evaluation in a random point is non-zero with high probability
        let random_point = felt!(363);
        assert_ne!(
            poly.evaluate(&random_point),
            ZERO,
            "fail interpolate test 2"
        );

        // Check that the polynomial's degree is correct
        assert_eq!(
            poly.degree(),
            (domain.len() as isize) - 1,
            "fail interpolate test 3"
        );
    }
}
