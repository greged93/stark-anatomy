use crate::felt;

use crate::field::types::field::{FieldElement, ONE, PRIME, ZERO};
use crate::field::utils::multiplicative_inverse;

#[derive(Clone, Debug)]
pub struct Polynomial(pub Vec<FieldElement>);

// mostly following the impl from:
//// https://github.com/lambdaclass/STARK101-rs/blob/main/stark101/src/polynomial.rs
//// and
//// https://github.com/aszepieniec/stark-anatomy/blob/master/code/univariate.py#L3

// univariate
impl Polynomial {
    pub fn new(coefficients: &[FieldElement]) -> Self {
        Polynomial(coefficients.into())
    }

    // degree is the length of the list minus the number of trailing zeroes
    pub fn degree(&self) -> isize {
        Self::trim_trailing_zeroes(&self.0).len() as isize - 1
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
    fn trim_trailing_zeroes(p: &[FieldElement]) -> Vec<FieldElement> {
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
    pub fn qdiv(&self, other: impl Into<Polynomial>) -> (Polynomial, Polynomial) {
        // Convert the input to a Polynomial and trim its trailing zeros.
        let other_poly: Polynomial = other.into();
        let other_elems = Polynomial::trim_trailing_zeroes(&other_poly.0);
        assert!(!other_elems.is_empty(), "Dividing by zero polynomial.");

        // Trim trailing zeros from the self polynomial.
        let self_elems = Polynomial::trim_trailing_zeroes(&self.0);

        // If the dividend polynomial is a zero polynomial, the quotient and remainder are both zero.
        if self_elems.is_empty() {
            return (Polynomial(vec![]), Polynomial(vec![]));
        }

        // Begin polynomial long division.
        // Initialize the remainder (rem) as the dividend polynomial.
        let mut rem = self_elems.clone();
        let mut degree_difference = rem.len() as isize - other_elems.len() as isize;

        // Initialize the quotient polynomial with enough zeros to match the potential degree.
        let mut quotient = if degree_difference > 0 {
            vec![ZERO].repeat((degree_difference + 1) as usize).to_vec()
        } else {
            vec![ZERO]
        };

        // Log initial values
        println!("Initial remainder: {:?}", rem);
        println!("Initial quotient: {:?}", quotient);
        println!("Initial divisor: {:?}", other_elems);
        println!("Initial degree_difference: {}", degree_difference);

        // Continue the division as long as the degree of the remainder is at least the degree of the divisor.
        let mut iteration = 1;
        while degree_difference >= 0 {
            println!("Starting iteration {}", iteration);

            // Compute the factor that makes the leading terms cancel each other out.
            let tmp = rem.last().unwrap().to_owned() * other_elems.last().unwrap().inverse();
            println!("Computed tmp: {:?}", tmp);

            // Update the quotient polynomial with this factor.
            quotient[degree_difference as usize] = quotient[degree_difference as usize] + tmp;
            println!("Quotient after update: {:?}", quotient);

            // Initialize the index of the last non-zero coefficient of the remainder.
            let mut last_non_zero = degree_difference as isize - 1;

            // Subtract the divisor polynomial times the computed factor from the remainder.
            for (i, coef) in other_elems.iter().enumerate() {
                let k = i + degree_difference as usize;
                rem[k] = rem[k] - (tmp * *coef);

                // Update the index if the coefficient is non-zero.
                if rem[k] != ZERO {
                    last_non_zero = k as isize
                }
            }

            println!("Remainder after subtraction: {:?}", rem);

            // Remove trailing zeros from the remainder.
            rem = rem.into_iter().take((last_non_zero + 1) as usize).collect();
            println!("Remainder after trimming: {:?}", rem);

            // Update the degree difference for the next iteration.
            degree_difference = rem.len() as isize - other_elems.len() as isize;
            println!(
                "Degree difference for next iteration: {}",
                degree_difference
            );

            iteration += 1;
        }

        // Return the quotient and remainder after removing any potential trailing zeros from the quotient.
        (
            Polynomial(Self::trim_trailing_zeroes(&quotient)),
            Polynomial(rem),
        )
    }

    /// `two_list_tuple_operation` takes two lists of field elements (representing polynomial coefficients)
    /// and an operation function, then returns a list resulting from applying the operation elementwise.
    /// This method also takes an `id_element` which is used to fill in missing coefficients when the
    /// two polynomials don't have the same degree.
    ///
    /// For instance, if you're adding two polynomials, the `id_element` would be zero.
    ///
    /// We've refactored the method for the following reasons:
    ///
    /// 1. Avoiding unnecessary length matching of the two lists by appending zeros.
    ///    This is because adding zeros doesn't change the value of a polynomial, but it does affect its
    ///    representation. A polynomial [1, 0, 0] is the same as [1].
    ///
    /// 2. Ensuring that the resulting polynomial has its trailing zeros removed.
    ///    This makes operations like equality checks and degree calculations consistent and accurate.
    fn two_list_tuple_operation<F>(
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
            let elem1 = l1.get(i).cloned().unwrap_or(id_element.clone());
            let elem2 = l2.get(i).cloned().unwrap_or(id_element.clone());

            result.push(operation(elem1, elem2));
        }

        // Ensure that the result doesn't have unnecessary trailing zeros.
        // This ensures that the polynomial's representation is as concise as possible.
        Self::trim_trailing_zeroes(&result)
    }

    /// Constructs the monomial coefficient * x^degree.
    pub fn monomial(degree: usize, coefficient: FieldElement) -> Self {
        let mut coefficients = [ZERO].repeat(degree);
        coefficients.push(coefficient);
        Polynomial::new(&coefficients)
    }

    /// Computes the product of the given polynomials.
    pub fn prod(values: &[Polynomial]) -> Polynomial {
        let len_values = values.len();
        if len_values == 0 {
            return Polynomial(vec![ONE]);
        };
        if len_values == 1 {
            return values.first().unwrap().to_owned().into();
        };
        let prod_left = values
            .iter()
            .take(len_values / 2)
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        let prod_right = values
            .iter()
            .skip(len_values / 2)
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        Self::prod(&prod_left) * Self::prod(&prod_right)
    }

    /// Given the x_values for evaluating some polynomials,
    /// it computes part of the lagrange polynomials required to interpolate a polynomial over this domain.
    pub fn calculate_lagrange_polynomials(x_values: &[FieldElement]) -> Vec<Self> {
        let mut lagrange_polynomials = vec![];
        let monomials = x_values
            .iter()
            .map(|x| Self::monomial(1, ONE) - Self::monomial(0, *x))
            .collect::<Vec<_>>();
        let numerator = Self::prod(&monomials);
        for j in 0..(x_values.len()) {
            // In the denominator, we have:
            // (x_j-x_0)(x_j-x_1)...(x_j-x_{j-1})(x_j-x_{j+1})...(x_j-x_{len(X)-1})
            let denominator_values = x_values
                .iter()
                .enumerate()
                .filter(|(i, _)| dbg!(*i) != j)
                .map(|(_, x)| {
                    let poly: Polynomial = Polynomial(vec![x_values[j]]);
                    let x_poly: Polynomial = Polynomial(vec![*x]);
                    dbg!(&poly, &x_poly);
                    dbg!(poly - x_poly)
                })
                .collect::<Vec<_>>();
            let denominator = Polynomial::prod(&denominator_values);
            // Numerator is a bit more complicated, since we need to compute a poly multiplication here.
            //  Similarly to the denominator, we have:
            // (x-x_0)(x-x_1)...(x-x_{j-1})(x-x_{j+1})...(x-x_{len(X)-1})
            dbg!(
                &j,
                &x_values,
                &numerator,
                &denominator,
                &denominator_values,
                &monomials[j]
            );
            let (cur_poly, _) = numerator.qdiv(dbg!(monomials[j].clone() * denominator));
            lagrange_polynomials.push(cur_poly);
        }

        lagrange_polynomials
    }

    fn scalar_operation<F>(
        elements: &[FieldElement],
        operation: F,
        scalar: impl Into<FieldElement>,
    ) -> Vec<FieldElement>
    where
        F: Fn(FieldElement, FieldElement) -> FieldElement,
    {
        let value: FieldElement = scalar.into();
        elements.into_iter().map(|e| operation(*e, value)).collect()
    }

    /// Multiplies polynomial by a scalar.
    pub fn scalar_mul(&self, scalar: usize) -> Self {
        Polynomial(Self::scalar_operation(
            &self.0,
            |x, y| x * y,
            felt!(scalar as u128),
        ))
    }

    /// Interpolate the polynomials given a set of y_values.
    /// - y_values: y coordinates of the points.
    /// - lagrange_polynomials: the polynomials obtained from calculate_lagrange_polynomials.
    ///
    /// Returns the interpolated polynomial.
    pub fn interpolate_poly_lagrange(
        y_values: &[FieldElement],
        lagrange_polynomials: Vec<Self>,
    ) -> Self {
        let mut poly = Polynomial(vec![]);
        for (j, y_value) in y_values.iter().enumerate() {
            poly = poly + lagrange_polynomials[j].scalar_mul(y_value.value as usize);
        }
        poly
    }

    /// Returns a polynomial of degree < len(x_values) that evaluates to y_values[i] on x_values[i] for all i.
    pub fn interpolate(x_values: &[FieldElement], y_values: &[FieldElement]) -> Polynomial {
        assert!(x_values.len() == y_values.len());
        let lp = Self::calculate_lagrange_polynomials(x_values);
        dbg!(&lp);
        Self::interpolate_poly_lagrange(y_values, lp)
    }

    fn interpolate_domain(domain: &[FieldElement], values: &[FieldElement]) -> Polynomial {
        println!("domain: {:?}", domain);
        println!("values: {:?}", values);

        assert_eq!(domain.len(), values.len());
        let x = Polynomial::new(&[ZERO, ONE]);
        let mut acc = Polynomial::new(&[]);

        for i in 0..domain.len() {
            println!("\nStart of iteration i={}", i);

            let mut prod = Polynomial::new(&[values[i].clone()]);
            println!("Prod initialized to: {:?}", prod);

            for j in 0..domain.len() {
                if j == i {
                    continue;
                }

                let diff = domain[i] - domain[j];
                let inverse_diff = diff.inverse();

                println!("\nCalculating term for i={}, j={}", i, j);
                println!("domain[i] = {:?}", domain[i]);
                println!("domain[j] = {:?}", domain[j]);
                println!("diff = {:?}", diff);
                println!("inverse_diff = {:?}", inverse_diff);

                let term = x.clone()
                    - Polynomial::new(&[domain[j].clone()]) * Polynomial::new(&[inverse_diff]);

                println!("term = {:?}", term);

                println!("\nMultiplying prod and term:");
                println!("prod before = {:?}", prod);
                prod = prod * term;
                println!("prod after = {:?}", prod);
            }

            let evaluation = acc.evaluate(&domain[i]);
            dbg!(&evaluation);
            assert_eq!(
                evaluation, values[i],
                "Polynomial evaluation at domain[{}] = {:?} is not equal to values[{}]: {:?}",
                i, evaluation, i, values[i]
            );

            println!("\nAccumulator before adding prod:");
            println!("acc = {:?}", acc);
            acc = acc + prod;
            println!("acc after = {:?}", acc);

            println!("\nEvaluating polynomial:");
            println!("poly({:?}) = {:?}", domain[i], acc.evaluate(&domain[i]));
        }

        println!("\nFinal result:");
        println!("acc = {:?}", acc);

        acc
    }

    /// Evaluates the polynomial at a given point.
    pub fn evaluate(&self, point: &FieldElement) -> FieldElement {
        let mut xi = ONE;
        let mut value = ZERO;

        for c in &self.0 {
            value = value + (*c) * xi;
            xi = xi * *point;
        }

        value
    }
}

impl PartialEq for Polynomial {
    fn eq(&self, other: &Self) -> bool {
        let trimmed_self = Self::trim_trailing_zeroes(&self.0);
        let trimmed_other = Self::trim_trailing_zeroes(&other.0);
        trimmed_self == trimmed_other
    }
}

impl std::ops::Add for Polynomial {
    type Output = Polynomial;

    fn add(self, other: Self) -> Self::Output {
        let result = Self::two_list_tuple_operation(&self.0, &other.0, |x, y| x + y, ZERO);
        println!("Before trimming: {:?}", result); // Debug the output

        Polynomial(Self::trim_trailing_zeroes(&result))
    }
}

impl std::ops::Sub for Polynomial {
    type Output = Polynomial;

    fn sub(self, other: Self) -> Self::Output {
        println!("Adding polynomials: {:?} and {:?}", self, other);
        dbg!(Polynomial(Self::two_list_tuple_operation(
            &self.0,
            &other.0,
            |x, y| x - y,
            ZERO,
        )))
    }
}

impl std::ops::Neg for Polynomial {
    type Output = Polynomial;

    fn neg(self) -> Self::Output {
        Polynomial(vec![]) - self
    }
}

impl std::ops::Mul for Polynomial {
    type Output = Polynomial;

    fn mul(self, other: Self) -> Self::Output {
        // Allocate space for the result polynomial. The maximum possible degree of the resulting polynomial
        // after multiplication would be the sum of the degrees of the two input polynomials.
        // We add 2 to cover both zero-based indexing and potential leading terms.
        let mut res = vec![ZERO; (self.degree() + other.degree() + 2) as usize];

        // Iterate through each term of the first polynomial.
        for (i, c1) in self.0.iter().enumerate() {
            // For each term of the first polynomial, iterate through each term of the second polynomial.
            for (j, c2) in other.0.iter().enumerate() {
                // The resulting coefficient's position after multiplying two terms would be the sum of their positions.
                // Add the product of the two coefficients to the appropriate position in the result polynomial.
                println!("Multiplying terms: {:?} * {:?}", c1, c2); // Logging the terms being multiplied.
                if let Some(value) = res.get_mut(i + j) {
                    *value += *c1 * *c2;
                }
            }
        }

        // Log the result of the multiplication before trimming
        println!("Result of multiplication before trimming: {:?}", res);

        // Trim trailing zeroes. After performing the multiplication, it's possible that the resulting polynomial
        // has some trailing coefficients that are zero. These do not affect the polynomial's behavior and can be removed.
        Polynomial(Self::trim_trailing_zeroes(&res))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO: FieldElement = felt!(2);
    const THREE: FieldElement = felt!(3);
    const FOUR: FieldElement = felt!(4);
    const FIVE: FieldElement = felt!(5);
    const SIX: FieldElement = felt!(6);
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
    fn test_trim_trailing_zeroes() {
        let vec_with_zeros = vec![ONE, TWO, ZERO, ZERO];
        let trimmed = Polynomial::trim_trailing_zeroes(&vec_with_zeros);
        assert_eq!(trimmed, vec![FieldElement::new(1), FieldElement::new(2)]);
    }

    #[test]
    fn test_basic_operations() {
        let a = Polynomial::new(&[ONE, TWO]);
        let b = Polynomial::new(&[TWO, ONE]);

        // Testing addition
        assert_eq!(a.clone() + b.clone(), Polynomial::new(&[THREE, THREE]));

        // Testing multiplication
        assert_eq!(a.clone() * b.clone(), Polynomial::new(&[TWO, FIVE, TWO]));
    }

    #[test]
    fn test_internal_methods() {
        let p = Polynomial::new(&[ZERO, ONE, ZERO, TWO, ZERO, ZERO]);

        // Testing trim_trailing_zeroes
        assert_eq!(
            Polynomial::trim_trailing_zeroes(&p.0),
            vec![ZERO, ONE, ZERO, TWO],
            "trim trailing zeroes failed"
        );

        // Testing two_list_tuple_operation for addition
        let q = Polynomial::new(&[TWO, TWO]);
        let result = Polynomial::two_list_tuple_operation(&p.0, &q.0, |x, y| x + y, ZERO);
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

        let actual_product = a.clone() * b.clone();

        assert_eq!(
            actual_product, expected_product,
            "Failed polynomial multiplication test"
        );
    }

    #[test]
    fn test_polynomial_subtraction() {
        let a = Polynomial::new(&[ONE, ZERO, FIVE, TWO]);
        let b = Polynomial::new(&[TWO, TWO, ONE]);
        let expected_difference =
            Polynomial::new(&[felt!((PRIME - 1)), felt!((PRIME - 2)), felt!(4), TWO]); // Adjusted for correct size.

        let actual_difference = a.clone() - b.clone();

        assert_eq!(
            actual_difference, expected_difference,
            "Failed polynomial subtraction test"
        );
    }

    #[test]
    fn test_divide_by_degree_zero() {
        let a = Polynomial::new(&[THREE, felt!(4), FIVE]);
        let b = Polynomial::new(&[TWO]);

        let inv_2 = felt!(multiplicative_inverse(2i64.into(), PRIME.into())
            .unwrap()
            .into());

        let expected_quo = Polynomial::new(&[THREE * inv_2, felt!(4) * inv_2, FIVE * inv_2]);

        let (quo, rem) = a.qdiv(b);

        assert_eq!(quo, expected_quo);
        assert!(rem.is_zero());
    }

    #[test]
    fn test_polynomial_division_quotient() {
        let product = Polynomial::new(&[TWO, TWO, felt!(11), felt!(14), felt!(9), TWO]);
        let a = Polynomial::new(&[ONE, ZERO, FIVE, TWO]);
        let expected_quo = Polynomial::new(&[TWO, TWO, ONE]);

        let (quo, _rem) = product.qdiv(a.clone());

        assert_eq!(quo, expected_quo, "Quotient did not match expected value");
    }

    #[test]
    fn test_division() {
        let a = Polynomial::new(&[ONE, ZERO, FIVE, TWO]);
        let b = Polynomial::new(&[TWO, TWO, ONE]);
        let c = Polynomial::new(&[ZERO, FIVE, TWO, FIVE, FIVE, ONE]);

        let (quo, rem) = (a.clone() * b.clone()).qdiv(a.clone());

        assert!(rem.is_zero(), "division a*b/a should have no remainder");
        assert_eq!(quo, b.clone(), "division a*b/a should have quotient of b");

        let (quo, rem) = (a.clone() * b.clone()).qdiv(b.clone());

        assert!(rem.is_zero(), "division a*b/b should have no remainder");
        assert_eq!(quo, a, "division a*b/b should have quotient of a");

        let (quo, rem) = (a.clone() * b.clone()).qdiv(c.clone());

        assert!(
            !rem.is_zero(),
            "division a*b/b should not be divisible by c"
        );

        assert!(
            quo * c.clone() + rem == a.clone() * b.clone(),
            "quo * c + rem == a*b"
        );
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
                b.clone()
            );

            // Additional checks
            assert!(quo != zero, "Quotient shouldn't be zero for these cases");
            assert!(rem != one, "Remainder shouldn't be one for these cases");
        }
    }

    #[test]
    #[ignore]
    fn test_interpolate() {
        let values = vec![FIVE, TWO, TWO, ONE, FIVE, ZERO];
        let domain: Vec<FieldElement> = (1..=6).map(|i| felt!(i)).collect();

        // call into interpolate_domain
        let poly = Polynomial::interpolate(&domain, &values);

        // Assuming you've already defined or imported the needed types

        // Check that the polynomial evaluates correctly on each point in the domain
        for i in 0..domain.len() {
            assert_eq!(
                poly.evaluate(&domain[i]),
                values[i],
                "polynomial fails at domain={:?} at i={} ",
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
