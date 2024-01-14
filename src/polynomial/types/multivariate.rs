use std::collections::BTreeMap;

use crate::field::types::field::{FieldElement, ONE, ZERO};
use crate::polynomial::types::univariate::Polynomial;
///
/// we can treat a multivariate polynomial as a composition of 'terms' where
/// a term, you guessed it, can have one or more variables, to a given degree,
/// multiplied by a given coefficient
///
/// we can represent these terms as key value pairs in a btree, i.e.
/// the degree of the variable is implicit in its index in a vector key
/// and the coefficient the term is multiplied by is the value
///
/// from stark-anatomy:
/// Where univariate polynomials are useful for
/// reducing big claims about large vectors to
/// small claims about scalar values in random points,
/// multivariate polynomials are useful for articulating
/// the arithmetic constraints that an integral computation satisfies.
///
pub struct MultiPolynomial(BTreeMap<Vec<usize>, FieldElement>);

impl MultiPolynomial {
    pub const ZERO: Self = Self(BTreeMap::new());

    pub fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    /// Creates a constant multivariate polynomial.
    /// This method generates a polynomial consisting of a single term where
    /// each variable has a zero exponent. The resulting polynomial's term
    /// is represented as a vector of zeros (with length equal to `var_count`)
    /// paired with the provided `coefficient`.
    ///
    /// Parameters:
    /// - `coefficient`: The scalar coefficient of the constant term.
    /// - `var_count`: The number of variables in the polynomial.
    ///
    /// Returns:
    /// A `MultiPolynomial` where the only term is the constant defined by `coefficient`.    
    pub fn constant(coefficient: FieldElement, var_count: usize) -> Self {
        Self(BTreeMap::from([vec![0; var_count], coefficient]))
    }

    fn num_variables(&self) -> usize {
        self.0.keys().map(|k| k.len()).max().unwrap_or(0)
    }

    fn canonize(&mut self) {
        self.0.retain(|_, &mut fe| fe != FieldElement::ZERO);
    }
}

#[cfg(test)]
mod tests {}
