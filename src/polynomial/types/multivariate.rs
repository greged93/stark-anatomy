use std::collections::BTreeMap;

use crate::field::types::field::{FieldElement, ONE, ZERO};
use crate::polynomial::types::univariate::Polynomial;

/// we can treat a multivariate polynomial as a composition of 'terms' where
/// a term, you guessed it, can have one or more variables, to a given degree,
/// multiplied by a given coefficient

/// we can represent these terms as key value pairs in a btree, i.e.
/// the degree of the variable is implicit in its index in a vector key
/// and the coefficient the term is multiplied by is the value

pub struct MultiPolynomial(BTreeMap<Vec<usize>, FieldElement>);

impl MultiPolynomial {}

#[cfg(test)]
mod tests {}
