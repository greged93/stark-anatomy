use std::collections::BTreeMap;

use crate::felt;
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
#[derive(Clone, Debug)]
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
        Self(BTreeMap::from([(vec![0; var_count], coefficient)]))
    }

    fn num_variables(&self) -> usize {
        self.0.keys().map(|k| k.len()).max().unwrap_or(0)
    }

    fn canonize(&mut self) {
        self.0.retain(|_, &mut fe| fe != ZERO);
    }

    pub fn univariate_variables(nvar: usize) -> Vec<Self> {
        assert!(nvar > 0);
        let mut res = vec![Self::ZERO; nvar];

        for i in 0..nvar {
            let mut ith_var = vec![0; nvar];
            ith_var[i] = 1;
            res[i].0.insert(ith_var, ONE);
        }

        res
    }

    pub fn lift_univariate(poly: &Polynomial, var_count: usize, index: usize) -> Self {
        assert!(var_count > 0);
        if poly.is_zero() {
            return Self::ZERO;
        }
        let univariates = Self::univariate_variables(var_count);
        let x = univariates[index].clone();
        let mut acc = Self::ZERO;
        let coeffs = poly.clone().0;

        for i in 0..coeffs.len() {
            acc = acc + (Self::constant(coeffs[i], var_count) * (x.pow(i as _)));
        }

        acc
    }

    pub fn evaluate(&self, fes: &Vec<FieldElement>) -> FieldElement {
        let mut acc = ZERO;

        for (k, &v) in self.0.iter() {
            let mut prod = v;
            for i in 0..k.len() {
                prod = prod * fes[i].pow(felt!(k[i] as u128));
            }
            acc = acc + prod;
        }

        acc
    }

    pub fn evaluate_symbolic(&self, pols: &Vec<Polynomial>) -> Polynomial {
        let mut acc = Polynomial::zero();

        for (k, &v) in self.0.iter() {
            let mut prod = Polynomial::new(&[v]);
            for i in 0..k.len() {
                prod = prod * (pols[i].pow(k[i]));
            }
            acc = acc + prod;
        }

        acc
    }

    pub fn pow(&self, n: u128) -> Self {
        if self.is_zero() {
            return Self::ZERO;
        }
        let nvar = self.num_variables();
        let vars = vec![0; nvar];
        let mut acc = Self::ZERO;
        acc.0.insert(vars, ONE);

        for b in (0..u128::BITS).rev().map(|i| (n >> i) & 1) {
            acc = acc.clone() * acc;
            if b != 0 {
                acc = acc * self.clone();
            }
        }

        acc
    }
}

impl std::ops::Add for MultiPolynomial {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut btree = BTreeMap::new();
        let nvar = std::cmp::max(self.num_variables(), rhs.num_variables());

        for (lk, &lv) in self.0.iter() {
            let mut newk = lk.clone();
            newk.append(&mut vec![0; nvar - lk.len()]);
            btree.insert(newk, lv);
        }
        for (rk, &rv) in rhs.0.iter() {
            let mut newk = rk.clone();
            newk.append(&mut vec![0; nvar - rk.len()]);
            if let Some(&prev) = btree.get(&newk) {
                btree.insert(newk, prev + rv);
            } else {
                btree.insert(newk, rv);
            }
        }

        let mut res = Self(btree);
        res.canonize();
        res
    }
}

impl std::ops::Sub for MultiPolynomial {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self + -rhs
    }
}

impl std::ops::Neg for MultiPolynomial {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut btree = BTreeMap::new();
        for (k, v) in self.0.iter() {
            btree.insert(k.clone(), v.neg());
        }
        Self(btree)
    }
}

impl std::ops::Mul for MultiPolynomial {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut btree = BTreeMap::new();
        let nvar = std::cmp::max(self.num_variables(), rhs.num_variables());

        for (lk, &lv) in self.0.iter() {
            for (rk, &rv) in rhs.0.iter() {
                let mut vars = vec![0; nvar];
                for i in 0..lk.len() {
                    vars[i] += lk[i];
                }
                for i in 0..rk.len() {
                    vars[i] += rk[i];
                }
                if let Some(&prev) = btree.get(&vars) {
                    btree.insert(vars, prev + lv * rv);
                } else {
                    btree.insert(vars, lv * rv);
                }
            }
        }

        let mut res = Self(btree);
        res.canonize();
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO: FieldElement = felt!(2);
    const FIVE: FieldElement = felt!(5);

    #[test]
    fn evaluate() {
        let univariate_vars = MultiPolynomial::univariate_variables(4);

        let mpoly_a = MultiPolynomial::constant(ONE, 4) * (univariate_vars[0].clone())
            + (MultiPolynomial::constant(TWO, 4) * (univariate_vars[1].clone()))
            + (MultiPolynomial::constant(FIVE, 4) * (univariate_vars[2].clone().pow(3)));

        let mpoly_b = MultiPolynomial::constant(ONE, 4)
            * (univariate_vars[0].clone())
            * (univariate_vars[3].clone())
            + (MultiPolynomial::constant(FIVE, 4) * (univariate_vars[3].clone().pow(3)))
            + (MultiPolynomial::constant(FIVE, 4));

        let mpoly_c = mpoly_a.clone() * mpoly_b.clone();
        let fes = vec![ZERO, FIVE, FIVE, ZERO];

        let eval_a = mpoly_a.evaluate(&fes);
        let eval_b = mpoly_b.evaluate(&fes);
        let eval_c = mpoly_c.evaluate(&fes);

        assert_eq!(eval_a * eval_b, eval_c);
        assert_eq!(eval_a + eval_b, (mpoly_a + mpoly_b).evaluate(&fes));
    }

    #[test]
    fn lift() {
        let upoly = Polynomial::interpolate_domain(&[ZERO, ONE, FIVE], &[TWO, FIVE, FIVE]);

        let mpoly = MultiPolynomial::lift_univariate(&upoly, 4, 3);

        assert_eq!(
            upoly.evaluate(&FIVE),
            mpoly.evaluate(&vec![ZERO, ZERO, ZERO, FIVE])
        );
    }
}
