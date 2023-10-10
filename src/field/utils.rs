use crate::field::types::base::I256;

pub(crate) fn extended_euclidean(a: I256, b: I256) -> (I256, I256, I256) {
    let (mut old_r, mut r) = (a, b);
    let (mut old_s, mut s) = (I256::ONE, I256::ZERO);
    let (mut old_t, mut t) = (I256::ZERO, I256::ONE);

    loop {
        let q = old_r / r;

        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
        (old_t, t) = (t, old_t - q * t);
        if r == I256::ZERO {
            return (old_r, old_s, old_t);
        }
    }
}

/// Calculates the multiplicative inverse of `a` modulo `m`.
///
/// # Arguments
/// * `a` - The number for which we want to find an inverse.
/// * `m` - The modulo under which the inverse should be found.
///
/// # Returns
/// * `Option<I256>` - The multiplicative inverse if it exists; `None` if `a` and `m` are not coprime.
///
/// # Explanation
/// Even though `x` is always non-negative in `I256`, it might represent a "negative" value
/// in the context of modular arithmetic. To ensure our result is in the standard range [0, m-1],
/// the following steps are performed:
///
/// 1. (x % m): This operation takes `x` modulo `m`.
///    * If `x` is positive, the result is in [0, m-1].
///    * If `x` is equivalent to a negative number in modular arithmetic, the result is in [-m+1, 0].
///
/// 2. (x % m + m): By adding `m`, we shift the result into a positive range.
///    * If `x` was already positive, the range becomes [m, 2m-1].
///    * If `x` was negative, the range becomes [0, m-1], which is our desired range.
///
/// 3. ((x % m + m) % m): Taking modulo `m` again ensures the result is in the desired range [0, m-1],
///    thereby guaranteeing a non-negative result that's also less than `m`.
pub(crate) fn multiplicative_inverse(a: I256, m: I256) -> Option<I256> {
    let (g, x, _) = extended_euclidean(a, m);
    if g != I256::ONE {
        // a and m are not coprime, thus a doesn't have an inverse modulo m
        None
    } else {
        Some((x % m + m) % m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_extended_euclidean() {
        // Given
        let a = I256::from(240u64);
        let b = I256::from(46u64);

        // When
        let (g, s, t) = extended_euclidean(a, b);

        // Then
        assert_eq!(I256::from(2u64), g);
        assert_eq!(I256::from(-9i64), s);
        assert_eq!(I256::from(47u64), t);
    }

    #[test]
    fn test_complex_extended_euclidean() {
        // Given
        let a = I256::from(6543211245u64);
        let b = I256::from(123456785u64);

        // When
        let (g, s, t) = extended_euclidean(a, b);

        // Then
        assert_eq!(I256::from(5u64), g);
        assert_eq!(I256::from(6850346u64), s);
        assert_eq!(I256::from(-363068429i64), t);
    }

    #[test]
    fn test_basic_multiplicative_inverse() {
        // Given
        let a = I256::from(3u64);
        let m = I256::from(7u64);

        // When
        let inverse = multiplicative_inverse(a, m);

        // Then
        assert_eq!(Some(I256::from(5u64)), inverse); // 3*5 % 7 == 1
    }

    #[test]
    fn test_no_multiplicative_inverse() {
        // Given
        let a = I256::from(2u64);
        let m = I256::from(4u64); // 2 and 4 are not co-prime

        // When
        let inverse = multiplicative_inverse(a, m);

        // Then
        assert_eq!(None, inverse);
    }

    #[test]
    fn test_edge_case_multiplicative_inverse() {
        // Given
        let a = I256::from(1u64);
        let m = I256::from(17u64);

        // When
        let inverse = multiplicative_inverse(a, m);

        // Then
        assert_eq!(inverse, Some(I256::from(1u64))); // The inverse of 1 mod anything is 1

        // Given
        let a = I256::from(-1i64);
        let m = I256::from(17u64);

        // When
        //// Convert the negative value to its positive modular equivalent.
        //// For a negative value 'a', its positive modular equivalent is 'a+m'.
        let inverse = multiplicative_inverse(a + m, m);

        // Then
        assert_eq!(Some(I256::from(16u64)), inverse); // The inverse of -1 mod 17 is 16
    }
}
