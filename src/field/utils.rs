use crate::field::types::base::I320;

pub(crate) fn extended_euclidean(a: I320, b: I320) -> (I320, I320, I320) {
    let (mut old_r, mut r) = (a, b);
    let (mut old_s, mut s) = (I320::ONE, I320::ZERO);
    let (mut old_t, mut t) = (I320::ZERO, I320::ONE);

    loop {
        let q = old_r / r;

        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
        (old_t, t) = (t, old_t - q * t);
        if r == I320::ZERO {
            return (old_r, old_s, old_t);
        }
    }
}

pub(crate) fn multiplicative_inverse(a: I256, m: I256) -> Option<I256> {
    let (g, x, _) = extended_euclidean(a, m);
    if g != I256::ONE {
        // a and m are not coprime, thus a doesn't have an inverse modulo m
        None
    } else {
        // x might be negative, so we standardize it to be between 0 and m-1
        Some((x % m + m) % m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_extended_euclidean() {
        // Given
        let a = I320::from(240u64);
        let b = I320::from(46u64);

        // When
        let (g, s, t) = extended_euclidean(a, b);

        // Then
        assert_eq!(g, I320::from(2u64));
        assert_eq!(s, I320::from(-9i64));
        assert_eq!(t, I320::from(47u64));
    }

    #[test]
    fn test_complex_extended_euclidean() {
        // Given
        let a = I320::from(6543211245u64);
        let b = I320::from(123456785u64);

        // When
        let (g, s, t) = extended_euclidean(a, b);

        // Then
        assert_eq!(g, I320::from(5u64));
        assert_eq!(s, I320::from(6850346u64));
        assert_eq!(t, I320::from(-363068429i64));
    }

    #[test]
<<<<<<< variant A
    fn test_complex_extended_eucledian() {
>>>>>>> variant B
    fn test_basic_multiplicative_inverse() {
======= end
        // Given
<<<<<<< variant A
        let a = I320::from(270497897142230380135924736767050121215u128);
        let b = I320::from(270497897142230380135924736767050121217u128);
>>>>>>> variant B
        let a = I256::from(3u64);
        let m = I256::from(7u64);
======= end

        // When
<<<<<<< variant A
        let (g, s, t) = extended_euclidean(a, b);
>>>>>>> variant B
        let inverse = multiplicative_inverse(a, m);
======= end

        // Then
<<<<<<< variant A
        assert_eq!(g, I320::from(1u64));
        assert_eq!(s, I320::from(135248948571115190067962368383525060608i128));
        assert_eq!(t, I320::from(-135248948571115190067962368383525060607i128));
>>>>>>> variant B
        assert_eq!(inverse, Some(I256::from(5u64))); // 3*5 % 7 == 1
======= end
    }
<<<<<<< variant A
>>>>>>> variant B

    #[test]
    fn test_no_multiplicative_inverse() {
        // Given
        let a = I256::from(2u64);
        let m = I256::from(4u64); // 2 and 4 are not co-prime

        // When
        let inverse = multiplicative_inverse(a, m);

        // Then
        assert_eq!(inverse, None);
    }

    #[test]
    fn test_edge_case_multiplicative_inverse() {
        // Given
        let a = I256::from(1u64);
        let m = I256::from(17u64); // using a smaller prime for clarity

        // When
        let inverse = multiplicative_inverse(a, m);

        // Then
        assert_eq!(inverse, Some(I256::from(1u64))); // The inverse of 1 mod anything is 1

        // Given
        let a = I256::from(-1i64);
        let m = I256::from(17u64);

        // When
        let inverse = multiplicative_inverse(m + a, m);

        // Then
        assert_eq!(inverse, Some(I256::from(16u64))); // The inverse of -1 mod 17 is 16
    }
======= end
}
