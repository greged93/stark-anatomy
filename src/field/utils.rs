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

pub(crate) fn multiplicative_inverse(a: I320, m: I320) -> Option<I320> {
    let (g, x, _) = extended_euclidean(a, m);
    if g != I320::ONE {
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
    fn test_multiplicative_inverse_of_2_mi() {
        // Given
        let prime = I320::from(270497897142230380135924736767050121217_u128);
        let a = I320::from(2_u128);
        let expected_inverse = I320::from(135248948571115190067962368383525060609_u128);

        // When
        let inverse = multiplicative_inverse(a, prime).unwrap();

        // Then
        assert_eq!(
            inverse, expected_inverse,
            "Multiplicative inverse is incorrect"
        );

        assert_eq!(
            (inverse * a) % prime,
            I320::ONE,
            "Multiplicative inverse check failed"
        );
    }
}
