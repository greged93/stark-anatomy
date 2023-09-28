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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_extended_eucledian() {
        // Given
        let a = I256::from(240u64);
        let b = I256::from(46u64);

        // When
        let (g, s, t) = extended_euclidean(a, b);

        // Then
        assert_eq!(g, I256::from(2u64));
        assert_eq!(s, I256::from(-9i64));
        assert_eq!(t, I256::from(47u64));
    }

    #[test]
    fn test_complex_extended_eucledian() {
        // Given
        let a = I256::from(6543211245u64);
        let b = I256::from(123456785u64);

        // When
        let (g, s, t) = extended_euclidean(a, b);

        // Then
        assert_eq!(g, I256::from(5u64));
        assert_eq!(s, I256::from(6850346u64));
        assert_eq!(t, I256::from(-363068429i64));
    }
}
