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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_extended_eucledian() {
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
    fn test_mid_extended_eucledian() {
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
    fn test_complex_extended_eucledian() {
        // Given
        let a = I320::from(270497897142230380135924736767050121215u128);
        let b = I320::from(270497897142230380135924736767050121217u128);

        // When
        let (g, s, t) = extended_euclidean(a, b);

        // Then
        assert_eq!(g, I320::from(1u64));
        assert_eq!(s, I320::from(135248948571115190067962368383525060608i128));
        assert_eq!(t, I320::from(-135248948571115190067962368383525060607i128));
    }
}
