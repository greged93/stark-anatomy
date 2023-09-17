pub(crate) fn extended_eucledian(a: i128, b: i128) -> (i128, i128, i128) {
    let (mut old_r, mut r) = (a, b);
    let (mut old_s, mut s) = (1, 0);
    let (mut old_t, mut t) = (0, 1);

    loop {
        let q = old_r / r;

        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
        (old_t, t) = (t, old_t - q * t);
        if r == 0 {
            return (old_r, old_s, old_t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extended_eucledian() {
        // Given
        let a = 240;
        let b = 46;

        // When
        let (g, s, t) = extended_eucledian(a, b);

        // Then
        assert_eq!(g, 2);
        assert_eq!(s, -9);
        assert_eq!(t, 47);
    }
}
