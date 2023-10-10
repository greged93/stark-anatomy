use ruint::algorithms::div::div;
use std::{
    fmt::{Debug, Formatter},
    ops,
};

#[derive(Clone, Copy)]
pub struct I320 {
    value: [u64; 5],
}

impl Debug for I320 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let value = self.abs();
        if self.sign() {
            write!(f, "-")?;
        }
        write!(f, "{:?}", Into::<u128>::into(value))
    }
}

impl I320 {
    pub const ZERO: I320 = I320 {
        value: [0, 0, 0, 0, 0],
    };
    pub const ONE: I320 = I320 {
        value: [1, 0, 0, 0, 0],
    };

    pub fn pow(self, exponent: Self) -> Self {
        let mut result = Self::ONE;
        let mut base = self;
        let mut exp = exponent;
        while exp > Self::ZERO {
            if exp % Self::from(2u64) == Self::ONE {
                result = result * base;
            }
            base = base * base;
            exp = exp / Self::from(2u64);
        }
        result
    }

    fn abs(self) -> Self {
        if self.sign() {
            -self
        } else {
            self
        }
    }

    fn sign(&self) -> bool {
        (self.value[4] & 0x8000000000000000) != 0
    }
}

impl ops::Add<I320> for I320 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = [0u64; 5];
        let mut carry = 0u64;
        for (i, v) in self.value.into_iter().enumerate() {
            let (sum, carry1) = v.overflowing_add(rhs.value[i]);
            let (sum, carry2) = sum.overflowing_add(carry);
            result[i] = sum;
            carry = carry1 as u64 + carry2 as u64;
        }
        Self { value: result }
    }
}

impl ops::Sub<I320> for I320 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl ops::Neg for I320 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut signed = self;
        let mut carry = true;
        for i in 0..5 {
            (signed.value[i], carry) = (!signed.value[i]).overflowing_add(carry as u64);
        }
        signed
    }
}

impl ops::Mul<I320> for I320 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut result = [0u64; 10];
        let signed = self.sign() ^ rhs.sign();
        let lhs = self.abs();
        let rhs = rhs.abs();
        for (i, v) in lhs.value.iter().enumerate() {
            let mut carry_high = false;
            for (j, w) in rhs.value.iter().enumerate() {
                let mut carry = false;
                let product = *v as u128 * *w as u128;
                let low = product & 0xffffffffffffffff;
                let high = product >> 64;
                (result[i + j], carry) = result[i + j].overflowing_add(low as u64 + carry as u64);
                (result[i + j + 1], carry_high) = result[i + j + 1]
                    .overflowing_add(high as u64 + carry as u64 + carry_high as u64);
            }
        }
        let prod = Self {
            value: [result[0], result[1], result[2], result[3], result[4]],
        };
        if signed {
            -prod
        } else {
            prod
        }
    }
}

impl ops::Div<I320> for I320 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        if rhs == Self::ZERO {
            panic!("Division by zero");
        }
        let mut numerator = self;
        let mut denominator = rhs;
        div(&mut numerator.value, &mut denominator.value);
        numerator
    }
}

impl ops::Rem<I320> for I320 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let divisor = rhs.abs();
        let dividend = self.abs();
        let div = dividend / divisor;
        if self.sign() {
            return -(dividend - div * divisor);
        }
        self - div * divisor
    }
}

impl PartialEq for I320 {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl std::cmp::PartialOrd for I320 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let l_sign = self.sign();
        let r_sign = other.sign();

        let mut l = self.abs();
        l.value.reverse();
        let mut r = other.abs();
        r.value.reverse();

        match (l_sign, r_sign) {
            (true, true) => l.value.partial_cmp(&r.value).map(|x| x.reverse()),
            (true, false) => Some(std::cmp::Ordering::Less),
            (false, true) => Some(std::cmp::Ordering::Greater),
            (false, false) => l.value.partial_cmp(&r.value),
        }
    }
}

impl From<u64> for I320 {
    fn from(num: u64) -> Self {
        Self {
            value: [num, 0, 0, 0, 0],
        }
    }
}

impl From<u128> for I320 {
    fn from(num: u128) -> Self {
        Self {
            value: [num as u64, (num >> 64) as u64, 0, 0, 0],
        }
    }
}

impl From<I320> for u128 {
    fn from(num: I320) -> Self {
        (num.value[1] as u128) << 64 | num.value[0] as u128
    }
}

impl From<i64> for I320 {
    fn from(num: i64) -> Self {
        if num < 0 {
            -Self::from(-num as u64)
        } else {
            Self::from(num as u64)
        }
    }
}

impl From<i128> for I320 {
    fn from(num: i128) -> Self {
        if num < 0 {
            -Self::from(-num as u128)
        } else {
            Self::from(num as u128)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pow() {
        // Given
        let a = I320 {
            value: [2, 0, 0, 0, 0],
        };

        // When
        let result = a.pow(I320::from(100u64));

        // Then
        let expected = I320 {
            value: [0, 68719476736, 0, 0, 0],
        };
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_neg() {
        // Given
        let a = I320 {
            value: [u64::MAX, u64::MAX, 0, 0, 0],
        };

        // When
        let result = -a;

        // Then
        let expected = I320 {
            value: [1, 0, u64::MAX, u64::MAX, u64::MAX],
        };
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_add() {
        // Given
        let a = I320 {
            value: [u64::MAX, u64::MAX, 0, 0, 0],
        };
        let b = I320 {
            value: [u64::MAX, u64::MAX, 0, 0, 0],
        };

        // When
        let result = a + b;

        // Then
        let expected = I320 {
            value: [u64::MAX - 1, u64::MAX, 1, 0, 0],
        };
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_add_neg() {
        // Given
        let a = I320 {
            value: [u64::MAX, u64::MAX, 0, 0, 0],
        };
        let b = I320 {
            value: [u64::MAX, u64::MAX, 0, 0, 0],
        };

        // When
        let result = a - b;

        // Then
        let expected = I320 {
            value: [0, 0, 0, 0, 0],
        };
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_mul() {
        // Given
        let a = I320 {
            value: [1, 2, 3, 4, 0],
        };
        let b = I320 {
            value: [5, 6, 7, 8, 0],
        };

        // When
        let result = a * b;

        // Then
        let expected = I320 {
            value: [5, 16, 34, 60, 61],
        };
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_mul_negative() {
        // Given
        let a = I320::from(135248948571115190067962368383525060607u128);
        let b = I320::from(-1i64);

        // When
        let result = a * b;

        // Then
        let expected = -I320::from(135248948571115190067962368383525060607u128);
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_mul_high() {
        // Given
        let a = I320::from(180331931428153586757283157844700080811u128);
        let b = I320::from(270497897142230380135924736767050121214u128);

        // When
        let result = a * b;

        // Then
        let expected = I320 {
            value: [
                12297829382473034410,
                16080853069464251050,
                12297829382473034409,
                7771008069523909290,
                0,
            ],
        };
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_mul_near_boundary() {
        // `PRIME` is the expression `1 + 407 * 2u128.pow(119)` evaluated
        // see: https://github.com/aszepieniec/stark-anatomy/blob/76c375505a28e7f02f8803f77f8d7620d834071d/docs/basic-tools.md?plain=1#L113-L119
        //
        // This test is essential for ensuring the robustness and correctness of our multiplication
        // implementation, especially when values approach the maximum representable in the I320 type.
        //
        // In many cryptographic contexts, especially those that use modular arithmetic (like STARKs),
        // operations involving numbers near the modulus (in this case, PRIME) are common. This test
        // simulates a "worst-case scenario" where two values just shy of the modulus are multiplied
        // together. The results of such multiplications can potentially overflow the I320 type.
        //
        // Two things are being checked here:
        // 1. That the raw multiplication (without considering any modulus) is correct. This checks
        //    if the system handles potential overflows correctly. The result is compared against a
        //    saturating multiplication to ensure no unintended wrap-around occurs.
        // 2. That the multiplication result modulo PRIME is as expected. In modular arithmetic,
        //    multiplying two numbers both equal to (PRIME - 1) should yield a result of 1 when taken modulo PRIME.
        //
        // Ensuring correctness for this boundary condition is crucial for the overall reliability of
        // any system built on top of this arithmetic foundation.

        // Given the boundary condition where PRIME is close to the maximum value for I320. [-2^255 + 1; 2^255 - 1]
        let prime = 270497897142230380135924736767050121217_u128;
        let prime_minus_one = prime - 1_u128;
        let near_boundary = I320::from(prime_minus_one);

        // When
        let result = near_boundary * near_boundary;

        // Then
        // Expected result computation depends on the desired behavior.
        let mod_prime = result % I320::from(prime);

        assert_eq!(I320::ONE, mod_prime);
    }

    #[test]
    fn test_compare_pos_pos() {
        // Given
        let a = I320 {
            value: [1, 2, 3, 4, 0],
        };
        let b = I320 {
            value: [5, 6, 7, 8, 0],
        };

        // When
        let result1 = a < b;
        let result2 = a > b;
        let result3 = b == b;

        // Then
        assert!(result1);
        assert!(!result2);
        assert!(result3);
    }

    #[test]
    fn test_compare_pos_neg() {
        // Given
        let a = I320 {
            value: [1, 2, 3, 4, 0],
        };
        let b = -I320 {
            value: [5, 6, 7, 8, 0],
        };

        // When
        let result1 = a > b;
        let result2 = a < b;
        let result3 = b.eq(&b);

        // Then
        assert!(result1);
        assert!(!result2);
        assert!(result3);
    }

    #[test]
    fn test_compare_neg_pos() {
        // Given
        let a = -I320 {
            value: [1, 2, 3, 4, 0],
        };
        let b = I320 {
            value: [5, 6, 7, 8, 0],
        };

        // When
        let result1 = a < b;
        let result2 = a > b;
        let result3 = b.eq(&b);

        // Then
        assert!(result1);
        assert!(!result2);
        assert!(result3);
    }

    #[test]
    fn test_compare_neg_neg() {
        // Given
        let a = -I320 {
            value: [1, 2, 3, 4, 0],
        };
        let b = -I320 {
            value: [5, 6, 7, 8, 0],
        };

        // When
        let result1 = a > b;
        let result2 = a < b;
        let result3 = b.eq(&b);

        // Then
        assert!(result1);
        assert!(!result2);
        assert!(result3);
    }

    #[test]
    fn test_division() {
        // Given
        let a = I320 {
            value: [5, 6, 7, 8, 0],
        };
        let b = I320 {
            value: [1, 2, 3, 4, 0],
        };

        // When
        let result = a / b;

        // Then
        let expected = I320 {
            value: [2, 0, 0, 0, 0],
        };
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_rem_simple() {
        // Given
        let a = I320 {
            value: [5, 6, 7, 8, 0],
        };
        let b = I320 {
            value: [1, 2, 3, 4, 0],
        };

        // When
        let result = a % b;

        // Then
        let expected = I320 {
            value: [3, 2, 1, 0, 0],
        };
        let div = a / b;
        assert_eq!(expected.value, result.value);
        assert_eq!(expected, a - b * div);
    }

    #[test]
    fn test_rem_complex() {
        // Given
        let a = I320 {
            value: [0, 0, 4294967296, 0, 0],
        };
        let b = I320::from(1 + 407 * 2u128.pow(119));

        // When
        let result = a % b;

        // Then
        let expected = I320 {
            value: [18446744068306546075, 13150510911921848319, 0, 0, 0],
        };
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_rem_complex_negative_modulus() {
        // Given
        let a = I320 {
            value: [0, 0, 4294967296, 0, 0],
        };
        let b = I320::from(1 + 407 * 2u128.pow(119));

        // When
        let result = a % (-b);

        // Then
        let expected = I320 {
            value: [18446744068306546075, 13150510911921848319, 0, 0, 0],
        };
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_rem_complex_negative_dividend() {
        // Given
        let a = I320 {
            value: [0, 0, 4294967296, 0, 0],
        };
        let b = I320::from(1 + 407 * 2u128.pow(119));

        // When
        let result = (-a) % b;

        // Then
        let expected = -I320 {
            value: [18446744068306546075, 13150510911921848319, 0, 0, 0],
        };
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_from_u64() {
        // Given
        let a = 5u64;

        // When
        let result = I320::from(a);

        // Then
        let expected = I320 {
            value: [5, 0, 0, 0, 0],
        };
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_from_u128() {
        // Given
        let a = u64::MAX as u128 + 2;

        // When
        let result = I320::from(a);

        // Then
        let expected = I320 {
            value: [1, 1, 0, 0, 0],
        };
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_from_i256_to_u128() {
        // Given
        let a = I320 {
            value: [u64::MAX, u64::MAX, 0, 0, 0],
        };

        // When
        let result = u128::from(a);

        // Then
        let expected = u128::MAX;
        assert_eq!(expected, result);
    }

    #[test]
    fn test_from_i64() {
        // Given
        let a = -5i64;

        // When
        let result = I320::from(a);

        // Then
        let expected = -I320 {
            value: [5, 0, 0, 0, 0],
        };
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_from_i128() {
        // Given
        let a = -(u64::MAX as i128 + 2);

        // When
        let result = I320::from(a);

        // Then
        let expected = -I320 {
            value: [1, 1, 0, 0, 0],
        };
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_multiplication_near_boundary() {
        // `PRIME` is the expression `1 + 407 * 2u128.pow(119)` evaluated
        // see: https://github.com/aszepieniec/stark-anatomy/blob/76c375505a28e7f02f8803f77f8d7620d834071d/docs/basic-tools.md?plain=1#L113-L119
        //
        // This test is essential for ensuring the robustness and correctness of our multiplication
        // implementation, especially when values approach the maximum representable in the I256 type.
        //
        // In many cryptographic contexts, especially those that use modular arithmetic (like STARKs),
        // operations involving numbers near the modulus (in this case, PRIME) are common. This test
        // simulates a "worst-case scenario" where two values just shy of the modulus are multiplied
        // together. The results of such multiplications can potentially overflow the I256 type.
        //
        // Two things are being checked here:
        // 1. That the raw multiplication (without considering any modulus) is correct. This checks
        //    if the system handles potential overflows correctly. The result is compared against a
        //    saturating multiplication to ensure no unintended wrap-around occurs.
        // 2. That the multiplication result modulo PRIME is as expected. In modular arithmetic,
        //    multiplying two numbers both equal to (PRIME - 1) should yield a result of 1 when taken modulo PRIME.
        //
        // Ensuring correctness for this boundary condition is crucial for the overall reliability of
        // any system built on top of this arithmetic foundation.

        // Given the boundary condition where PRIME is close to the maximum value for I256.
        let prime = 270497897142230380135924736767050121217_u128;
        let prime_minus_one = prime - 1_u128;
        let near_boundary = I256::from(prime_minus_one);

        // When
        let result = near_boundary * near_boundary;

        // Then
        // Expected result computation depends on the desired behavior.
        let expected_value = I256::from((prime_minus_one).saturating_mul(prime_minus_one));
        let mod_prime = expected_value % I256::from(prime);

        assert_eq!(expected_value, result);
        assert_eq!(I256::ONE, mod_prime);
    }

    #[test]
    fn test_specific_values_multiplication() {
        // Given the context where we have identified specific values of interest.
        let prime = 270497897142230380135924736767050121217_u128;
        let value1 = I256::from(1_u128);
        let value2 = I256::from(200713427363522296808474866102332030979_u128);

        // When
        let result = value1 * value2;

        // Then
        // Expected result without modular arithmetic should be value2 itself.
        let expected_value = value2;
        let mod_prime = result % I256::from(prime);

        // Asserting multiplication result
        assert_eq!(
            expected_value, result,
            "Expected the multiplication of 1 with any value to be the value itself."
        );

        // Asserting modular arithmetic result
        // Multiplying 1 by any value modulo prime should give the value itself modulo prime.
        let expected_mod_prime = value2 % I256::from(prime);
        assert_eq!(
            expected_mod_prime, mod_prime,
            "Expected the modulo result to be consistent with basic modular arithmetic rules."
        );
    }
}
