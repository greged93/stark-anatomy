use ruint::algorithms::div::div;
use std::ops;

#[derive(Clone)]
struct I256 {
    value: [u64; 4],
}

impl I256 {
    const ZERO: I256 = I256 {
        value: [0, 0, 0, 0],
    };
}

impl ops::Add<I256> for I256 {
    type Output = I256;

    fn add(self, rhs: I256) -> Self::Output {
        let mut result = [0u64; 4];
        let mut carry = 0u64;
        for (i, v) in self.value.into_iter().enumerate() {
            let (sum, carry1) = v.overflowing_add(rhs.value[i]);
            let (sum, carry2) = sum.overflowing_add(carry);
            result[i] = sum;
            carry = carry1 as u64 + carry2 as u64;
        }
        I256 { value: result }
    }
}

impl ops::Sub<I256> for I256 {
    type Output = I256;

    fn sub(self, rhs: I256) -> Self::Output {
        self + (-rhs)
    }
}

impl ops::Neg for I256 {
    type Output = I256;

    fn neg(self) -> Self::Output {
        let mut signed = self;
        let mut carry = true;
        for i in 0..4 {
            (signed.value[i], carry) = (!signed.value[i]).overflowing_add(carry as u64);
        }
        signed
    }
}

impl ops::Mul<I256> for I256 {
    type Output = I256;

    fn mul(self, rhs: I256) -> Self::Output {
        let mut result = [0u64; 9];
        for (i, v) in self.value.iter().enumerate() {
            for (j, w) in rhs.value.iter().enumerate() {
                let product = *v as u128 * *w as u128;
                let low = product & 0xffffffffffffffff;
                let high = product >> 64;
                result[i + j] += low as u64;
                result[i + j + 1] += high as u64;
            }
        }
        I256 {
            value: [result[0], result[1], result[2], result[3]],
        }
    }
}

impl PartialEq for I256 {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl std::cmp::PartialOrd for I256 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl ops::Div<I256> for I256 {
    type Output = I256;

    fn div(self, rhs: I256) -> Self::Output {
        if rhs == I256::ZERO {
            panic!("Division by zero");
        }
        if rhs > self {
            return I256::ZERO;
        }
        let mut numerator = self;
        let mut denominator = rhs;
        div(&mut numerator.value, &mut denominator.value);
        numerator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neg_i256() {
        // Given
        let a = I256 {
            value: [u64::MAX, u64::MAX, 0, 0],
        };

        // When
        let result = -a;

        // Then
        let expected = I256 {
            value: [1, 0, u64::MAX, u64::MAX],
        };
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_add() {
        // Given
        let a = I256 {
            value: [u64::MAX, u64::MAX, 0, 0],
        };
        let b = I256 {
            value: [u64::MAX, u64::MAX, 0, 0],
        };

        // When
        let result = a + b;

        // Then
        let expected = I256 {
            value: [u64::MAX - 1, u64::MAX, 1, 0],
        };
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_add_neg() {
        // Given
        let a = I256 {
            value: [u64::MAX, u64::MAX, 0, 0],
        };
        let b = I256 {
            value: [u64::MAX, u64::MAX, 0, 0],
        };

        // When
        let result = a - b;

        // Then
        let expected = I256 {
            value: [0, 0, 0, 0],
        };
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_mul() {
        // Given
        let a = I256 {
            value: [1, 2, 3, 4],
        };
        let b = I256 {
            value: [5, 6, 7, 8],
        };

        // When
        let result = a * b;

        // Then
        let expected = I256 {
            value: [5, 16, 34, 60],
        };
        assert_eq!(expected.value, result.value);
    }

    #[test]
    fn test_compare() {
        // Given
        let a = I256 {
            value: [1, 2, 3, 4],
        };
        let b = I256 {
            value: [5, 6, 7, 8],
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
    fn test_division() {
        // Given
        let a = I256 {
            value: [5, 6, 7, 8],
        };
        let b = I256 {
            value: [1, 2, 3, 4],
        };

        // When
        let result = a / b;

        // Then
    }
}
