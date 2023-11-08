use std::{fmt::Debug, mem};

use serde::Serialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

type ProofStreamResult<T> = Result<T, ProofStreamError>;

#[derive(Debug, Error)]
pub enum ProofStreamError {
    #[error("Read index is out of bounds")]
    OutOfBoundsReadIndexError,
}

#[derive(Default, Debug, PartialEq, Eq)]
pub struct ProofStream<S: Serialize> {
    items: Vec<S>,
    read_index: usize,
}

impl<S: Serialize + Clone + Debug> ProofStream<S> {
    pub fn push(&mut self, item: S) {
        self.items.push(item);
    }

    pub fn pull(&mut self) -> ProofStreamResult<&S> {
        if self.read_index >= self.items.len() {
            return Err(ProofStreamError::OutOfBoundsReadIndexError);
        }
        let val = &self.items[self.read_index];
        self.read_index += 1;
        Ok(val)
    }

    pub fn serialize(&mut self) -> &[u8] {
        serialize_inner(&self.items)
    }

    pub fn deserialize(items: &[u8]) -> Self {
        let items: &[S] = unsafe {
            core::slice::from_raw_parts(
                items as *const [u8] as *const S,
                items.len() / mem::size_of::<S>(),
            )
        };
        Self {
            items: items.to_vec(),
            read_index: 0,
        }
    }

    pub fn prover_fiat_shamir(&mut self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&mut self.serialize().to_vec());
        hasher.finalize().into()
    }

    pub fn verifier_fiat_shamir(&mut self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        let items = &self.items[0..self.read_index];
        hasher.update(&mut serialize_inner(items).to_vec());
        hasher.finalize().into()
    }
}

const fn serialize_inner<T>(items: &[T]) -> &[u8] {
    unsafe {
        core::slice::from_raw_parts(
            (items as *const [T]) as *const u8,
            mem::size_of::<T>() * items.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use serde_derive::Serialize;

    use super::*;

    #[derive(Serialize, Clone, Default, PartialEq, Eq, Debug)]
    struct TestStruct {
        pub a: u8,
        pub b: u8,
    }

    #[derive(Serialize, Clone, Default, PartialEq, Eq, Debug)]
    struct TestStructComplex {
        pub a: Vec<u8>,
        pub b: u8,
        pub c: TestStruct,
    }

    #[test]
    fn test_serialize_inner() {
        // Given
        let items = vec![
            TestStruct { a: 1, b: 2 },
            TestStruct { a: 2, b: 3 },
            TestStruct { a: 3, b: 4 },
            TestStruct { a: 4, b: 5 },
        ];

        // When
        let serialized = serialize_inner(&items);

        // Then
        let expected = vec![1, 2, 2, 3, 3, 4, 4, 5];
        assert_eq!(serialized, expected.as_slice());
    }

    #[test]
    fn test_deserialize() {
        // Given
        let mut ps = ProofStream::default();
        ps.push(TestStruct { a: 1, b: 2 });
        ps.push(TestStruct { a: 2, b: 3 });
        ps.push(TestStruct { a: 3, b: 4 });
        ps.push(TestStruct { a: 4, b: 5 });

        let serialized = ps.serialize();

        // When
        let deserialized = ProofStream::<TestStruct>::deserialize(serialized);

        // Then
        assert_eq!(ps, deserialized);
    }

    #[test]
    fn test_deserialize_complex() {
        // Given
        let mut ps = ProofStream::default();
        ps.push(TestStructComplex {
            a: vec![1, 2, 3, 4],
            b: 5,
            c: TestStruct { a: 6, b: 7 },
        });
        ps.push(TestStructComplex {
            a: vec![8, 9, 10, 11],
            b: 12,
            c: TestStruct { a: 13, b: 14 },
        });
        ps.push(TestStructComplex {
            a: vec![15, 16, 17, 18],
            b: 19,
            c: TestStruct { a: 20, b: 21 },
        });
        ps.push(TestStructComplex {
            a: vec![22, 23, 24, 25],
            b: 26,
            c: TestStruct { a: 27, b: 28 },
        });

        let serialized = ps.serialize();

        // When
        let deserialized = ProofStream::<TestStructComplex>::deserialize(serialized);

        // Then
        assert_eq!(ps, deserialized);
    }

    #[test]
    fn test_prover_fiat_shamir() {
        // Given
        let mut ps = ProofStream::default();
        ps.push(TestStruct { a: 1, b: 2 });
        ps.push(TestStruct { a: 2, b: 3 });
        ps.push(TestStruct { a: 3, b: 4 });
        ps.push(TestStruct { a: 4, b: 5 });

        // When
        let fs = ps.prover_fiat_shamir();

        // Then
        // Made using
        // ```
        // from hashlib import sha256
        // h = sha256()
        // h.update(bytes([1,2,2,3,3,4,4,5]))
        // list(h.digest())
        // ````
        let expected = [
            93, 42, 1, 224, 75, 233, 101, 227, 125, 162, 216, 6, 228, 174, 108, 97, 47, 133, 105,
            139, 185, 172, 216, 9, 111, 253, 76, 41, 218, 63, 129, 104,
        ];
        assert_eq!(fs, expected);
    }

    #[test]
    fn test_verifier_fiat_shamir() {
        // Given
        let mut ps = ProofStream::default();
        ps.push(TestStruct { a: 1, b: 2 });
        ps.push(TestStruct { a: 2, b: 3 });
        ps.push(TestStruct { a: 3, b: 4 });
        ps.push(TestStruct { a: 4, b: 5 });

        // When
        ps.pull().unwrap();
        ps.pull().unwrap();
        ps.pull().unwrap();
        let fs = ps.verifier_fiat_shamir();

        // Then
        // Made using
        // ```
        // from hashlib import sha256
        // h = sha256()
        // h.update(bytes([1,2,2,3,3,4]))
        // list(h.digest())
        // ````
        let expected = [
            75, 202, 77, 187, 114, 48, 150, 65, 196, 29, 129, 127, 207, 1, 44, 173, 19, 244, 179,
            187, 59, 51, 209, 6, 10, 103, 150, 249, 200, 234, 12, 42,
        ];
        assert_eq!(fs, expected);
    }
}
