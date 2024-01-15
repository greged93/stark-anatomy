use std::{fmt::Debug, marker::PhantomData};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

type ProofStreamResult<T> = Result<T, ProofStreamError>;

#[derive(Debug, Error)]
pub enum ProofStreamError {
    #[error("Serialization error")]
    ErrorSerializingProofItems(#[from] serde_json::Error),
    #[error("Read index is out of bounds")]
    OutOfBoundsReadIndexError,
}

#[derive(Default, Debug, PartialEq, Eq)]
pub struct ProofStream<'de, S: Serialize + Deserialize<'de>> {
    items: Vec<S>,
    read_index: usize,
    _phantom: PhantomData<&'de ()>,
}

impl<'de, S: Serialize + Deserialize<'de> + Clone + Debug> ProofStream<'de, S> {
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

    pub fn serialize(&mut self) -> ProofStreamResult<Vec<u8>> {
        serialize_inner(&self.items)
    }

    pub fn deserialize(items: &'de [u8]) -> ProofStreamResult<Self> {
        let items: Vec<S> = serde_json::from_slice(items)?;
        Ok(Self {
            items: items.to_vec(),
            read_index: 0,
            _phantom: PhantomData,
        })
    }

    pub fn prover_fiat_shamir(&mut self) -> ProofStreamResult<[u8; 32]> {
        let mut hasher = Sha256::new();
        hasher.update(&mut self.serialize()?.to_vec());
        Ok(hasher.finalize().into())
    }

    pub fn verifier_fiat_shamir(&mut self) -> ProofStreamResult<[u8; 32]> {
        let mut hasher = Sha256::new();
        let items = &self.items[0..self.read_index];

        hasher.update(&mut serialize_inner(items)?.to_vec());
        Ok(hasher.finalize().into())
    }
}

fn serialize_inner<T: Serialize>(items: &[T]) -> ProofStreamResult<Vec<u8>> {
    Ok(serde_json::to_vec(items)?)
}

#[cfg(test)]
mod tests {
    use serde_derive::{Deserialize, Serialize};

    use super::*;

    #[derive(Serialize, Deserialize, Clone, Default, PartialEq, Eq, Debug)]
    struct TestStruct {
        pub a: u8,
        pub b: u8,
    }

    #[derive(Serialize, Deserialize, Clone, Default, PartialEq, Eq, Debug)]
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
        let serialized = serialize_inner(&items).expect("Failed to serialize");

        // Then
        let expected = vec![
            91, 123, 34, 97, 34, 58, 49, 44, 34, 98, 34, 58, 50, 125, 44, 123, 34, 97, 34, 58, 50,
            44, 34, 98, 34, 58, 51, 125, 44, 123, 34, 97, 34, 58, 51, 44, 34, 98, 34, 58, 52, 125,
            44, 123, 34, 97, 34, 58, 52, 44, 34, 98, 34, 58, 53, 125, 93,
        ];
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

        let serialized = ps.serialize().expect("Failed to serialize");

        // When
        let deserialized =
            ProofStream::<TestStruct>::deserialize(&serialized).expect("Failed to deserialize");

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

        let serialized = ps.serialize().expect("Failed to serialize");

        // When
        let deserialized = ProofStream::<TestStructComplex>::deserialize(&serialized)
            .expect("Failed to deserialize");

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
        let fs = ps
            .prover_fiat_shamir()
            .expect("Failed to compute prover fiat shamir");

        // Then
        // Made using
        // ```
        // from hashlib import sha256
        // h = sha256()
        // h.update(bytes([91,123,34,97,34,58,49,44,34,98,34,58,50,125,44,123,34,97,34,58,50,44,34,98,34,58,51,125,44,123,34,97,34,58,51,44,34,98,34,58,52,125,44,123,34,97,34,58,52,44,34,98,34,58,53,125,93,]))
        // list(h.digest())
        // ````
        let expected = [
            201, 176, 198, 41, 77, 42, 190, 176, 93, 90, 51, 57, 129, 77, 162, 158, 96, 4, 167,
            126, 67, 85, 94, 5, 241, 172, 158, 164, 239, 74, 93, 192,
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
        let fs = ps
            .verifier_fiat_shamir()
            .expect("Failed to compute verifier fiat shamir");

        // Then
        // Made using
        // ```
        // from hashlib import sha256
        // h = sha256()
        // h.update(bytes([91,123,34,97,34,58,49,44,34,98,34,58,50,125,44,123,34,97,34,58,50,44,34,98,34,58,51,125,44,123,34,97,34,58,51,44,34,98,34,58,52,125,93,]))
        // list(h.digest())
        // ````
        let expected = [
            188, 90, 181, 158, 133, 61, 76, 0, 246, 85, 241, 132, 91, 7, 84, 157, 111, 193, 104,
            105, 236, 15, 89, 163, 86, 118, 20, 24, 98, 195, 116, 174,
        ];
        assert_eq!(fs, expected);
    }
}
