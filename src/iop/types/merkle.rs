use blake2::{Blake2b512, Digest};
use serde_big_array::BigArray;
use serde_derive::{Deserialize, Serialize};

pub type Hash = [u8; 64];

/// A 512-bit hash, used for Merkle tree leaves and internal nodes.
// BigArray is used to handle the serialization of arrays with over 32 elements
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MerkleHash {
    #[serde(with = "BigArray")]
    inner: Hash,
}

impl From<[u8; 64]> for MerkleHash {
    fn from(inner: [u8; 64]) -> Self {
        Self { inner }
    }
}

impl MerkleHash {
    pub fn as_bytes(&self) -> &[u8; 64] {
        &self.inner
    }
}

/// A Merkle tree, which owns a vector of (already hashed) leaves, from which a root hash is derived.
pub struct MerkleTree {
    leaves: Vec<MerkleHash>,
    pub root: MerkleHash,
}

impl MerkleTree {
    /// Creates a new Merkle tree given a list of hashed leaves. Computes the root hash.    
    pub fn commit(leaves: &[MerkleHash]) -> Self {
        assert!(leaves.len().is_power_of_two());

        let mut hashes: Vec<MerkleHash> = leaves.to_vec();

        // A buffer is used to store the results of each round of hashing.
        // By allocating the buffer once and reusing it, we avoid
        // the overhead of allocating a new vector in each iteration.
        // Using clear() just resets the length of the vector without
        // deallocating the memory, making it more efficient for our purposes.
        let mut buffer = Vec::with_capacity(hashes.len() / 2);

        let mut hasher = Blake2b512::new();
        while hashes.len() > 1 {
            buffer.clear();

            for pair in hashes.chunks(2) {
                hasher.update(pair[0].as_bytes());
                hasher.update(pair[1].as_bytes());
                let result: [u8; 64] = hasher.finalize_reset().into();
                buffer.push(MerkleHash::from(result));
            }
            // Swapping the references of hashes and buffer avoids
            // copying the data and prepares for the next round.
            std::mem::swap(&mut buffer, &mut hashes);
        }

        let root = hashes[0];

        Self {
            leaves: leaves.to_vec(),
            root,
        }
    }

    pub fn open(&self, index: usize) -> Vec<MerkleHash> {
        // Ensure that the number of leaves is a power of 2.
        assert_eq!(self.leaves.len() & (self.leaves.len() - 1), 0);

        // Ensure the provided index is within the bounds of the leaves.
        assert!(index < self.leaves.len());

        // This will store our results as we traverse the tree.
        let mut results = Vec::new();

        // Start with the entire set of leaves.
        let mut current_slice = &self.leaves[..];
        let mut current_index = index;

        // Traverse the tree until we reach a level with only 2 leaves.
        while current_slice.len() > 2 {
            let mid = current_slice.len() / 2;

            // If our desired leaf is in the left half...
            if current_index < mid {
                // Add the root of the right half to the results.
                results.push(MerkleTree::commit(&current_slice[mid..]).root);
                // Narrow our focus to the left half.
                current_slice = &current_slice[..mid];
            } else {
                // Otherwise, add the root of the left half to the results.
                results.push(MerkleTree::commit(&current_slice[..mid]).root);
                // Narrow our focus to the right half and adjust the current index.
                current_slice = &current_slice[mid..];
                current_index -= mid;
            }
        }

        // Add the sibling of our desired leaf to the results.
        results.push(current_slice[1 - current_index]);
        // Reverse the order of results to match the path from leaf to root.
        results.reverse();
        results
    }

    pub fn verify(&self, leaf: MerkleHash, path: &[MerkleHash], index: usize) -> bool {
        Self::is_leaf_in_tree(self.root, leaf, path, index)
    }

    fn is_leaf_in_tree(
        root: MerkleHash,
        mut leaf: MerkleHash,
        path: &[MerkleHash],
        mut index: usize,
    ) -> bool {
        // Assert that the index is less than 2 to the power of the path's length.
        // This ensures the index is within the expected range based on the path.
        assert!(index < (1 << path.len()));

        // Assert that the path is not empty. We need to have at least one hash
        // in the path to perform the verification.
        assert!(!path.is_empty());

        // Iterate over each hash in the path.
        for path_hash in path.iter() {
            // Depending on the parity of the index (even/odd), compute the current hash.
            // If the index is even (index % 2 == 0), the leaf comes first, then the path's hash.
            // Otherwise, the path's hash comes first followed by the leaf.
            let current_hash: Hash = if index % 2 == 0 {
                Blake2b512::digest([&leaf.as_bytes()[..], path_hash.as_bytes()].concat()).into()
            } else {
                Blake2b512::digest([path_hash.as_bytes(), &leaf.as_bytes()[..]].concat()).into()
            };

            // Update the leaf to be the recently computed hash.
            // This step effectively "moves" us up the Merkle tree.
            leaf = MerkleHash::from(current_hash);

            // Right shift the index for the next iteration.
            // This step divides the index by 2, essentially determining the
            // position of the next hash in the path (left or right child).
            index >>= 1;
        }

        // After iterating over the entire path and computing the final hash,
        // compare it to the given root to see if they match.
        // If they do, this confirms that the given leaf is part of the Merkle tree
        // represented by the root.
        root.as_bytes().as_ref() == leaf.as_bytes().as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::OsRng, Rng};

    fn random_leaf() -> MerkleHash {
        let mut leaf = [0u8; 64];
        OsRng.fill(&mut leaf);
        MerkleHash::from(leaf)
    }

    fn get_different_index(i: usize) -> usize {
        let mut j = i;
        while j == i {
            j = (i + 1 + (OsRng.gen_range(0..N))) % N;
        }
        j
    }

    const N: usize = 64;

    #[test]
    fn verify_path_for_each_leaf() {
        let leaves: Vec<_> = (0..N).map(|_| random_leaf()).collect();
        let tree = MerkleTree::commit(&leaves);

        leaves.iter().enumerate().for_each(|(i, leaf)| {
            let path = tree.open(i);
            assert!(tree.verify(*leaf, &path, i));
        });
    }

    #[test]
    fn random_leaf_with_original_path() {
        let leaves: Vec<_> = (0..N).map(|_| random_leaf()).collect();
        let tree = MerkleTree::commit(&leaves);

        test_random_leaf_with_original_path(&tree);
    }

    #[test]
    fn different_leaves_with_original_path() {
        let leaves: Vec<_> = (0..N).map(|_| random_leaf()).collect();
        let tree = MerkleTree::commit(&leaves);

        test_different_leaves_with_original_path(&tree, &leaves);
    }

    #[test]
    fn original_leaf_with_different_path_index() {
        let leaves: Vec<_> = (0..N).map(|_| random_leaf()).collect();
        let tree = MerkleTree::commit(&leaves);

        test_original_leaf_with_different_path_index(&tree, &leaves);
    }

    #[test]
    fn with_changed_root() {
        let leaves: Vec<_> = (0..N).map(|_| random_leaf()).collect();
        let mut tree = MerkleTree::commit(&leaves);

        test_with_changed_root(&mut tree, &leaves);
    }

    #[test]
    fn with_tampered_path() {
        let leaves: Vec<_> = (0..N).map(|_| random_leaf()).collect();
        let tree = MerkleTree::commit(&leaves);

        test_with_tampered_path(&tree, &leaves);
    }

    #[test]
    fn different_tree_original_leaves_and_paths() {
        let leaves: Vec<_> = (0..N).map(|_| random_leaf()).collect();
        let tree = MerkleTree::commit(&leaves);
        let fake_tree = MerkleTree::commit(&vec![random_leaf(); N]);
        leaves.iter().enumerate().for_each(|(i, leaf)| {
            let path = tree.open(i);
            assert!(!fake_tree.verify(*leaf, &path, i));
        });
    }

    fn test_random_leaf_with_original_path(tree: &MerkleTree) {
        for i in 0..N {
            let path = tree.open(i);
            assert!(!tree.verify(random_leaf(), &path, i));
        }
    }

    fn test_different_leaves_with_original_path(tree: &MerkleTree, leaves: &[MerkleHash]) {
        for i in 0..N {
            let path = tree.open(i);
            let j = get_different_index(i);
            assert!(!tree.verify(leaves[j], &path, i));
        }
    }

    fn test_original_leaf_with_different_path_index(tree: &MerkleTree, leaves: &[MerkleHash]) {
        leaves.iter().enumerate().for_each(|(i, leaf)| {
            let path = tree.open(i);
            let j = get_different_index(i);
            assert!(!tree.verify(*leaf, &path, j));
        });
    }

    fn test_with_changed_root(tree: &mut MerkleTree, leaves: &[MerkleHash]) {
        let original_root = tree.root;
        leaves.iter().enumerate().for_each(|(i, leaf)| {
            let path = tree.open(i);
            tree.root = random_leaf();
            assert!(!tree.verify(*leaf, &path, i));
        });
        tree.root = original_root;
    }

    fn test_with_tampered_path(tree: &MerkleTree, leaves: &[MerkleHash]) {
        leaves.iter().enumerate().for_each(|(i, leaf)| {
            let path = tree.open(i);
            for j in 0..path.len() {
                let mut fake_path = path.clone();
                fake_path[j] = random_leaf();
                assert!(!tree.verify(*leaf, &fake_path, i));
            }
        });
    }
}
