clippy:
	cargo clippy --workspace --all-features --benches --examples --tests -- -D warnings

check-fmt:
	cargo fmt --all -- --check

test: 
	cargo nextest run
