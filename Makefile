# autonomous-videography-tracking — dev Makefile.
#
# `tracking-core` + `tracking-eval` build with zero system dependencies.
# `tracking-cv` links opencv-rust (real CSRT) and needs libclang + libopencv
# (+ contrib) dev packages on the host — see README "Building tracking-cv"
# if `build`/`test` fails only on that crate.

.DEFAULT_GOAL := help

help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

build:  ## cargo build --workspace
	cargo build --workspace

test:  ## cargo test --workspace
	cargo test --workspace

clippy:  ## cargo clippy --workspace, warnings as errors
	cargo clippy --workspace --all-targets -- -D warnings

fmt:  ## cargo fmt --all
	cargo fmt --all

fmt-check:  ## cargo fmt --all --check
	cargo fmt --all --check

eval:  ## Run the synthetic-data evaluation harness (tracking-eval)
	cargo run -p tracking-eval --bin eval

clean:  ## cargo clean
	cargo clean

.PHONY: help build test clippy fmt fmt-check eval clean
