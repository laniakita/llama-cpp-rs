# Rust `common_chat_parse` and `common_chat_msg_diff::compute_diffs` implementations

This example demonstrates the usage of the `ChatParser` and `ChatDiff` struct methods, to parse chat messages and compute diffs between them for UI token routing. 

```shell
usage: chat_parser [options]

options:
	-m, --model arg        Path to the gguf file
	-p, --prompt arg       Prompt to use for the test
	-t, --template arg     Chat template to use for the test
	-r, --reasoning        Include reasoning tokens in the prompt
	-c, --continue         Continue the conversation
	-n, --n-tokens arg     Number of tokens to generate
	-h, --help             Show this help message and exit
```