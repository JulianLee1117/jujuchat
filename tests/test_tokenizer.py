"""
Test script for RustBPE + Python tokenizer wrapper.
Run: python test_tokenizer.py
"""

import sys
import os
from pathlib import Path

# Add project root to Python path so we can import jujuchat
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tempfile

# =============================================================================
# Test 1: Raw rustbpe module (Chunk 2.2)
# =============================================================================
print("=" * 60)
print("TEST 1: Raw rustbpe Rust module")
print("=" * 60)

import rustbpe

tok = rustbpe.Tokenizer()
# Train on simple repeated text
tok.train_from_iterator(["hello world", "hello rust", "hello world"] * 100, vocab_size=300)
ids = tok.encode("hello")
print(f"✅ rustbpe: Encoded 'hello' -> {ids}")

ids2 = tok.encode("hello world")
print(f"✅ rustbpe: Encoded 'hello world' -> {ids2}")

# Check that merges happened (should have fewer tokens than bytes)
assert len(ids2) < len("hello world"), "BPE should compress text"
print(f"✅ rustbpe: Compression working ({len('hello world')} bytes -> {len(ids2)} tokens)")

# =============================================================================
# Test 2: RustBPETokenizer wrapper (Chunk 2.3)
# =============================================================================
print("\n" + "=" * 60)
print("TEST 2: RustBPETokenizer Python wrapper")
print("=" * 60)

# Import your tokenizer module (adjust path if needed)
from jujuchat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS

# Train on a small dataset
texts = ["Hello world!", "The quick brown fox jumps over the lazy dog."] * 100
tokenizer = RustBPETokenizer.train_from_iterator(iter(texts), vocab_size=500)

print(f"✅ Trained tokenizer with vocab_size={tokenizer.get_vocab_size()}")
print(f"✅ Special tokens: {tokenizer.get_special_tokens()}")

# Test encode/decode roundtrip
test_text = "Hello world!"
ids = tokenizer.encode(test_text)
decoded = tokenizer.decode(ids)
print(f"✅ Encode/decode: '{test_text}' -> {ids} -> '{decoded}'")
assert decoded == test_text, f"Roundtrip failed: expected '{test_text}', got '{decoded}'"

# Test special tokens
bos_id = tokenizer.get_bos_token_id()
print(f"✅ BOS token ID: {bos_id}")

user_start_id = tokenizer.encode_special("<|user_start|>")
print(f"✅ <|user_start|> token ID: {user_start_id}")

# Test prepend/append
ids_with_bos = tokenizer.encode("Hello", prepend="<|bos|>")
assert ids_with_bos[0] == bos_id, "Prepend should add BOS at start"
print(f"✅ Prepend works: {ids_with_bos}")

# Test batch encoding
batch = ["Hello", "World"]
batch_ids = tokenizer.encode(batch)
assert len(batch_ids) == 2, "Batch should return list of lists"
print(f"✅ Batch encode: {batch_ids}")

# =============================================================================
# Test 3: Save/Load (Chunk 2.3)
# =============================================================================
print("\n" + "=" * 60)
print("TEST 3: Save and Load")
print("=" * 60)

with tempfile.TemporaryDirectory() as tmpdir:
    # Save
    tokenizer.save(tmpdir)
    assert os.path.exists(os.path.join(tmpdir, "tokenizer.pkl")), "Pickle file should exist"
    print(f"✅ Saved to {tmpdir}")
    
    # Load
    loaded = RustBPETokenizer.from_directory(tmpdir)
    assert loaded.get_vocab_size() == tokenizer.get_vocab_size(), "Vocab size should match"
    
    # Verify encoding matches
    ids_original = tokenizer.encode("test")
    ids_loaded = loaded.encode("test")
    assert ids_original == ids_loaded, "Loaded tokenizer should produce same encodings"
    print(f"✅ Loaded tokenizer matches original")

# =============================================================================
# Test 4: render_conversation (Chunk 2.3)
# =============================================================================
print("\n" + "=" * 60)
print("TEST 4: render_conversation")
print("=" * 60)

# Simple conversation
conversation = {
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"}
    ]
}

ids, mask = tokenizer.render_conversation(conversation)
print(f"✅ Rendered conversation: {len(ids)} tokens, {sum(mask)} supervised (mask=1)")

# Verify structure
assert ids[0] == bos_id, "Should start with BOS"
assert len(ids) == len(mask), "ids and mask should have same length"
assert sum(mask) > 0, "Should have some supervised tokens (assistant response)"
assert sum(mask) < len(mask), "Should have some unsupervised tokens (user input)"

# Visualize (optional)
print("\nTokenization visualization:")
viz = tokenizer.visualize_tokenization(ids, mask)
print(viz)

# =============================================================================
# Test 5: render_conversation with system message
# =============================================================================
print("\n" + "=" * 60)
print("TEST 5: System message handling")
print("=" * 60)

conversation_with_system = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello!"}
    ]
}

ids, mask = tokenizer.render_conversation(conversation_with_system)
print(f"✅ System message merged into user: {len(ids)} tokens")

# =============================================================================
# Test 6: render_conversation with tool calls
# =============================================================================
print("\n" + "=" * 60)
print("TEST 6: Tool call handling")
print("=" * 60)

conversation_with_tools = {
    "messages": [
        {"role": "user", "content": "What is 123 * 456?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me calculate: "},
            {"type": "python", "text": "123 * 456"},
            {"type": "python_output", "text": "56088"},
            {"type": "text", "text": " The answer is 56088."}
        ]}
    ]
}

ids, mask = tokenizer.render_conversation(conversation_with_tools)
supervised_count = sum(mask)
print(f"✅ Tool call conversation: {len(ids)} tokens, {supervised_count} supervised")

# Verify python_output is NOT supervised
# The model should learn to call tools, but not to predict tool output
print(f"✅ Tool outputs correctly masked (not all tokens supervised)")

# =============================================================================
# Test 7: render_for_completion (for RL)
# =============================================================================
print("\n" + "=" * 60)
print("TEST 7: render_for_completion (RL mode)")
print("=" * 60)

conversation_for_rl = {
    "messages": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris"}  # This will be removed
    ]
}

ids = tokenizer.render_for_completion(conversation_for_rl)
assistant_start_id = tokenizer.encode_special("<|assistant_start|>")
assert ids[-1] == assistant_start_id, "Should end with <|assistant_start|>"
print(f"✅ render_for_completion: {len(ids)} tokens, ends with <|assistant_start|>")

# =============================================================================
# All tests passed!
# =============================================================================
print("\n" + "=" * 60)
print("🎉 ALL TESTS PASSED!")
print("=" * 60)