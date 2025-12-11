"""
BPE Tokenizer in the style of GPT-4.

Our own RustBPE Tokenizer for training and tiktoken for efficient inference
"""
import os
import copy
from functools import lru_cache

SPECIAL_TOKENS = [
    # beginning of sequence token that delimits documents
    "<|bos|>",
    # tokens below used only in finetuning to render convos into token ids
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>", 
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL output
    "<|output_end|>",
]

# split pattern from GPT-4 but use \p{N}{1,2} instead of \p{N}{1,3} for numbers
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# rustbpe + tiktoken tokenizer for training and inference
import pickle
import rustbpe
import tiktoken

class RustBPETokenizer:
    """light wrapper around tiktoken (for efficient inference) and rustbpe (for training)"""
    def __init__(self, enc, bos_token):
        # Store the tiktoken encoding object (handles actual tokenization)
        self.enc = enc
        # Convert the BOS (beginning of sequence) token string to its integer ID
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        """Train a new tokenizer from scratch using a text iterator (e.g., lines from a file)"""
        # 1. train w/ rustbpe
        tokenizer = rustbpe.Tokenizer()
        # special tokens inserted later in __init__, we don't train them here
        # Reserve space in vocab for special tokens (they get added at the end)
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        # Train the BPE tokenizer: learns which byte pairs to merge based on frequency in training data
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        
        # 2. construct associated tiktoken encoding for inference
        # Extract the trained regex pattern used for splitting text
        pattern = tokenizer.get_pattern()
        # Get the learned merge rules: each token gets a rank (priority for merging)
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        # Convert to dict format expected by tiktoken: token bytes -> rank
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        # Special tokens go after regular tokens in the vocabulary
        tokens_offset = len(mergeable_ranks)
        # Assign IDs to special tokens (BOS, user_start, etc.)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        # Create tiktoken encoding (fast inference engine) with our trained vocabulary
        enc = tiktoken.Encoding(
            name = "rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>")
    
    @classmethod
    def from_directory(cls, tokenizer_dir):
        """Load a previously saved tokenizer from disk"""
        # Build path to the pickled encoding file
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        # Load the tiktoken encoding object from disk
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")
    
    @classmethod
    def from_pretrained(cls, tiktoken_name):
        """Load a pretrained tokenizer from OpenAI (e.g., 'gpt2', 'cl100k_base' for GPT-4)"""
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        # Get the encoding by name (downloads vocab if needed)
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls special document delimiter token "<|endoftext|>"
        # this is confusing because this token is almost always prepended to beginning of doc
        # most often used to signal start of new sequence to LLM during inference etc.
        # so in jujuchat (nanochat) we use "<|bos|>" instead
        return cls(enc, "<|endoftext|>")
    
    def get_vocab_size(self):
        """Return total number of tokens in vocabulary (regular + special tokens)"""
        return self.enc.n_vocab
    
    def get_special_tokens(self):
        """Return set of all special token strings (e.g., '<|bos|>', '<|user_start|>')"""
        return self.enc.special_tokens_set
    
    def id_to_token(self, id):
        """Convert a single token ID back to its string representation"""
        return self.enc.decode([id])
    
    @lru_cache(maxsize=32)
    def encode_special(self, text):
        """Encode a special token string to its ID (cached for performance)"""
        return self.enc.encode_single_token(text)
    
    def get_bos_token_id(self):
        """Return the ID of the beginning-of-sequence token"""
        return self.bos_token_id
    
    def encode(self, text, prepend=None, append=None, num_threads=8):
        """
        Encode text into token IDs with optional prepend/append tokens.
        Args:
            text: str or list[str] - text to tokenize
            prepend: optional token ID (int) or special token (str) to add at start
            append: optional token ID (int) or special token (str) to add at end
            num_threads: number of threads for batch processing
        Returns:
            list[int] if text is str, or list[list[int]] if text is list
        """
        # Convert prepend/append to token IDs if they're strings (e.g., "<|bos|>")
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
        
        # Single string: tokenize and add prepend/append
        if isinstance(text, str):
            # Tokenize the text (doesn't include special tokens)
            ids = self.enc.encode_ordinary(text)
            # Add prepend token at the beginning
            if prepend is not None:
                ids.insert(0, prepend_id)
            # Add append token at the end
            if append is not None:
                ids.append(append_id)
            return ids
        
        # Batch of strings: tokenize all in parallel, then add prepend/append to each
        elif isinstance(text, list):
            # Batch tokenization is faster than processing each string individually
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            # Add prepend token to the start of each sequence
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            # Add append token to the end of each sequence
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
            
        return ids
    
    def __call__(self, *args, **kwargs):
        """Allow using tokenizer as a function: tokenizer(text) same as tokenizer.encode(text)"""
        return self.encode(*args, **kwargs)
    
    def decode(self, ids):
        """Convert token IDs back to text string"""
        return self.enc.decode(ids)
    
    def save(self, tokenizer_dir):
        """Save tokenizer to disk for later use"""
        # Create directory if it doesn't exist
        os.makedirs(tokenizer_dir, exist_ok=True)
        # Save the tiktoken encoding object as a pickle file
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")
    
    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single chat conversation for training.
        This converts a conversation into token IDs with a mask indicating which tokens to train on.
        
        Args:
            conversation: dict with "messages" list containing role/content pairs
            max_tokens: maximum sequence length (truncate if longer)
        
        Returns:
            - ids: list[int] - token IDs of the full conversation
            - mask: list[int] - 1 for tokens the model should predict (assistant output), 0 otherwise
        """
        # Initialize lists to store token IDs and training masks
        ids, mask = [], []
        
        def add_tokens(token_ids, mask_val):
            """Helper to add tokens and their corresponding mask values"""
            # Convert single token ID to list for uniform handling
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            # Add token IDs to the sequence
            ids.extend(token_ids)
            # Add mask values (1 = train on this, 0 = don't train on this)
            mask.extend([mask_val] * len(token_ids))

        # sometimes first message is a system message => just merge w/ second (user) message
        # System messages provide instructions/context that should be part of the user prompt
        if conversation["messages"][0]["role"] == "system":
            # some conversation surgery here
            conversation = copy.deepcopy(conversation) # avoid mutating original
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by user message"
            # Merge system message into user message (system instructions + user query)
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            # Remove the system message (now merged into user message)
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all special tokens needed to structure the conversation
        # These tokens mark different parts of the conversation (user vs assistant, tool calls, etc.)
        bos = self.get_bos_token_id()  # Beginning of sequence
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # now we can tokenize the convo
        # Start with BOS token (mask=0 because we don't train on it)
        add_tokens(bos, 0)
        
        # Process each message in the conversation
        for i, message in enumerate(messages):
            # Ensure messages alternate between user and assistant (user first)
            # This prevents malformed conversations where roles are out of order
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} must be from {must_be_from}, got {message['role']}"

            # content can either be a simple string or list of parts (ex. containing tool calls)
            content = message["content"]
            
            # USER MESSAGES: wrap content in <|user_start|> ... <|user_end|>
            # We don't train on user messages (mask=0) since the model only predicts assistant responses
            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = self.encode(content)  # Tokenize user text
                add_tokens(user_start, 0)  # User start marker (not trained on)
                add_tokens(value_ids, 0)   # User message content (not trained on)
                add_tokens(user_end, 0)    # User end marker (not trained on)
            # ASSISTANT MESSAGES: wrap content in <|assistant_start|> ... <|assistant_end|>
            # We DO train on assistant messages (mask=1) since this is what the model should learn to generate
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)  # Start marker (not trained on)
                
                # Simple string response (most common case)
                if isinstance(content, str):
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)  # Train on assistant's text
                
                # Complex response with multiple parts (text + tool calls + outputs)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        
                        if part["type"] == "text":
                            # Regular text from assistant
                            add_tokens(value_ids, 1)  # Train on this
                        
                        elif part["type"] == "python":
                            # Assistant invoking Python code (tool call)
                            add_tokens(python_start, 1)  # Train on python markers
                            add_tokens(value_ids, 1)     # Train on the code itself
                            add_tokens(python_end, 1)    # Train on python markers
                        
                        elif part["type"] == "python_output":
                            # Output from Python REPL (determined by environment, not model)
                            # Don't train on this since the model doesn't generate it
                            add_tokens(output_start, 0)  # Don't train on output markers
                            add_tokens(value_ids, 0)     # Don't train on REPL output
                            add_tokens(output_end, 0)    # Don't train on output markers
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                
                add_tokens(assistant_end, 1)  # End marker (trained on)

        # truncate to max_tokens tokens MAX (helps prevent OOMs during training)
        # Longer sequences use more memory, so we limit them
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask
    
    def visualize_tokenization(self, ids, mask):
        """
        Small helper function useful in debugging: visualize tokenization of render_conversation.
        Shows tokens in GREEN if trained on (mask=1) or RED if not trained on (mask=0).
        """
        # ANSI color codes for terminal output
        RED = '\033[91m'    # Not trained on (user messages, special tokens, etc.)
        GREEN = '\033[92m'  # Trained on (assistant responses)
        RESET = '\033[0m'   # Reset to default color
        
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            # Convert token ID back to text
            token_str = self.decode([token_id])
            # Color based on whether we train on it
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
        # Join with '|' to clearly separate tokens
        return '|'.join(tokens)

    def render_for_completion(self, conversation):
        """
        Used for reinforcement learning / inference where we want the model to complete a conversation.
        Removes the last assistant message and adds <|assistant_start|> to prompt the model to respond.
        Unlike supervised training (render_conversation), we don't need the mask here.
        
        Args:
            conversation: dict with "messages", last message must be from assistant
        Returns:
            ids: list[int] - token IDs ending with <|assistant_start|> to prompt completion
        """
        # Make a copy to avoid modifying the original conversation
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from assistant"
        # Remove the assistant's last message (we want the model to generate it)
        messages.pop()
        # Tokenize the conversation up to (but not including) the assistant's response
        ids, mask = self.render_conversation(conversation)
        # Add the assistant start token to prompt the model to generate a response
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids
        
# jujuchat (nanochat)-specific convenience functions

def get_tokenizer():
    """
    Load the jujuchat project's trained tokenizer from the standard location.
    This is a convenience function to avoid manually specifying the tokenizer path.
    """
    from jujuchat.common import get_base_dir
    # Get the project's base directory
    base_dir = get_base_dir()
    # Standard location where trained tokenizer is saved
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    return RustBPETokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    """
    Load the token bytes tensor used for fast decoding in the model.
    Each token ID maps to its byte representation for efficient text generation.
    
    Args:
        device: 'cpu' or 'cuda' - where to load the tensor
    Returns:
        torch.Tensor: shape (vocab_size, max_token_length) with byte values
    """
    import torch
    from jujuchat.common import get_base_dir
    # Get the project's base directory
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    # token_bytes.pt is generated during tokenizer training
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    # Load the tensor to the specified device (CPU or GPU)
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
