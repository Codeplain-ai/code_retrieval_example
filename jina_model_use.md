# Jina Code Embeddings: A Small but Performant Code Embedding Model

## Intended Usage & Model Info

jina-code-embeddings is an embedding model for code retrieval. The model supports various types of code retrieval (text-to-code, code-to-code, code-to-text, code-to-completion) and technical question answering across 15+ programming languages.

Built on Qwen/Qwen2.5-Coder-0.5B, jina-code-embeddings-0.5b features:

- Multilingual support (15+ programming languages) and compatibility with a wide range of domains, including web development, software development, machine learning, data science, and educational coding problems.
- Task-specific instruction prefixes for NL2Code, Code2Code, Code2NL, Code2Completion, and Technical QA, which can be selected at inference time.
- Flexible embedding size: dense embeddings are 896-dimensional by default but can be truncated to as low as 64 with minimal performance loss.

## Summary of Features

| Feature | Jina Code Embeddings 0.5B |
|---------|---------------------------|
| Base Model | Qwen2.5-Coder-0.5B |
| Supported Tasks | nl2code, code2code, code2nl, code2completion, qa |
| Model DType | BFloat 16 |
| Max Sequence Length | 32768 |
| Embedding Vector Dimension | 896 |
| Matryoshka dimensions | 64, 128, 256, 512, 896 |
| Pooling Strategy | Last-token pooling |
| Attention Mechanism | FlashAttention2 |

## Usage

### Requirements

#### via transformers

```bash
# !pip install transformers>=4.53.0 torch>=2.7.1
```

```python
import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

INSTRUCTION_CONFIG = {
    "nl2code": {
        "query": "Find the most relevant code snippet given the following query:\n",
        "passage": "Candidate code snippet:\n"
    },
    "qa": {
        "query": "Find the most relevant answer given the following question:\n",
        "passage": "Candidate answer:\n"
    },
    "code2code": {
        "query": "Find an equivalent code snippet given the following code snippet:\n",
        "passage": "Candidate code snippet:\n"
    },
    "code2nl": {
        "query": "Find the most relevant comment given the following code snippet:\n",
        "passage": "Candidate comment:\n"
    },
    "code2completion": {
        "query": "Find the most relevant completion given the following start of code snippet:\n",
        "passage": "Candidate completion:\n"
    }
}

MAX_LENGTH = 8192

def cosine_similarity(x,y):
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return x @ y.T

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def add_instruction(instruction, query):
    return f'{instruction}{query}'

# The queries and documents to embed
queries = [
    add_instruction(INSTRUCTION_CONFIG["nl2code"]["query"], "print hello world in python"),
    add_instruction(INSTRUCTION_CONFIG["nl2code"]["query"], "initialize array of 5 zeros in c++")
]
documents = [
    add_instruction(INSTRUCTION_CONFIG["nl2code"]["passage"], "print('Hello World!')"),
    add_instruction(INSTRUCTION_CONFIG["nl2code"]["passage"], "int arr[5] = {0, 0, 0, 0, 0};")
]
all_inputs = queries + documents

tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-code-embeddings-0.5b')
model = AutoModel.from_pretrained('jinaai/jina-code-embeddings-0.5b')

batch_dict = tokenizer(
    all_inputs,
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
batch_dict.to(model.device)
outputs = model(**batch_dict)
embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
query_embeddings = embeddings[:2]
passage_embeddings = embeddings[2:]

# Compute the (cosine) similarity between the query and document embeddings
scores = cosine_similarity(query_embeddings, passage_embeddings)
print(scores)
# tensor([[0.8168, 0.1236],
#         [0.1204, 0.5525]], grad_fn=<MmBackward0>)
```

#### via SentenceTransformers

```python
# !pip install sentence_transformers>=5.0.0 torch>=2.7.1

import torch
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer(
    "jinaai/jina-code-embeddings-0.5b",
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "device_map": "cuda"
    },
    tokenizer_kwargs={"padding_side": "left"},
)

# The queries and documents to embed
queries = [
    "print hello world in python",
    "initialize array of 5 zeros in c++"
]
documents = [
    "print('Hello World!')",
    "int arr[5] = {0, 0, 0, 0, 0};"
]

query_embeddings = model.encode(queries, prompt_name="nl2code_query")
document_embeddings = model.encode(documents, prompt_name="nl2code_document")

# Compute the (cosine) similarity between the query and document embeddings
similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)
# tensor([[0.8169, 0.1214],
#         [0.1190, 0.5500]])

```

For testing, using a smaller model may be faster. Note that this smaller model does not support the prompt_name parameter in the model.encode function.

```python
!pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer(
    "jinaai/jina-embeddings-v2-small-en", # switch to en/zh for English or Chinese
    trust_remote_code=True
)

# control your input sequence length up to 8192
model.max_seq_length = 1024

embeddings = model.encode([
    'How is the weather today?',
    'What is the current weather like today?'
])
print(cos_sim(embeddings[0], embeddings[1]))
```


To prevent segfaults when loading models from within the CLI, torch thread limits must be set before any model loading occurs. This should be done in the CLI entry point as follows:

```python
if __name__ == '__main__':
    import torch
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # If torch has already been initialized (e.g., during testing),
        # we can't change thread settings, but we can continue
        pass
    
    cli()
```


# Approximate indexing of a codebase
```python
# 1. Indexing codebase
def index_codebase(path):
    files = list(Path(path).rglob("*.py"))
    embeddings, metadata = [], []

    for file in files:
        code = file.read_text()
        chunks = code.split("\n\n")  # simple function/class split
        for i, chunk in enumerate(chunks):
            emb = model.encode(chunk)
            embeddings.append(emb)
            metadata.append({
                "file": str(file),
                "chunk_index": i,
                "snippet": chunk[:100]  # first 100 chars
            })

    # Create FAISS index
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    # Save index and metadata
    faiss.write_index(index, "code_index.faiss")
    with open("metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

# 2. Search
def search_code(query, top_k=5):
    index = faiss.read_index("code_index.faiss")
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    q_emb = model.encode([query]).astype("float32")
    distances, indices = index.search(q_emb, top_k)
    for idx in indices[0]:
        print(metadata[idx]["file"], metadata[idx]["chunk_index"])
        print(metadata[idx]["snippet"])
        print("-"*50)
```