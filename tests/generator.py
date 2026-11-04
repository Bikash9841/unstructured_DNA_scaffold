from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

checkpoint = "LongSafari/hyenadna-medium-160k-seqlen-hf"

model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    trust_remote_code=True
)

model.eval()

def generate_dna(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=1.0,
    top_k=4
):
    model.eval()

    # Encode DNA prompt
    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    )["input_ids"]

    # Model device
    input_ids = input_ids.to(model.device)

    generated = input_ids.clone()

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Forward pass
            outputs = model(input_ids=generated)

            # Get logits for the last nucleotide
            logits = outputs.logits[:, -1, :]

            # Temperature
            logits = logits / temperature

            # Top-k sampling
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                min_value = values[:, -1].unsqueeze(-1)
                logits[logits < min_value] = -float("Inf")

            # Probability distribution
            probs = F.softmax(logits, dim=-1)

            # Sample next nucleotide
            next_token = torch.multinomial(probs, num_samples=1)

            # Append nucleotide
            generated = torch.cat(
                [generated, next_token],
                dim=-1
            )

    # Decode
    sequence = tokenizer.decode(
        generated[0],
        skip_special_tokens=True
    )

    return sequence


# generate sequence

prompt = "[BOS]"

sequence = generate_dna(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_new_tokens=100,
    temperature=1.0,
    top_k=4
)

print(sequence)
print("Length:", len(sequence))