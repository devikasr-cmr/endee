from transformers import pipeline
from app.retrieval.retrieve import retrieve_context


def generate_answer(query, top_k=3):
    contexts = retrieve_context(query, top_k=top_k)

    if not contexts:
        return "I could not find relevant information in the documents."

    context_text = "\n".join(contexts)

    prompt = (
        "You are an assistant that answers questions strictly using the context below.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n"
        "Answer (one concise paragraph):"
    )

    generator = pipeline(
        "text-generation",
        model="distilgpt2"
    )

    output = generator(
        prompt,
        max_new_tokens=80,
        do_sample=False
    )[0]["generated_text"]

    # Extract only the answer part
    answer = output.split("Answer (one concise paragraph):")[-1].strip()
    answer = answer.split("\n")[0].strip()

    return answer


if __name__ == "__main__":
    query = "What is Endee?"
    answer = generate_answer(query)
    print("\n🤖 Answer:\n")
    print(answer)
