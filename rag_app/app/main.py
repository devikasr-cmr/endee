from app.generation.generate import generate_answer


def main():
    print("\n🧠 RAG System using Endee Vector Database")
    print("-" * 40)

    while True:
        query = input("\n❓ Ask a question (or type 'exit'): ")

        if query.lower() in {"exit", "quit"}:
            print("\n👋 Exiting RAG system. Bye!")
            break

        answer = generate_answer(query)

        print("\n🤖 Answer:")
        print(answer)
        print("-" * 40)


if __name__ == "__main__":
    main()
