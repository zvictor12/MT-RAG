from mtrag.config import settings
from mtrag.llm import AnswerGenerator, OllamaClient, QueryRewriter
from mtrag.schemas import BenchmarkTask, Context, Message


task = BenchmarkTask(
    task_id="smoke<::>2",
    conversation_id="smoke",
    turn=2,
    collection="mt-rag-clapnq-elser-512-100-20240503",
    domain="clapnq",
    messages=(
        Message("user", "Tell me about the 2012 Summer Olympics."),
        Message("agent", "What would you like to know?"),
        Message("user", "Which city hosted them?"),
    ),
)
contexts = (
    Context(
        document_id="smoke-document",
        title="2012 Summer Olympics",
        text="The 2012 Summer Olympics were held in London, United Kingdom.",
    ),
)

client = OllamaClient(settings)
try:
    rewritten = QueryRewriter(
        client,
        model_name=client.model_name,
        max_tokens=64,
    ).rewrite(task)
    answer = AnswerGenerator(
        client,
        model_name=client.model_name,
        max_tokens=64,
    ).generate(task, contexts)
finally:
    client.unload()

print("rewritten query:", rewritten)
print("grounded answer:", answer)
