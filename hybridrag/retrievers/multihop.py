from hybridrag.retrievers.retriever_hybrid import HybridRetriever
import dspy
from dsp.utils import deduplicate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerateQuery(dspy.Signature):
    context = dspy.InputField(desc = "may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()


class GenerateAnswer(dspy.Signature):
    context = dspy.InputField(desc = "may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc = "often between 1 and 5 words")

class MultiHop(dspy.Module):
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        passages_per_hop: int = 3,
        max_hops: int = 3,
    ):
        super().__init__()
        self.hybrid_retriever = hybrid_retriever
        self.generate_query = [
            dspy.ChainOfThought(GenerateQuery) for _ in range(max_hops)
        ]
        self.retrieve = lambda query: self.hybrid_retriever.retrieve(query)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops
        self.passages_per_hop = passages_per_hop

    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            try:
                query_response = self.generate_query[hop](
                    context=" ".join(context) if context else "",
                    question=question
                )
                query = query_response.query
            except Exception as e:
                logger.error(f"Error generating query in hop {hop}: {e}")
                query = question

            passages = self.retrieve(query)[: self.passages_per_hop]

            context = deduplicate(context + [p.page_content for p in passages])

            if len(context) > 20:
                break

        try:
            answer_prompt = f"""
            Context Information:
            {' '.join(context)}

            Question: {question}

            Based on the above context, provide a concise answer.
            """
            pred = self.generate_answer(
                context=answer_prompt,
                question=question
            )
            return dspy.Prediction(context=context, answer=pred.answer)
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return dspy.Prediction(context=context, answer="Unable to generate answer")
