import asyncio
from enum import Enum
from typing import Dict, List, Optional, Tuple

from llama_index.core.async_utils import run_async_tasks
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import IndexNode, NodeWithScore, QueryBundle
from langchain.schema import Document

QUERY_GEN_PROMPT = (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Generate {num_queries} search queries, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
)


class FUSION_MODES(str, Enum):
    """Enum for different fusion modes."""

    RECIPROCAL_RANK = "reciprocal_rerank"  # apply reciprocal rank fusion
    RELATIVE_SCORE = "relative_score"  # apply relative score fusion
    DIST_BASED_SCORE = "dist_based_score"  # apply distance-based score fusion
    SIMPLE = "simple"  # simple re-ordering of results based on original scores


class LocalQueryFusionRetriever(BaseRetriever):
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        local_llm: callable,  # Function to call your local LLM
        query_gen_prompt: Optional[str] = None,
        mode: FUSION_MODES = FUSION_MODES.SIMPLE,
        similarity_top_k: int = 5,
        num_queries: int = 4,
        use_async: bool = True,
        verbose: bool = False,
        retriever_weights: Optional[List[float]] = None,
    ) -> None:
        self.num_queries = num_queries
        self.query_gen_prompt = query_gen_prompt or QUERY_GEN_PROMPT
        self.similarity_top_k = similarity_top_k
        self.mode = mode
        self.use_async = use_async
        self.local_llm = local_llm  # Callable for local LLM
        self.verbose = verbose

        self._retrievers = retrievers
        if retriever_weights is None:
            self._retriever_weights = [1.0 / len(retrievers)] * len(retrievers)
        else:
            # Normalize weights to ensure they sum to 1
            total_weight = sum(retriever_weights)
            self._retriever_weights = [w / total_weight for w in retriever_weights]

        super().__init__()

    def _get_queries(self, original_query: str) -> List[QueryBundle]:
        """Generate query variations using the local LLM."""
        prompt = self.query_gen_prompt.format(
            num_queries=self.num_queries - 1,
            query=original_query,
        )

        try:
            # Call the local LLM
            response = self.local_llm(prompt)
            queries = response.split("\n")
            queries = [q.strip() for q in queries if q.strip()]

            if self.verbose:
                print(f"Generated queries:\n{queries}")

            return [QueryBundle(q) for q in queries[: self.num_queries - 1]]

        except Exception as e:
            raise RuntimeError(f"Error generating queries with local LLM: {e}")

    def _reciprocal_rerank_fusion(
        self, results: Dict[Tuple[str, int], List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        """Apply reciprocal rank fusion."""
        k = 60.0  # Controls the impact of outlier rankings
        fused_scores = {}
        hash_to_node = {}

        for nodes_with_scores in results.values():
            for rank, node_with_score in enumerate(
                sorted(nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True)
            ):
                hash = node_with_score.node.hash
                hash_to_node[hash] = node_with_score
                if hash not in fused_scores:
                    fused_scores[hash] = 0.0
                fused_scores[hash] += 1.0 / (rank + k)

        # Sort results by score and return
        reranked_results = sorted(
            fused_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [hash_to_node[hash] for hash, _ in reranked_results][: self.similarity_top_k]

    def _relative_score_fusion(
        self, results: Dict[Tuple[str, int], List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        """Apply relative score fusion."""
        min_max_scores = {}

        # Min-max scale each retriever's scores
        for query_tuple, nodes_with_scores in results.items():
            if not nodes_with_scores:
                min_max_scores[query_tuple] = (0.0, 0.0)
                continue
            scores = [node.score or 0.0 for node in nodes_with_scores]
            min_score, max_score = min(scores), max(scores)
            min_max_scores[query_tuple] = (min_score, max_score)

        # Adjust scores based on min-max scaling
        for query_tuple, nodes_with_scores in results.items():
            for node in nodes_with_scores:
                min_score, max_score = min_max_scores[query_tuple]
                if max_score == min_score:
                    node.score = 1.0 if max_score > 0 else 0.0
                else:
                    node.score = (node.score - min_score) / (max_score - min_score)
                retriever_idx = query_tuple[1]
                node.score *= self._retriever_weights[retriever_idx]
                node.score /= self.num_queries

        all_nodes = {}
        for nodes_with_scores in results.values():
            for node in nodes_with_scores:
                hash = node.node.hash
                if hash in all_nodes:
                    all_nodes[hash].score += node.score
                else:
                    all_nodes[hash] = node

        return sorted(all_nodes.values(), key=lambda x: x.score or 0.0, reverse=True)

    def _run_sync_queries(
        self, queries: List[QueryBundle]
    ) -> Dict[Tuple[str, int], List[NodeWithScore]]:
        results = {}
        for query in queries:
            for i, retriever in enumerate(self._retrievers):
                results[(query.query_str, i)] = retriever.retrieve(query)
        return results

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries = [query_bundle]
        if self.num_queries > 1:
            queries.extend(self._get_queries(query_bundle.query_str))

        results = self._run_sync_queries(queries)

        if self.mode == FUSION_MODES.RECIPROCAL_RANK:
            return self._reciprocal_rerank_fusion(results)
        elif self.mode == FUSION_MODES.RELATIVE_SCORE:
            return self._relative_score_fusion(results)
        elif self.mode == FUSION_MODES.SIMPLE:
            return sorted(
                sum(results.values(), []),
                key=lambda x: x.score or 0.0,
                reverse=True,
            )[: self.similarity_top_k]
        else:
            raise ValueError(f"Invalid fusion mode: {self.mode}")
