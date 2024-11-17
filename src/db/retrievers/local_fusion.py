from typing import List, Optional, Dict, Tuple
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore

class LocalQueryFusionRetriever(QueryFusionRetriever):
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        local_llm_callable,
        query_gen_prompt: Optional[str] = None,
        mode: str = "simple",
        similarity_top_k: int = 5,
        num_queries: int = 4,
        use_async: bool = True,
        verbose: bool = False,
    ):
        self.local_llm_callable = local_llm_callable
        super().__init__(
            retrievers=retrievers,
            query_gen_prompt=query_gen_prompt,
            mode=mode,
            similarity_top_k=similarity_top_k,
            num_queries=num_queries,
            use_async=use_async,
            verbose=verbose,
        )

    def _get_queries(self, original_query: str) -> List[QueryBundle]:
        prompt_str = self.query_gen_prompt.format(
            num_queries=self.num_queries - 1,
            query=original_query,
        )
        print(f"Sending prompt to local LLM: {prompt_str}")
        response_text = self.local_llm_callable(prompt_str)
        print(f"Received response from local LLM: {response_text}")
        queries = response_text.split("\n")
        queries = [q.strip() for q in queries if q.strip()]
        if self.verbose:
            queries_str = "\n".join(queries)
            print(f"Generated queries:\n{queries_str}")
        return [QueryBundle(q) for q in queries]

    def _reciprocal_rerank_fusion(
        self, results: Dict[Tuple[str, int], List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        k = 60.0
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

        reranked_results = dict(
            sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        )

        reranked_nodes: List[NodeWithScore] = []
        for hash, score in reranked_results.items():
            reranked_nodes.append(hash_to_node[hash])
            reranked_nodes[-1].score = score

        return reranked_nodes

    def _relative_score_fusion(
        self,
        results: Dict[Tuple[str, int], List[NodeWithScore]],
        dist_based: Optional[bool] = False,
    ) -> List[NodeWithScore]:
        min_max_scores = {}
        for query_tuple, nodes_with_scores in results.items():
            if not nodes_with_scores:
                min_max_scores[query_tuple] = (0.0, 0.0)
                continue
            scores = [
                node_with_score.score or 0.0 for node_with_score in nodes_with_scores
            ]
            if dist_based:
                mean_score = sum(scores) / len(scores)
                std_dev = (
                    sum((x - mean_score) ** 2 for x in scores) / len(scores)
                ) ** 0.5
                min_score = mean_score - 3 * std_dev
                max_score = mean_score + 3 * std_dev
            else:
                min_score = min(scores)
                max_score = max(scores)
            min_max_scores[query_tuple] = (min_score, max_score)

        for query_tuple, nodes_with_scores in results.items():
            for node_with_score in nodes_with_scores:
                min_score, max_score = min_max_scores[query_tuple]
                if max_score == min_score:
                    node_with_score.score = 1.0 if max_score > 0 else 0.0
                else:
                    node_with_score.score = (node_with_score.score - min_score) / (
                        max_score - min_score
                    )
                retriever_idx = query_tuple[1]
                existing_score = node_with_score.score or 0.0
                node_with_score.score = (
                    existing_score * self._retriever_weights[retriever_idx]
                )
                node_with_score.score /= self.num_queries

        all_nodes: Dict[str, NodeWithScore] = {}

        for nodes_with_scores in results.values():
            for node_with_score in nodes_with_scores:
                hash = node_with_score.node.hash
                if hash in all_nodes:
                    cur_score = all_nodes[hash].score or 0.0
                    all_nodes[hash].score = cur_score + (node_with_score.score or 0.0)
                else:
                    all_nodes[hash] = node_with_score

        return sorted(all_nodes.values(), key=lambda x: x.score or 0.0, reverse=True)

    def _simple_fusion(
        self, results: Dict[Tuple[str, int], List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        all_nodes: Dict[str, NodeWithScore] = {}
        for nodes_with_scores in results.values():
            for node_with_score in nodes_with_scores:
                hash = node_with_score.node.hash
                if hash in all_nodes:
                    max_score = max(
                        node_with_score.score or 0.0, all_nodes[hash].score or 0.0
                    )
                    all_nodes[hash].score = max_score
                else:
                    all_nodes[hash] = node_with_score

        return sorted(all_nodes.values(), key=lambda x: x.score or 0.0, reverse=True)

    def _run_nested_async_queries(
        self, queries: List[QueryBundle]
    ) -> Dict[Tuple[str, int], List[NodeWithScore]]:
        tasks, task_queries = [], []
        for query in queries:
            for i, retriever in enumerate(self._retrievers):
                tasks.append(retriever.aretrieve(query))
                task_queries.append((query.query_str, i))

        task_results = run_async_tasks(tasks)

        results = {}
        for query_tuple, query_result in zip(task_queries, task_results):
            results[query_tuple] = query_result

        return results

    async def _run_async_queries(
        self, queries: List[QueryBundle]
    ) -> Dict[Tuple[str, int], List[NodeWithScore]]:
        tasks, task_queries = [], []
        for query in queries:
            for i, retriever in enumerate(self._retrievers):
                tasks.append(retriever.aretrieve(query))
                task_queries.append((query.query_str, i))

        task_results = await asyncio.gather(*tasks)

        results = {}
        for query_tuple, query_result in zip(task_queries, task_results):
            results[query_tuple] = query_result

        return results

    def _run_sync_queries(
        self, queries: List[QueryBundle]
    ) -> Dict[Tuple[str, int], List[NodeWithScore]]:
        results = {}
        for query in queries:
            for i, retriever in enumerate(self._retrievers):
                results[(query.query_str, i)] = retriever.retrieve(query)

        return results

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: List[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            queries.extend(self._get_queries(query_bundle.query_str))

        if self.use_async:
            results = self._run_nested_async_queries(queries)
        else:
            results = self._run_sync_queries(queries)

        if self.mode == "reciprocal_rerank":
            return self._reciprocal_rerank_fusion(results)[: self.similarity_top_k]
        elif self.mode == "relative_score":
            return self._relative_score_fusion(results)[: self.similarity_top_k]
        elif self.mode == "dist_based_score":
            return self._relative_score_fusion(results, dist_based=True)[
                : self.similarity_top_k
            ]
        elif self.mode == "simple":
            return self._simple_fusion(results)[: self.similarity_top_k]
        else:
            raise ValueError(f"Invalid fusion mode: {self.mode}")

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: List[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            queries.extend(self._get_queries(query_bundle.query_str))

        results = await self._run_async_queries(queries)

        if self.mode == "reciprocal_rerank":
            return self._reciprocal_rerank_fusion(results)[: self.similarity_top_k]
        elif self.mode == "relative_score":
            return self._relative_score_fusion(results)[: self.similarity_top_k]
        elif self.mode == "dist_based_score":
            return self._relative_score_fusion(results, dist_based=True)[
                : self.similarity_top_k
            ]
        elif self.mode == "simple":
            return self._simple_fusion(results)[: self.similarity_top_k]
        else:
            raise ValueError(f"Invalid fusion mode: {self.mode}")