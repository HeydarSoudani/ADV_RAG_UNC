import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import random

from run_rag_methods.src.rag_methods import *
from run_mcts.src.models.semantic_equivalence import SemanticEquivalenceGenerator
from run_rag_methods.src.retrievers_local import BM25Retriever, ContrieverRetriever, RerankRetriever, DenseRetriever
from run_mcts.src.models.generate_paraphrase import (
    SearchQueryGenerator, ThinkGenerator,
    get_paraphrased_query, get_paraphrased_think
)


class RagConsistency:
    def __init__(self, 
        rag_model,
        secondary_model,
        secondary_tokenizer,
        device, args, 
    ):
        self.args = args
        # === Static Retriever =====================
        if args.retriever_name == 'bm25':
            self.retriever = BM25Retriever(args)  
        elif args.retriever_name == 'contriever':
            self.retriever = ContrieverRetriever(args)
        elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
            self.retriever = RerankRetriever(args)
        elif args.retriever_name in ['e5', 'bge']:
            self.retriever = DenseRetriever(args)
            
        # === Models ===============================
        self.rag_model = rag_model
        self.se_model = SemanticEquivalenceGenerator(args, device, self.rag_model.generator.generator, self.rag_model.generator.tokenizer)
        self.paraphrase_model = secondary_model
        self.paraphrase_tokenizer = secondary_tokenizer
        self.search_query_generator = SearchQueryGenerator(args, self.paraphrase_model, self.paraphrase_tokenizer)
        self.think_generator = ThinkGenerator(args, self.paraphrase_model, self.paraphrase_tokenizer)
    
    
    def get_masked_traces(self, qid, question, trace):
        masked_traces, answer_output_list = [], []
        has_search = len(trace) > 1
        
        if has_search:
            think_search_indices = range(0, len(trace)-1)
            selected_indices = random.choices(think_search_indices, k=self.args.n_generations)
            selected_indices_group = [(x, selected_indices.count(x)) for x in sorted(set(selected_indices))]
            
            for (selected_index, repeat) in selected_indices_group:
                
                #! Step 1: Generating paraphrased search queries 
                original_sq = trace[selected_index].get('search_query', '')
                
                sq_prompt = self.search_query_generator.get_instruction(original_sq, n=repeat)
                sq_output = self.search_query_generator.generate(sq_prompt, temperature=0.7)[0]
                paraphrased_queries = get_paraphrased_query(sq_output)
        
                # check if paraphrased_queries are None
                if paraphrased_queries == None:
                    print(f"Paraphrased queries are not provided for query {qid} ...")
                    for i in range(self.args.retry):
                        print(f"Paraphrased queries, try {i+1} ...")
                        sq_output = self.search_query_generator.generate(sq_prompt, temperature=1.0)[0]
                        paraphrased_queries = get_paraphrased_query(sq_output)
                        if paraphrased_queries != None:
                            break
                    else:
                        print(f"Failed to generate 'paraphrased queries' after all retries for query {qid}!!!")
                        paraphrased_queries = []
                
                # Check if the number of paraphrased_queries is equal to "repeat"
                if paraphrased_queries is None:
                    paraphrased_queries = []
        
                max_iterations = 10
                iteration = 0
                while len(paraphrased_queries) != repeat and iteration < max_iterations:
                    remaining = repeat - len(paraphrased_queries)        
                    extra_prompt = self.search_query_generator.get_instruction(original_sq, n=remaining)
                    extra_output = self.search_query_generator.generate(extra_prompt, temperature=1.0)[0]
                    extra_queries = get_paraphrased_query(extra_output)

                    if extra_queries:
                        paraphrased_queries.extend(extra_queries)
                        paraphrased_queries = paraphrased_queries[:repeat]  # trim if over
                    else:
                        print(f"Failed to generate extra queries on iteration {iteration + 1}")
                    iteration += 1
                if len(paraphrased_queries) != repeat:
                    print(f"Warning: Only generated {len(paraphrased_queries)} queries out of {repeat} after {iteration} iterations.")
                
                
                #! Step 2: Generating new masked traces
                for paraphrased_query in paraphrased_queries:
                    paraphrased_query = paraphrased_query.strip()
                    new_trace = []
                    
                    # Before break point: Keep steps excluding the selected one
                    new_trace = trace[:selected_index]
                    
                    # On break point
                    retrieved_docs = self.retriever.search(paraphrased_query) if paraphrased_query else []
                    new_trace.append({
                        "think": trace[selected_index].get('think', ''),
                        "search_query": paraphrased_query,
                        "docs": retrieved_docs,
                    })
                    
                    # After break point: ask searchR1 to generate
                    pred_answer, rest_of_trace = self.rag_model.inference_with_partial_trace(question, new_trace)
                    new_trace.extend(rest_of_trace)
                    
                    masked_traces.append(new_trace)
                    answer_output_list.append(pred_answer.strip() if pred_answer else '')
                
        else:
            ## Step 1: Generating paraphrased thinks
            original_think = trace[0].get('think', '')
            think_prompt = self.think_generator.get_instruction(original_think, n=self.args.n_generations)
            think_output = self.think_generator.generate(think_prompt, temperature=0.7)[0]
            paraphrased_thinks = get_paraphrased_think(think_output)
            
            # check if paraphrased_thinks are None
            if paraphrased_thinks == None:
                print(f"Paraphrased thinks are not provided for query {qid} ...")
                for i in range(self.args.retry):
                    print(f"Paraphrased thinks, try {i+1} ...")
                    think_output = self.think_generator.generate(think_prompt, temperature=1.0)[0]
                    paraphrased_thinks = get_paraphrased_query(think_output)
                    if paraphrased_thinks != None:
                        break
                else:
                    print(f"Failed to generate 'paraphrased thinks' after all retries for query {qid}!!!")
                    paraphrased_thinks = []
        
            ## Step 2: Generating new masked traces
            input_prompt_text = self.rag_model.prompt.format(question=question)
            for pt in paraphrased_thinks:
                input_text_pt = input_prompt_text + f"<think> {pt} </think>\n<answer>"
                messages = [{"role": "user", "content": input_text_pt}]
                _, output_text = self.rag_model.generator.generate(
                    messages,
                    self.rag_model.generator.searchr1_stopping_criteria,
                    temperature=self.args.consistency_temperature
                )
                pred_answer = self.get_partial_answer(output_text)
                new_trace = [{'think': pt, 'answer': pred_answer}]
                masked_traces.append(new_trace)
                answer_output_list.append(pred_answer.strip())
        
        
        # Convert mased trace to text
        masked_traces_text = [
            self.rag_model.get_input_prompt_self_consistency(question, masked_trace)
            for masked_trace in masked_traces
        ]
        
        return masked_traces, masked_traces_text, answer_output_list