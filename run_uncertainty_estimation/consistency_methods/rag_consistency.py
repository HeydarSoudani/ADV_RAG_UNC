import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import random

from run_rag_methods.src.rag_methods import *
from run_mcts_two_actions.src.models.semantic_equivalence import SemanticEquivalenceGenerator
from run_rag_methods.src.retrievers_local import BM25Retriever, ContrieverRetriever, RerankRetriever, DenseRetriever

from run_uncertainty_estimation.consistency_methods.models.trace_augmentor import (
    SearchQueryParaphraser, ThinkParaphraser, RetrievalPerturber,
    get_paraphrased_query, get_paraphrased_think
)
# from run_mcts.src.models.generate_paraphrase import (
#     SearchQueryParaphraser, ThinkGenerator,
#     get_paraphrased_query, get_paraphrased_think
# )
# from run_mcts.src.models.query_rewriter import QueryRewriter, get_rewritten_queries


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
        self.secondary_model = secondary_model
        self.secondary_tokenizer = secondary_tokenizer
        self.se_model = SemanticEquivalenceGenerator(args, device, self.rag_model.generator.generator, self.rag_model.generator.tokenizer)
        self.search_query_paraphraser = SearchQueryParaphraser(args, self.secondary_model, self.secondary_tokenizer)
        self.think_paraphraser = ThinkParaphraser(args, self.secondary_model, self.secondary_tokenizer)
        self.retrieval_perturber = RetrievalPerturber(args, self.retriever)    
    
    
    def get_masked_traces(self, qid, question, trace):
        masked_traces, answer_output_list = [], []
        has_search = len(trace) > 1
        
        if has_search:
            think_search_indices = range(0, len(trace)-1)
            # V1: random choice
            # selected_indices = random.choices(think_search_indices, k=self.args.n_generations)
            # selected_indices_group = [(x, selected_indices.count(x)) for x in sorted(set(selected_indices))]
            # V2: fix number form each depth
            action = 'doc_reodering'
            selected_indices_group = [(x, 2, action) for x in sorted(set(think_search_indices))]
            
            for (selected_index, repeat, action) in selected_indices_group:
                
                #! Step 1: Generating paraphrased search queries
                #! Step 1: Applying actions: 
                original_sq = trace[selected_index].get('search_query', None)
                # print(original_sq)
                # print('---')
                
                paraphrased_queries = self.search_query_paraphraser.inference(qid, original_sq, repeat=repeat) if original_sq else []
                retrieved_docs_list = [self.retrieval_perturber.inference(paraphrased_query) for paraphrased_query in paraphrased_queries]
                # print(paraphrased_queries)
                # print('---')
                    
                
                #! Step 2: Generating new masked traces
                for i in range(repeat):
                # for paraphrased_query in paraphrased_queries:
                    paraphrased_query = paraphrased_queries[i]
                    paraphrased_query = paraphrased_query.strip()
                    retrieved_docs = retrieved_docs_list[i]
                    new_trace = []
                    
                    # Before break point: Keep steps excluding the selected one
                    new_trace = trace[:selected_index]
                    
                    # On break point
                    # retrieved_docs = self.retriever.search(paraphrased_query) if paraphrased_query else []
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
            #! Step 1: Generating paraphrased thinks
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
        
            #! Step 2: Generating new masked traces
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
    










# history = [
#     (key, item[key])
#     for item in trace[:selected_index]
#     for key in ('think', 'search_query')
#     if item.get(key)
# ] + (
#     [('think', trace[selected_index]['think'])] if trace[selected_index].get('think') else []
# ) 
# paraphrased_queries = self.query_rewriter.inference(qid, original_sq, history, repeat=repeat)


