import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import random
from collections import Counter

from run_rag_methods.src.rag_methods import *
from run_mcts_two_actions.src.models.semantic_equivalence import SemanticEquivalenceGenerator
from run_rag_methods.src.retrievers_local import BM25Retriever, ContrieverRetriever, RerankRetriever, DenseRetriever

from run_uncertainty_estimation.consistency_methods.models.trace_augmentor import (
    SearchQueryParaphraser, ThinkParaphraser, RetrievalPerturber, CriticalThinkGenerator,
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
        self.critical_think_generator = CriticalThinkGenerator(args, self.secondary_model, self.secondary_tokenizer)
    
    def get_masked_traces(self, qid, question, trace):
        masked_traces, answer_output_list = [], []
        has_search = len(trace) > 1
        
        if has_search:
            think_search_indices = range(0, len(trace)-1)
            actions = ['query_paraphrasing', 'adding_critical_thought'] # 'doc_shuffling'
            # --- V1: random choice
            # selected_indices = random.choices(think_search_indices, k=self.args.n_generations)
            # selected_indices_group = [(x, selected_indices.count(x), action) for x in sorted(set(selected_indices))]
            
            # --- V2: fix number form each depth
            # selected_indices_group = [(x, 2, action) for x in sorted(set(think_search_indices))]
            
            random_pairs = [(random.choice(think_search_indices), random.choice(actions)) for _ in range(self.args.n_generations)]
            pair_counts = Counter(random_pairs)
            selected_indices_group = [(index, repeat, action) for (index, action), repeat in pair_counts.items()]
            print(selected_indices_group)
            print('----')
            
            # TODO: actions combinition
            
            for (selected_index, repeat, action) in selected_indices_group:
                original_think = trace[selected_index].get('think', '')
                original_sq = trace[selected_index].get('search_query', None)
                original_docs = trace[selected_index].get('docs', [])
                
                #! Step 1: Applying actions
                if action == 'query_paraphrasing':
                    paraphrased_queries = self.search_query_paraphraser.inference(qid, original_sq, repeat=repeat) if original_sq else []
                    # retrieved_docs_list = [self.retrieval_perturber.inference(paraphrased_query) for paraphrased_query in paraphrased_queries]
                    retrieved_docs_list = [self.retriever.search(paraphrased_query) if paraphrased_query else [] for paraphrased_query in paraphrased_queries]
                elif action == 'adding_critical_thought':
                    critical_thinks, critical_search_queries = self.critical_think_generator.inference(qid, original_sq, repeat=repeat) if original_sq else []
                    retrieved_docs_list = [self.retriever.search(critical_query) if critical_query else [] for critical_query in critical_search_queries]

                #! Step 2: Generating new masked traces
                for i in range(repeat):
                    new_trace = []
                    
                    # - A) Before the break point: Keep steps excluding the selected one
                    new_trace = trace[:selected_index]
                    
                    # - B) On the break point
                    if action == 'query_paraphrasing':
                        paraphrased_query = paraphrased_queries[i].strip()
                        retrieved_docs = retrieved_docs_list[i]
                        new_trace.append({
                            "think": original_think,
                            "search_query": paraphrased_query,
                            "docs": retrieved_docs,
                        })
                    elif action == 'adding_critical_thought':
                        critical_think = critical_thinks[i].strip()
                        critical_query = critical_search_queries[i].strip()
                        critical_docs = retrieved_docs_list[i]
                        new_trace.append({"think": original_think, "search_query": original_sq, "docs": original_docs})
                        new_trace.append({"think": critical_think, "search_query": critical_query, "docs": critical_docs})
                        
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


