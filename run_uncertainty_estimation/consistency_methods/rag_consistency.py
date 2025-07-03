import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import math
import random
from collections import Counter

from run_rag_methods.src.rag_methods import *
from run_mcts_two_actions.src.models.semantic_equivalence import SemanticEquivalenceGenerator
from run_rag_methods.src.retrievers_local import BM25Retriever, ContrieverRetriever, RerankRetriever, DenseRetriever

from run_uncertainty_estimation.consistency_methods.models.trace_augmentor import (
    SearchQueryParaphraser, ThinkParaphraser, RetrievalPerturber, CriticalThinkGenerator, AnswerValidator,
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
        self.answer_validator = AnswerValidator(args, self.secondary_model, self.secondary_tokenizer)
    
    def get_masked_traces(self, qid, question, prediction, trace):
        # actions = ['query_paraphrasing', 'adding_critical_thought', 'answer_validation'] #  'answer_validation', 'doc_shuffling'
        actions = ['query_paraphrasing']
        
        masked_traces, answer_output_list = [], []
        if self.args.rag_method == 'search_r1':
            has_search = len(trace) > 1
            think_search_indices = range(0, len(trace)-1)
        elif self.args.rag_method == 'self_ask':
            has_search = len(trace) > 2
            think_search_indices = range(1, len(trace)-1)
        
        if has_search:
            # --- V1: random choice
            # selected_indices = random.choices(think_search_indices, k=self.args.n_generations)
            # selected_indices_group = [(x, selected_indices.count(x), action) for x in sorted(set(selected_indices))]
            # --- V2: fix number form each depth
            # selected_indices_group = [(x, 2, action) for x in sorted(set(think_search_indices))]
            
            random_pairs = [(random.choice(think_search_indices), random.choice(actions)) for _ in range(self.args.n_generations)]
            pair_counts = Counter(random_pairs)
            selected_indices_group = [(index, repeat, action) for (index, action), repeat in pair_counts.items()]
            
            for (selected_index, repeat, action) in selected_indices_group:
                original_think = trace[selected_index].get('think', '')
                original_sq = trace[selected_index].get('search_query', None)
                original_docs = trace[selected_index].get('docs', [])
                
                if original_sq:
                    #! Step 1: Applying actions
                    if action == 'query_paraphrasing':
                        paraphrased_queries = self.search_query_paraphraser.inference(qid, original_sq, repeat=repeat) if original_sq else []
                        # retrieved_docs_list = [self.retrieval_perturber.inference(paraphrased_query) for paraphrased_query in paraphrased_queries]
                        retrieved_docs_list = [self.retriever.search(paraphrased_query) if paraphrased_query else [] for paraphrased_query in paraphrased_queries]
                    elif action == 'adding_critical_thought':
                        critical_thinks, critical_search_queries = self.critical_think_generator.inference(qid, original_sq, repeat=repeat) if original_sq else []
                        retrieved_docs_list = [self.retriever.search(critical_query) if critical_query else [] for critical_query in critical_search_queries]
                    elif action == 'answer_validation':
                        summarization_list = self.answer_validator.summarization_inference(qid, question, trace, repeat=repeat)
                        validation_thinks, search_queries = self.answer_validator.validation_inference(qid, question, prediction, summarization_list, repeat=repeat)
                        retrieved_docs_list = [self.retriever.search(search_query) if search_query else [] for search_query in search_queries]

                    #! Step 2: Generating new masked traces
                    for i in range(repeat):
                        new_trace = []
                        
                        # - A) Before the break point: Keep steps excluding the selected one
                        # ...
                        # - B) On the break point
                        if action == 'query_paraphrasing':
                            new_trace = copy.deepcopy(trace[:selected_index])
                            paraphrased_query = paraphrased_queries[i].strip()
                            retrieved_docs = retrieved_docs_list[i]
                            new_trace.append({"think": original_think, "search_query": paraphrased_query, "docs": retrieved_docs})
                        
                        elif action == 'adding_critical_thought':
                            new_trace = copy.deepcopy(trace[:selected_index])
                            critical_think = critical_thinks[i].strip()
                            critical_query = critical_search_queries[i].strip()
                            critical_docs = retrieved_docs_list[i]
                            new_trace.append({"think": original_think, "search_query": original_sq, "docs": original_docs})
                            new_trace.append({"think": critical_think, "search_query": critical_query, "docs": critical_docs})
                        
                        elif action == 'answer_validation':
                            new_trace = []
                            summarization = summarization_list[i]
                            validation_think = validation_thinks[i].strip()
                            search_query = search_queries[i].strip()
                            docs = retrieved_docs_list[i]
                            new_trace.append({"think": '', "search_query": '', "docs": [{'id':'000', 'contents': f"\n{summarization}"}]})
                            new_trace.append({"think": validation_think, "search_query": search_query, "docs": docs})
                        
                        # After break point: ask searchR1 to generate
                        pred_answer, rest_of_trace = self.rag_model.partial_inference_rag_consistency(question, new_trace)
                        
                        new_trace.extend(rest_of_trace)
                        masked_traces.append(new_trace)
                        answer_output_list.append(pred_answer.strip() if pred_answer else '')

        else:
            # ==============================
            # -- DO reasoning-consistency --
            # ==============================        
            #! 1) Create partial think
            if self.args.n_generations == 1:
                interval = 0
            else:
                assert self.args.n_generations > 1
                assert self.args.mask_right_boundary >= self.args.mask_left_boundary, f"right_boundary: {self.args.mask_right_boundary} < left_boundary: {self.args.mask_left_boundary}"
                interval = (self.args.mask_right_boundary - self.args.mask_left_boundary) / (self.args.n_generations - 1)
            
            last_think = trace[-1].get('think', '')
            if last_think:
                words_in_last_think = last_think.split(" ")
                mask_len = len(words_in_last_think)
            
                masked_last_thinks = []
                for i in range(self.args.n_generations):
                    prefix_part_ratio = self.args.mask_left_boundary + i * interval
                    prefix_part_num_words = math.ceil(mask_len * prefix_part_ratio) + 1
                    prefix_part_str = " ".join(words_in_last_think[:prefix_part_num_words])
                    masked_last_thinks.append(prefix_part_str)
            else:
                masked_last_thinks = [' ']*self.args.n_generations
            
            masked_traces_ = []
            for masked_last_think in masked_last_thinks:
                new_trace = copy.deepcopy(trace)
                new_trace[-1]['think'] = masked_last_think
                masked_traces_.append(new_trace)
        
            #! 2) Generate rest
            answer_output_list, masked_traces = [], []
            for partial_trace in masked_traces_: 
                last_think_first_part = partial_trace[-1].get('think', '')
                input_prompt_text = self.rag_model.get_input_prompt_reasoning_consistency(question, partial_trace)
                last_think_second_part, final_ans = self.rag_model.partial_inference_reasoning_consistency(input_prompt_text)
                
                new_trace = copy.deepcopy(trace)
                new_trace[-1]['think'] = f"{last_think_first_part.strip()} {last_think_second_part.strip()}".strip()
                new_trace[-1]['answer'] = final_ans
                masked_traces.append(new_trace)
                answer_output_list.append(final_ans)
        
        
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


