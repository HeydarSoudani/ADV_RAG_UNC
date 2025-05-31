from copy import deepcopy
from typing import List, Dict

from run_mcts.src.MCTS_backbone import MCTS_Node
from run_mcts.src.generate_node import Generator
from utils.mcts_utils import Node_Type, reach_terminal_subquestion


class Reasoning_MCTS_Node(MCTS_Node):
    def __init__(
        self,
        parent: "Reasoning_MCTS_Node",
        depth: int,
        node_type: Node_Type,
        verbose: bool = False,
        enable_critique: bool = False,
        enable_doc_generation: bool = False,
        
        # --- For instantiating root node ---
        question_id: str = None,
        user_question: str = None,
        gt_answer: List[str] = None,
        gt_reasoning_steps: List[str] = None,
        answer_candidates: List[str] = None,
        generator: Generator = None,
        node_reward: float = None,
        scores = None,
        max_depth_allowed: int = None,
        
        # --- My Actions ---------------------
        think: str = None,
        critique: str = None,
        search_query: str = None,
        retrieved_documents: List[str] = None,
        generated_documents: List[str] = None,
        answer: str = None,
        
        # --- For node selection (not in sanity checks yet) ---
        enable_potential_score: bool = None,
        potential_answers: List[str] = None,
        
        
    ) -> None:
        """params:
        subquestion: the node is proposing a new subquestion
        subanswer: the answer corresponding to the new subquestion the node proposed
        re_subanswer: the node is proposing a new subanswer to the parent's subquestion
        """
        super().__init__()
        
        #! sanity checks
        try:
            assert depth is not None
            assert node_type is not None
            if node_reward is not None:
                assert node_reward > 0, breakpoint()
        
            if node_type is Node_Type.USER_QUESTION:
                assert depth == 0
                assert all(
                    attr is None
                    for attr in [
                        parent,
                        node_reward,
                        scores,
                        think,
                        critique,
                        search_query,
                        retrieved_documents,
                        generated_documents,
                        answer,
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        generator,
                        question_id,
                        user_question,
                        gt_answer,
                        gt_reasoning_steps,
                        answer_candidates,
                        max_depth_allowed,
                    ]
                )
            elif node_type is Node_Type.THINK_GENERATE:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        question_id,
                        user_question,
                        gt_answer,
                        gt_reasoning_steps,
                        answer_candidates,
                        max_depth_allowed,
                        node_reward,
                        scores,
                        answer,   
                        critique,
                        retrieved_documents,
                        
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        think,
                        search_query,
                        generated_documents
                    ]
                )
            elif node_type is Node_Type.THINK_SERACH:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        question_id,
                        user_question,
                        gt_answer,
                        gt_reasoning_steps,
                        answer_candidates,
                        max_depth_allowed,
                        node_reward,
                        scores,
                        answer,   
                        critique,
                        generated_documents
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        think,
                        search_query,
                        retrieved_documents,
                    ]
                )
            
            elif node_type is Node_Type.THINK_ANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        question_id,
                        user_question,
                        gt_answer,
                        gt_reasoning_steps,
                        answer_candidates,
                        max_depth_allowed,
                        search_query,
                        retrieved_documents,
                        critique,
                        generated_documents
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        node_reward,
                        scores,
                        think,
                        answer,
                    ]
                )

            elif node_type is Node_Type.CRITIQUE_SEARCH:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        question_id,
                        user_question,
                        gt_answer,
                        gt_reasoning_steps,
                        answer_candidates,
                        max_depth_allowed,
                        node_reward,
                        scores,
                        answer,  
                        think, 
                        generated_documents
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        critique,
                        search_query,
                        retrieved_documents,
                    ]
                )
            
            elif node_type is Node_Type.CRITIQUE_ANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        question_id,
                        user_question,
                        gt_answer,
                        gt_reasoning_steps,
                        answer_candidates,
                        max_depth_allowed,
                        search_query,
                        retrieved_documents,
                        think,
                        generated_documents
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        node_reward,
                        scores,
                        critique,
                        answer,
                    ]
                )
        
        except AssertionError:
            print(f"Instantiating node with type {node_type} failed!")
            breakpoint()
            exit()
        
        #! attributes
        self.parent = parent  # if parent is None, then the node is the root
        self.children: List["Reasoning_MCTS_Node"] = []
        self.depth = depth
        self.node_type = node_type
        self.node_reward = node_reward
        self.scores = scores
        self.think = think
        self.critique = critique
        self.search_query = search_query
        self.retrieved_documents = retrieved_documents
        self.generated_documents = generated_documents
        self.answer = answer
        
        if parent is None:  # root
            self.verbose = verbose
            self.enable_critique = enable_critique
            self.enable_doc_generation = enable_doc_generation
            self.question_id = question_id
            self.user_question = user_question
            self.gt_answer = gt_answer
            self.gt_reasoning_steps = gt_reasoning_steps
            self.answer_candidates = answer_candidates
            self.generator = generator
            self.max_depth_allowed = max_depth_allowed
            self.enable_potential_score = enable_potential_score
        else:  # inherit from parent
            self.verbose = parent.verbose
            self.enable_critique = parent.enable_critique
            self.enable_doc_generation = parent.enable_doc_generation
            self.question_id = parent.question_id
            self.user_question = parent.user_question
            self.gt_answer = parent.gt_answer
            self.gt_reasoning_steps = parent.gt_reasoning_steps
            self.answer_candidates = parent.answer_candidates
            self.generator = parent.generator
            self.max_depth_allowed = parent.max_depth_allowed
            self.enable_potential_score = parent.enable_potential_score
        
        #! record solution trace from root to the current node. key: subquestion id
        if parent is None:  # root
            assert self.node_type is Node_Type.USER_QUESTION
            self.solution_trace: Dict[int, Dict[str, str]] = {
                0: {
                    "user_question": user_question,
                    "qid": question_id,
                    "ground_truth": gt_answer,
                    "answer_candidates": answer_candidates,
                    "reasoning_steps": gt_reasoning_steps
                }
            }
        else:
            assert self.node_type is not Node_Type.USER_QUESTION
            self.solution_trace = deepcopy(parent.solution_trace)
            
            if node_type is Node_Type.THINK_GENERATE:
                self.solution_trace[max(self.solution_trace.keys())+1] = {
                    "think_generate": {
                        "think": think,
                        "search_query": search_query,
                        "generated_documents": generated_documents
                    }
                }
            
            elif node_type is Node_Type.THINK_SERACH:
                self.solution_trace[max(self.solution_trace.keys())+1] = {
                    "think_search": {
                        "think": think,
                        "search_query": search_query,
                        "retrieved_documents": retrieved_documents
                    }
                }
            elif node_type is Node_Type.THINK_ANSWER:
                self.solution_trace[max(self.solution_trace.keys())+1] = {
                    "think_answer": {
                        "think": think,
                        "answer": answer,
                        "node_reward": node_reward,
                        "scores": scores,
                    }
                }
            elif node_type is Node_Type.CRITIQUE_SEARCH:
                self.solution_trace[max(self.solution_trace.keys())+1] = {
                    "critique_search": {
                        "critique": critique,
                        "search_query": search_query,
                        "retrieved_documents": retrieved_documents
                    }
                }
            elif node_type is Node_Type.CRITIQUE_ANSWER:
                self.solution_trace[max(self.solution_trace.keys())+1] = {
                    "critique_answer": {
                        "critique": critique,
                        "answer": answer,
                        "node_reward": node_reward,
                    }
                }
        
        #! potential_score for intermediate nodes (only used for node selection)
        if self.enable_potential_score:
            self.potential_answers = potential_answers
            self.potential_score = 0
            if parent is None:  # root
                assert self.node_type is Node_Type.USER_QUESTION
                self.potential_answers_history = {}
            else:
                assert self.node_type is not Node_Type.USER_QUESTION
                self.potential_answers_history = deepcopy(parent.potential_answers_history)
                self.potential_answers_history[self.depth] = potential_answers
    
    def __str__(self) -> str:
        type2str = {
            Node_Type.USER_QUESTION: "UQ",
            Node_Type.THINK_GENERATE: "TG",
            Node_Type.THINK_SERACH: "TS",
            Node_Type.THINK_ANSWER: "TA",
            Node_Type.CRITIQUE_SEARCH: "CS",
            Node_Type.CRITIQUE_ANSWER: "CA",
        }
        return f"{type2str[self.node_type]}-{self.id}"
    
    def _create_children(self):
        #! Action Functions
        def do_action_think_search():
            print(f"---- Generating think search for node {self.id}...")
            think, search_query, retrieved_docs = self.generator.generate_think_search(solution_trace=self.solution_trace)
            self.children.append(
                Reasoning_MCTS_Node(
                    parent=self,
                    depth=self.depth + 1,
                    node_type=Node_Type.THINK_SERACH,
                    think=think,
                    search_query=search_query,
                    retrieved_documents=retrieved_docs
                )
            )
        
        def do_action_think_answer():
            print(f"---- Generating think answer for node {self.id}...")
            think, answer, node_reward, scores = self.generator.generate_think_answer(solution_trace=self.solution_trace)
            self.children.append(
                Reasoning_MCTS_Node(
                    parent=self,
                    depth=self.depth + 1,
                    node_type=Node_Type.THINK_ANSWER,
                    think=think,
                    answer=answer,
                    node_reward=node_reward,
                    scores=scores
                )
            )
    
        def do_action_think_generate():
            print(f"---- Generating think generate for node {self.id}...")
            think, search_query, generated_docs = self.generator.generate_think_generate(solution_trace=self.solution_trace)
            self.children.append(
                Reasoning_MCTS_Node(
                    parent=self,
                    depth=self.depth + 1,
                    node_type=Node_Type.THINK_GENERATE,
                    think=think,
                    search_query=search_query,
                    generated_documents=generated_docs
                )
            )
    
    
        def do_action_critique_search():
            print(f"---- Generating critique search for node {self.id}...")
            critique, search_query, retrieved_docs = self.generator.generate_critique_search(solution_trace=self.solution_trace)
            self.children.append(
                Reasoning_MCTS_Node(
                    parent=self,
                    depth=self.depth + 1,
                    node_type=Node_Type.CRITIQUE_SEARCH,
                    critique=critique,
                    search_query=search_query,
                    retrieved_documents=retrieved_docs
                )
            )
        
        def do_action_critique_answer():
            print(f"---- Generating critique answer for node {self.id}...")
            critique, answer, value = self.generator.generate_critique_answer(solution_trace=self.solution_trace)
            self.children.append(
                Reasoning_MCTS_Node(
                    parent=self,
                    depth=self.depth + 1,
                    node_type=Node_Type.CRITIQUE_ANSWER,
                    critique=critique,
                    answer=answer,
                    sc_value=value
                )
            )
    
        #! create children
        if self.node_type is Node_Type.USER_QUESTION:
            if self.enable_doc_generation:
                do_action_think_generate()
            do_action_think_search()
            do_action_think_answer()
        elif self.node_type is Node_Type.THINK_SERACH:
            if self.enable_doc_generation:
                do_action_think_generate()
            do_action_think_search()
            do_action_think_answer()
            if self.enable_critique:
                do_action_critique_search()
                do_action_critique_answer()
        
        elif self.node_type is Node_Type.THINK_GENERATE:
            do_action_think_generate()
            do_action_think_search()
            do_action_think_answer()
        elif self.node_type is Node_Type.CRITIQUE_SEARCH:
            do_action_think_search()
            do_action_think_answer()
            do_action_critique_answer()
            
        elif self.node_type is Node_Type.CRITIQUE_ANSWER:
            raise ValueError("CRITIQUE_ANSWER node cannot create children!!")  
        elif self.node_type is Node_Type.THINK_ANSWER:
            raise ValueError("THINK_ANSWER node cannot create children!!")    
        
        assert self.children
        return self.children
        
    def is_valid_leaf_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
        return self.node_type is Node_Type.THINK_ANSWER or self.node_type is Node_Type.CRITIQUE_ANSWER

    def is_valid_solution_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
        return self.node_type is Node_Type.THINK_ANSWER or self.node_type is Node_Type.CRITIQUE_ANSWER

    def set_potential_score(self, score: float):
        self.potential_score = score

    def find_children(self, rollout_id: int):
        self.children = self.children or self._create_children()
        for child in self.children:
            child.set_rollout_id(rollout_id)
        assert self.children
        return self.children

    def is_terminal(self):
        return self.depth >= self.max_depth_allowed or self.is_valid_leaf_node()

    def calculate_reward(self):
        if self.is_valid_leaf_node():
            assert self.node_reward is not None, breakpoint()
            return self.node_reward
        else:
            return 0

    def skip_backprop(self):
        return self.node_type is Node_Type.USER_QUESTION

