
from copy import deepcopy
from typing import List, Dict, Tuple

from src_mcts.MCTS_backbone import MCTS_Node
from src_mcts.generate_node import Generator
from utils.mcts_utils import (
    Node_Type,
    reach_terminal_subquestion
)

class Reasoning_MCTS_Node(MCTS_Node):
    def __init__(
        self,
        parent: "Reasoning_MCTS_Node",
        depth: int,
        node_type: Node_Type,
        verbose: bool = False,
        
        # --- For instantiating root node ---
        question_id: str = None,
        user_question: str = None,
        gt_answer: List[str] = None,
        generator: Generator = None,
        node_value: float = None,
        max_depth_allowed: int = None,
        
        # --- My Actions ---------------------
        rephrased_query: str = None,
        direct_answer: str = None,
        rag_answer: str = None,
        retrieved_documents: List[str] = None,
        subquestions: List[str] = None,
        
        subquestion: str = None,
        subq_retrieved_documents: List[str] = None,
        subanswer: str = None,
        subquestion_pointer: int = None,
        len_subqs: int = None,
        
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
            if node_value is not None:
                assert node_value > 0, breakpoint()
                
            if node_type is Node_Type.USER_QUESTION:
                assert depth == 0
                assert all(
                    attr is None
                    for attr in [
                        parent,
                        node_value,
                        rephrased_query,
                        direct_answer,
                        rag_answer,
                        retrieved_documents,
                        subquestions,
                        subquestion,
                        subanswer,
                        subq_retrieved_documents,
                        subquestion_pointer,
                        len_subqs
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        generator,
                        question_id,
                        user_question,
                        gt_answer,
                        max_depth_allowed,
                    ]
                )
                
            elif node_type is Node_Type.REPHRASED_QUERY:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        question_id,
                        user_question,
                        gt_answer,
                        max_depth_allowed,
                        node_value,
                        direct_answer,
                        rag_answer,
                        retrieved_documents,
                        subquestions,
                        subquestion,
                        subanswer,
                        subq_retrieved_documents,
                        subquestion_pointer,
                        len_subqs
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        rephrased_query,
                    ]
                )

            elif node_type is Node_Type.DIRECT_ANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        question_id,
                        user_question,
                        gt_answer,
                        max_depth_allowed,
                        rephrased_query,
                        rag_answer,
                        retrieved_documents,
                        subquestions,
                        subquestion,
                        subanswer,
                        subq_retrieved_documents,
                        subquestion_pointer,
                        len_subqs
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        node_value,
                        direct_answer
                    ]
                )
                
            elif node_type is Node_Type.RAG_ANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        question_id,
                        user_question,
                        gt_answer,
                        max_depth_allowed,
                        rephrased_query,
                        direct_answer,
                        subquestions,
                        subquestion,
                        subanswer,
                        subq_retrieved_documents,
                        subquestion_pointer,
                        len_subqs
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        node_value,
                        rag_answer,
                        retrieved_documents,
                    ]
                )

            elif node_type is Node_Type.SUBQUESTIONS:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        question_id,
                        user_question,
                        gt_answer,
                        max_depth_allowed,
                        node_value,
                        rephrased_query,
                        direct_answer,
                        retrieved_documents,
                        subquestion,
                        subanswer,
                        subq_retrieved_documents,
                        subquestion_pointer,
                        len_subqs
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        subquestions,   
                    ]
                )
            
            elif node_type is Node_Type.SUBQ_DIRECT_ANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        question_id,
                        user_question,
                        gt_answer,
                        max_depth_allowed,
                        node_value,
                        rephrased_query,
                        direct_answer,
                        retrieved_documents,
                        subq_retrieved_documents,
                        subquestions,
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        subquestion,
                        subanswer,
                        subquestion_pointer,
                        len_subqs
                    ]
                )
                
            elif node_type is Node_Type.SUBQ_RAG_ANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        question_id,
                        user_question,
                        gt_answer,
                        max_depth_allowed,
                        node_value,
                        rephrased_query,
                        direct_answer,
                        retrieved_documents,
                        subquestions
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        subquestion,
                        subanswer,
                        subq_retrieved_documents,
                        subquestion_pointer,
                        len_subqs
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
        self.node_value = node_value
        self.rephrased_query = rephrased_query
        self.direct_answer = direct_answer
        self.retrieved_documents = retrieved_documents
        self.rag_answer = rag_answer
        self.subquestions = subquestions
        self.subanswer = subanswer
        self.subquestion = subquestion
        self.subq_retrieved_documents = subq_retrieved_documents
        self.subquestion_pointer = subquestion_pointer
        self.len_subqs = len_subqs
        
        if parent is None:  # root
            self.verbose = verbose
            self.question_id = question_id
            self.user_question = user_question
            self.gt_answer = gt_answer
            self.generator = generator
            self.max_depth_allowed = max_depth_allowed
            self.enable_potential_score = enable_potential_score
        else:  # inherit from parent
            self.verbose = parent.verbose
            self.question_id = parent.question_id
            self.user_question = parent.user_question
            self.gt_answer = parent.gt_answer
            self.generator = parent.generator
            self.max_depth_allowed = parent.max_depth_allowed
            self.enable_potential_score = parent.enable_potential_score

        #! keep track of paraphrasing
        #! record number of subquestions till now

        #! record solution trace from root to the current node. key: subquestion id
        if parent is None:  # root
            assert self.node_type is Node_Type.USER_QUESTION
            self.solution_trace: Dict[int, Dict[str, str]] = {
                0: {"user_question": user_question, "qid": question_id, "ground_truth": gt_answer}
            }
        else:
            assert self.node_type is not Node_Type.USER_QUESTION
            self.solution_trace = deepcopy(parent.solution_trace)
            
            if node_type is Node_Type.DIRECT_ANSWER:
                self.solution_trace[max(self.solution_trace.keys())+1] = {
                    "direct_answer": {"text": direct_answer, "value": node_value}
                }
                
            elif node_type is Node_Type.RAG_ANSWER:
                self.solution_trace[max(self.solution_trace.keys())+1]= {
                    "rag_answer": {"text": rag_answer, "documents": retrieved_documents, "value": node_value}
                }
            
            elif node_type is Node_Type.REPHRASED_QUERY:
                self.solution_trace[max(self.solution_trace.keys())+1] = {
                    "rephrased_query": rephrased_query,
                }
                
            elif node_type is Node_Type.SUBQUESTIONS:
                self.solution_trace[max(self.solution_trace.keys())+1] = {
                    "subquestions": subquestions,
                }
            
            elif node_type is Node_Type.SUBQ_DIRECT_ANSWER:
                self.solution_trace[max(self.solution_trace.keys())+1] = {
                    "subq_direct_answer": {
                        "subq_pointer": subquestion_pointer,
                        "subquestion": subquestion,
                        "subanswer": subanswer,
                    }
                }
            
            elif node_type is Node_Type.SUBQ_RAG_ANSWER:
                self.solution_trace[max(self.solution_trace.keys())+1] = {
                    "subq_rag_answer": {
                        "subq_pointer": subquestion_pointer,
                        "subquestion": subquestion,
                        "documents": subq_retrieved_documents,
                        "subanswer": subanswer,
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
            Node_Type.REPHRASED_QUERY: "RQ",
            Node_Type.DIRECT_ANSWER: "DA",
            Node_Type.RAG_ANSWER: "RA",
            Node_Type.SUBQUESTIONS: "SQ",
            Node_Type.SUBQ_DIRECT_ANSWER: "SQDA",
            Node_Type.SUBQ_RAG_ANSWER: "SQRA"
        }
        return f"{type2str[self.node_type]}-{self.id}"

    def _create_children(self):
        #! Action Functions
        def do_action_generate_rephrased_question():
            print(f"---- Generating rephrased question for node {self.id}...")
            if self.node_type is Node_Type.USER_QUESTION:
                query = self.user_question
            elif self.node_type is Node_Type.SUBQUESTIONS:
                query = self.subquestion
            rephrased_query = self.generator.generate_rephrased_question(solution_trace=self.solution_trace)
            self.children.append(
                Reasoning_MCTS_Node(
                    parent=self,
                    depth=self.depth + 1,
                    node_type=Node_Type.REPHRASED_QUERY,
                    rephrased_query=rephrased_query
                )
            )
        
        def do_action_generate_direct_answer():
            print(f"---- Generating direct answer for node {self.id}...")
            direct_answer, value = self.generator.generate_direct_answer(solution_trace=self.solution_trace)
            self.children.append(
                Reasoning_MCTS_Node(
                    parent=self,
                    depth=self.depth + 1,
                    node_type=Node_Type.DIRECT_ANSWER,
                    direct_answer=direct_answer,
                    node_value=value,
                )
            )
        
        def do_action_generate_rag_answer():
            print(f"---- Retrieving documents for node {self.id}...")
            docs, rag_answer, value = self.generator.generate_rag_answer(solution_trace=self.solution_trace)
            self.children.append(
                Reasoning_MCTS_Node(
                    parent=self,
                    depth=self.depth + 1,
                    node_type=Node_Type.RAG_ANSWER,
                    retrieved_documents=docs,
                    rag_answer=rag_answer,
                    node_value=value
                )
            )

        def do_action_generate_subquestions():
            print(f"---- Generating subquestions for node {self.id}...")
            if self.node_type is Node_Type.USER_QUESTION:
                query = self.user_question
            elif self.node_type is Node_Type.SUBQUESTIONS:
                query = self.subquestion
            elif self.node_type is Node_Type.REPHRASED_QUERY:
                query = self.rephrased_query
            
            subquestions = self.generator.generate_query_decomposition(solution_trace=self.solution_trace)
            self.children.append(
                Reasoning_MCTS_Node(
                    parent=self,
                    depth=self.depth + 1,
                    node_type=Node_Type.SUBQUESTIONS,
                    subquestions=subquestions,
                )
            )
        
        def do_action_generate_subq_direct_answer():
            print(f"---- Generating subq direct answer for node {self.id}...")
            subquestion, subanswer, subquestion_pointer, len_subqs = self.generator.generate_subq_direct_answer(solution_trace=self.solution_trace)
            self.children.append(
                Reasoning_MCTS_Node(
                    parent=self,
                    depth=self.depth + 1,
                    node_type=Node_Type.SUBQ_DIRECT_ANSWER,
                    subquestion=subquestion,
                    subanswer=subanswer,
                    subquestion_pointer=subquestion_pointer,
                    len_subqs=len_subqs
                )
            )
        
        def do_action_generate_subq_rag_answer():
            print(f"---- Generating subq rag answer for node {self.id}...")
            subq_retrieved_documents, subquestion, subanswer, subquestion_pointer, len_subqs = self.generator.generate_subq_rag_answer(solution_trace=self.solution_trace)
            self.children.append(
                Reasoning_MCTS_Node(
                    parent=self,
                    depth=self.depth + 1,
                    node_type=Node_Type.SUBQ_RAG_ANSWER,
                    subq_retrieved_documents=subq_retrieved_documents,
                    subquestion=subquestion,
                    subanswer=subanswer,
                    subquestion_pointer=subquestion_pointer,
                    len_subqs=len_subqs
                )
            )
        
        #! create children
        if self.node_type is Node_Type.USER_QUESTION:
            do_action_generate_direct_answer() # A1
            do_action_generate_rag_answer()  # A2
            do_action_generate_subquestions() # A3
            do_action_generate_rephrased_question() # A4

        elif self.node_type is Node_Type.REPHRASED_QUERY:
            do_action_generate_direct_answer() # A1
            do_action_generate_rag_answer()  # A2
            do_action_generate_subquestions() # A3

        elif self.node_type is Node_Type.DIRECT_ANSWER:
            raise ValueError("DIRECT_ANSWER node cannot create children!!")

        elif self.node_type is Node_Type.RAG_ANSWER:
            raise ValueError("RAG_ANSWER node cannot create children!!")

        elif self.node_type is Node_Type.SUBQUESTIONS:
            do_action_generate_subq_direct_answer() # A5
            do_action_generate_subq_rag_answer() # A6
        
        elif self.node_type is Node_Type.SUBQ_DIRECT_ANSWER:
            if self.subquestion_pointer == self.len_subqs:
                do_action_generate_direct_answer() # A1
            else:
                do_action_generate_subq_direct_answer() # A5
                do_action_generate_subq_rag_answer() # A6
                
        elif self.node_type is Node_Type.SUBQ_RAG_ANSWER:
            if self.subquestion_pointer == self.len_subqs:
                do_action_generate_direct_answer() # A1
            else:
                do_action_generate_subq_direct_answer() # A4
                do_action_generate_subq_rag_answer() # A5
        
        assert self.children
        return self.children

    def is_valid_leaf_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
        return self.node_type is Node_Type.RAG_ANSWER or self.node_type is Node_Type.DIRECT_ANSWER

    def is_valid_solution_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
        return self.node_type is Node_Type.RAG_ANSWER or self.node_type is Node_Type.DIRECT_ANSWER

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
            assert self.node_value is not None, breakpoint()
            return self.node_value
        else:
            return 0

    def skip_backprop(self):
        return self.node_type is Node_Type.USER_QUESTION

