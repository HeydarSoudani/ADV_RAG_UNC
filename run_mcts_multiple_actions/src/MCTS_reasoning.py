from copy import deepcopy
from typing import List, Dict

from run_mcts_multiple_actions.src.MCTS_backbone import MCTS_Node
from run_mcts_multiple_actions.src.generate_node import Generator
from utils.mcts_multi_actions_utils import Node_Type


class Reasoning_MCTS_Node(MCTS_Node):
    def __init__(
        self,
        parent: "Reasoning_MCTS_Node",
        depth: int,
        node_type: Node_Type,
        verbose: bool = False,
        
        # --- For instantiating root node ---
        qid: str = None,
        user_query: str = None,
        gt_answers: List[str] = None,
        gt_reasoning_steps: List[str] = None,
        generator: Generator = None,
        node_reward: float = None,
        scores = None,
        max_depth_allowed: int = None,

        # --- Actions -----------------------
        think: str = None,
        search_query: str = None,
        retrieved_documents: List[str] = None,
        answer: str = None,
        is_finished:bool = None,
        documents_analysis: str = None,
        critical_rethinking: str = None,
        answer_validation: str = None,
        
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
        
            if node_type is Node_Type.USER_QUERY:
                assert depth == 0
                assert all(
                    attr is None
                    for attr in [
                        parent,
                        node_reward,
                        scores,
                        think,
                        search_query,
                        retrieved_documents,
                        answer,
                        is_finished,
                        documents_analysis,
                        critical_rethinking,
                        answer_validation
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        generator,
                        qid,
                        user_query,
                        gt_answers,
                        gt_reasoning_steps,
                        max_depth_allowed,
                        enable_potential_score
                    ]
                )
        
            elif node_type is Node_Type.RETRIEVE_DOCUMENTS:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        qid,
                        user_query,
                        gt_answers,
                        gt_reasoning_steps,
                        max_depth_allowed,
                        node_reward,
                        scores,
                        answer,
                        is_finished,
                        documents_analysis,
                        critical_rethinking,
                        answer_validation
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
            
            elif node_type is Node_Type.DOCUMENTS_ANALYSIS:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        qid,
                        user_query,
                        gt_answers,
                        gt_reasoning_steps,
                        max_depth_allowed,
                        node_reward,
                        scores,
                        answer,
                        search_query,
                        retrieved_documents,
                        is_finished,
                        think,
                        critical_rethinking,
                        answer_validation
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        documents_analysis
                    ]
                )
                
            elif node_type is Node_Type.DOCUMENTS_RETHINKING:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        qid,
                        user_query,
                        gt_answers,
                        gt_reasoning_steps,
                        max_depth_allowed,
                        node_reward,
                        scores,
                        answer,
                        search_query,
                        retrieved_documents,
                        is_finished,
                        documents_analysis,
                        think,
                        answer_validation
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        critical_rethinking,
                    ]
                )
            
            elif node_type is Node_Type.ANSWER_GENERATION:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        qid,
                        user_query,
                        gt_answers,
                        gt_reasoning_steps,
                        max_depth_allowed,
                        node_reward,
                        scores,
                        search_query,
                        retrieved_documents,
                        is_finished,
                        documents_analysis,
                        critical_rethinking,
                        answer_validation
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        think,
                        answer,
                    ]
                )
            
            elif node_type is Node_Type.ANSWER_VALIDATION:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        qid,
                        user_query,
                        gt_answers,
                        gt_reasoning_steps,
                        max_depth_allowed,
                        node_reward,
                        scores,
                        search_query,
                        retrieved_documents,
                        answer,
                        is_finished,
                        documents_analysis,
                        critical_rethinking,
                        think,
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        answer_validation
                    ]
                )
            
            elif node_type is Node_Type.FINISH:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        qid,
                        user_query,
                        gt_answers,
                        gt_reasoning_steps,
                        max_depth_allowed,
                        node_reward,
                        scores,
                        search_query,
                        retrieved_documents,
                        answer,
                        think,
                        documents_analysis,
                        critical_rethinking,
                        answer_validation
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        is_finished
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
        # -
        self.think = think
        self.search_query = search_query
        self.retrieved_documents = retrieved_documents
        self.answer = answer
        self.is_finished = is_finished
        self.documents_analysis = documents_analysis
        self.critical_rethinking = critical_rethinking
        self.answer_validation = answer_validation

        if parent is None:  # root
            self.verbose = verbose
            self.qid = qid
            self.user_query = user_query
            self.gt_answers = gt_answers
            self.gt_reasoning_steps = gt_reasoning_steps
            self.generator = generator
            self.max_depth_allowed = max_depth_allowed
            self.enable_potential_score = enable_potential_score
        else:  # inherit from parent
            self.verbose = parent.verbose
            self.qid = parent.qid
            self.user_query = parent.user_query
            self.gt_answers = parent.gt_answers
            self.gt_reasoning_steps = parent.gt_reasoning_steps
            self.generator = parent.generator
            self.max_depth_allowed = parent.max_depth_allowed
            self.enable_potential_score = parent.enable_potential_score

        #! record solution trace from root to the current node. key: subquestion id
        if parent is None:  # root
            assert self.node_type is Node_Type.USER_QUERY
            self.solution_trace: Dict[int, Dict[str, str]] = {
                0: {
                    "user_query": user_query,
                    "qid": qid,
                    "ground_truth": gt_answers,
                    "reasoning_steps": gt_reasoning_steps
                }
            }
        
        else:
            assert self.node_type is not Node_Type.USER_QUERY
            self.solution_trace = deepcopy(parent.solution_trace)
            
            if node_type is Node_Type.RETRIEVE_DOCUMENTS:
                self.solution_trace[max(self.solution_trace.keys())+1] = {
                    "retrieve_documents": {
                        "think": think,
                        "search_query": search_query,
                        "docs": retrieved_documents
                    }
                }
            elif node_type is Node_Type.DOCUMENTS_ANALYSIS:
                self.solution_trace[max(self.solution_trace.keys())+1] = {
                    "documents_analysis": {
                        "documents_analysis": documents_analysis
                    }
                }
            elif node_type is Node_Type.DOCUMENTS_RETHINKING:
                self.solution_trace[max(self.solution_trace.keys())+1] = {
                    "documents_rethinking": {
                        "critical_rethinking": critical_rethinking
                    }
                }
            elif node_type is Node_Type.ANSWER_GENERATION:
                self.solution_trace[max(self.solution_trace.keys())+1] = {
                    "answer_generation": {
                        "think": think,
                        "answer": answer
                    }
                }
            elif node_type is Node_Type.ANSWER_VALIDATION:
                self.solution_trace[max(self.solution_trace.keys())+1] = {
                    "answer_validation": {
                        "answer_validation": answer_validation
                    }
                }
            elif node_type is Node_Type.FINISH:
                self.solution_trace[max(self.solution_trace.keys())+1] = {
                    "finish": {
                        "is_finished": is_finished
                    }
                }

        #! potential_score for intermediate nodes (only used for node selection)
        if self.enable_potential_score:
            self.potential_answers = potential_answers
            self.potential_score = 0
            if parent is None:  # root
                assert self.node_type is Node_Type.USER_QUERY
                self.potential_answers_history = {}
            else:
                assert self.node_type is not Node_Type.USER_QUERY
                self.potential_answers_history = deepcopy(parent.potential_answers_history)
                self.potential_answers_history[self.depth] = potential_answers

    def __str__(self) -> str:
        type2str = {
            Node_Type.USER_QUERY: "UQ",
            Node_Type.RETRIEVE_DOCUMENTS: "RD",
            Node_Type.DOCUMENTS_ANALYSIS: "DA",
            Node_Type.DOCUMENTS_RETHINKING: "DR",
            Node_Type.ANSWER_GENERATION: "AG",
            Node_Type.ANSWER_VALIDATION: "AV",
            Node_Type.FINISH: "FI"
        }
        return f"{type2str[self.node_type]}-{self.id}"


    def _create_children(self):
        #! action Functions
        def do_action_retrieve_documents(): #A1
            print(f"---- Retrieving documents for node {self.id}...")
            think, search_query, retrieved_docs = self.generator.retrieve_documents(solution_trace=self.solution_trace)
            self.children.append(
                Reasoning_MCTS_Node(
                    parent=self,
                    depth=self.depth + 1,
                    node_type=Node_Type.RETRIEVE_DOCUMENTS,
                    think=think,
                    search_query=search_query,
                    retrieved_documents=retrieved_docs
                )
            )
            
        def do_action_documents_analysis(): #A2
            print(f"---- Documents analysis for node {self.id}...")
            documents_analysis = self.generator.documents_analysis(solution_trace=self.solution_trace)
            self.children.append(
                Reasoning_MCTS_Node(
                    parent=self,
                    depth=self.depth + 1,
                    node_type=Node_Type.DOCUMENTS_ANALYSIS,
                    documents_analysis=documents_analysis
                )
            )

        def do_action_documents_rethinking(): #A3
            print(f"---- Documents rethinking for node {self.id}...")
            critical_rethinking = self.generator.documents_rethinking(solution_trace=self.solution_trace)
            self.children.append(
                Reasoning_MCTS_Node(
                    parent=self,
                    depth=self.depth + 1,
                    node_type=Node_Type.DOCUMENTS_RETHINKING,
                    critical_rethinking=critical_rethinking
                )
            )

        def do_action_answer_generation(): #A4
            print(f"---- Answer generation for node {self.id}...")
            think, answer = self.generator.answer_generation(solution_trace=self.solution_trace)
            self.children.append(
                Reasoning_MCTS_Node(
                    parent=self,
                    depth=self.depth + 1,
                    node_type=Node_Type.ANSWER_GENERATION,
                    think=think,
                    answer=answer
                )
            )

        def do_action_answer_validation(): #A5
            print(f"---- Answer validation for node {self.id}...")
            answer_validation = self.generator.answer_validation(solution_trace=self.solution_trace)
            self.children.append(
                Reasoning_MCTS_Node(
                    parent=self,
                    depth=self.depth + 1,
                    node_type=Node_Type.ANSWER_VALIDATION,
                    answer_validation=answer_validation
                )
            )

        def do_action_finish(): #A6
            print(f"---- Finish for node {self.id}...")
            is_finished = self.generator.finish(solution_trace=self.solution_trace)
            self.children.append(
                Reasoning_MCTS_Node(
                    parent=self,
                    depth=self.depth + 1,
                    node_type=Node_Type.FINISH,
                    is_finished=is_finished
                )
            )

        #! create children
        if self.node_type is Node_Type.USER_QUERY:
            do_action_retrieve_documents()
            do_action_answer_generation()
        
        elif self.node_type is Node_Type.RETRIEVE_DOCUMENTS:
            do_action_documents_analysis()
            do_action_documents_rethinking()
            do_action_answer_generation()
        
        elif self.node_type is Node_Type.DOCUMENTS_ANALYSIS:
            do_action_documents_rethinking()
            do_action_retrieve_documents()
            do_action_answer_generation()
        
        elif self.node_type is Node_Type.DOCUMENTS_RETHINKING:
            do_action_retrieve_documents()
            do_action_answer_generation()
        
        elif self.node_type is Node_Type.ANSWER_GENERATION:
            do_action_answer_validation()
            do_action_finish()
            do_action_retrieve_documents()
            
        elif self.node_type is Node_Type.ANSWER_VALIDATION:
            do_action_retrieve_documents()
            do_action_finish()

        elif self.node_type is Node_Type.FINISH:
            raise ValueError("FINISH node cannot create children!!")  

        assert self.children
        return self.children

    def is_valid_leaf_node(self):
        #! a valid solution can only be in FINISH
        return self.node_type is Node_Type.FINISH

    def is_valid_solution_node(self):
        #! a valid solution can only be in FINISH
        return self.node_type is Node_Type.FINISH

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
        return self.node_type is Node_Type.USER_QUERY