"""LangGraph workflow definitions for EduAgent.

Two compiled graphs are exposed:

* :func:`build_tutoring_graph` — the main chat-turn graph:
  ``planner → (teach | quiz | rag → teach) → memory → recommendation``
* :func:`build_evaluation_graph` — the quiz-submission graph:
  ``evaluation → memory → recommendation``

Design rationale
----------------
The tutoring graph's conditional edge lives at ``planner``: the planner outputs
a deterministic ``next_agent`` value (computed from the classified intent), and
the graph routes accordingly. This keeps routing out of the LLM's hands while
allowing the model to still decide the *topic* and *goal*.

When ``with_quiz`` is ``True`` on a teach turn, the graph chains
``teacher → quiz`` before the memory/recommendation tail — this lets a single
API call deliver a lesson followed by a practice quiz.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.context import AgentContext
from agents.evaluation import EvaluationAgent
from agents.memory import MemoryAgent
from agents.planner import PlannerAgent
from agents.quiz import QuizAgent
from agents.rag import RAGAgent
from agents.recommendation import RecommendationAgent
from agents.teacher import TeacherAgent
from config.constants import Routes
from models.state import EduAgentState

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------- #
# Tutoring graph (chat turn)
# -------------------------------------------------------------------------- #
def build_tutoring_graph(
    ctx: AgentContext, *, checkpointer=None, interrupt_before=None
) -> CompiledStateGraph:
    """Build and compile the main tutoring workflow graph."""
    planner = PlannerAgent(ctx)
    rag = RAGAgent(ctx)
    teacher = TeacherAgent(ctx)
    quiz = QuizAgent(ctx)
    memory = MemoryAgent(ctx)
    recommendation = RecommendationAgent(ctx)

    graph = StateGraph(EduAgentState)

    # --- nodes ---
    for agent in (planner, rag, teacher, quiz, memory, recommendation):
        if ctx.harness:
            ctx.harness.bind(ctx)
        node = ctx.harness.wrap_node(agent.name, agent.run) if ctx.harness else agent.run
        graph.add_node(agent.name, node)

    # --- start → planner ---
    graph.add_edge(START, planner.name)

    # --- planner → conditional routing ---
    graph.add_conditional_edges(
        planner.name,
        lambda state: state.get("next_agent", Routes.END),
        {
            Routes.TEACH: teacher.name,
            Routes.ANSWER_QUESTION: rag.name,
            Routes.QUIZ: quiz.name,
            Routes.END: END,
        },
    )

    # --- rag → teacher (answer grounded in retrieved notes) ---
    graph.add_edge(rag.name, teacher.name)

    # --- teacher → optional quiz, else memory ---
    graph.add_conditional_edges(
        teacher.name,
        lambda state: quiz.name if state.get("with_quiz") else memory.name,
        {
            quiz.name: quiz.name,
            memory.name: memory.name,
        },
    )

    # --- quiz / memory / recommendation tail ---
    graph.add_edge(quiz.name, memory.name)
    graph.add_edge(memory.name, recommendation.name)
    graph.add_edge(recommendation.name, END)

    logger.debug("Tutoring graph compiled")
    return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


# -------------------------------------------------------------------------- #
# Evaluation graph (quiz submission)
# -------------------------------------------------------------------------- #
def build_evaluation_graph(
    ctx: AgentContext, *, checkpointer=None, interrupt_before=None
) -> CompiledStateGraph:
    """Build and compile the quiz-evaluation workflow graph."""
    evaluation = EvaluationAgent(ctx)
    memory = MemoryAgent(ctx)
    recommendation = RecommendationAgent(ctx)

    graph = StateGraph(EduAgentState)

    for agent in (evaluation, memory, recommendation):
        if ctx.harness:
            ctx.harness.bind(ctx)
        node = ctx.harness.wrap_node(agent.name, agent.run) if ctx.harness else agent.run
        graph.add_node(agent.name, node)

    graph.add_edge(START, evaluation.name)
    graph.add_edge(evaluation.name, memory.name)
    graph.add_edge(memory.name, recommendation.name)
    graph.add_edge(recommendation.name, END)

    logger.debug("Evaluation graph compiled")
    return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
