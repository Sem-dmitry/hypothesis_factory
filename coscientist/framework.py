# -*- coding: utf-8 -*-

"""
The overall framework that takes a CoscientistStateManager from global_state.py,
setups the agents, and organizes the multi-agent system. The framework will be controlled
by a supervisor agent.
"""

import logging
import math
import random
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from coscientist.common import RETRY_FAILED, retry_call
from coscientist.evolution_agent import build_evolution_agent
from coscientist.final_report_agent import build_final_report_agent
from coscientist.generation_agent import (
    CollaborativeConfig,
    IndependentConfig,
    build_generation_agent,
)
from coscientist.global_state import CoscientistStateManager
from coscientist.hypothesis_assessment import (
    AssessmentWeights,
    HypothesisAssessment,
    assess_hypothesis,
    rank_assessments,
)
from coscientist.literature_review_agent import build_literature_review_agent
from coscientist.meta_review_agent import build_meta_review_agent
from coscientist.model_factory import (
    cheaper_pool,
    get_embeddings,
    smarter_pool,
)
from coscientist.reasoning_types import ReasoningType
from coscientist.reflection_agent import build_deep_verification_agent
from coscientist.supervisor_agent import build_supervisor_agent
from coscientist.web_literature import extract_web_literature_sources

if TYPE_CHECKING:
    from coscientist.corpus.retrieval import CorpusRetriever

# A single generation call can return empty/So-only-thinking content (a transient
# provider behavior, esp. with reasoning models). Retry a few times before giving
# up on that one hypothesis — and never let one failure abort the whole run.
_GENERATION_MAX_ATTEMPTS = 3


def _collect_guidance(pull_messages: Optional[Callable[[], list[str]]]) -> list[str]:
    """
    Pull any pending live-steering messages from the provider, cleaned and
    non-empty. Never raises — a broken provider just yields no messages.
    """
    if pull_messages is None:
        return []
    try:
        msgs = pull_messages() or []
    except Exception:
        return []
    return [m.strip() for m in msgs if isinstance(m, str) and m.strip()]


def _hypothesis_cap_override(
    action: str,
    *,
    total: int,
    max_total: int,
    num_unranked: int,
) -> Optional[str]:
    """
    Enforce the hypothesis-pool cap deterministically. When the pool is full,
    a pool-growing action (generate/evolve) is redirected so the run converges:
    rank any stragglers first (run_tournament), otherwise finish. Returns the
    replacement action, or None to leave the decision unchanged.

    ``max_total <= 0`` means unlimited (never overrides).
    """
    if max_total <= 0 or total < max_total:
        return None
    if action not in ("generate_new_hypotheses", "evolve_hypotheses"):
        return None
    return "run_tournament" if num_unranked > 0 else "finish"


def _termination_reason(
    *,
    has_clear_leader: bool,
    is_plateau: bool,
    recent_actions: list[str],
    num_actions: int,
    budget: int,
    max_consecutive_evolve: int,
) -> Optional[str]:
    """
    Decide, deterministically, whether the run must finish now — without asking
    the LLM supervisor. Returns a short reason string, or None to keep going.

    Guards (any one triggers a forced finish):
      * budget spent — the existing hard cap that guarantees termination;
      * settled tournament — a clear leader that has plateaued (the intended
        stopping point, made deterministic instead of relying on the LLM);
      * runaway evolution — the last ``max_consecutive_evolve`` actions are all
        ``evolve_hypotheses`` (each blocks on the reflection queue, so the
        supervisor can otherwise loop on them indefinitely).
    """
    if num_actions >= budget:
        return f"Action budget of {budget} reached; finishing."
    if has_clear_leader and is_plateau:
        return "Clear tournament leader has plateaued; finishing."
    if (
        max_consecutive_evolve > 0
        and len(recent_actions) >= max_consecutive_evolve
        and all(a == "evolve_hypotheses" for a in recent_actions[-max_consecutive_evolve:])
    ):
        return (
            f"{max_consecutive_evolve} consecutive evolve_hypotheses actions; "
            "finishing to avoid a non-terminating improvement loop."
        )
    return None


class CoscientistConfig:
    """
    Configuration for the Coscientist system.

    Note that the config for GPTResearcher which is used throughout the system
    is defined in `researcher_config.json`.

    Attributes
    ----------
    literature_review_agent_llm : BaseChatModel
        The language model for the literature review. This LLM decides on the research
        subtopics for GPTResearcher.
    generation_agent_llms : dict[str, BaseChatModel]
        The language models for the generation agents
    reflection_agent_llms : dict[str, BaseChatModel]
        The language models for the reflection agents
    evolution_agent_llms : dict[str, BaseChatModel]
        The language models for the evolution agents
    meta_review_agent_llm : BaseChatModel
        The language model for the meta-review. Gemini works best because of the long
        context window that isn't severely rate limited like other providers.
    proximity_agent_embedding_model : Embeddings
        The embedding model for the proximity agent
    specialist_fields : list[str]
        The fields of expertise for generation agents. This list should be expanded
        by the configuration agent.

    """

    def __init__(
        self,
        literature_review_agent_llm: BaseChatModel | None = None,
        generation_agent_llms: dict[str, BaseChatModel] | None = None,
        reflection_agent_llms: dict[str, BaseChatModel] | None = None,
        evolution_agent_llms: dict[str, BaseChatModel] | None = None,
        meta_review_agent_llm: BaseChatModel | None = None,
        supervisor_agent_llm: BaseChatModel | None = None,
        final_report_agent_llm: BaseChatModel | None = None,
        proximity_agent_embedding_model: Embeddings | None = None,
        specialist_fields: list[str] | None = None,
        retriever: Optional["CorpusRetriever"] = None,
        assessment_llm: BaseChatModel | None = None,
        assessment_weights: AssessmentWeights | None = None,
        ranking_agent_llm: BaseChatModel | None = None,
        max_supervisor_actions: int = 40,
        max_consecutive_evolve: int = 3,
        max_total_hypotheses: int = 0,
        web_research_enabled: bool = True,
        pull_messages: Optional[Callable[[], list[str]]] = None,
        reflection_concurrency: int = 4,
        literature_subtopics: int = 5,
        max_assumptions_researched: int = 0,
    ):
        # Models are built lazily here (not at import time) so importing this
        # module never requires API keys or network. Every model is routed
        # through RouterAI / an API-based embedding via `model_factory`.
        # Passing an explicit model overrides the default for that role.
        # TODO: Add functionality for overriding GPTResearcher config.
        smarter = None  # built on demand to avoid redundant client construction

        def _smarter() -> dict[str, BaseChatModel]:
            nonlocal smarter
            if smarter is None:
                smarter = smarter_pool()
            return smarter

        self.literature_review_agent_llm = (
            literature_review_agent_llm
            or _smarter()["claude-sonnet-4-20250514"]
        )
        self.generation_agent_llms = generation_agent_llms or _smarter()
        self.reflection_agent_llms = reflection_agent_llms or _smarter()
        self.evolution_agent_llms = evolution_agent_llms or _smarter()
        self.meta_review_agent_llm = (
            meta_review_agent_llm or cheaper_pool()["gemini-2.5-flash"]
        )
        # The Elo tournament (Ranking agent). Defaults to the meta-review LLM to
        # preserve prior behaviour when not explicitly configured.
        self.ranking_agent_llm = ranking_agent_llm or self.meta_review_agent_llm
        # Soft cap on supervisor iterations — a safety net that guarantees the
        # run terminates (the finish decision itself is relative, not budget-based).
        self.max_supervisor_actions = max_supervisor_actions
        # Deterministic anti-runaway guard: force finish once this many
        # `evolve_hypotheses` actions run back-to-back (they block on the
        # reflection queue, so the LLM supervisor can otherwise loop on them).
        self.max_consecutive_evolve = max_consecutive_evolve
        # Hard cap on how many hypotheses the pool may accumulate (0 = unlimited).
        # Once reached, pool-growing actions (generate/evolve) are redirected so
        # the run converges instead of exploring forever.
        self.max_total_hypotheses = max_total_hypotheses
        # Explicit gate for GPTResearcher/web-backed literature and reflection.
        # When False, the run stays on local LLM + private-corpus retrieval only.
        self.web_research_enabled = web_research_enabled
        # How many hypotheses to deep-verify concurrently. Web-backed reflection
        # (GPTResearcher per assumption) is the run's bottleneck; processing the
        # reflection queue with a bounded thread pool overlaps those web waits.
        # 1 restores the previous strictly-sequential behaviour.
        self.reflection_concurrency = max(1, int(reflection_concurrency))
        # LITE-mode knobs: literature breadth (subtopics dispatched to web
        # research) and a cap on how many assumptions per hypothesis get
        # web-verified (0 = all). Both shrink the web-search fan-out.
        self.literature_subtopics = max(1, int(literature_subtopics))
        self.max_assumptions_researched = max(0, int(max_assumptions_researched))
        # Optional provider of live steering messages from the human expert.
        # Drained at each supervisor-loop boundary and folded into agent prompts
        # (chat-to-running-agents); None ⇒ no steering, unchanged behaviour.
        self.pull_messages = pull_messages
        self.supervisor_agent_llm = (
            supervisor_agent_llm or _smarter()["claude-sonnet-4-20250514"]
        )
        self.final_report_agent_llm = (
            final_report_agent_llm or _smarter()["claude-sonnet-4-20250514"]
        )
        self.proximity_agent_embedding_model = (
            proximity_agent_embedding_model or get_embeddings()
        )
        if specialist_fields is None:
            # Default specialist fields for the Nornickel ore-beneficiation /
            # metallurgy domain (was "biology" in the upstream biomedical demo).
            self.specialist_fields = [
                "mineral processing",
                "flotation",
                "extractive metallurgy",
            ]
        else:
            self.specialist_fields = specialist_fields

        # Private-corpus grounding + structured assessment (Phases 2-4). All
        # opt-in: when `retriever` is None the system behaves exactly as before.
        self.retriever = retriever
        # Falls back to the supervisor LLM at assessment time if not set.
        self.assessment_llm = assessment_llm
        self.assessment_weights = assessment_weights or AssessmentWeights()


class CoscientistFramework:
    """
    The framework that takes a CoscientistStateManager from global_state.py,
    setups the agents, and organizes the multi-agent system. The framework will be controlled
    by a supervisor agent.
    """

    def __init__(
        self, config: CoscientistConfig, state_manager: CoscientistStateManager
    ):
        self.config = config
        self.state_manager = state_manager

    def list_generation_llm_names(self) -> list[str]:
        """
        List the names of the generation agents.
        """
        return list(self.config.generation_agent_llms.keys())

    def list_generation_modes(self) -> list[str]:
        """
        List the names of the generation modes.
        """
        return ["independent", "collaborative"]

    def list_reflection_llm_names(self) -> list[str]:
        """
        List the names of the reflection agents.
        """
        return list(self.config.reflection_agent_llms.keys())

    def list_evolution_llm_names(self) -> list[str]:
        """
        List the names of the evolution agents.
        """
        return list(self.config.evolution_agent_llms.keys())

    def list_evolution_modes(self) -> list[str]:
        """
        List the names of the evolution modes.
        """
        return ["evolve_from_feedback", "out_of_the_box"]

    def list_specialist_fields(self) -> list[str]:
        """
        List the names of the specialist fields.
        """
        return self.config.specialist_fields

    def list_reasoning_types(self) -> list[str]:
        """
        List the names of the reasoning types.
        """
        return list(ReasoningType.__members__.keys())

    def get_semantic_communities(
        self, resolution: float = 1.0, min_weight: float = 0.85
    ) -> list[set[str]]:
        """
        Get the semantic communities of the hypotheses.
        """
        self.state_manager.proximity_graph.update_edges()
        return self.state_manager.proximity_graph.get_semantic_communities(
            resolution=resolution, min_weight=min_weight
        )

    def _deep_verify(self, initial_reflection_state):
        """Run deep verification for one hypothesis (with retries). Pure w.r.t.
        shared state — returns the final reflection state or RETRY_FAILED — so it
        is safe to call concurrently from a worker thread."""
        def _attempt(initial=initial_reflection_state):
            llm_name = random.choice(self.list_reflection_llm_names())
            reflection_agent = build_deep_verification_agent(
                llm=self.config.reflection_agent_llms[llm_name],
                review_llm=self.config.meta_review_agent_llm,
                parallel=False,
                checkpointer=None,
                retriever=self.config.retriever,
                web_research_enabled=self.config.web_research_enabled,
                max_assumptions=self.config.max_assumptions_researched,
            )
            return reflection_agent.invoke(initial)

        return retry_call(_attempt, label="Deep verification")

    def process_reflection_queue(self) -> None:
        """
        Process all hypotheses in the reflection queue through deep verification.

        Web-backed reflection (a GPTResearcher call per assumption) is the run's
        bottleneck, so hypotheses are verified CONCURRENTLY with a bounded thread
        pool (``config.reflection_concurrency``). Queue draining and every
        state-manager mutation stay on the calling thread; only the heavy,
        state-free ``invoke()`` runs in workers. Results are applied in queue
        order so behaviour matches the previous sequential loop.
        """
        # Drain the queue up front (single-threaded); nothing enqueues during
        # reflection, so this is equivalent to the old pop-as-you-go loop.
        initial_states = []
        while not self.state_manager.reflection_queue_is_empty:
            initial_states.append(self.state_manager.next_reflection_state())
        if not initial_states:
            return

        workers = min(self.config.reflection_concurrency, len(initial_states))
        if workers <= 1:
            results = [self._deep_verify(s) for s in initial_states]
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(self._deep_verify, initial_states))

        # Apply outcomes sequentially on the calling thread, in queue order.
        for final_reflection_state in results:
            # On repeated failure, skip this hypothesis rather than abort the run.
            if final_reflection_state is RETRY_FAILED:
                self.state_manager.record_execution_issue("Deep verification skipped")
                continue
            if final_reflection_state["passed_initial_filter"]:
                self.state_manager.add_reviewed_hypothesis(
                    final_reflection_state["reviewed_hypothesis"]
                )
                self.state_manager.advance_reviewed_hypothesis()

    def _build_independent_config(self) -> IndependentConfig:
        """Build a randomized independent-generation config, grounded when a
        private-corpus retriever is configured."""
        llm_name = random.choice(self.list_generation_llm_names())
        reasoning_type = random.choice(self.list_reasoning_types())
        specialist_field = random.choice(self.list_specialist_fields())
        return IndependentConfig(
            llm=self.config.generation_agent_llms[llm_name],
            reasoning_type=getattr(ReasoningType, reasoning_type),
            field=specialist_field,
            retriever=self.config.retriever,
        )

    def _build_collaborative_config(self) -> CollaborativeConfig:
        """Build a randomized collaborative-generation config, grounded when a
        private-corpus retriever is configured."""
        llm_names = np.random.choice(self.list_generation_llm_names(), 2).tolist()
        specialist_fields = np.random.choice(self.list_specialist_fields(), 2).tolist()
        reasoning_types = np.random.choice(self.list_reasoning_types(), 2).tolist()

        agent_names = [
            f"{llm_name}_{field}"
            for llm_name, field in zip(llm_names, specialist_fields)
        ]
        return CollaborativeConfig(
            agent_names=agent_names,
            agent_fields=dict(zip(agent_names, specialist_fields)),
            agent_reasoning_types={
                name: getattr(ReasoningType, reasoning_type)
                for name, reasoning_type in zip(agent_names, reasoning_types)
            },
            llms={
                name: self.config.generation_agent_llms[llm_name]
                for name, llm_name in zip(agent_names, llm_names)
            },
            max_turns=10,
            retriever=self.config.retriever,
        )

    def _generate_new_hypothesis(self) -> bool:
        """
        Generate one hypothesis, retrying on empty/unparseable model output.

        A model call can occasionally return empty content (e.g. it spent the
        turn "thinking" and produced no answer). Rather than let that crash the
        whole tournament, we retry a few times with a fresh call and, if all
        attempts fail, return False so the caller can skip this one hypothesis.

        Returns True if a hypothesis was produced and stored, False otherwise.
        """
        # TODO: The mode and roles should be selected by the supervisor agent.
        def _attempt():
            # Randomly pick a mode, reasoning type and specialist field each try;
            # a fresh call usually recovers from a transient empty response.
            mode = random.choice(self.list_generation_modes())
            if mode == "independent":
                config = self._build_independent_config()
                first_agent_name = None
            else:
                config = self._build_collaborative_config()
                first_agent_name = config.agent_names[0]

            generation_agent = build_generation_agent(mode, config)
            initial_generation_state = self.state_manager.next_generation_state(
                mode, first_agent_name
            )
            final_generation_state = generation_agent.invoke(initial_generation_state)
            hypothesis = final_generation_state.get("hypothesis")
            if hypothesis is None or not str(
                getattr(hypothesis, "hypothesis", "")
            ).strip():
                raise ValueError("generation returned an empty hypothesis")
            return hypothesis

        hypothesis = retry_call(
            _attempt, label="Generation", attempts=_GENERATION_MAX_ATTEMPTS
        )
        if hypothesis is RETRY_FAILED:
            self.state_manager.record_execution_issue("Generation skipped")
            return False
        self.state_manager.add_generated_hypothesis(hypothesis)
        return True

    async def start(self, n_hypotheses: int = 8) -> None:
        """
        Starts the Coscientist system with a fixed number of initial
        hypotheses.
        """
        assert n_hypotheses >= 2, "Must generate at least two hypotheses to start"
        if self.state_manager.is_started:
            raise ValueError(
                "Coscientist system has already been started. "
                f"Use one of {self.available_actions()} instead!"
            )

        # Perform the initial literature review.
        if not self.state_manager.has_literature_review:
            if not self.config.web_research_enabled:
                from coscientist.corpus.literature import build_corpus_literature_review

                final_lit_review_state = build_corpus_literature_review(
                    goal=self.state_manager.goal,
                    retriever=self.config.retriever,
                    k=getattr(self.config.retriever, "default_k", 6)
                    if self.config.retriever is not None
                    else 6,
                )
            else:
                literature_review_agent = build_literature_review_agent(
                    self.config.literature_review_agent_llm
                )
                initial_lit_review_state = self.state_manager.next_literature_review_state(
                    max_subtopics=self.config.literature_subtopics
                )
                final_lit_review_state = await literature_review_agent.ainvoke(
                    initial_lit_review_state
                )
            self.state_manager.update_literature_review(final_lit_review_state)

        # TODO: Make this async
        _ = await self.generate_new_hypotheses(
            n_hypotheses=max(0, n_hypotheses - self.state_manager.total_hypotheses)
        )

        # Guard against an empty tournament (all generations failed or every
        # hypothesis was desk-rejected) — fail clearly instead of a cryptic
        # math domain error downstream.
        if self.state_manager.num_tournament_hypotheses == 0:
            raise RuntimeError(
                "No hypotheses reached the tournament (generation returned empty "
                "output or all candidates were filtered). Try rerunning, lowering "
                "the thinking level, or switching the generation model."
            )

        # Run the EloTournament
        # The top k for the bracket should the nearest power of
        # 2 less than the number of hypotheses and no more than 16.
        k_bracket = min(16, 2 ** math.floor(math.log2(n_hypotheses)))
        # TODO: Figure out the right LLM for this job; should it be different from meta-review?
        # Feels like it should be fixed for the sake of consistency though
        _ = await self.run_tournament(k_bracket=k_bracket)
        _ = await self.run_meta_review(k_bracket=k_bracket)

    async def generate_new_hypotheses(self, n_hypotheses: int = 2) -> int:
        """
        Generate up to ``n_hypotheses`` new hypotheses.

        Each slot is generated with retries; a slot that keeps failing is
        skipped rather than aborting the run. Returns the number produced.
        """
        produced = 0
        for _ in range(n_hypotheses):
            if self._generate_new_hypothesis():
                self.state_manager.advance_hypothesis(kind="generated")
                produced += 1

        # Now run through the review queue and perform deep verification
        self.process_reflection_queue()
        if produced:
            self.state_manager.update_proximity_graph_edges()
        return produced

    async def evolve_hypotheses(self, n_hypotheses: int = 4) -> None:
        """
        Takes the top (n_hypotheses // 2) hypotheses and evolves them. Also
        randomly selects (n_hypotheses // 2) hypotheses to evolve.
        """
        assert n_hypotheses >= 2, "Must evolve at least two hypotheses"
        assert self.state_manager.is_started, "Coscientist system must be started first"
        evolution_candidate_uids = (
            self.state_manager.get_tournament_hypotheses_for_evolution()
        )
        if len(evolution_candidate_uids) < n_hypotheses:
            logging.warning(
                f"Only {len(evolution_candidate_uids)} hypotheses are qualified for evolution. "
                f"Evolving {len(evolution_candidate_uids)} hypotheses."
            )
            n_hypotheses = len(evolution_candidate_uids)

        # The first uids are the top ranked hypotheses
        top_ranked_uids = evolution_candidate_uids[: (n_hypotheses // 2)]
        # The rest are randomly selected
        random_uids = np.random.choice(
            evolution_candidate_uids[(n_hypotheses // 2) :],
            size=n_hypotheses // 2,
            replace=False,
        ).tolist()

        # Evolve the top ranked and random hypotheses based on feedback
        for uid in top_ranked_uids + random_uids:
            initial_evolution_state = self.state_manager.next_evolution_state(
                mode="evolve_from_feedback", uid_to_evolve=uid
            )

            def _attempt(initial=initial_evolution_state):
                llm_name = random.choice(self.list_evolution_llm_names())
                evolution_agent = build_evolution_agent(
                    mode="evolve_from_feedback",
                    llm=self.config.evolution_agent_llms[llm_name],
                )
                return evolution_agent.invoke(initial)

            final_evolution_state = retry_call(_attempt, label="Evolution (feedback)")
            if final_evolution_state is RETRY_FAILED:
                self.state_manager.record_execution_issue("Evolution skipped")
                continue  # skip this one, keep evolving the rest
            self.state_manager.add_evolved_hypothesis(
                final_evolution_state["evolved_hypothesis"]
            )
            self.state_manager.advance_hypothesis(kind="evolved")

        # Run one round instance of evolving the top ranked hypotheses
        # into something new
        out_of_box_initial_state = self.state_manager.next_evolution_state(
            mode="out_of_the_box",
            top_k=n_hypotheses // 2,
        )

        def _attempt_oob():
            llm_name = random.choice(self.list_evolution_llm_names())
            evolution_agent = build_evolution_agent(
                mode="out_of_the_box", llm=self.config.evolution_agent_llms[llm_name]
            )
            return evolution_agent.invoke(out_of_box_initial_state)

        out_of_box_state = retry_call(_attempt_oob, label="Evolution (out-of-the-box)")
        if out_of_box_state is RETRY_FAILED:
            self.state_manager.record_execution_issue("Evolution skipped")
        else:
            self.state_manager.add_evolved_hypothesis(
                out_of_box_state["evolved_hypothesis"]
            )
            # Move the out-of-the-box hypothesis to the reflection queue.
            self.state_manager.advance_hypothesis(kind="evolved")

        # TODO: Do we have to worry about reflecting on hypotheses that are
        # already in the reflection queue but weren't advanced yet?
        # Do we always want to run reflection immediately after a hypothesis
        # is generated?
        self.process_reflection_queue()

        # Move the reviewed hypothesis to the EloTournament.
        self.state_manager.update_proximity_graph_edges()

    async def expand_literature_review(self) -> None:
        """
        Expands the literature review by adding more subtopics.
        """
        if not self.config.web_research_enabled:
            from coscientist.corpus.literature import build_corpus_literature_review

            literature_state = build_corpus_literature_review(
                goal=self.state_manager.goal,
                retriever=self.config.retriever,
                k=getattr(self.config.retriever, "default_k", 6)
                if self.config.retriever is not None
                else 6,
            )
            self.state_manager.update_literature_review(literature_state)
            return
        initial_lit_review_state = self.state_manager.next_literature_review_state(
            max_subtopics=self.config.literature_subtopics
        )
        literature_review_agent = build_literature_review_agent(
            self.config.literature_review_agent_llm
        )
        final_lit_review_state = await literature_review_agent.ainvoke(
            initial_lit_review_state
        )
        self.state_manager.update_literature_review(final_lit_review_state)

    async def run_tournament(self, k_bracket: int = 8) -> None:
        k_bracket = min(
            k_bracket,
            2 ** math.floor(math.log2(self.state_manager.num_tournament_hypotheses)),
        )
        self.state_manager.run_tournament(
            llm=self.config.ranking_agent_llm, k_bracket=k_bracket
        )

    async def run_meta_review(self, k_bracket: int = 8) -> None:
        initial_meta_review_state = self.state_manager.next_meta_review_state(
            top_k=k_bracket
        )

        def _attempt():
            meta_review_agent = build_meta_review_agent(
                self.config.meta_review_agent_llm
            )
            return meta_review_agent.invoke(initial_meta_review_state)

        final_meta_review_state = retry_call(_attempt, label="Meta-review")
        if final_meta_review_state is RETRY_FAILED:
            self.state_manager.record_execution_issue("Meta-review failed (placeholder)")
            # Keep a placeholder so the run can proceed and finish.
            final_meta_review_state = {
                **initial_meta_review_state,
                "result": "(meta-review unavailable after retries)",
            }
        self.state_manager.update_meta_review(final_meta_review_state)

    def _assessment_llm(self) -> BaseChatModel:
        """LLM used for structured assessment (assessment_llm or supervisor)."""
        return self.config.assessment_llm or self.config.supervisor_agent_llm

    def assess_hypotheses(
        self,
        hypotheses: Optional[list[str]] = None,
        *,
        top_k: int = 5,
    ) -> list[HypothesisAssessment]:
        """
        Produce structured, cited, ranked assessments for the top hypotheses.

        Grounds each assessment in the private corpus (``config.retriever``) and
        scores it with ``config.assessment_weights``. When ``hypotheses`` is not
        given, the top-``top_k`` hypothesis statements are read from the
        tournament. Results are stored on the state and returned ranked.
        """
        if hypotheses is None:
            hypotheses = self.state_manager.top_tournament_hypotheses(top_k)
        llm = self._assessment_llm()
        web_sources = extract_web_literature_sources(
            self.state_manager.literature_review_reports
        )

        assessments = []
        for h in hypotheses:
            result = retry_call(
                lambda h=h: assess_hypothesis(
                    h,
                    goal=self.state_manager.goal,
                    retriever=self.config.retriever,
                    llm=llm,
                    weights=self.config.assessment_weights,
                    constraints=self.state_manager.constraints,
                    web_sources=web_sources,
                ),
                label="Hypothesis assessment",
            )
            # Never drop a hypothesis: fall back to a minimal assessment.
            if result is RETRY_FAILED:
                self.state_manager.record_execution_issue("Assessment fallback (minimal)")
                result = HypothesisAssessment(hypothesis=h)
            assessments.append(result)
        ranked = rank_assessments(assessments)
        self.state_manager.set_assessments(ranked)
        return ranked

    def export_results(self, out_dir: str) -> dict[str, str]:
        """
        Write business deliverables (report + tasks + graph) from the stored
        assessments. Runs assessment first if none are stored yet.
        """
        import json as _json
        import os as _os

        from coscientist import viz
        from coscientist.export import (
            assessments_to_csv,
            assessments_to_jira,
            assessments_to_json,
            write_report,
        )

        assessments = self.state_manager.assessments
        if not assessments:
            assessments = self.assess_hypotheses()

        _os.makedirs(out_dir, exist_ok=True)
        paths = write_report(assessments, out_dir, goal=self.state_manager.goal)

        def _w(name: str, text: str) -> str:
            p = _os.path.join(out_dir, name)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text)
            return p

        paths["csv"] = _w("tasks.csv", assessments_to_csv(assessments))
        paths["json"] = _w("tasks.json", assessments_to_json(assessments))
        paths["jira"] = _w(
            "jira_tasks.json",
            _json.dumps(assessments_to_jira(assessments), ensure_ascii=False, indent=2),
        )
        paths["graph"] = _w("graph.html", viz.to_html(assessments))
        return paths

    async def finish(self) -> None:
        # Structured, cited assessment of the top hypotheses (case output shape).
        self.assess_hypotheses(top_k=min(self.state_manager.num_tournament_hypotheses, 10))

        initial_final_report_state = self.state_manager.next_final_report_state(top_k=3)

        def _attempt():
            final_report_agent = build_final_report_agent(
                self.config.final_report_agent_llm
            )
            return final_report_agent.invoke(initial_final_report_state)

        final_report_state = retry_call(_attempt, label="Final report")
        if final_report_state is RETRY_FAILED:
            self.state_manager.record_execution_issue("Final report failed (placeholder)")
            final_report_state = {
                **initial_final_report_state,
                "result": "(final report unavailable after retries)",
            }
        self.state_manager.update_final_report(final_report_state)

    @classmethod
    def available_actions(self) -> list[str]:
        """
        List the available actions for the Coscientist system.
        """
        return [
            "generate_new_hypotheses",
            "evolve_hypotheses",
            "expand_literature_review",
            "run_tournament",
            "run_meta_review",
            "finish",
        ]

    async def run(self, n_hypotheses: int = 4) -> tuple[str, str]:
        """
        Runs the coscientist system until it is finished.

        Parameters
        ----------
        n_hypotheses : int
            Number of initial hypotheses to seed the tournament with.
        """
        if not self.state_manager.is_started:
            _ = await self.start(n_hypotheses=max(2, n_hypotheses))

        supervisor_agent = build_supervisor_agent(self.config.supervisor_agent_llm)

        while not self.state_manager.is_finished:
            # Fold any live steering the human sent since the last step into the
            # shared state (chat-to-running-agents). Applied at this boundary — the
            # in-flight step is never interrupted; subsequent agents see it.
            for message in _collect_guidance(self.config.pull_messages):
                self.state_manager.add_guidance(message)
                logging.info("Live guidance folded in: %s", message[:200])

            budget = self.config.max_supervisor_actions
            initial_supervisor_state = self.state_manager.next_supervisor_state(
                max_actions=budget, max_hypotheses=self.config.max_total_hypotheses
            )

            # Deterministic termination guards (do NOT rely on the LLM choosing
            # "finish"): spent budget, a settled tournament, or a runaway
            # evolve-loop. Each guarantees the run reaches finish() in bounded steps.
            forced_reason = _termination_reason(
                has_clear_leader=initial_supervisor_state.get("has_clear_leader") == "yes",
                is_plateau=initial_supervisor_state.get("is_plateau") == "yes",
                recent_actions=self.state_manager.recent_actions(
                    self.config.max_consecutive_evolve
                ),
                num_actions=self.state_manager.num_actions,
                budget=budget,
                max_consecutive_evolve=self.config.max_consecutive_evolve,
            )
            if forced_reason is not None:
                logging.warning("Forced finish: %s", forced_reason)
                final_supervisor_state = {
                    **initial_supervisor_state,
                    "action": "finish",
                    "decision_reasoning": forced_reason,
                }
                self.state_manager.update_supervisor_decision(final_supervisor_state)
                self.state_manager.add_action("finish")
                _ = await self.finish()
                break

            def _attempt(initial=initial_supervisor_state):
                state = supervisor_agent.invoke(initial)
                if state.get("action") not in self.available_actions():
                    raise ValueError(f"Invalid supervisor action: {state.get('action')}")
                return state

            final_supervisor_state = retry_call(_attempt, label="Supervisor decision")
            if final_supervisor_state is RETRY_FAILED:
                self.state_manager.record_execution_issue(
                    "Supervisor decision failed (forced finish)"
                )
                # Graceful termination: if the supervisor can't produce a valid
                # action, finish the run with whatever we have.
                final_supervisor_state = {
                    **initial_supervisor_state,
                    "action": "finish",
                    "decision_reasoning": "Supervisor failed to decide; finishing gracefully.",
                }
            current_action = final_supervisor_state["action"]

            # Deterministic hypothesis-pool cap: never let generate/evolve grow
            # the pool past max_total_hypotheses — redirect to ranking/finishing.
            capped_action = _hypothesis_cap_override(
                current_action,
                total=self.state_manager.total_hypotheses,
                max_total=self.config.max_total_hypotheses,
                num_unranked=self.state_manager.num_unranked_hypotheses,
            )
            if capped_action is not None and capped_action != current_action:
                logging.warning(
                    "Hypothesis cap (%d) reached; %s -> %s",
                    self.config.max_total_hypotheses, current_action, capped_action,
                )
                final_supervisor_state = {
                    **final_supervisor_state,
                    "action": capped_action,
                    "decision_reasoning": (
                        f"Hypothesis cap of {self.config.max_total_hypotheses} reached; "
                        f"{capped_action} instead of {current_action}."
                    ),
                }
                current_action = capped_action

            self.state_manager.update_supervisor_decision(final_supervisor_state)
            self.state_manager.add_action(current_action)
            _ = await getattr(self, current_action)()

        return self.state_manager.final_report, self.state_manager.meta_review
