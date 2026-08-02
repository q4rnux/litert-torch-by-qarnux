"""
Base Agent Definition.

All agents in the orchestration pipeline inherit from this base class,
which provides common logging, status tracking, and message passing
interfaces.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class AgentStatus(Enum):
    """Possible states of an agent during execution."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"


@dataclass
class AgentMessage:
    """Message passed between agents in the pipeline."""
    source: str
    target: str
    data: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


class BaseAgent(abc.ABC):
    """
    Abstract base class for all pipeline agents.

    Each agent implements a single responsibility in the conversion
    pipeline and communicates results through AgentMessage objects.
    """

    def __init__(self, agent_id: str):
        """
        Initialize the agent.

        Args:
            agent_id: Unique identifier for this agent instance.
        """
        self.agent_id = agent_id
        self.status = AgentStatus.IDLE
        self.logger = logging.getLogger(f"agent.{agent_id}")

    @abc.abstractmethod
    def execute(self, message: AgentMessage) -> AgentMessage:
        """
        Execute the agent's task.

        Args:
            message: Input message containing data from a previous agent.

        Returns:
            Output message containing results for the next agent.
        """
        ...

    def start(self, message: AgentMessage) -> AgentMessage:
        """
        Start execution with proper status management.

        Args:
            message: Input message.

        Returns:
            Output message from execute().
        """
        self.status = AgentStatus.RUNNING
        self.logger.info("Agent '%s' started execution", self.agent_id)

        try:
            result = self.execute(message)
            self.status = AgentStatus.COMPLETED
            self.logger.info("Agent '%s' completed successfully", self.agent_id)
            return result
        except Exception as e:
            self.status = AgentStatus.FAILED
            self.logger.error("Agent '%s' failed: %s", self.agent_id, str(e))
            return AgentMessage(
                source=self.agent_id,
                target="orchestrator",
                success=False,
                error_message=str(e),
            )

    def get_status(self) -> AgentStatus:
        """Return the current execution status."""
        return self.status
