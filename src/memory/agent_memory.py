"""
Tier 2: Agent Memory Store (SQLite)

Persistent memory for the multi-agent system that logs:
- User interactions (queries, answers, agents involved)
- Agent decisions (individual agent choices and reasoning)
- Learned patterns (security threats, query patterns, etc.)

This enables:
- Cross-session memory
- Pattern recognition
- Agent collaboration history
- Query analysis and optimization

Usage:
    from src.memory.agent_memory import AgentMemoryStore
    
    memory = AgentMemoryStore()
    
    # Log an interaction
    interaction_id = memory.log_interaction(
        query="What's the weather in Seattle?",
        answer="Currently 65°F and sunny...",
        agents=["SecurityAgent", "IntentAgent", "IPAgent", "WeatherAgent"],
        success=True,
        time_ms=2500
    )
    
    # Log agent decisions
    memory.log_agent_decision(
        interaction_id=interaction_id,
        agent_name="SecurityAgent",
        decision="ALLOW",
        reasoning="No threats detected in query"
    )
    
    # Learn patterns
    memory.learn_pattern(
        pattern_type="security_threat",
        pattern_text="sql injection attempt",
        metadata={"severity": "high", "blocked": True}
    )
    
    # Retrieve similar queries
    similar = memory.get_similar_past_queries("weather in Portland")
"""

import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AgentMemoryStore:
    """Persistent SQLite-based memory for multi-agent system."""
    
    def __init__(self, db_path: str = "memory/agent_memory.db"):
        """
        Initialize memory store with SQLite database.
        
        Args:
            db_path: Path to SQLite database file (created if doesn't exist)
        """
        # Create memory directory if it doesn't exist
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to database (thread-safe for multi-agent access)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row  # Enable column access by name
        
        self._init_schema()
        logger.info(f"AgentMemoryStore initialized at {db_path}")
    
    def _init_schema(self):
        """Create database tables for different memory types."""
        
        # Table 1: Interactions (complete user interactions)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_query TEXT NOT NULL,
                final_answer TEXT,
                agents_involved TEXT,  -- JSON array of agent names
                was_successful BOOLEAN,
                execution_time_ms INTEGER,
                error_message TEXT,
                security_blocked BOOLEAN DEFAULT 0,
                intent_rejected BOOLEAN DEFAULT 0
            )
        """)
        
        # Index for faster query searches
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_interactions_timestamp 
            ON interactions(timestamp DESC)
        """)
        
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_interactions_success 
            ON interactions(was_successful)
        """)
        
        # Table 2: Agent Decisions (individual agent choices)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS agent_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER,
                agent_name TEXT NOT NULL,
                decision TEXT NOT NULL,
                reasoning TEXT,
                timestamp TEXT NOT NULL,
                execution_time_ms INTEGER,
                FOREIGN KEY (interaction_id) REFERENCES interactions(id)
            )
        """)
        
        # Index for agent performance analysis
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_decisions_agent 
            ON agent_decisions(agent_name, timestamp DESC)
        """)
        
        # Table 3: Learned Patterns (incremental learning)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS learned_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,  -- 'security_threat', 'query_pattern', 'weather_location', etc.
                pattern_text TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                metadata TEXT,  -- JSON with additional details
                UNIQUE(pattern_type, pattern_text)
            )
        """)
        
        # Index for pattern retrieval
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_patterns_type 
            ON learned_patterns(pattern_type, frequency DESC)
        """)
        
        self.db.commit()
        logger.info("Database schema initialized successfully")
    
    def log_interaction(
        self, 
        query: str, 
        answer: Optional[str], 
        agents: List[str], 
        success: bool, 
        time_ms: int,
        error_message: Optional[str] = None,
        security_blocked: bool = False,
        intent_rejected: bool = False
    ) -> int:
        """
        Log a complete user interaction.
        
        Args:
            query: User's original query
            answer: Final answer (None if failed)
            agents: List of agent names that participated
            success: Whether interaction completed successfully
            time_ms: Total execution time in milliseconds
            error_message: Error message if failed
            security_blocked: True if blocked by security agent
            intent_rejected: True if rejected by intent agent
        
        Returns:
            interaction_id: ID of logged interaction for referencing
        """
        cursor = self.db.execute("""
            INSERT INTO interactions 
            (timestamp, user_query, final_answer, agents_involved, 
             was_successful, execution_time_ms, error_message,
             security_blocked, intent_rejected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            query,
            answer,
            json.dumps(agents),
            success,
            time_ms,
            error_message,
            security_blocked,
            intent_rejected
        ))
        self.db.commit()
        
        interaction_id = cursor.lastrowid
        logger.debug(f"Logged interaction {interaction_id}: success={success}, agents={len(agents)}")
        return interaction_id
    
    def log_agent_decision(
        self, 
        interaction_id: Optional[int], 
        agent_name: str, 
        decision: str, 
        reasoning: str,
        execution_time_ms: Optional[int] = None
    ):
        """
        Log an individual agent's decision.
        
        Args:
            interaction_id: ID from log_interaction() (can be None if interaction logging failed)
            agent_name: Name of the agent (e.g., "SecurityAgent")
            decision: Agent's decision (e.g., "ALLOW", "BLOCK", "weather")
            reasoning: Why the agent made this decision
            execution_time_ms: Agent's execution time (optional)
        """
        if interaction_id is None:
            logger.warning(f"Skipping agent decision logging - no interaction_id for {agent_name}")
            return
        
        self.db.execute("""
            INSERT INTO agent_decisions 
            (interaction_id, agent_name, decision, reasoning, timestamp, execution_time_ms)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            interaction_id,
            agent_name,
            decision,
            reasoning,
            datetime.now().isoformat(),
            execution_time_ms
        ))
        self.db.commit()
        logger.debug(f"Logged {agent_name} decision: {decision}")
    
    def get_similar_past_queries(
        self, 
        query: str, 
        limit: int = 5,
        only_successful: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar past queries (simple keyword matching).
        
        Note: This is basic implementation. For semantic similarity,
        use Tier 3 RAG system.
        
        Args:
            query: Query to find similar matches for
            limit: Maximum number of results
            only_successful: Only return successful interactions
        
        Returns:
            List of similar interactions with query, answer, agents
        """
        # Extract keywords from query (simple approach)
        keywords = [word.lower() for word in query.split() if len(word) > 3]
        
        # Build SQL query with keyword matching
        where_clause = "was_successful = 1" if only_successful else "1=1"
        
        if keywords:
            keyword_conditions = " OR ".join([
                f"LOWER(user_query) LIKE '%{keyword}%'" 
                for keyword in keywords[:3]  # Use top 3 keywords
            ])
            where_clause += f" AND ({keyword_conditions})"
        
        cursor = self.db.execute(f"""
            SELECT user_query, final_answer, agents_involved, 
                   timestamp, execution_time_ms
            FROM interactions
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "query": row["user_query"],
                "answer": row["final_answer"],
                "agents": json.loads(row["agents_involved"]),
                "timestamp": row["timestamp"],
                "execution_time_ms": row["execution_time_ms"]
            })
        
        logger.debug(f"Found {len(results)} similar queries for: {query[:50]}...")
        return results
    
    def learn_pattern(
        self, 
        pattern_type: str, 
        pattern_text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Learn or increment a pattern (incremental learning).
        
        If pattern already exists, increments frequency.
        If new, creates new entry.
        
        Args:
            pattern_type: Type of pattern (e.g., "security_threat", "query_pattern")
            pattern_text: The pattern itself (e.g., "sql injection")
            metadata: Additional info (e.g., {"severity": "high"})
        """
        now = datetime.now().isoformat()
        metadata_json = json.dumps(metadata or {})
        
        # Try to update existing pattern
        cursor = self.db.execute("""
            SELECT id, frequency FROM learned_patterns
            WHERE pattern_type = ? AND pattern_text = ?
        """, (pattern_type, pattern_text))
        
        existing = cursor.fetchone()
        
        if existing:
            # Increment frequency
            new_frequency = existing["frequency"] + 1
            self.db.execute("""
                UPDATE learned_patterns
                SET frequency = ?, last_seen = ?, metadata = ?
                WHERE id = ?
            """, (new_frequency, now, metadata_json, existing["id"]))
            logger.debug(f"Updated pattern '{pattern_text}': frequency={new_frequency}")
        else:
            # Create new pattern
            self.db.execute("""
                INSERT INTO learned_patterns 
                (pattern_type, pattern_text, frequency, first_seen, last_seen, metadata)
                VALUES (?, ?, 1, ?, ?, ?)
            """, (pattern_type, pattern_text, now, now, metadata_json))
            logger.debug(f"Learned new pattern: {pattern_type} - {pattern_text}")
        
        self.db.commit()
    
    def get_top_patterns(
        self, 
        pattern_type: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get most frequent patterns of a given type.
        
        Args:
            pattern_type: Type to filter by (e.g., "security_threat")
            limit: Maximum number of results
        
        Returns:
            List of patterns sorted by frequency
        """
        cursor = self.db.execute("""
            SELECT pattern_text, frequency, first_seen, last_seen, metadata
            FROM learned_patterns
            WHERE pattern_type = ?
            ORDER BY frequency DESC
            LIMIT ?
        """, (pattern_type, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "pattern": row["pattern_text"],
                "frequency": row["frequency"],
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
                "metadata": json.loads(row["metadata"])
            })
        
        return results
    
    def get_agent_statistics(self, agent_name: str) -> Dict[str, Any]:
        """
        Get performance statistics for a specific agent.
        
        Args:
            agent_name: Name of agent (e.g., "SecurityAgent")
        
        Returns:
            Dict with stats: total_decisions, avg_time, common_decisions
        """
        # Count total decisions
        cursor = self.db.execute("""
            SELECT COUNT(*) as count, AVG(execution_time_ms) as avg_time
            FROM agent_decisions
            WHERE agent_name = ?
        """, (agent_name,))
        
        row = cursor.fetchone()
        total = row["count"] if row else 0
        avg_time = row["avg_time"] if row and row["avg_time"] else 0
        
        # Get common decisions
        cursor = self.db.execute("""
            SELECT decision, COUNT(*) as count
            FROM agent_decisions
            WHERE agent_name = ?
            GROUP BY decision
            ORDER BY count DESC
            LIMIT 5
        """, (agent_name,))
        
        common_decisions = [
            {"decision": row["decision"], "count": row["count"]}
            for row in cursor.fetchall()
        ]
        
        return {
            "agent_name": agent_name,
            "total_decisions": total,
            "average_execution_time_ms": round(avg_time, 2),
            "common_decisions": common_decisions
        }
    
    def get_recent_interactions(
        self, 
        limit: int = 10,
        only_successful: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get most recent interactions.
        
        Args:
            limit: Maximum number of results
            only_successful: Filter to only successful interactions
        
        Returns:
            List of recent interactions
        """
        where_clause = "was_successful = 1" if only_successful else "1=1"
        
        cursor = self.db.execute(f"""
            SELECT user_query, final_answer, agents_involved, 
                   was_successful, execution_time_ms, timestamp,
                   security_blocked, intent_rejected
            FROM interactions
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "query": row["user_query"],
                "answer": row["final_answer"],
                "agents": json.loads(row["agents_involved"]),
                "success": bool(row["was_successful"]),
                "execution_time_ms": row["execution_time_ms"],
                "timestamp": row["timestamp"],
                "security_blocked": bool(row["security_blocked"]),
                "intent_rejected": bool(row["intent_rejected"])
            })
        
        return results
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get overall system statistics.
        
        Returns:
            Dict with total interactions, success rate, avg time, etc.
        """
        # Total interactions
        cursor = self.db.execute("SELECT COUNT(*) as count FROM interactions")
        total = cursor.fetchone()["count"]
        
        # Success rate
        cursor = self.db.execute("""
            SELECT 
                COUNT(*) as successful_count,
                AVG(execution_time_ms) as avg_time
            FROM interactions
            WHERE was_successful = 1
        """)
        row = cursor.fetchone()
        successful = row["successful_count"] if row else 0
        avg_time = row["avg_time"] if row and row["avg_time"] else 0
        
        # Security blocks
        cursor = self.db.execute("""
            SELECT COUNT(*) as count FROM interactions WHERE security_blocked = 1
        """)
        security_blocks = cursor.fetchone()["count"]
        
        # Intent rejections
        cursor = self.db.execute("""
            SELECT COUNT(*) as count FROM interactions WHERE intent_rejected = 1
        """)
        intent_rejections = cursor.fetchone()["count"]
        
        # Most active agents
        cursor = self.db.execute("""
            SELECT agent_name, COUNT(*) as count
            FROM agent_decisions
            GROUP BY agent_name
            ORDER BY count DESC
            LIMIT 5
        """)
        top_agents = [
            {"agent": row["agent_name"], "decisions": row["count"]}
            for row in cursor.fetchall()
        ]
        
        return {
            "total_interactions": total,
            "successful_interactions": successful,
            "success_rate_percent": round((successful / total * 100) if total > 0 else 0, 2),
            "average_execution_time_ms": round(avg_time, 2),
            "security_blocks": security_blocks,
            "intent_rejections": intent_rejections,
            "top_agents": top_agents
        }
    
    def close(self):
        """Close database connection."""
        self.db.close()
        logger.info("AgentMemoryStore connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function for creating shared memory instance
_shared_memory: Optional[AgentMemoryStore] = None

def get_memory_store(db_path: str = "memory/agent_memory.db") -> AgentMemoryStore:
    """
    Get or create shared memory store instance (singleton pattern).
    
    Args:
        db_path: Path to SQLite database
    
    Returns:
        Shared AgentMemoryStore instance
    """
    global _shared_memory
    if _shared_memory is None:
        _shared_memory = AgentMemoryStore(db_path)
    return _shared_memory
